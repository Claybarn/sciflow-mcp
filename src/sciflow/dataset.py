"""Datasets (experiment > subject > session) and pipelines over them.

A **dataset** lists sessions. Each session has a subject, free-form
attributes (condition, day, genotype, ...) and named inputs, given either as
artifact ids or as files to import::

    name: sleep-deprivation
    sessions:
      - id: m1-d1
        subject: m1
        attributes: {condition: control, day: 1}
        inputs:
          wheel: {path: /data/m1/d1/wheel.npz}
          widefield: {path: /data/m1/d1/widefield.npz, options: {parametric_axis: 2, axes: [time, y, x]}}
          sync_img: sha256:5be1...

Registering a dataset imports any files and stores the resolved manifest
(every input an artifact id) under a content-derived ``dataset:`` id, so a
pipeline that names it is pinned to exact data.

A **pipeline** runs the same steps on every session, stacks chosen
per-session results along a ``session`` axis, and runs aggregate steps on
the stacks::

    name: running-response-by-condition
    dataset: dataset:4be0c3...
    session_steps:            # an ordinary workflow body; inputs.* come from each session
      - {id: running, op: threshold_intervals, inputs: {series: inputs.wheel}, params: {threshold: 2}}
      - ...
      - {id: response, op: epoch_average, inputs: {epochs: steps.epochs}}
    collect: {response: response}           # stack name -> session step
    aggregate_steps:
      - {id: by_condition, op: group_reduce, inputs: {series: inputs.response},
         params: {group_by: [condition], within: [subject]}}

A pipeline compiles to one flat workflow (session steps prefixed
``<session>__``), so caching, provenance, validation and the Slurm / Argo
exporters apply unchanged: sessions run in parallel on a cluster, and adding
a session to the dataset runs only that session's steps before re-aggregating.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from . import io
from .store import Store
from .workflow import Step, WorkflowSpec, _parse_ref, refs

SEP = "__"


class ImportSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str
    kind: str = "series"
    options: dict[str, Any] = Field(default_factory=dict, description="Keyword options for import_data.")


class SessionSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    subject: str | None = None
    attributes: dict[str, str | int | float | bool] = Field(default_factory=dict)
    inputs: dict[str, str | ImportSpec]


class DatasetSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    description: str = ""
    sessions: list[SessionSpec]


class PipelineSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    description: str = ""
    dataset: str | DatasetSpec = Field(description="A registered dataset id, or an inline dataset.")
    where: dict[str, list[str | int | float | bool]] = Field(
        default_factory=dict, description="Only include sessions whose attribute (or subject / id) is in the list.")
    session_steps: list[Step]
    collect: dict[str, str] = Field(description="Stack name -> session step whose output is stacked across sessions.")
    aggregate_steps: list[Step] = Field(default_factory=list,
                                        description="Steps over the stacks; reference a stack as inputs.<stack name>.")
    outputs: list[str] | None = None


def _yaml_or_dict(source) -> dict:
    if isinstance(source, dict):
        return source
    text = str(source)
    if isinstance(source, Path) or ("\n" not in text and Path(text).suffix.lower() in (".json", ".yaml", ".yml")):
        if not Path(text).is_file():
            raise FileNotFoundError(f"file {text!r} not found")
        text = Path(source).read_text()
    return yaml.safe_load(text)


def register_dataset(source: dict | str | Path | DatasetSpec, store: Store) -> str:
    spec = source if isinstance(source, DatasetSpec) else DatasetSpec.model_validate(_yaml_or_dict(source))
    ids = [s.id for s in spec.sessions]
    dups = sorted({i for i in ids if ids.count(i) > 1})
    if dups:
        raise ValueError(f"duplicate session ids: {dups}")
    sessions = []
    for s in spec.sessions:
        inputs = {}
        for name, value in s.inputs.items():
            if isinstance(value, ImportSpec):
                art = io.import_file(value.path, value.kind, **value.options)
                inputs[name] = store.put(art, provenance={"source": "import", "path": value.path, "kind": value.kind,
                                                          "options": value.options, "dataset": spec.name, "session": s.id})
            else:
                inputs[name] = store.resolve(value)
        sessions.append({"id": s.id, "subject": s.subject, "attributes": s.attributes, "inputs": inputs})
    return store.put_dataset({"name": spec.name, "description": spec.description, "sessions": sessions})


def _session_key(session_id: str, used: set[str]) -> str:
    key = re.sub(r"[^A-Za-z0-9_\-]", "_", session_id).replace(SEP, "_")
    if not re.match(r"[A-Za-z_]", key):
        key = "s_" + key
    base, i = key, 1
    while key in used:
        i += 1
        key = f"{base}_{i}"
    used.add(key)
    return key


def _session_matches(s: dict, where: dict) -> bool:
    for attr, allowed in where.items():
        value = s["id"] if attr == "session" else s["subject"] if attr == "subject" else s["attributes"].get(attr)
        if value not in allowed:
            return False
    return True


def is_pipeline(source) -> bool:
    if isinstance(source, (WorkflowSpec, PipelineSpec)):
        return isinstance(source, PipelineSpec)
    try:
        d = _yaml_or_dict(source)
    except Exception:
        return False
    return isinstance(d, dict) and "session_steps" in d


def load_pipeline(source) -> PipelineSpec:
    return source if isinstance(source, PipelineSpec) else PipelineSpec.model_validate(_yaml_or_dict(source))


def compile_pipeline(source, store: Store) -> WorkflowSpec:
    """Expand a pipeline into a flat workflow over every selected session."""
    p = load_pipeline(source)
    dataset = store.get_dataset(p.dataset) if isinstance(p.dataset, str) else store.get_dataset(register_dataset(p.dataset, store))
    sessions = [s for s in dataset["sessions"] if _session_matches(s, p.where)]
    if not sessions:
        raise ValueError(f"no sessions of dataset {dataset['name']!r} match {p.where}")

    for step in p.aggregate_steps:
        if SEP in step.id:
            raise ValueError(f"aggregate step id {step.id!r} must not contain {SEP!r}")
    for name in p.collect:
        if SEP in name:
            raise ValueError(f"stack name {name!r} must not contain {SEP!r}")
    session_step_ids = {s.id for s in p.session_steps}
    for name, sid in p.collect.items():
        if sid not in session_step_ids:
            raise ValueError(f"collect {name!r}: no session step {sid!r}")

    needed = sorted({_parse_ref(r)[1] for st in p.session_steps for v in st.inputs.values() for r in refs(v)
                     if r.startswith("inputs.")})
    used: set[str] = set()
    inputs: dict[str, str] = {}
    steps: list[Step] = []
    keys = []
    for s in sessions:
        key = _session_key(s["id"], used)
        keys.append(key)
        missing = [n for n in needed if n not in s["inputs"]]
        if missing:
            raise ValueError(f"session {s['id']!r} has no inputs {missing}; it has {sorted(s['inputs'])}")
        for n in needed:
            inputs[f"{key}{SEP}{n}"] = s["inputs"][n]

        def rewrite(ref: str, key=key) -> str:
            scope, name = _parse_ref(ref)
            return f"{scope}.{key}{SEP}{name}"

        for st in p.session_steps:
            new_inputs = {port: [rewrite(r) for r in v] if isinstance(v, list) else rewrite(v) for port, v in st.inputs.items()}
            steps.append(Step(id=f"{key}{SEP}{st.id}", op=st.op, inputs=new_inputs, params=st.params))

    attr_names = sorted({a for s in sessions for a in s["attributes"]})
    attributes = {"subject": [s["subject"] for s in sessions],
                  **{a: [s["attributes"].get(a) for s in sessions] for a in attr_names}}
    for name, sid in p.collect.items():
        steps.append(Step(id=name, op="stack_sessions",
                          inputs={"items": [f"steps.{k}{SEP}{sid}" for k in keys]},
                          params={"sessions": [s["id"] for s in sessions], "attributes": attributes}))

    def rewrite_agg(ref: str) -> str:
        scope, name = _parse_ref(ref)
        return f"steps.{name}" if scope == "inputs" else ref

    for st in p.aggregate_steps:
        new_inputs = {port: [rewrite_agg(r) for r in v] if isinstance(v, list) else rewrite_agg(v) for port, v in st.inputs.items()}
        steps.append(Step(id=st.id, op=st.op, inputs=new_inputs, params=st.params))

    outputs = p.outputs
    if outputs is None:
        used_agg = {_parse_ref(r)[1] for st in p.aggregate_steps for v in st.inputs.values() for r in refs(v)}
        agg_leaves = [st.id for st in p.aggregate_steps if st.id not in used_agg]
        outputs = agg_leaves or list(p.collect)
    return WorkflowSpec(name=p.name, description=p.description + f" [dataset {dataset['id']}, {len(sessions)} sessions]",
                        inputs=inputs, steps=steps, outputs=outputs)


def load_any(source, store: Store) -> WorkflowSpec:
    """A workflow, or a pipeline compiled to one."""
    from .workflow import load_spec

    if is_pipeline(source):
        return compile_pipeline(source, store)
    return load_spec(source)


def run_any(source, store: Store, force: bool = False) -> dict[str, Any]:
    """Run a workflow or a pipeline; pipeline runs record the pipeline and dataset in their provenance."""
    from .workflow import run

    if not is_pipeline(source):
        return run(source, store, force=force)
    p = load_pipeline(source)
    spec = compile_pipeline(p, store)
    dataset_id = p.dataset if isinstance(p.dataset, str) else register_dataset(p.dataset, store)
    extra = {"pipeline": p.model_dump(mode="json"), "dataset": store.get_dataset(dataset_id)["id"]}
    return run(spec, store, force=force, extra=extra)
