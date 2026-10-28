"""Workflow specs, static validation and execution.

A workflow is a DAG of steps. Each step names an operation, binds each of
the operation's inputs to either a workflow input (``inputs.<name>``, an
artifact id) or another step's output (``steps.<id>``), and supplies
parameters. Example (YAML or JSON)::

    name: running-locked-lfp
    inputs:
      wheel: sha256:3f2a...
      lfp: sha256:91bc...
    steps:
      - id: running
        op: threshold_intervals
        inputs: {series: inputs.wheel}
        params: {threshold: 2.0}
      - id: epochs
        op: peri_event
        inputs: {series: inputs.lfp, events: steps.running}
        params: {pre: 1.0, post: 2.0}
      - id: response
        op: epoch_average
        inputs: {epochs: steps.epochs}

Every step's cache key is a hash of (operation, operation version,
parameters, input artifact ids). Because artifact ids are content hashes,
re-running a workflow on the same data reuses results, and any two machines
that produce the same key produce the same artifact.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import platform
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError as PydanticValidationError

from . import ops  # noqa: F401  (registers operations)
from .registry import get_op, REGISTRY
from .store import Store, canonical_json

FORMAT_VERSION = 1


class Step(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_\-]*$")
    op: str
    # A reference, or a list of references for ports declared as "<kind>[]".
    inputs: dict[str, str | list[str]] = Field(default_factory=dict)
    params: dict[str, Any] = Field(default_factory=dict)


def refs(value: str | list[str]) -> list[str]:
    return list(value) if isinstance(value, list) else [value]


class WorkflowSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    description: str = ""
    inputs: dict[str, str] = Field(default_factory=dict, description="Workflow input name -> artifact id.")
    steps: list[Step]
    outputs: list[str] | None = Field(None, description="Step ids whose results are the workflow's outputs (default: leaf steps).")

    def step(self, step_id: str) -> Step:
        for s in self.steps:
            if s.id == step_id:
                return s
        raise KeyError(f"no step {step_id!r}")


def load_spec(source: str | Path | dict | WorkflowSpec) -> WorkflowSpec:
    if isinstance(source, WorkflowSpec):
        return source
    if isinstance(source, dict):
        return WorkflowSpec.model_validate(source)
    text = str(source)
    if isinstance(source, Path) or ("\n" not in text and Path(text).suffix.lower() in (".json", ".yaml", ".yml")):
        if not Path(text).is_file():
            raise FileNotFoundError(f"workflow file {text!r} not found")
        text = Path(source).read_text()
    return WorkflowSpec.model_validate(yaml.safe_load(text))


def _parse_ref(ref: str) -> tuple[str, str]:
    scope, _, name = ref.partition(".")
    if scope not in ("inputs", "steps") or not name:
        raise ValueError(f"reference {ref!r} must look like 'inputs.<name>' or 'steps.<id>'")
    return scope, name


def _kinds_compatible(expected: str, actual: str) -> bool:
    expected = expected.removesuffix("[]")
    return expected == "any" or actual == "any" or expected == actual


def validate(spec: WorkflowSpec, store: Store) -> tuple[list[str], list[str], dict[str, str]]:
    """Statically check a workflow without running it.

    Returns ``(errors, execution order, output kind of each step)``.
    """
    errors: list[str] = []
    input_kinds: dict[str, str] = {}
    for name, ref in spec.inputs.items():
        try:
            input_kinds[name] = store.header(ref)["kind"]
        except KeyError as e:
            errors.append(f"input {name!r}: {e.args[0]}")
            input_kinds[name] = "any"

    ids = [s.id for s in spec.steps]
    for dup in {i for i in ids if ids.count(i) > 1}:
        errors.append(f"duplicate step id {dup!r}")
    by_id = {s.id: s for s in spec.steps}

    deps: dict[str, set[str]] = {}
    for s in spec.steps:
        deps[s.id] = set()
        for port, value in s.inputs.items():
            for ref in refs(value):
                try:
                    scope, name = _parse_ref(ref)
                except ValueError as e:
                    errors.append(f"step {s.id!r} input {port!r}: {e}")
                    continue
                if scope == "inputs" and name not in spec.inputs:
                    errors.append(f"step {s.id!r} input {port!r}: workflow has no input {name!r}")
                elif scope == "steps":
                    if name not in by_id:
                        errors.append(f"step {s.id!r} input {port!r}: no step {name!r}")
                    else:
                        deps[s.id].add(name)

    # Kahn's algorithm, keeping spec order among ready steps for stable output.
    order: list[str] = []
    remaining = {k: set(v) for k, v in deps.items()}
    while True:
        ready = [i for i in ids if i in remaining and not remaining[i]]
        if not ready:
            break
        for r in ready:
            order.append(r)
            del remaining[r]
            for v in remaining.values():
                v.discard(r)
    if remaining:
        errors.append(f"cycle detected among steps {sorted(remaining)}")

    step_kinds: dict[str, str] = {}
    for sid in order:
        s = by_id[sid]
        try:
            spec_op = get_op(s.op)
        except KeyError as e:
            errors.append(f"step {sid!r}: {e.args[0]}")
            step_kinds[sid] = "any"
            continue
        missing = set(spec_op.inputs) - set(s.inputs)
        extra = set(s.inputs) - set(spec_op.inputs)
        if missing:
            errors.append(f"step {sid!r} ({s.op}): missing inputs {sorted(missing)}")
        if extra:
            errors.append(f"step {sid!r} ({s.op}): unknown inputs {sorted(extra)}; expected {sorted(spec_op.inputs)}")
        actual: dict[str, str] = {}
        for port, value in s.inputs.items():
            if port not in spec_op.inputs:
                continue
            expected = spec_op.inputs[port]
            if expected.endswith("[]") != isinstance(value, list):
                errors.append(f"step {sid!r} ({s.op}): input {port!r} takes "
                              + ("a list of references" if expected.endswith("[]") else "a single reference"))
                continue
            for ref in refs(value):
                try:
                    scope, name = _parse_ref(ref)
                except ValueError:
                    continue
                kind = input_kinds.get(name, "any") if scope == "inputs" else step_kinds.get(name, "any")
                actual[port] = kind
                if not _kinds_compatible(expected, kind):
                    errors.append(f"step {sid!r} ({s.op}): input {port!r} expects {expected.removesuffix('[]')} "
                                  f"but {ref} is {kind}")
        try:
            spec_op.params.model_validate(s.params)
        except PydanticValidationError as e:
            for err in e.errors():
                loc = ".".join(str(x) for x in err["loc"]) or "(params)"
                errors.append(f"step {sid!r} ({s.op}): param {loc}: {err['msg']}")
        step_kinds[sid] = spec_op.output_kind(actual)

    for o in spec.outputs or []:
        if o not in by_id:
            errors.append(f"output {o!r} is not a step id")
    return errors, order, step_kinds


def step_key(step: Step, input_ids: dict[str, str | list[str]]) -> str:
    spec_op = get_op(step.op)
    params = spec_op.params.model_validate(step.params).model_dump(mode="json")
    payload = {"format": FORMAT_VERSION, "op": step.op, "version": spec_op.version, "params": params, "inputs": input_ids}
    return hashlib.sha256(canonical_json(payload).encode()).hexdigest()


def environment() -> dict[str, Any]:
    versions = {}
    for pkg in ("sciflow-mcp", "eventlogic", "numpy", "scipy", "pydantic"):
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            versions[pkg] = None
    return {"python": platform.python_version(), "platform": platform.platform(), "packages": versions,
            "operations": {n: o.version for n, o in sorted(REGISTRY.items())}}


class WorkflowError(RuntimeError):
    def __init__(self, message: str, errors: list[str] | None = None):
        super().__init__(message)
        self.errors = errors or []


def _leaf_steps(spec: WorkflowSpec) -> list[str]:
    used = {_parse_ref(r)[1] for s in spec.steps for v in s.inputs.values() for r in refs(v) if r.startswith("steps.")}
    return [s.id for s in spec.steps if s.id not in used]


def _resolve_inputs(step: Step, spec: WorkflowSpec, store: Store, outputs: dict[str, str]) -> dict[str, str | list[str]]:
    def one(ref: str) -> str:
        scope, name = _parse_ref(ref)
        return store.resolve(spec.inputs[name]) if scope == "inputs" else outputs[name]

    return {port: [one(r) for r in v] if isinstance(v, list) else one(v) for port, v in step.inputs.items()}


def _run_one(step: Step, input_ids: dict[str, str | list[str]], store: Store, run_id: str, force: bool) -> dict[str, Any]:
    spec_op = get_op(step.op)
    key = step_key(step, input_ids)
    record = {"id": step.id, "op": step.op, "op_version": spec_op.version,
              "params": spec_op.params.model_validate(step.params).model_dump(mode="json"),
              "inputs": input_ids, "cache_key": key}
    cached = None if force else store.cache_get(key)
    if cached:
        return {**record, "output": cached, "cached": True, "seconds": 0.0}
    t0 = time.perf_counter()
    args = {port: [store.get(a) for a in aid] if isinstance(aid, list) else store.get(aid)
            for port, aid in input_ids.items()}
    result = spec_op.fn(spec_op.params.model_validate(step.params), **args)
    seconds = time.perf_counter() - t0
    aid = store.put(result, provenance={"source": "step", "run_id": run_id, "step": step.id, "op": step.op,
                                        "op_version": spec_op.version, "params": record["params"],
                                        "inputs": input_ids, "cache_key": key})
    store.cache_put(key, aid)
    return {**record, "output": aid, "cached": False, "seconds": round(seconds, 6)}


def run(spec: WorkflowSpec | dict | str | Path, store: Store, force: bool = False,
        extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Validate and execute a workflow locally, returning its provenance record.

    ``extra`` is merged into the record (pipelines add the pipeline spec and dataset id).
    """
    spec = load_spec(spec)
    errors, order, _ = validate(spec, store)
    if errors:
        raise WorkflowError(f"workflow {spec.name!r} is invalid", errors)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    resolved_spec = spec.model_copy(update={"inputs": {k: store.resolve(v) for k, v in spec.inputs.items()}})
    record: dict[str, Any] = {
        "run_id": run_id, "workflow_name": spec.name, "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "workflow": resolved_spec.model_dump(mode="json"),
        "workflow_hash": hashlib.sha256(canonical_json(resolved_spec.model_dump(mode="json")).encode()).hexdigest(),
        "environment": environment(), "steps": [], "outputs": {}, **(extra or {}),
    }
    outputs: dict[str, str] = {}
    by_id = {s.id: s for s in spec.steps}
    try:
        for sid in order:
            step = by_id[sid]
            rec = _run_one(step, _resolve_inputs(step, spec, store, outputs), store, run_id, force)
            outputs[sid] = rec["output"]
            record["steps"].append(rec)
        record["status"] = "succeeded"
    except Exception as e:  # recorded, then re-raised with the run id attached
        record["status"] = "failed"
        record["error"] = {"step": sid, "type": type(e).__name__, "message": str(e),
                           "traceback": traceback.format_exc(limit=5)}
    record["outputs"] = {sid: outputs[sid] for sid in (spec.outputs or _leaf_steps(spec)) if sid in outputs}
    record["finished_at"] = datetime.now(timezone.utc).isoformat()
    store.put_run(record)
    if record["status"] == "failed":
        raise WorkflowError(f"step {record['error']['step']!r} failed: {record['error']['message']} (run {run_id})")
    return record


def run_step(spec: WorkflowSpec | dict | str | Path, step_id: str, store: Store, force: bool = False) -> dict[str, Any]:
    """Execute one step, locating upstream outputs through the step cache.

    This is the entry point cluster jobs call: each Slurm job or Argo task
    runs one step against a shared store, after its dependencies finished.
    """
    spec = load_spec(spec)
    errors, order, _ = validate(spec, store)
    if errors:
        raise WorkflowError(f"workflow {spec.name!r} is invalid", errors)
    by_id = {s.id: s for s in spec.steps}
    outputs: dict[str, str] = {}
    if step_id not in by_id:
        raise KeyError(f"no step {step_id!r}")
    for sid in order:
        step = by_id[sid]
        try:
            input_ids = _resolve_inputs(step, spec, store, outputs)
        except KeyError:
            if sid == step_id:
                raise WorkflowError(f"upstream results for step {step_id!r} are missing; run its dependencies first")
            continue  # an unrelated branch that hasn't run; skip it
        if sid == step_id:
            return _run_one(step, input_ids, store, run_id=f"step-{step_id}", force=force)
        aid = store.cache_get(step_key(step, input_ids))
        if aid is not None:
            outputs[sid] = aid
    raise WorkflowError(f"step {step_id!r} is not reachable")
