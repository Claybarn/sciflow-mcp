"""Compile a workflow into cluster jobs.

Both targets run each step as ``sciflow run-step <workflow> <step>`` against a
shared store, so a step's results are found by later steps through the
content-addressed cache exactly as in a local run.

- Slurm: one ``.sbatch`` file per step and a ``submit.sh`` that chains them
  with ``--dependency=afterok``. The store must be on a filesystem every node
  can see.
- Kubernetes: an Argo Workflows DAG plus a ConfigMap carrying the workflow
  spec. The store is a PersistentVolumeClaim (ReadWriteMany) mounted at
  ``/store``.
"""

from __future__ import annotations

import json
import shlex
from pathlib import Path

import yaml

from .store import Store
from .workflow import WorkflowError, WorkflowSpec, _parse_ref, refs, validate


def _deps(spec: WorkflowSpec) -> dict[str, list[str]]:
    return {s.id: sorted({_parse_ref(r)[1] for v in s.inputs.values() for r in refs(v) if r.startswith("steps.")})
            for s in spec.steps}


def _checked(spec: WorkflowSpec, store: Store) -> list[str]:
    errors, order, _ = validate(spec, store)
    if errors:
        raise WorkflowError(f"workflow {spec.name!r} is invalid", errors)
    return order


def _resolved(spec: WorkflowSpec, store: Store) -> WorkflowSpec:
    return spec.model_copy(update={"inputs": {k: store.resolve(v) for k, v in spec.inputs.items()}})


def export_slurm(
    spec: WorkflowSpec, store: Store, out_dir: str | Path, *,
    store_path: str | None = None, partition: str | None = None, time: str = "01:00:00",
    mem: str = "8G", cpus: int = 1, account: str | None = None,
    setup: str = "", sciflow_cmd: str = "sciflow", extra_sbatch: list[str] | None = None,
) -> dict:
    order = _checked(spec, store)
    out = Path(out_dir).expanduser().resolve()
    (out / "steps").mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(exist_ok=True)
    wf_path = out / "workflow.json"
    wf_path.write_text(json.dumps(_resolved(spec, store).model_dump(mode="json"), indent=2))
    store_path = store_path or str(store.root)
    deps = _deps(spec)

    for sid in order:
        lines = ["#!/bin/bash", f"#SBATCH --job-name={spec.name}-{sid}",
                 f"#SBATCH --output={out}/logs/{sid}-%j.out", f"#SBATCH --time={time}",
                 f"#SBATCH --mem={mem}", f"#SBATCH --cpus-per-task={cpus}"]
        if partition:
            lines.append(f"#SBATCH --partition={partition}")
        if account:
            lines.append(f"#SBATCH --account={account}")
        lines += [f"#SBATCH {x}" for x in extra_sbatch or []]
        lines += ["set -euo pipefail", setup,
                  f"{sciflow_cmd} run-step {shlex.quote(str(wf_path))} {shlex.quote(sid)} --store {shlex.quote(store_path)}", ""]
        (out / "steps" / f"{sid}.sbatch").write_text("\n".join(l for l in lines if l is not None))

    submit = ["#!/bin/bash", "# Submits every step with afterok dependencies. Re-running is safe: finished steps hit the cache.",
              "set -euo pipefail", f'cd "$(dirname "$0")"', "declare -A JOB"]
    for sid in order:
        dep = ":".join(f"${{JOB[{d}]}}" for d in deps[sid])
        flag = f"--dependency=afterok:{dep} " if dep else ""
        submit.append(f'JOB[{sid}]=$(sbatch --parsable {flag}steps/{sid}.sbatch)')
        submit.append(f'echo "{sid}: ${{JOB[{sid}]}}"')
    (out / "submit.sh").write_text("\n".join(submit) + "\n")
    (out / "submit.sh").chmod(0o755)
    return {"target": "slurm", "dir": str(out), "submit": str(out / "submit.sh"), "steps": order,
            "note": f"store must be visible to compute nodes at {store_path}"}


def _k8s_name(s: str) -> str:
    out = "".join(c if c.isalnum() else "-" for c in s.lower()).strip("-")
    return out[:63] or "step"


def export_argo(
    spec: WorkflowSpec, store: Store, out_dir: str | Path, *,
    image: str = "ghcr.io/example/sciflow:latest", namespace: str = "argo",
    store_pvc: str = "sciflow-store", cpu: str = "1", memory: str = "4Gi",
) -> dict:
    order = _checked(spec, store)
    out = Path(out_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    name = _k8s_name(spec.name)
    deps = _deps(spec)
    configmap = {
        "apiVersion": "v1", "kind": "ConfigMap",
        "metadata": {"name": f"{name}-spec", "namespace": namespace},
        "data": {"workflow.json": json.dumps(_resolved(spec, store).model_dump(mode="json"), indent=2)},
    }
    templates = [{
        "name": "run-step",
        "inputs": {"parameters": [{"name": "step"}]},
        "container": {
            "image": image,
            "command": ["sciflow", "run-step", "/workflow/workflow.json", "{{inputs.parameters.step}}", "--store", "/store"],
            "resources": {"requests": {"cpu": cpu, "memory": memory}},
            "volumeMounts": [{"name": "store", "mountPath": "/store"}, {"name": "spec", "mountPath": "/workflow"}],
        },
    }]
    tasks = []
    for sid in order:
        t = {"name": _k8s_name(sid), "template": "run-step",
             "arguments": {"parameters": [{"name": "step", "value": sid}]}}
        if deps[sid]:
            t["dependencies"] = [_k8s_name(d) for d in deps[sid]]
        tasks.append(t)
    templates.insert(0, {"name": "main", "dag": {"tasks": tasks}})
    workflow = {
        "apiVersion": "argoproj.io/v1alpha1", "kind": "Workflow",
        "metadata": {"generateName": f"{name}-", "namespace": namespace},
        "spec": {
            "entrypoint": "main",
            "volumes": [{"name": "store", "persistentVolumeClaim": {"claimName": store_pvc}},
                        {"name": "spec", "configMap": {"name": f"{name}-spec"}}],
            "templates": templates,
        },
    }
    path = out / f"{name}.argo.yaml"
    path.write_text(yaml.safe_dump_all([configmap, workflow], sort_keys=False))
    return {"target": "argo", "file": str(path), "steps": order,
            "submit": f"kubectl create -f {path.name}",
            "note": f"PVC {store_pvc!r} must contain the store's input artifacts and be ReadWriteMany"}
