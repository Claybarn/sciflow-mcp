"""MCP server exposing sciflow to agents.

Agents work in four moves:

1. ``import_data`` raw files into stamped artifacts (content-addressed ids),
   or ``register_dataset`` a whole experiment (subjects > sessions).
2. ``list_operations`` / ``describe_operation`` to discover typed operations.
3. ``validate_workflow`` then ``run_workflow`` a DAG of operations, or a
   pipeline that runs per session and aggregates across them; every run
   writes a provenance record.
4. ``export_workflow`` to Slurm or Argo (Kubernetes) to run the same spec on
   a cluster, or ``export_artifact`` to hand results to MATLAB / pandas / R.

The store location comes from ``SCIFLOW_STORE`` (default ``./.sciflow``).
"""

from __future__ import annotations

from typing import Any, Literal

from mcp.server.mcpserver import MCPServer

from . import compare as _compare, exporters, io, viz
from .dataset import is_pipeline, load_any, register_dataset as _register_dataset, run_any
from .registry import REGISTRY, get_op
from .store import Store
from .workflow import WorkflowError, validate

mcp = MCPServer(
    "sciflow",
    instructions=(
        "Build reproducible analyses of stamped scientific data. All data is either a 'series' (an N-D "
        "array whose axis 0 is a named parametric axis, usually time but also depth, frequency or "
        "position, paired with one stamp per sample) or 'intervals' (sorted [on, off] ranges on a "
        "parametric axis; point events have on == off). Every operation consumes and produces these kinds, "
        "so steps compose regardless of data dimensionality; combining artifacts requires the same "
        "parametric axis and unit (use reparametrize / rescale_stamps to convert). Workflow step inputs reference "
        "'inputs.<name>' (an artifact id) or 'steps.<id>'. For multi-session experiments, register_dataset "
        "once, then write a pipeline (session_steps run on every session, collect stacks a per-session result "
        "along a 'session' axis, aggregate_steps such as group_reduce combine sessions -> subjects -> conditions); "
        "the workflow tools accept pipelines too. Always validate_workflow before run_workflow, "
        "and cite the run_id of results you report."
    ),
)

_store: Store | None = None


def store() -> Store:
    global _store
    if _store is None:
        _store = Store()
    return _store


def _error(e: Exception) -> dict[str, Any]:
    out = {"ok": False, "error": str(e)}
    if isinstance(e, WorkflowError) and e.errors:
        out["errors"] = e.errors
    return out


@mcp.tool()
def list_operations(category: str | None = None) -> list[dict[str, Any]]:
    """List available operations with their input kinds and output kind. Optionally filter by category
    (intervals, series, filtering, peri-event, correlation, alignment)."""
    return [o.describe() for o in sorted(REGISTRY.values(), key=lambda o: (o.category, o.name))
            if category is None or o.category == category]


@mcp.tool()
def describe_operation(name: str) -> dict[str, Any]:
    """Full documentation and JSON schema of an operation's parameters."""
    try:
        return get_op(name).describe(full=True)
    except KeyError as e:
        return _error(e)


@mcp.tool()
def import_data(
    path: str,
    kind: Literal["series", "intervals"] = "series",
    name: str | None = None,
    units: str | None = None,
    stream: str | None = None,
    data_key: str = "data",
    stamps_key: str | None = None,
    ons_key: str = "ons",
    offs_key: str = "offs",
    times_key: str | None = None,
    stamp_column: str | None = None,
    parametric_axis: int = 0,
    sample_rate: float | None = None,
    start: float = 0.0,
    ticks_per_unit: float | None = None,
    axes: list[str] | None = None,
    axis: str | None = None,
    unit: str | None = None,
) -> dict[str, Any]:
    """Import a .npy/.npz/.mat/.csv file into the store and return its artifact id.

    Series: the array under `data_key` with its parametric axis at index `parametric_axis`, paired with
    `stamps_key` (default 'stamps' or Collate's 'timestamps'), or stamps generated from `sample_rate` and
    `start`. `ticks_per_unit` divides raw stamps (e.g. 30000 for sample numbers at 30 kHz). `axes` labels
    every axis, the first naming the parametric axis; or pass just `axis` (e.g. "depth") and its `unit`.
    Intervals: `ons_key`/`offs_key`, or `times_key` for point events, on parametric `axis`.
    `stream` selects one struct in a Collate-style .mat file. CSV series use `stamp_column` (default: first).
    """
    try:
        art = io.import_file(path, kind, name=name, units=units, stream=stream, data_key=data_key,
                             stamps_key=stamps_key, ons_key=ons_key, offs_key=offs_key,
                             times_key=times_key, stamp_column=stamp_column, parametric_axis=parametric_axis,
                             sample_rate=sample_rate, start=start, ticks_per_unit=ticks_per_unit,
                             axes=axes, axis=axis, unit=unit)
        aid = store().put(art, provenance={"source": "import", "path": str(path), "kind": kind, "stream": stream})
        return {"ok": True, "id": aid, **art.summary()}
    except Exception as e:
        return _error(e)


@mcp.tool()
def list_artifacts() -> list[dict[str, Any]]:
    """List every artifact in the store."""
    return store().list()


@mcp.tool()
def inspect_artifact(artifact_id: str) -> dict[str, Any]:
    """Summarise an artifact (parametric axis and unit, shape, axes, extent, sample rate, NaN fraction) and how it was produced.
    Accepts a full id or a unique prefix."""
    try:
        s = store()
        aid = s.resolve(artifact_id)
        return {"ok": True, "id": aid, **s.get(aid).summary(), "provenance": s.header(aid)["provenance"]}
    except Exception as e:
        return _error(e)


@mcp.tool()
def register_dataset(dataset: dict[str, Any] | str) -> dict[str, Any]:
    """Register an experiment: {name, sessions: [{id, subject, attributes: {condition: ..., day: ...},
    inputs: {<name>: <artifact id> | {path, kind, options}}}]}. File inputs are imported (options are
    import_data's keyword arguments). Returns a content-derived dataset id that pins the exact data."""
    try:
        did = _register_dataset(dataset, store())
        d = store().get_dataset(did)
        return {"ok": True, "id": did, "n_sessions": len(d["sessions"]),
                "subjects": sorted({s["subject"] for s in d["sessions"] if s["subject"] is not None}),
                "attributes": sorted({a for s in d["sessions"] for a in s["attributes"]}),
                "inputs": sorted({i for s in d["sessions"] for i in s["inputs"]})}
    except Exception as e:
        return _error(e)


@mcp.tool()
def get_dataset(dataset_id: str | None = None) -> dict[str, Any] | list[dict[str, Any]]:
    """A registered dataset's sessions, subjects, attributes and input artifact ids. With no id, lists datasets."""
    try:
        return store().get_dataset(dataset_id) if dataset_id else store().list_datasets()
    except Exception as e:
        return _error(e)


@mcp.tool()
def compare_artifacts(a: str, b: str, rtol: float = 1e-9, atol: float = 0.0) -> dict[str, Any]:
    """Check whether two artifacts agree numerically, e.g. the same step run on a laptop and on a cluster.
    Ids are hashes of exact bytes, so different hardware can give different ids for results that agree to
    ~1e-15; this compares kind, axes, stamps, coordinates and data within |a - b| <= atol + rtol * |b|."""
    try:
        return {"ok": True, **_compare.compare(store().get(a), store().get(b), rtol=rtol, atol=atol)}
    except Exception as e:
        return _error(e)


@mcp.tool()
def validate_workflow(workflow: dict[str, Any] | str) -> dict[str, Any]:
    """Statically check a workflow or pipeline (JSON object, YAML/JSON text, or file path): references,
    cycles, input/output kinds and parameters. Returns the execution order and each step's output kind;
    pipelines are compiled first (session steps appear as '<session>__<step>')."""
    try:
        spec = load_any(workflow, store())
        errors, order, kinds = validate(spec, store())
        out = {"ok": not errors, "errors": errors, "order": order, "step_kinds": kinds}
        if is_pipeline(workflow):
            out["n_steps"] = len(order)
            out["outputs"] = spec.outputs
            out["order"] = [s for s in order if "__" not in s] + [f"... {sum('__' in s for s in order)} session steps"]
        return out
    except Exception as e:
        return _error(e)


@mcp.tool()
def run_workflow(workflow: dict[str, Any] | str, force: bool = False) -> dict[str, Any]:
    """Run a workflow or pipeline locally. Steps whose (operation, version, params, inputs) already ran are
    reused from cache unless `force`, so adding sessions to a dataset only runs the new ones. Returns run_id,
    per-step outputs (summarised for pipelines) and the workflow outputs."""
    try:
        rec = run_any(workflow, store(), force=force)
        steps = [{k: s[k] for k in ("id", "op", "output", "cached", "seconds")} for s in rec["steps"]]
        out = {"ok": True, "run_id": rec["run_id"], "outputs": rec["outputs"]}
        if "pipeline" in rec:
            out["dataset"] = rec["dataset"]
            out["session_steps"] = {"run": sum(not s["cached"] for s in steps if "__" in s["id"]),
                                    "cached": sum(s["cached"] for s in steps if "__" in s["id"])}
            out["steps"] = [s for s in steps if "__" not in s["id"]]
        else:
            out["steps"] = steps
        return out
    except Exception as e:
        return _error(e)


@mcp.tool()
def visualize_workflow(workflow: dict[str, Any] | str, format: Literal["mermaid", "graph"] = "mermaid") -> dict[str, Any]:
    """Draw a workflow or pipeline: Mermaid flowchart text (render it for the user), or the node / edge
    graph as JSON for a visual editor. Pipelines show their session steps once, inside an
    'each of N sessions' block, with the artifact kind on every edge."""
    try:
        if format == "graph":
            return {"ok": True, **viz.graph(workflow, store())}
        return {"ok": True, "mermaid": viz.mermaid(workflow, store())}
    except Exception as e:
        return _error(e)


@mcp.tool()
def get_run(run_id: str | None = None) -> dict[str, Any] | list[dict[str, Any]]:
    """Full provenance record of a run (resolved workflow, environment and package versions, every step's
    params, inputs, outputs and cache key). With no run_id, lists recent runs."""
    try:
        return store().get_run(run_id) if run_id else store().list_runs()
    except Exception as e:
        return _error(e)


@mcp.tool()
def export_workflow(
    workflow: dict[str, Any] | str,
    target: Literal["slurm", "argo"],
    out_dir: str,
    options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compile a workflow or pipeline into cluster jobs that each run one step against a shared store
    (a pipeline's sessions run in parallel; its aggregate steps wait for all of them).

    slurm options: store_path, partition, time, mem, cpus, account, setup (shell lines, e.g. module load),
    sciflow_cmd, extra_sbatch. argo options: image, namespace, store_pvc, cpu, memory.
    """
    try:
        spec = load_any(workflow, store())
        fn = exporters.export_slurm if target == "slurm" else exporters.export_argo
        return {"ok": True, **fn(spec, store(), out_dir, **(options or {}))}
    except Exception as e:
        return _error(e)


@mcp.tool()
def export_artifact(artifact_id: str, path: str,
                    format: Literal["npz", "mat", "csv", "table"] | None = None) -> dict[str, Any]:
    """Write an artifact to disk as .npz, .mat (Collate struct layout, readable in MATLAB), .csv, or 'table':
    a long-format CSV with one row per element and columns for every axis plus subject / condition / other
    attributes of stacked sessions (ready for mixed-effects models in R or statsmodels)."""
    try:
        art = store().get(artifact_id)
        if format == "table":
            return {"ok": True, "path": str(io.export_table(art, path))}
        return {"ok": True, "path": str(io.export_artifact(art, path, format))}
    except Exception as e:
        return _error(e)


def main():
    mcp.run()


if __name__ == "__main__":
    main()
