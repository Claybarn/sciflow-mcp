# sciflow-mcp

An MCP server that lets scientists and their agents build **reproducible, interoperable analysis workflows** over stamped data: data positioned along time, cortical depth, frequency, or any other parametric axis.

It builds on three earlier pieces of work:

- **[Collate](https://github.com/cardin-higley-lab/Collate)**: every array is paired with a stamp per sample, and multiple acquisition systems are aligned to one clock. Its `GeneralizedData` successor generalizes time to any parametric axis.
- **[EventLogic](https://github.com/Claybarn/EventLogic)**: logical operations (AND, OR, NOT, merge, containment) on `[on, off]` intervals.
- **The generalized analysis functions** from [Barnes et al. 2026](https://github.com/cardin-higley-lab/Barnes_et_al_2026): peri-event collation, normalization and correlation that work whatever the data's dimensionality.

## Install

```bash
pip install sciflow-mcp
```

To run the MCP server without installing anything, use `uvx sciflow-mcp`, for example in a client config: `{"command": "uvx", "args": ["sciflow-mcp"], "env": {"SCIFLOW_STORE": "/path/to/store"}}`.

For development, clone the repo:

```bash
git clone https://github.com/Claybarn/sciflow-mcp && cd sciflow-mcp && uv sync
```

Opening the clone in Claude Code picks up the server from [.mcp.json](.mcp.json).

## Why it interoperates

There are exactly two artifact kinds. Both lie on a named **parametric axis** with a unit (`time` in `s` by default):

| kind | contents | invariant |
|---|---|---|
| `series` | N-D `data`, 1-D `stamps`, axis labels, coordinates, `stamp_unit` | `len(stamps) == data.shape[0]`, stamps finite and non-decreasing, **the parametric axis is always axis 0** |
| `intervals` | sorted `ons`, `offs`, `axis`, `unit` | `on <= off`; point events have `on == off` |

- **Workflows are type-checked before they run.** Every operation declares which kinds it takes and which it returns.
- **No `parametricAxis` argument.** The parametric axis is always axis 0, so `peri_event` works the same on a 1-D wheel trace, a channels × time LFP and a 32 × 32 × time imaging stack.
- **Axes can't be mixed silently.** Operations that combine artifacts require the same parametric axis and unit. Time intervals can't be applied to a depth profile, and spike sample numbers can't be mixed with seconds. The error names the fix: `reparametrize`, `rescale_stamps` or `align_clock`.
- **Any axis can become the parametric one.** `reparametrize` turns a time × depth CSD into a depth × time series. Every operation then works along depth, for example `restrict` to a cortical layer defined with `define_intervals`.
- **Derived results stay stamped.** Peri-event epochs are stamped by event position and get a `lag` axis. Their average is a series whose parametric axis is `lag`, which can be windowed with `lag` intervals. Per-interval correlations are stamped by interval onset. Any output can feed any later step.

## Why it's reproducible

- **Content-addressed artifacts.** An artifact's id is the SHA-256 of its arrays and header, so identical bytes always get identical ids.
  - **Same platform and image:** results are bit-identical, so cache hits and ids carry over. This holds for a local Docker run and the same image on Slurm or Kubernetes.
  - **Different hardware or OS:** floating-point operations can differ in the last bits. A Butterworth filter computed on macOS and in the Linux image differs by up to ~7e-15 relative, which changes its id. `compare_artifacts` checks that two results agree within a tolerance, and every run records Python, platform and package versions.
- **Deterministic step keys.** Each step's cache key is `hash(op, op version, validated params, input ids)`. Re-running reuses results, and changing one parameter re-runs only that step and the steps downstream of it.
- **Provenance records.** Every run writes `runs/<run_id>.json`. It holds the resolved workflow, Python/package/operation versions, and each step's params, inputs, output, cache key and timing. Each output artifact also records the step that made it.

## Operations

`sciflow ops` (or the `list_operations` tool) prints the full list. The current 34:

- **intervals:** `threshold_intervals` (fixed or Collate's `midpoint` threshold), `define_intervals`, `event_windows`, `interval_union` / `intersection` / `difference` / `xor` / `complement`, `merge_intervals`, `filter_intervals`, `select_contained`
- **series:** `restrict`, `select`, `reduce_axis`, `reparametrize`, `events_where` (e.g. spikes of chosen clusters), `arithmetic`, `standardize`
- **filtering:** `butter_filter`, `median_filter`, `bin_average` (downsample along any axis, with a circular mean for phase)
- **peri-event:** `peri_event`, `peri_event_normalize` (dff / zscore / subtract), `epoch_average`, `event_histogram` (PSTH)
- **correlation:** `interval_correlation`, `pairwise_interval_correlation`
- **dataset:** `stack_sessions`, `group_reduce` (hierarchical: sessions → subjects → conditions)
- **stats:** `permutation_test` (stratified, from [Claybarn/permute](https://github.com/Claybarn/permute)), `mixed_model` (statsmodels MixedLM)
- **alignment:** `align_clock` (Collate's pulse cross-correlation plus linear fit; converts ticks or frame numbers to the reference unit), `rescale_stamps`, `interpolate_to`

The filtering, threshold, rescaling and `events_where` operations cover the preprocessing that each `AcquisitionSystem` subclass in [Collate's examples](https://github.com/cardin-higley-lab/Collate/tree/main/examples) re-implemented by hand. Examples are LFP/EMG high-pass filtering, median-filtered sync lines, OpenEphys sample numbers, Kilosort cluster labels and the miniscope's low-pass dF/F baseline. Each becomes a workflow step with recorded parameters.

## Workflows

```yaml
name: running-onset-widefield
inputs: {wheel: sha256:…, widefield: sha256:…, sync_ref: sha256:…, sync_img: sha256:…}
steps:
  - {id: aligned, op: align_clock, inputs: {target: inputs.widefield, local_clock: inputs.sync_img, reference_clock: inputs.sync_ref}}
  - {id: running, op: threshold_intervals, inputs: {series: inputs.wheel}, params: {threshold: 2.0}}
  - {id: epochs, op: peri_event_normalize, inputs: {series: steps.aligned, events: steps.running},
     params: {pre: 2, post: 4, baseline_start: -2, baseline_stop: 0, method: dff}}
  - {id: response, op: epoch_average, inputs: {epochs: steps.epochs}}
```

See [examples/running_response.yaml](examples/running_response.yaml). To run it end to end on synthetic two-clock data:

```bash
uv run python examples/make_demo_data.py
```

```bash
uv run python examples/run_demo.py
```

## Datasets: experiment → subject → session

A **dataset** is a manifest of sessions. Each session has a subject, attributes (condition, day, genotype, …) and named inputs, given as artifact ids or as files to import. `register_dataset` imports the files and stores the resolved manifest under a content-derived `dataset:` id, so the id pins the exact data.

A **pipeline** runs the same `session_steps` on every session. `collect` stacks a chosen per-session result along a new `session` axis, carrying each session's subject and attributes. `aggregate_steps` then combine the stacks:

```yaml
name: running-onset-lfp-by-condition
dataset: dataset:b30a456710f97312
where: {day: [1, 2]}                       # optional session filter
session_steps:
  - {id: running, op: threshold_intervals, inputs: {series: inputs.wheel}, params: {threshold: 2}}
  - {id: epochs, op: peri_event_normalize, inputs: {series: inputs.lfp, events: steps.running},
     params: {pre: 1, post: 2, baseline_start: -1, baseline_stop: 0, method: zscore}}
  - {id: response, op: epoch_average, inputs: {epochs: steps.epochs}}
collect: {responses: response}
aggregate_steps:
  - {id: condition_mean, op: group_reduce, inputs: {series: inputs.responses},
     params: {group_by: [condition], within: [subject], statistic: mean}}
```

- **Averaging is hierarchical.** `group_reduce` with `within: [subject]` averages each subject's sessions before combining subjects. A subject with more sessions doesn't dominate, and `statistic: sem` gives the standard error across subjects, not sessions.
- **Pipelines compile to ordinary workflows.** Session steps become `<session>__<step>`, so validation, caching, provenance and the Slurm/Argo exporters all apply unchanged. On a cluster, sessions run in parallel and the aggregate steps wait for all of them.
- **Adding a session only runs that session.** Everything else is a cache hit, and because results are keyed by content, sessions with identical inputs even share results. A pipeline's run record includes the pipeline and dataset id.
- **Results export as a long-format table.** `export_artifact(format="table")` writes one row per element, with columns for every axis plus subject, condition and other attributes. That's ready for mixed-effects models in R or statsmodels.

### Statistics that use every session

Both tests run on a stacked series and test every element (each lag × channel, each pixel) separately, without first averaging sessions within subject:

- **`permutation_test`** calls your `multitest_stratified_permutationtest`. It shuffles condition labels only within each `strata` attribute (default `subject`), so repeated sessions are never treated as independent.
  - `correction: max` (the default) is the permutation max-statistic (Westfall–Young) FWER correction. Each test's statistic is first standardized by its own null, so channels on different scales compete fairly. Holm, Bonferroni and FDR (BH) are also available.
  - Strata missing a condition are dropped and listed in the metadata, and elements with missing data get NaN. Without this, the raw function marks every test as significant at the minimum p-value.
  - The seed is a parameter, so the result is cacheable and reproducible. With `strata: null` it falls back to `multitest_two_sample`.
- **`mixed_model`** fits `formula` (e.g. `value ~ condition + C(day)`) with a random intercept per `groups` (or `re_formula` slopes) at every element. Output adds `term` and `statistic` axes (estimate, se, z, p, Holm-corrected p). Failed fits and convergence warnings are counted in the metadata.

Results stay on the parametric axis (`lag`), so p-values can be restricted to a window, plotted or exported to a table.

### Seeing a pipeline

`visualize_workflow` returns Mermaid text, or the node/edge graph as JSON for a visual editor such as the Genepi rete-js frontend. Pipelines are drawn with session steps collapsed into one "each of N sessions" block, and every edge is labeled with its artifact kind. [examples/onset_pipeline.mmd](examples/onset_pipeline.mmd) is the rendered demo pipeline.

[examples/dataset_demo.py](examples/dataset_demo.py) builds a synthetic 4-subject × 2-condition × 2-day experiment and runs [examples/onset_pipeline.yaml](examples/onset_pipeline.yaml) on it. The result recovers the planted 1.6× condition effect (5.05 vs 3.19). The permutation test (max-corrected) and the mixed model both find the condition effect in the first 100–150 ms after onset.

## MCP tools

| tool | purpose |
|---|---|
| `list_operations`, `describe_operation` | discover operations and the JSON schema of their params |
| `register_dataset`, `get_dataset` | register an experiment's sessions (importing files) and inspect it |
| `import_data` | `.npy` / `.npz` / `.mat` (incl. Collate structs) / `.csv` into the store; `ticks_per_unit` converts sample numbers |
| `list_artifacts`, `inspect_artifact` | parametric axis and unit, shape, axes, extent, sample rate, NaN fraction, provenance |
| `validate_workflow` | static check of a workflow or pipeline: references, cycles, kinds, params |
| `compare_artifacts` | check two results agree within a tolerance, e.g. a laptop run vs a cluster run |
| `visualize_workflow` | Mermaid flowchart or node/edge graph of a workflow or pipeline |
| `run_workflow`, `get_run` | execute a workflow or pipeline locally and read the provenance record |
| `export_workflow` | compile to Slurm (`sbatch` chain) or Argo Workflows (Kubernetes) |
| `export_artifact` | `.npz`, `.mat` (opens in MATLAB), `.csv`, or a long-format `table` for statistics |

This repo's [.mcp.json](.mcp.json) registers the server for Claude Code, with the store in `./.sciflow`. For other clients, use absolute paths:

```json
{"command": "uv", "args": ["run", "--directory", "/path/to/sciflow-mcp", "sciflow-mcp"], "env": {"SCIFLOW_STORE": "/path/to/store"}}
```

## Running on a cluster

Both exporters run each step as `sciflow run-step <workflow.json> <step> --store <store>` against a shared store. Results pass between jobs through the content-addressed cache, so a cluster run and a local run produce identical artifact ids.

- **Slurm:** `export_workflow(target="slurm")` writes one `.sbatch` per step and a `submit.sh` that chains them with `--dependency=afterok`. The store must be on a shared filesystem.
- **Kubernetes:** `export_workflow(target="argo")` writes a ConfigMap (the workflow) plus an Argo `Workflow` DAG. The store is a ReadWriteMany PVC mounted at `/store`. Build the step image with the included `Dockerfile`. It installs exactly the versions in `uv.lock`.

## Development

```bash
uv sync
```

```bash
uv run pytest
```

To develop against a local EventLogic checkout, uncomment `[tool.uv.sources]` in `pyproject.toml`.

### Fixes relative to the MATLAB generalized functions

- `binEventDataToEvents` / `collate_event_data_to_events` passed every event's window on every loop iteration. `event_histogram` uses each event's own window.
- `collatePairwiseCorrContinuousDataToEvents` correlated `data1` with itself. The port correlates `data1` with `data2`.
- The correlation helpers' exponent `0.args.post_event_seconds` (a find-and-replace accident) is back to a square root.

- The Python translations matched each event to the first sample at or after it (`searchsorted`). sciflow defaults to the nearest sample; `peri_event` and `peri_event_normalize` take `match: "next"` to reproduce the original. The windows' 0-based bounds check (`ind - pre >= 0`) also fixes the off-by-one carried over from MATLAB's 1-based `> 0`.
- `binEventDataToEvents` summed over events. `event_histogram` keeps one row per event; `epoch_average` with `statistic: sum` gives the original total.

`datetime64` timestamps and ISO-date CSV columns are converted to seconds since the Unix epoch, recorded as `time_origin` in the artifact's `meta`, so daily or market data works too (e.g. `pre: 2592000` for 30 days).

### Workarounds for EventLogic 0.2.0

- `Events.__xor__` is broken. XOR is composed as `(a | b) & ~(a & b)`.
- The sweeps assume each operand is internally non-overlapping, so inputs are normalized with `self_or()` first.
