import json

import numpy as np
import pytest
import yaml

from sciflow import IntervalSet, Store, Series, run, run_step, validate, load_spec
from sciflow import io
from sciflow.exporters import export_argo, export_slurm
from sciflow.store import artifact_id
from sciflow.workflow import WorkflowError


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / "store")


@pytest.fixture
def inputs(store):
    rng = np.random.default_rng(0)
    fs = 50.0
    t = np.arange(3000) / fs
    speed = np.where((t % 10) < 3, 5.0, 0.0) + rng.normal(scale=0.1, size=t.size)
    lfp = rng.normal(size=(t.size, 8))
    return {
        "wheel": store.put(Series(t, speed, name="wheel", units="cm/s"), {"source": "test"}),
        "lfp": store.put(Series(t, lfp, name="lfp", axes=("time", "channel")), {"source": "test"}),
    }


def spec(inputs, **over):
    d = {
        "name": "running-locked-lfp",
        "inputs": inputs,
        "steps": [
            {"id": "running", "op": "threshold_intervals", "inputs": {"series": "inputs.wheel"}, "params": {"threshold": 2.0}},
            {"id": "bouts", "op": "filter_intervals", "inputs": {"a": "steps.running"}, "params": {"min_duration": 1.0}},
            {"id": "epochs", "op": "peri_event", "inputs": {"series": "inputs.lfp", "events": "steps.bouts"},
             "params": {"pre": 1.0, "post": 2.0}},
            {"id": "response", "op": "epoch_average", "inputs": {"epochs": "steps.epochs"}},
        ],
    }
    d.update(over)
    return d


def test_artifact_ids_are_content_hashes(store):
    a = Series(np.arange(3.0), np.ones(3), name="x")
    b = Series(np.arange(3.0), np.ones(3), name="x")
    assert artifact_id(a) == artifact_id(b) == store.put(a)
    assert artifact_id(Series(np.arange(3.0), np.ones(3) * 2, name="x")) != artifact_id(a)
    got = store.get(store.resolve(artifact_id(a)[7:19]))  # prefix lookup
    np.testing.assert_array_equal(got.data, a.data)


def test_validate_catches_errors(store, inputs):
    bad = spec(inputs)
    bad["steps"][1]["inputs"] = {"a": "inputs.lfp"}                 # timeseries into intervals port
    bad["steps"][2]["params"]["pre"] = -1                             # violates ge=0
    bad["steps"][3]["inputs"] = {"epochs": "steps.nope"}              # dangling ref
    bad["steps"].append({"id": "x", "op": "no_such_op", "inputs": {}})
    errors, _, _ = validate(load_spec(bad), store)
    text = "\n".join(errors)
    assert "expects intervals but inputs.lfp is series" in text
    assert "param pre" in text
    assert "no step 'nope'" in text
    assert "unknown operation 'no_such_op'" in text


def test_validate_detects_cycles(store, inputs):
    s = spec(inputs)
    s["steps"][0]["inputs"] = {"series": "steps.response"}
    errors, _, _ = validate(load_spec(s), store)
    assert any("cycle" in e for e in errors)


def test_run_records_provenance_and_caches(store, inputs):
    rec = run(spec(inputs), store)
    assert rec["status"] == "succeeded"
    out = store.get(rec["outputs"]["response"])
    assert out.data.shape == (150, 8) and out.axes == ("lag", "channel")
    # 6 running bouts (3 s of every 10 s); the one at t=0 has no 1 s pre-window
    assert out.meta["n_valid"] == 5
    hdr = store.header(rec["outputs"]["response"])
    assert hdr["provenance"]["op"] == "epoch_average" and hdr["provenance"]["run_id"] == rec["run_id"]
    assert rec["environment"]["packages"]["eventlogic"]
    saved = store.get_run(rec["run_id"])
    assert [s["id"] for s in saved["steps"]] == ["running", "bouts", "epochs", "response"]

    again = run(spec(inputs), store)
    assert all(s["cached"] for s in again["steps"])
    assert again["outputs"] == rec["outputs"]

    changed = spec(inputs)
    changed["steps"][2]["params"]["post"] = 1.0
    third = run(changed, store)
    assert [s["cached"] for s in third["steps"]] == [True, True, False, False]


def test_failed_run_is_recorded(store, inputs):
    s = spec(inputs)
    s["steps"][0]["params"] = {"threshold": 2.0, "channel": [0]}  # wheel is 1-D: indexing fails at runtime
    with pytest.raises(Exception):
        run(s, store)
    runs = store.list_runs()
    assert runs[0]["status"] == "failed"


def test_run_step_matches_local_run(store, inputs, tmp_path):
    s = spec(inputs)
    with pytest.raises(WorkflowError, match="upstream"):
        run_step(s, "epochs", store)
    for sid in ["running", "bouts", "epochs", "response"]:
        last = run_step(s, sid, store)
    # a full local run now hits the cache populated by the step-by-step (cluster-style) execution
    rec = run(s, store)
    assert all(x["cached"] for x in rec["steps"]) and rec["outputs"]["response"] == last["output"]


def test_export_slurm(store, inputs, tmp_path):
    out = export_slurm(load_spec(spec(inputs)), store, tmp_path / "slurm", partition="day", setup="module load miniconda")
    submit = (tmp_path / "slurm" / "submit.sh").read_text()
    assert "JOB[epochs]=$(sbatch --parsable --dependency=afterok:${JOB[bouts]} steps/epochs.sbatch)" in submit
    sb = (tmp_path / "slurm" / "steps" / "epochs.sbatch").read_text()
    assert "#SBATCH --partition=day" in sb and "run-step" in sb and "module load miniconda" in sb
    wf = json.loads((tmp_path / "slurm" / "workflow.json").read_text())
    assert wf["inputs"]["lfp"].startswith("sha256:")
    assert out["steps"] == ["running", "bouts", "epochs", "response"]


def test_export_argo(store, inputs, tmp_path):
    out = export_argo(load_spec(spec(inputs)), store, tmp_path / "k8s", image="sciflow:dev")
    cm, wf = list(yaml.safe_load_all(open(out["file"])))
    assert cm["kind"] == "ConfigMap" and "workflow.json" in cm["data"]
    tasks = {t["name"]: t for t in wf["spec"]["templates"][0]["dag"]["tasks"]}
    assert tasks["epochs"]["dependencies"] == ["bouts"]
    assert wf["spec"]["templates"][1]["container"]["image"] == "sciflow:dev"


def test_io_roundtrip(tmp_path):
    ts = Series(np.arange(4.0) / 10, np.arange(8.0).reshape(4, 2), name="lfp")
    for fmt in ("npz", "mat", "csv"):
        p = io.export_artifact(ts, tmp_path / f"x.{fmt}")
        kw = {"stream": "lfp"} if fmt == "mat" else {}
        back = io.import_file(p, **kw)
        np.testing.assert_allclose(back.stamps, ts.stamps)
        np.testing.assert_allclose(back.data, ts.data)
    ev = IntervalSet([1, 3], [2, 4], name="ev")
    back = io.import_file(io.export_artifact(ev, tmp_path / "ev.npz"), kind="intervals")
    np.testing.assert_array_equal(back.offs, ev.offs)


def test_import_npy_with_sample_rate_and_time_axis(tmp_path):
    np.save(tmp_path / "stack.npy", np.zeros((3, 4, 100)))
    ts = io.import_file(tmp_path / "stack.npy", parametric_axis=2, sample_rate=20.0, start=5.0)
    assert ts.data.shape == (100, 3, 4) and ts.stamps[1] == pytest.approx(5.05)


def test_csv_with_dates(tmp_path):
    (tmp_path / "px.csv").write_text("date,close,volume\n2024-01-02,10.5,100\n2024-01-03,10.7,120\n2024-01-04,10.6,90\n")
    ts = io.import_file(tmp_path / "px.csv")
    assert ts.data.shape == (3, 2) and ts.stamps[1] - ts.stamps[0] == 86400
    assert list(ts.coords["channel"]) == ["close", "volume"]
    (tmp_path / "ev.csv").write_text("date\n2024-01-03\n")
    ev = io.import_file(tmp_path / "ev.csv", kind="intervals", times_key="date")
    assert ev.ons[0] == ts.stamps[1]


def test_missing_workflow_file_is_reported(tmp_path):
    with pytest.raises(FileNotFoundError, match="not found"):
        load_spec(str(tmp_path / "nope.json"))
