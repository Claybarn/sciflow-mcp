"""Datasets (experiment > subject > session) and pipelines."""

import asyncio
import csv
import json

import numpy as np
import pytest
from mcp import Client

from sciflow import Series, Store, server, validate
from sciflow.dataset import compile_pipeline, register_dataset, run_any
from sciflow.exporters import export_slurm
from sciflow.io import export_table


def session_files(tmp_path, sid, amplitude, seed):
    """30 s at 100 Hz: running bouts at 5, 15, 25 s; a 2-channel response of `amplitude` after each onset."""
    rng = np.random.default_rng(seed)
    t = np.arange(3000) / 100
    onsets = np.array([5.0, 15.0, 25.0])
    running = np.any((t[:, None] >= onsets) & (t[:, None] < onsets + 2), axis=1)
    lag = t[:, None] - onsets
    resp = amplitude * np.sum(np.where(lag >= 0, np.exp(-lag / 0.5), 0.0), axis=1)
    signal = np.stack([resp, 2 * resp], 1) + rng.normal(scale=0.01, size=(len(t), 2))
    d = tmp_path / sid
    d.mkdir()
    np.savez(d / "wheel.npz", timestamps=t, data=running * 5.0)
    np.savez(d / "lfp.npz", timestamps=t, data=signal)
    return {"wheel": {"path": str(d / "wheel.npz")}, "lfp": {"path": str(d / "lfp.npz"), "options": {"axes": ["time", "channel"]}}}


# m1 has three control sessions (amplitude 1), m2 one (amplitude 3): subject-level mean 2, flat mean 1.5.
SESSIONS = [
    ("m1-a", "m1", "control", 1.0), ("m1-b", "m1", "control", 1.0), ("m1-c", "m1", "control", 1.0),
    ("m2-a", "m2", "control", 3.0),
    ("m1-sd", "m1", "sleep_dep", 4.0), ("m2-sd", "m2", "sleep_dep", 6.0),
]


def manifest(tmp_path, sessions=SESSIONS):
    return {"name": "toy", "sessions": [
        {"id": sid, "subject": subj, "attributes": {"condition": cond},
         "inputs": session_files(tmp_path, sid, amp, i)}
        for i, (sid, subj, cond, amp) in enumerate(sessions)]}


def pipeline(dataset, **over):
    p = {
        "name": "onset-response",
        "dataset": dataset,
        "session_steps": [
            {"id": "running", "op": "threshold_intervals", "inputs": {"series": "inputs.wheel"}, "params": {"threshold": 1}},
            {"id": "epochs", "op": "peri_event", "inputs": {"series": "inputs.lfp", "events": "steps.running"},
             "params": {"pre": 0.5, "post": 1.0}},
            {"id": "response", "op": "epoch_average", "inputs": {"epochs": "steps.epochs"}},
        ],
        "collect": {"responses": "response"},
        "aggregate_steps": [
            {"id": "by_condition", "op": "group_reduce", "inputs": {"series": "inputs.responses"},
             "params": {"group_by": ["condition"], "within": ["subject"]}},
            {"id": "flat", "op": "group_reduce", "inputs": {"series": "inputs.responses"},
             "params": {"group_by": ["condition"]}},
        ],
    }
    p.update(over)
    return p


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / "store")


def peak(series, group_index):
    return float(series.data[:, group_index, 0].max())


def test_register_is_content_addressed(tmp_path, store):
    m = manifest(tmp_path)
    a = register_dataset(m, store)
    assert register_dataset(m, store) == a
    d = store.get_dataset(a)
    assert len(d["sessions"]) == 6 and all(v.startswith("sha256:") for v in d["sessions"][0]["inputs"].values())
    assert store.list_datasets()[0]["n_subjects"] == 2


def test_pipeline_hierarchical_vs_flat(tmp_path, store):
    did = register_dataset(manifest(tmp_path), store)
    rec = run_any(pipeline(did), store)
    assert rec["dataset"] == did and rec["pipeline"]["name"] == "onset-response"

    stacked = store.get(rec["steps"][-3]["output"])
    assert stacked.axes == ("lag", "session", "channel") and stacked.data.shape[1] == 6
    assert stacked.meta["unit_attributes"]["session"]["subject"][:3] == ["m1", "m1", "m1"]

    hier = store.get(rec["outputs"]["by_condition"])
    flat = store.get(rec["outputs"]["flat"])
    assert list(hier.coords["group"]) == ["condition=control", "condition=sleep_dep"]
    assert hier.meta["unit_attributes"]["group"]["n"] == [2, 2]          # two subjects per condition
    assert flat.meta["unit_attributes"]["group"]["n"] == [4, 2]          # four sessions in control
    assert peak(hier, 0) == pytest.approx(2.0, abs=0.05)                 # mean of subject means (1, 3)
    assert peak(flat, 0) == pytest.approx(1.5, abs=0.05)                 # session-weighted
    assert peak(hier, 1) == pytest.approx(5.0, abs=0.05)
    np.testing.assert_allclose(hier.data[:, :, 1], 2 * hier.data[:, :, 0], atol=0.05)  # channel 2 is 2x


def test_adding_a_session_only_runs_that_session(tmp_path, store):
    did = register_dataset(manifest(tmp_path, SESSIONS[:5]), store)
    run_any(pipeline(did), store)
    (tmp_path / "more").mkdir()
    new = manifest(tmp_path / "more", SESSIONS[5:])["sessions"]   # imported from files
    old = store.get_dataset(did)["sessions"]                       # already registered: inputs are artifact ids
    did2 = register_dataset({"name": "toy", "sessions": old + new}, store)
    assert did2 != did
    rec = run_any(pipeline(did2), store)
    session_steps = [s for s in rec["steps"] if "__" in s["id"]]
    ran = [s["id"] for s in session_steps if not s["cached"]]
    # Only the new session runs. Its wheel trace is identical to the others', so even its
    # threshold step is a cache hit: results are keyed by content, not by session.
    assert ran == ["m2-sd__epochs", "m2-sd__response"]
    assert not any(s["cached"] for s in rec["steps"] if s["id"] in ("responses", "by_condition"))


def test_where_filters_sessions(tmp_path, store):
    did = register_dataset(manifest(tmp_path), store)
    spec = compile_pipeline(pipeline(did, where={"subject": ["m2"]}), store)
    stack = spec.step("responses")
    assert stack.params["sessions"] == ["m2-a", "m2-sd"]


def test_pipeline_errors(tmp_path, store):
    did = register_dataset(manifest(tmp_path), store)
    bad = pipeline(did)
    bad["session_steps"][0]["inputs"] = {"series": "inputs.speed"}
    with pytest.raises(ValueError, match="has no inputs \\['speed'\\]"):
        compile_pipeline(bad, store)
    # interval results can't be stacked, and that's caught before anything runs
    errors, _, _ = validate(compile_pipeline(pipeline(did, collect={"bouts": "running"}, aggregate_steps=[]), store), store)
    assert any("expects series but steps.m1-a__running is intervals" in e for e in errors)


def test_stack_rejects_mismatched_sessions():
    from sciflow.ops.aggregate import StackParams, stack_sessions

    a = Series(np.arange(5.0), np.zeros(5))
    b = Series(np.arange(5.0) + 0.5, np.zeros(5))
    with pytest.raises(Exception, match="session 's2' doesn't match.*interpolate_to"):
        stack_sessions(StackParams(sessions=["s1", "s2"]), [a, b])


def test_export_pipeline_to_slurm(tmp_path, store):
    did = register_dataset(manifest(tmp_path), store)
    out = export_slurm(compile_pipeline(pipeline(did), store), store, tmp_path / "slurm")
    submit = (tmp_path / "slurm" / "submit.sh").read_text()
    stack_line = next(l for l in submit.splitlines() if l.startswith("JOB[responses]"))
    assert stack_line.count("${JOB[") == 6                               # waits for every session's response
    assert "JOB[m1-a__epochs]=$(sbatch --parsable --dependency=afterok:${JOB[m1-a__running]}" in submit
    assert len(out["steps"]) == 6 * 3 + 3


def test_table_export_has_attributes(tmp_path, store):
    did = register_dataset(manifest(tmp_path), store)
    rec = run_any(pipeline(did), store)
    stacked_id = next(s["output"] for s in rec["steps"] if s["id"] == "responses")
    path = export_table(store.get(stacked_id), tmp_path / "long.csv")
    rows = list(csv.DictReader(open(path)))
    assert set(rows[0]) == {"lag", "session", "subject", "condition", "channel", "value"}
    assert len(rows) == 150 * 6 * 2
    assert {r["condition"] for r in rows} == {"control", "sleep_dep"}


def test_pipeline_over_mcp(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "_store", Store(tmp_path / "store"))
    m = manifest(tmp_path)

    async def go():
        async with Client(server.mcp) as c:
            def payload(r):
                assert not r.is_error, r.content
                sc = r.structured_content
                return sc.get("result", sc) if isinstance(sc, dict) else json.loads(r.content[0].text)

            reg = payload(await c.call_tool("register_dataset", {"dataset": m}))
            assert reg["ok"] and reg["subjects"] == ["m1", "m2"] and reg["attributes"] == ["condition"]
            p = pipeline(reg["id"])
            v = payload(await c.call_tool("validate_workflow", {"workflow": p}))
            assert v["ok"] and v["n_steps"] == 21
            r = payload(await c.call_tool("run_workflow", {"workflow": p}))
            # identical wheel traces: 5 of the 6 threshold steps reuse the first one's result in the same run
            assert r["ok"] and r["session_steps"] == {"run": 13, "cached": 5}
            t = payload(await c.call_tool("export_artifact", {"artifact_id": r["outputs"]["by_condition"],
                                                               "path": str(tmp_path / "groups.csv"), "format": "table"}))
            assert t["ok"]
            assert payload(await c.call_tool("get_dataset", {}))[0]["n_sessions"] == 6

    asyncio.run(go())


def test_pipeline_graph_and_mermaid(tmp_path, store):
    from sciflow import viz

    did = register_dataset(manifest(tmp_path), store)
    p = pipeline(did)
    p["aggregate_steps"].append({"id": "perm", "op": "permutation_test", "inputs": {"series": "inputs.responses"},
                                 "params": {"condition": "condition", "reps": 200}})
    g = viz.graph(p, store)
    assert g["n_sessions"] == 6
    session_nodes = {n["id"] for n in g["nodes"] if n.get("group") == "session"}
    assert session_nodes == {"in:wheel", "in:lfp", "running", "epochs", "response"}
    fan_in = next(e for e in g["edges"] if e["target"] == "responses")
    assert fan_in == {"source": "response", "target": "responses", "port": "items", "kind": "series", "fan_in": 6}
    assert {"source": "responses", "target": "perm", "port": "series", "kind": "series"} in g["edges"]
    text = viz.mermaid(p, store)
    assert 'subgraph per_session["each of 6 sessions"]' in text
    assert 'response -- "series x6" --> responses' in text
    assert ":::stats" in text
    # a flat workflow renders too
    wf = {"name": "w", "inputs": {"x": store.get_dataset(did)["sessions"][0]["inputs"]["wheel"]},
          "steps": [{"id": "run", "op": "threshold_intervals", "inputs": {"series": "inputs.x"}, "params": {"threshold": 1}}]}
    assert 'in_x -- "series" --> run' in viz.mermaid(wf, store)
