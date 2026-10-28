import asyncio
import json

import numpy as np
from mcp import Client

from sciflow import Store, server


def _payload(result):
    assert not result.is_error, result.content
    sc = result.structured_content
    return sc.get("result", sc) if isinstance(sc, dict) else json.loads(result.content[0].text)


def test_agent_flow_over_mcp(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "_store", Store(tmp_path / "store"))
    t = np.arange(2000) / 100
    np.savez(tmp_path / "wheel.npz", timestamps=t, data=np.where((t % 5) < 1, 4.0, 0.0))
    np.save(tmp_path / "stack.npy", np.random.default_rng(0).normal(size=(6, 6, 2000)))

    async def go():
        async with Client(server.mcp) as c:
            tools = {t.name for t in (await c.list_tools()).tools}
            assert {"import_data", "validate_workflow", "run_workflow", "export_workflow", "get_run"} <= tools

            ops = _payload(await c.call_tool("list_operations", {"category": "peri-event"}))
            assert {o["name"] for o in ops} >= {"peri_event", "epoch_average"}
            desc = _payload(await c.call_tool("describe_operation", {"name": "peri_event"}))
            assert "pre" in desc["params_schema"]["properties"]

            wheel = _payload(await c.call_tool("import_data", {"path": str(tmp_path / "wheel.npz")}))
            stack = _payload(await c.call_tool("import_data", {
                "path": str(tmp_path / "stack.npy"), "parametric_axis": 2, "sample_rate": 100.0, "axes": ["time", "y", "x"]}))
            assert stack["shape"] == [2000, 6, 6]

            wf = {"name": "demo", "inputs": {"wheel": wheel["id"], "img": stack["id"]}, "steps": [
                {"id": "run", "op": "threshold_intervals", "inputs": {"series": "inputs.wheel"}, "params": {"threshold": 1}},
                {"id": "ep", "op": "peri_event", "inputs": {"series": "inputs.img", "events": "steps.run"},
                 "params": {"pre": 0.2, "post": 0.5}},
                {"id": "avg", "op": "epoch_average", "inputs": {"epochs": "steps.ep"}},
            ]}
            bad = json.loads(json.dumps(wf))
            bad["steps"][1]["inputs"]["events"] = "inputs.img"
            v = _payload(await c.call_tool("validate_workflow", {"workflow": bad}))
            assert not v["ok"] and "expects intervals" in v["errors"][0]

            assert _payload(await c.call_tool("validate_workflow", {"workflow": wf}))["ok"]
            r = _payload(await c.call_tool("run_workflow", {"workflow": wf}))
            assert r["ok"], r
            info = _payload(await c.call_tool("inspect_artifact", {"artifact_id": r["outputs"]["avg"]}))
            assert info["shape"] == [70, 6, 6] and info["provenance"]["op"] == "epoch_average"
            run = _payload(await c.call_tool("get_run", {"run_id": r["run_id"]}))
            assert run["status"] == "succeeded"

            ex = _payload(await c.call_tool("export_workflow", {
                "workflow": wf, "target": "slurm", "out_dir": str(tmp_path / "slurm")}))
            assert ex["ok"] and (tmp_path / "slurm" / "submit.sh").exists()
            out = _payload(await c.call_tool("export_artifact", {
                "artifact_id": r["outputs"]["avg"], "path": str(tmp_path / "avg.mat")}))
            assert out["ok"]

    asyncio.run(go())
