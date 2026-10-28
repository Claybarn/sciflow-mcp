"""Render workflows and pipelines as graphs.

``graph`` returns a plain node / edge description (what a visual editor such
as the Genepi rete-js frontend would consume); ``mermaid`` renders it as
Mermaid flowchart text for READMEs, notebooks and chat. Pipelines are drawn
with their session steps collapsed into one "each session" block, since the
compiled graph repeats them once per session.
"""

from __future__ import annotations

import re
from typing import Any

from .dataset import SEP, compile_pipeline, is_pipeline, load_pipeline
from .registry import get_op
from .store import Store
from .workflow import _parse_ref, load_spec, refs, validate


def _param_summary(params: dict[str, Any], limit: int = 3) -> str:
    parts = []
    for k, v in list(params.items())[:limit]:
        if isinstance(v, list) and len(v) > 4:
            v = f"[{len(v)} items]"
        parts.append(f"{k}={v}")
    return ", ".join(parts) + (", ..." if len(params) > limit else "")


def graph(source, store: Store) -> dict[str, Any]:
    """Nodes (inputs, steps) and kind-labelled edges; pipelines carry a ``group`` for session steps."""
    if is_pipeline(source):
        p = load_pipeline(source)
        compiled = compile_pipeline(p, store)
        _, _, kinds = validate(compiled, store)
        first = next(s.id.split(SEP)[0] for s in compiled.steps if SEP in s.id)
        n_sessions = len({s.id.split(SEP)[0] for s in compiled.steps if SEP in s.id})
        input_kinds = {}
        for name, aid in compiled.inputs.items():
            key, _, short = name.partition(SEP)
            if key == first:
                input_kinds[short] = store.header(aid)["kind"]
        nodes, edges = [], []
        for name, kind in sorted(input_kinds.items()):
            nodes.append({"id": f"in:{name}", "label": name, "kind": kind, "type": "input", "group": "session"})
        for st in p.session_steps:
            nodes.append({"id": st.id, "label": st.id, "op": st.op, "params": st.params, "group": "session",
                          "type": "step", "kind": kinds.get(f"{first}{SEP}{st.id}", "?")})
        for name, sid in p.collect.items():
            nodes.append({"id": name, "label": name, "op": "stack_sessions", "params": {"sessions": n_sessions},
                          "type": "step", "kind": kinds.get(name, "?")})
        for st in p.aggregate_steps:
            nodes.append({"id": st.id, "label": st.id, "op": st.op, "params": st.params, "type": "step",
                          "kind": kinds.get(st.id, "?")})
        kind_of = {n["id"]: n["kind"] for n in nodes}
        for st in p.session_steps:
            for port, v in st.inputs.items():
                for r in refs(v):
                    scope, name = _parse_ref(r)
                    src = f"in:{name}" if scope == "inputs" else name
                    edges.append({"source": src, "target": st.id, "port": port, "kind": kind_of.get(src, "?")})
        for name, sid in p.collect.items():
            edges.append({"source": sid, "target": name, "port": "items", "kind": kind_of.get(sid, "?"), "fan_in": n_sessions})
        for st in p.aggregate_steps:
            for port, v in st.inputs.items():
                for r in refs(v):
                    _, name = _parse_ref(r)  # inputs.<stack> and steps.<id> both name a node
                    src = name
                    edges.append({"source": src, "target": st.id, "port": port, "kind": kind_of.get(src, "?")})
        return {"name": p.name, "n_sessions": n_sessions, "nodes": nodes, "edges": edges, "outputs": compiled.outputs}

    spec = load_spec(source)
    _, _, kinds = validate(spec, store)
    nodes = [{"id": f"in:{n}", "label": n, "type": "input", "kind": _header_kind(store, a)} for n, a in spec.inputs.items()]
    nodes += [{"id": s.id, "label": s.id, "op": s.op, "params": s.params, "type": "step", "kind": kinds.get(s.id, "?")}
              for s in spec.steps]
    kind_of = {n["id"]: n["kind"] for n in nodes}
    edges = []
    for s in spec.steps:
        for port, v in s.inputs.items():
            for r in refs(v):
                scope, name = _parse_ref(r)
                src = f"in:{name}" if scope == "inputs" else name
                edges.append({"source": src, "target": s.id, "port": port, "kind": kind_of.get(src, "?")})
    outputs = spec.outputs or [s.id for s in spec.steps if not any(e["source"] == s.id for e in edges)]
    return {"name": spec.name, "nodes": nodes, "edges": edges, "outputs": outputs}


def _header_kind(store: Store, aid: str) -> str:
    try:
        return store.header(aid)["kind"]
    except KeyError:
        return "?"


def _mid(node_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", node_id)


def mermaid(source, store: Store) -> str:
    g = graph(source, store)
    lines = ["flowchart LR",
             "  classDef input fill:#eef,stroke:#669", "  classDef stats fill:#fee,stroke:#a55",
             "  classDef dataset fill:#efe,stroke:#5a5"]

    def node_line(n):
        if n["type"] == "input":
            return f'  {_mid(n["id"])}[/"{n["label"]}<br/><small>{n["kind"]}</small>"/]:::input'
        cat = get_op(n["op"]).category
        detail = _param_summary(n.get("params") or {})
        label = f'<b>{n["label"]}</b><br/>{n["op"]}' + (f"<br/><small>{detail}</small>" if detail else "")
        cls = ":::stats" if cat == "stats" else ":::dataset" if cat == "dataset" else ""
        return f'  {_mid(n["id"])}["{label}"]{cls}'

    session = [n for n in g["nodes"] if n.get("group") == "session"]
    if session:
        lines.append(f'  subgraph per_session["each of {g["n_sessions"]} sessions"]')
        lines += ["  " + node_line(n) for n in session]
        lines.append("  end")
    lines += [node_line(n) for n in g["nodes"] if n.get("group") != "session"]
    for e in g["edges"]:
        label = e["kind"] + (f" x{e['fan_in']}" if e.get("fan_in") else "")
        lines.append(f'  {_mid(e["source"])} -- "{label}" --> {_mid(e["target"])}')
    return "\n".join(lines)
