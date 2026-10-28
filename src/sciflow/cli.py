"""Command line entry point; ``run-step`` is what cluster jobs execute."""

from __future__ import annotations

import argparse
import json
import sys

from .dataset import load_any, register_dataset, run_any
from .store import Store
from .workflow import WorkflowError, run_step, validate


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="sciflow")
    ap.add_argument("--store", default=None, help="store directory (default $SCIFLOW_STORE or ./.sciflow)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("validate", "run"):
        p = sub.add_parser(name)
        p.add_argument("workflow")
        p.add_argument("--store", default=argparse.SUPPRESS)
        if name == "run":
            p.add_argument("--force", action="store_true")
    p = sub.add_parser("run-step")
    p.add_argument("workflow")
    p.add_argument("step")
    p.add_argument("--store", default=argparse.SUPPRESS)
    p.add_argument("--force", action="store_true")
    p = sub.add_parser("register-dataset", help="import a dataset manifest (YAML/JSON) and print its id")
    p.add_argument("manifest")
    p.add_argument("--store", default=argparse.SUPPRESS)
    p = sub.add_parser("ops")
    p.add_argument("name", nargs="?")
    sub.add_parser("serve", help="run the MCP server over stdio")
    args = ap.parse_args(argv)

    if args.cmd == "serve":
        from .server import main as serve
        serve()
        return 0
    if args.cmd == "ops":
        from .registry import REGISTRY, get_op
        out = get_op(args.name).describe(full=True) if args.name else [o.describe() for o in REGISTRY.values()]
        print(json.dumps(out, indent=2))
        return 0

    store = Store(args.store)
    try:
        if args.cmd == "register-dataset":
            print(register_dataset(args.manifest, store))
            return 0
        if args.cmd == "validate":
            errors, order, kinds = validate(load_any(args.workflow, store), store)
            print(json.dumps({"ok": not errors, "errors": errors, "order": order, "step_kinds": kinds}, indent=2))
            return 1 if errors else 0
        if args.cmd == "run":
            rec = run_any(args.workflow, store, force=args.force)
            print(json.dumps({"run_id": rec["run_id"], "outputs": rec["outputs"]}, indent=2))
        else:
            rec = run_step(load_any(args.workflow, store), args.step, store, force=args.force)
            print(json.dumps({k: rec[k] for k in ("id", "output", "cached", "seconds")}))
    except WorkflowError as e:
        print(f"error: {e}", file=sys.stderr)
        for err in e.errors:
            print(f"  - {err}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
