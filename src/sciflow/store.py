"""Content-addressed artifact store.

Artifacts are identified by the SHA-256 of their canonical contents, so the
same data always gets the same id on any machine. Each artifact lives in
``<root>/objects/<id>/`` as ``arrays.npz`` plus ``meta.json``. Step results
are memoised in ``<root>/cache/<step key>`` and run provenance records are
written to ``<root>/runs/<run id>.json``.

The store is a plain directory so it can sit on a shared filesystem used by
Slurm jobs or a Kubernetes PersistentVolume.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from .types import Artifact, IntervalSet, Series

ID_PREFIX = "sha256:"


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=_json_default)


def _json_default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"{type(o).__name__} is not JSON serialisable")


def _hash_array(h, name: str, a: np.ndarray):
    a = np.ascontiguousarray(a)
    h.update(f"{name}|{a.dtype.str}|{a.shape}|".encode())
    h.update(a.tobytes())


def _arrays_and_header(art: Artifact) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if isinstance(art, Series):
        arrays = {"stamps": art.stamps, "data": art.data}
        arrays.update({f"coord:{k}": v for k, v in art.coords.items()})
        header = {"kind": art.kind, "name": art.name, "units": art.units, "axes": list(art.axes),
                  "stamp_unit": art.stamp_unit, "meta": art.meta}
    elif isinstance(art, IntervalSet):
        arrays = {"ons": art.ons, "offs": art.offs}
        header = {"kind": art.kind, "name": art.name, "axis": art.axis, "unit": art.unit, "meta": art.meta}
    else:
        raise TypeError(f"not an artifact: {type(art).__name__}")
    return arrays, header


def artifact_id(art: Artifact) -> str:
    arrays, header = _arrays_and_header(art)
    h = hashlib.sha256()
    h.update(canonical_json(header).encode())
    for name in sorted(arrays):
        _hash_array(h, name, arrays[name])
    return ID_PREFIX + h.hexdigest()


def _atomic_write(path: Path, write):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-")
    try:
        with os.fdopen(fd, "wb") as f:
            write(f)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


class Store:
    def __init__(self, root: str | os.PathLike | None = None):
        self.root = Path(root or os.environ.get("SCIFLOW_STORE", ".sciflow")).expanduser().resolve()
        for sub in ("objects", "cache", "runs", "datasets"):
            (self.root / sub).mkdir(parents=True, exist_ok=True)

    # -- artifacts ---------------------------------------------------------
    def _dir(self, aid: str) -> Path:
        return self.root / "objects" / aid.removeprefix(ID_PREFIX)

    def resolve(self, ref: str) -> str:
        """Accept a full id or a unique prefix (with or without ``sha256:``)."""
        digest = ref.removeprefix(ID_PREFIX)
        if (self.root / "objects" / digest).is_dir():
            return ID_PREFIX + digest
        matches = [p.name for p in (self.root / "objects").glob(digest + "*") if p.is_dir()]
        if len(matches) == 1:
            return ID_PREFIX + matches[0]
        raise KeyError(f"artifact {ref!r} " + ("is ambiguous" if matches else "not found"))

    def put(self, art: Artifact, provenance: dict[str, Any] | None = None) -> str:
        aid = artifact_id(art)
        d = self._dir(aid)
        if (d / "meta.json").exists():
            return aid
        arrays, header = _arrays_and_header(art)
        header["id"] = aid
        header["provenance"] = provenance or {"source": "unknown"}
        _atomic_write(d / "arrays.npz", lambda f: np.savez(f, **arrays))
        # meta.json is written last: its presence marks the artifact complete.
        _atomic_write(d / "meta.json", lambda f: f.write(json.dumps(header, indent=2, default=_json_default).encode()))
        return aid

    def header(self, ref: str) -> dict[str, Any]:
        return json.loads((self._dir(self.resolve(ref)) / "meta.json").read_text())

    def get(self, ref: str) -> Artifact:
        aid = self.resolve(ref)
        header = self.header(aid)
        with np.load(self._dir(aid) / "arrays.npz", allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files}
        if header["kind"] == "series":
            coords = {k.removeprefix("coord:"): v for k, v in arrays.items() if k.startswith("coord:")}
            return Series(
                arrays["stamps"], arrays["data"], name=header["name"], units=header["units"],
                axes=tuple(header["axes"]), stamp_unit=header["stamp_unit"], coords=coords, meta=header["meta"],
            )
        return IntervalSet(arrays["ons"], arrays["offs"], name=header["name"], axis=header["axis"],
                           unit=header["unit"], meta=header["meta"])

    def list(self) -> list[dict[str, Any]]:
        out = []
        for meta in sorted((self.root / "objects").glob("*/meta.json")):
            h = json.loads(meta.read_text())
            axis = h["axes"][0] if h["kind"] == "series" else h["axis"]
            out.append({"id": h["id"], "kind": h["kind"], "name": h["name"], "parametric_axis": axis,
                        "source": h["provenance"].get("source")})
        return out

    # -- step cache --------------------------------------------------------
    def cache_get(self, key: str) -> str | None:
        p = self.root / "cache" / key
        if not p.exists():
            return None
        aid = p.read_text().strip()
        return aid if (self._dir(aid) / "meta.json").exists() else None

    def cache_put(self, key: str, aid: str) -> None:
        _atomic_write(self.root / "cache" / key, lambda f: f.write(aid.encode()))

    # -- datasets ----------------------------------------------------------
    def put_dataset(self, manifest: dict[str, Any]) -> str:
        """Store a resolved dataset manifest; its id is the hash of its contents."""
        did = "dataset:" + hashlib.sha256(canonical_json(manifest).encode()).hexdigest()[:16]
        p = self.root / "datasets" / f"{did.removeprefix('dataset:')}.json"
        if not p.exists():
            _atomic_write(p, lambda f: f.write(json.dumps({**manifest, "id": did}, indent=2, default=_json_default).encode()))
        return did

    def get_dataset(self, ref: str) -> dict[str, Any]:
        digest = ref.removeprefix("dataset:")
        matches = list((self.root / "datasets").glob(digest + "*.json"))
        if len(matches) != 1:
            raise KeyError(f"dataset {ref!r} " + ("is ambiguous" if matches else "not found"))
        return json.loads(matches[0].read_text())

    def list_datasets(self) -> list[dict[str, Any]]:
        out = []
        for p in sorted((self.root / "datasets").glob("*.json")):
            d = json.loads(p.read_text())
            subjects = {s.get("subject") for s in d["sessions"]}
            out.append({"id": d["id"], "name": d["name"], "n_sessions": len(d["sessions"]),
                        "n_subjects": len(subjects - {None})})
        return out

    # -- runs --------------------------------------------------------------
    def put_run(self, run: dict[str, Any]) -> None:
        _atomic_write(self.root / "runs" / f"{run['run_id']}.json",
                      lambda f: f.write(json.dumps(run, indent=2, default=_json_default).encode()))

    def get_run(self, run_id: str) -> dict[str, Any]:
        return json.loads((self.root / "runs" / f"{run_id}.json").read_text())

    def list_runs(self) -> list[dict[str, Any]]:
        runs = [json.loads(p.read_text()) for p in (self.root / "runs").glob("*.json")]
        runs.sort(key=lambda r: r["started_at"], reverse=True)
        return [{k: r[k] for k in ("run_id", "workflow_name", "started_at", "status")} for r in runs]
