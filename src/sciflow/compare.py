"""Compare artifacts numerically.

Artifact ids are hashes of exact bytes. The same computation on a different
CPU, OS or math library can differ in the last few bits (about 1e-15
relative), which gives a different id. ``compare`` answers the scientific
question instead: do two results agree within a tolerance?
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .types import Artifact, IntervalSet, Series, axis_of


def _close(a: np.ndarray, b: np.ndarray, rtol: float, atol: float) -> dict[str, Any]:
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        return {"match": False, "reason": f"shape {a.shape} vs {b.shape}"}
    if a.dtype.kind not in "fc" or b.dtype.kind not in "fc":
        equal = bool(np.array_equal(a, b))
        return {"match": equal, **({} if equal else {"reason": "values differ"})}
    nan_a, nan_b = np.isnan(a), np.isnan(b)
    if not np.array_equal(nan_a, nan_b):
        return {"match": False, "reason": f"NaN in different places ({int((nan_a ^ nan_b).sum())} elements)"}
    ok = ~nan_a
    diff = np.abs(a[ok] - b[ok])
    scale = np.abs(b[ok])
    out = {"max_abs_diff": float(diff.max()) if diff.size else 0.0,
           "max_rel_diff": float((diff / np.where(scale > 0, scale, 1)).max()) if diff.size else 0.0,
           "fraction_differing": float(np.mean(diff > 0)) if diff.size else 0.0}
    out["match"] = bool(np.all(diff <= atol + rtol * scale))
    return out


def compare(a: Artifact, b: Artifact, rtol: float = 1e-9, atol: float = 0.0) -> dict[str, Any]:
    """Structural equality plus element-wise ``|a - b| <= atol + rtol * |b|`` for floating-point arrays."""
    if a.kind != b.kind:
        return {"match": False, "reason": f"kinds differ: {a.kind} vs {b.kind}"}
    if axis_of(a) != axis_of(b):
        return {"match": False, "reason": f"parametric axis differs: {axis_of(a)} vs {axis_of(b)}"}
    if isinstance(a, Series):
        if a.axes != b.axes:
            return {"match": False, "reason": f"axes differ: {a.axes} vs {b.axes}"}
        parts = {"stamps": _close(a.stamps, b.stamps, rtol, atol), "data": _close(a.data, b.data, rtol, atol)}
        for k in sorted(set(a.coords) | set(b.coords)):
            if k not in a.coords or k not in b.coords:
                parts[f"coord:{k}"] = {"match": False, "reason": "missing on one side"}
            else:
                parts[f"coord:{k}"] = _close(a.coords[k], b.coords[k], rtol, atol)
    else:
        assert isinstance(a, IntervalSet) and isinstance(b, IntervalSet)
        parts = {"ons": _close(a.ons, b.ons, rtol, atol), "offs": _close(a.offs, b.offs, rtol, atol)}
    return {"match": all(p["match"] for p in parts.values()), "rtol": rtol, "atol": atol, "parts": parts}
