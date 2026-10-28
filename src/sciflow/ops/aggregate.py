"""Dataset-level operations: stack per-session results and reduce over groups.

Stacked series carry a ``session`` axis. Per-session attributes (subject,
condition, day, genotype, ...) are stored in
``meta["unit_attributes"][<axis label>]`` as one list per attribute,
aligned with that axis, so later steps can group by them.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import Field

from ..registry import Params, op
from ..types import Series, ValidationError
from .series import Statistic, _as_float, _axis_index, _reduce


class StackParams(Params):
    sessions: list[str] = Field(description="Session id of each stacked item, in order.")
    attributes: dict[str, list[Any]] = Field(default_factory=dict,
                                             description="Attribute name -> one value per session (subject, condition, ...).")


@op("stack_sessions", category="dataset", inputs={"items": "series[]"}, output="series", params=StackParams)
def stack_sessions(p: StackParams, items: list[Series]) -> Series:
    """Stack the same result from many sessions along a new ``session`` axis (axis 1).

    Every item must share stamps, axes and sample shape. That holds for
    event-relative results such as ``epoch_average`` output, but usually not
    for absolute-time series, which should be resampled onto a common grid
    with ``interpolate_to`` first.
    """
    if len(items) != len(p.sessions):
        raise ValidationError(f"{len(items)} items but {len(p.sessions)} session ids")
    for name, values in p.attributes.items():
        if len(values) != len(items):
            raise ValidationError(f"attribute {name!r} has {len(values)} values for {len(items)} sessions")
    if not items:
        raise ValidationError("nothing to stack")
    first = items[0]
    if "session" in first.axes:
        raise ValidationError("items already have a 'session' axis")
    for sid, it in zip(p.sessions[1:], items[1:]):
        problem = None
        if it.axes != first.axes or it.stamp_unit != first.stamp_unit:
            problem = f"axes {it.axes} ({it.stamp_unit}) vs {first.axes} ({first.stamp_unit})"
        elif it.data.shape != first.data.shape:
            problem = f"shape {it.data.shape} vs {first.data.shape}"
        elif not np.allclose(it.stamps, first.stamps, rtol=0, atol=1e-9):
            problem = "different stamps (resample onto a common grid with `interpolate_to`, or stack event-relative results)"
        if problem:
            raise ValidationError(f"session {sid!r} doesn't match session {p.sessions[0]!r}: {problem}")
    data = np.stack([_as_float(it.data) for it in items], axis=1)
    coords = {**first.coords, "session": np.array(p.sessions)}
    attrs = {"session": {"session": list(p.sessions), **p.attributes}}
    meta = {k: v for k, v in first.meta.items() if k not in ("n_valid", "unit_attributes")}
    meta["unit_attributes"] = {**first.meta.get("unit_attributes", {}), **attrs}
    meta["n_valid"] = [it.meta.get("n_valid") for it in items]
    return Series(first.stamps, data, name=first.name, units=first.units,
                  axes=(first.axes[0], "session") + first.axes[1:], stamp_unit=first.stamp_unit,
                  coords=coords, meta=meta)


class GroupReduceParams(Params):
    axis: str = Field("session", description="Axis whose elements are grouped (e.g. 'session').")
    group_by: list[str] = Field(default_factory=list,
                                description="Attributes defining the output groups, e.g. ['condition']. Empty = one group of everything.")
    within: list[str] = Field(default_factory=list, description=(
        "Nesting levels averaged first, outermost first, e.g. ['subject'] averages each subject's sessions before "
        "combining subjects, so subjects with more sessions don't dominate."))
    statistic: Statistic = Field("mean", description="Statistic across the top-level units (e.g. subjects) of each group; "
                                                     "'sem' gives the standard error across subjects.")
    nan_ignore: bool = True


def _group_keys(attrs: dict[str, list], names: list[str], n: int) -> list[tuple]:
    return [tuple(attrs[a][i] for a in names) for i in range(n)]


@op("group_reduce", category="dataset", inputs={"series": "series"}, output="series", params=GroupReduceParams)
def group_reduce(p: GroupReduceParams, series: Series) -> Series:
    """Reduce a stacked axis by group, hierarchically (e.g. sessions -> subjects -> conditions).

    With ``group_by: [condition]`` and ``within: [subject]``, each subject's
    sessions are first averaged, then ``statistic`` is taken across subjects
    in each condition. The output replaces the axis with a ``group`` axis whose
    coordinates are the group labels; ``meta["unit_attributes"]["group"]``
    holds the group attribute values and ``n``, the number of top-level units
    (e.g. subjects) in each group.
    """
    ax = _axis_index(series, p.axis)
    attrs = series.meta.get("unit_attributes", {}).get(p.axis)
    n = series.data.shape[ax]
    if attrs is None:
        raise ValidationError(f"axis {p.axis!r} has no unit attributes; stack it with stack_sessions first")
    missing = [a for a in p.group_by + p.within if a not in attrs]
    if missing:
        raise ValidationError(f"unknown attributes {missing}; available: {sorted(attrs)}")
    x = np.moveaxis(_as_float(series.data), ax, 0)  # units first

    # Collapse nesting levels innermost-first, always keeping group_by + outer levels as the key.
    keys = _group_keys(attrs, p.group_by + p.within, n)
    for depth in range(len(p.within), 0, -1):
        level_keys = [k[:len(p.group_by) + depth] for k in keys]
        uniq = list(dict.fromkeys(level_keys))
        x = np.stack([_reduce(x[[i for i, k in enumerate(level_keys) if k == u]], "mean", 0, p.nan_ignore) for u in uniq])
        keys = uniq
    top = [k[:len(p.group_by)] for k in keys]
    groups = list(dict.fromkeys(top))
    out, counts = [], []
    for g in groups:
        members = [i for i, k in enumerate(top) if k == g]
        out.append(_reduce(x[members], p.statistic, 0, p.nan_ignore))
        counts.append(len(members))
    data = np.moveaxis(np.stack(out), 0, ax)
    labels = np.array([",".join(f"{a}={v}" for a, v in zip(p.group_by, g)) or "all" for g in groups])
    unit_attrs = {k: v for k, v in series.meta.get("unit_attributes", {}).items() if k != p.axis}
    unit_attrs["group"] = {**{a: [g[i] for g in groups] for i, a in enumerate(p.group_by)}, "n": counts}
    meta = {**series.meta, "unit_attributes": unit_attrs,
            "group_reduce": {"over": p.axis, "group_by": p.group_by, "within": p.within, "statistic": p.statistic}}
    meta.pop("n_valid", None)
    axes = series.axes[:ax] + ("group",) + series.axes[ax + 1:]
    coords = {k: v for k, v in series.coords.items() if k != p.axis}
    coords["group"] = labels
    return series.replace(data=data, axes=axes, coords=coords, meta=meta)
