"""Operations that produce interval sets (along time, depth, frequency, ...)."""

from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import Field

from .. import intervals as iv
from ..registry import NoParams, Params, op
from ..types import IntervalSet, Series, ValidationError


class ThresholdParams(Params):
    threshold: float | Literal["midpoint"] = Field(
        description="Value the signal is compared against, or 'midpoint' for (min + max) / 2 as Collate does for "
                    "analog sync and photodiode lines."
    )
    direction: Literal["above", "below"] = Field("above", description="Keep periods where the signal is above or below threshold.")
    channel: list[int] | None = Field(
        None, description="Index into the non-parametric axes selecting a single channel. Required unless the series is 1-D."
    )


@op("threshold_intervals", category="intervals", inputs={"series": "series"}, output="intervals", params=ThresholdParams)
def threshold_intervals(p: ThresholdParams, series: Series) -> IntervalSet:
    """Convert a continuous signal into intervals where it crosses a threshold.

    Each interval runs from the stamp of the first sample past the threshold
    to the stamp of the last. This generalises Collate's
    ``DataStream.continuous_to_event``.
    """
    x = series.data
    if p.channel is not None:
        x = x[(slice(None), *p.channel)]
    if x.ndim != 1:
        raise ValidationError(f"series has sample shape {series.sample_shape}; pass `channel` to select one signal")
    thr = (np.nanmin(x) + np.nanmax(x)) / 2 if p.threshold == "midpoint" else p.threshold
    mask = (x > thr) if p.direction == "above" else (x < thr)
    edges = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    starts = np.flatnonzero(edges == 1)
    stops = np.flatnonzero(edges == -1) - 1
    st = series.stamps
    return IntervalSet(st[starts], st[stops], name=f"{series.name}:{p.direction}{thr:g}",
                       axis=series.axis, unit=series.stamp_unit, meta={"threshold": float(thr)})


class WindowParams(Params):
    pre: float = Field(ge=0, description="Extent before each event, in units of the parametric axis.")
    post: float = Field(ge=0, description="Extent after each event, in units of the parametric axis.")
    anchor: Literal["on", "off"] = Field("on", description="Build windows around interval onsets or offsets.")


@op("event_windows", category="intervals", inputs={"events": "intervals"}, output="intervals", params=WindowParams)
def event_windows(p: WindowParams, events: IntervalSet) -> IntervalSet:
    """Create a fixed window ``[t - pre, t + post]`` around every event."""
    t = events.ons if p.anchor == "on" else events.offs
    return events.like(t - p.pre, t + p.post, name=f"{events.name}:window")


@op("interval_union", category="intervals", inputs={"a": "intervals", "b": "intervals"}, output="intervals")
def interval_union(p: NoParams, a: IntervalSet, b: IntervalSet) -> IntervalSet:
    """Extent covered by either a or b (logical OR)."""
    return iv.union(a, b)


@op("interval_intersection", category="intervals", inputs={"a": "intervals", "b": "intervals"}, output="intervals")
def interval_intersection(p: NoParams, a: IntervalSet, b: IntervalSet) -> IntervalSet:
    """Extent covered by both a and b (logical AND)."""
    return iv.intersection(a, b)


@op("interval_difference", category="intervals", inputs={"a": "intervals", "b": "intervals"}, output="intervals")
def interval_difference(p: NoParams, a: IntervalSet, b: IntervalSet) -> IntervalSet:
    """Extent covered by a but not b (a AND NOT b)."""
    return iv.difference(a, b)


@op("interval_xor", category="intervals", inputs={"a": "intervals", "b": "intervals"}, output="intervals")
def interval_xor(p: NoParams, a: IntervalSet, b: IntervalSet) -> IntervalSet:
    """Extent covered by exactly one of a or b (logical XOR)."""
    return iv.symmetric_difference(a, b)


class ComplementParams(Params):
    start: float | None = Field(None, description="Clip the complement to start here (e.g. session start or pial surface).")
    stop: float | None = Field(None, description="Clip the complement to stop here (e.g. session end).")


@op("interval_complement", category="intervals", inputs={"a": "intervals"}, output="intervals", params=ComplementParams)
def interval_complement(p: ComplementParams, a: IntervalSet) -> IntervalSet:
    """Extent not covered by a (logical NOT), optionally clipped to [start, stop]."""
    lo = -np.inf if p.start is None else p.start
    hi = np.inf if p.stop is None else p.stop
    return iv.complement(a, domain=None if (p.start is None and p.stop is None) else (lo, hi))


class MergeParams(Params):
    max_gap: float = Field(ge=0, description="Merge neighbouring intervals separated by less than this many seconds.")


@op("merge_intervals", category="intervals", inputs={"a": "intervals"}, output="intervals", params=MergeParams)
def merge_intervals(p: MergeParams, a: IntervalSet) -> IntervalSet:
    """Merge intervals separated by short gaps (e.g. to debounce state detection)."""
    return iv.merge_gaps(a, p.max_gap)


class DurationParams(Params):
    min_duration: float = Field(0.0, ge=0, description="Shortest interval kept, in seconds.")
    max_duration: float | None = Field(None, description="Longest interval kept, in seconds. Omit for no limit.")


@op("filter_intervals", category="intervals", inputs={"a": "intervals"}, output="intervals", params=DurationParams)
def filter_intervals(p: DurationParams, a: IntervalSet) -> IntervalSet:
    """Keep only intervals whose duration falls within [min_duration, max_duration]."""
    return iv.filter_duration(a, p.min_duration, np.inf if p.max_duration is None else p.max_duration)


@op("select_contained", category="intervals", inputs={"items": "intervals", "container": "intervals"}, output="intervals")
def select_contained(p: NoParams, items: IntervalSet, container: IntervalSet) -> IntervalSet:
    """Keep the intervals (or point events) of ``items`` that lie entirely inside ``container``.

    For example: licks that occurred during running bouts.
    """
    keep = iv.contained_mask(container, items)
    return items.like(items.ons[keep], items.offs[keep])


class DefineParams(Params):
    ons: list[float] = Field(description="Interval starts. For point events, give the same values as `offs`.")
    offs: list[float] = Field(description="Interval ends.")
    axis: str = Field("time", description="Parametric axis the intervals lie on, e.g. 'time', 'depth', 'frequency', 'lag'.")
    unit: str | None = Field(None, description="Unit of the bounds (defaults to 's' on the time axis).")
    name: str = ""


@op("define_intervals", category="intervals", inputs={}, output="intervals", params=DefineParams)
def define_intervals(p: DefineParams) -> IntervalSet:
    """Intervals written directly into the workflow: frequency bands, cortical layers, a lag window, trial epochs."""
    if len(p.ons) != len(p.offs):
        raise ValidationError("ons and offs must have the same length")
    return IntervalSet(p.ons, p.offs, name=p.name, axis=p.axis, unit=p.unit)
