"""Interval algebra, delegated to EventLogic.

``IntervalSet`` stores plain arrays; set operations convert to
``eventlogic.Events`` and back. Inputs are normalised with ``self_or`` first
because EventLogic's linear-time sweeps assume each operand is internally
non-overlapping.

XOR is composed as ``(a | b) & ~(a & b)`` rather than using
``Events.__xor__``, which is broken in eventlogic 0.2.0.

Results keep the parametric axis and unit of their first operand; binary
operations require both operands to share them.
"""

from __future__ import annotations

import numpy as np
from eventlogic import Event, Events

from .types import IntervalSet, check_same_axis


def to_events(s: IntervalSet) -> Events:
    if len(s) == 0:
        return Events([])
    return Events.from_arrays(s.ons, s.offs)


def from_events(e: Events, like: IntervalSet, **changes) -> IntervalSet:
    # Drop the Event(None, None) placeholders EventLogic uses for "empty".
    pairs = [(ev.on, ev.off) for ev in e if ev.on is not None and ev.off is not None]
    if not pairs:
        return like.like(np.empty(0), np.empty(0), **changes)
    ons, offs = zip(*pairs)
    return like.like(np.asarray(ons, float), np.asarray(offs, float), **changes)


def _empty(like: IntervalSet) -> IntervalSet:
    return like.like(np.empty(0), np.empty(0))


def normalize(s: IntervalSet) -> IntervalSet:
    """Merge overlapping or touching intervals."""
    return from_events(to_events(s).self_or(), like=s)


def union(a: IntervalSet, b: IntervalSet) -> IntervalSet:
    check_same_axis(a=a, b=b)
    return from_events(to_events(normalize(a)) | to_events(normalize(b)), like=a)


def intersection(a: IntervalSet, b: IntervalSet) -> IntervalSet:
    check_same_axis(a=a, b=b)
    if len(a) == 0 or len(b) == 0:
        return _empty(a)
    return from_events(to_events(normalize(a)) & to_events(normalize(b)), like=a)


def complement(a: IntervalSet, domain: tuple[float, float] | None = None) -> IntervalSet:
    if len(a) == 0:
        out = a.like([-np.inf], [np.inf])
    else:
        out = from_events(~to_events(normalize(a)), like=a)
    if domain is not None:
        out = intersection(out, a.like([domain[0]], [domain[1]]))
    # Closed intervals share their endpoints with the complement, so clipping
    # can leave zero-width slivers such as [0, 0]; they carry no extent.
    keep = out.offs > out.ons
    return a.like(out.ons[keep], out.offs[keep])


def difference(a: IntervalSet, b: IntervalSet) -> IntervalSet:
    check_same_axis(a=a, b=b)
    return intersection(a, complement(b)) if len(b) else normalize(a)


def symmetric_difference(a: IntervalSet, b: IntervalSet) -> IntervalSet:
    return difference(union(a, b), intersection(a, b))


def merge_gaps(a: IntervalSet, max_gap: float) -> IntervalSet:
    """Merge intervals separated by gaps shorter than ``max_gap``."""
    if len(a) == 0:
        return a
    return from_events(to_events(normalize(a)).merge(max_gap), like=a)


def filter_duration(a: IntervalSet, min_duration: float = 0.0, max_duration: float = np.inf) -> IntervalSet:
    d = a.durations()
    keep = (d >= min_duration) & (d <= max_duration)
    return a.like(a.ons[keep], a.offs[keep])


def contained_mask(container: IntervalSet, items: IntervalSet) -> np.ndarray:
    """For each interval in ``items``, is it fully inside some interval of ``container``?"""
    check_same_axis(container=container, items=items)
    if len(container) == 0 or len(items) == 0:
        return np.zeros(len(items), dtype=bool)
    return np.asarray(to_events(container).contains_events(to_events(items)), dtype=bool)


__all__ = [
    "Event", "Events", "to_events", "from_events", "normalize", "union", "intersection",
    "complement", "difference", "symmetric_difference", "merge_gaps", "filter_duration",
    "contained_mask",
]
