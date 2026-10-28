"""Core data types.

Every artifact in sciflow is one of two kinds, and both are positioned along
a named *parametric axis*: usually time, but equally cortical depth,
frequency, wavelength or position on a track.

- ``Series``: an N-dimensional array paired sample-for-sample with a 1-D
  vector of stamps along the parametric axis (Collate's ``DataStream``
  invariant, generalised as in ``GeneralizedData.Data``). The parametric axis
  is always axis 0, so analysis functions never need a ``parametricAxis``
  argument. Other axes can be labelled and given coordinates (e.g. the
  ``lag`` axis of peri-event epochs, or electrode depths).
- ``IntervalSet``: sorted ``[on, off]`` intervals along a parametric axis
  (EventLogic ``Events``): behavioural states in time, cortical layers in
  depth, frequency bands. Point events are intervals with ``on == off``.

Because every operation consumes and produces these two kinds, any step's
output can feed any other step that accepts that kind, whatever the
dimensionality of the data. Operations that combine artifacts check that they
share a parametric axis and unit, so time intervals can't silently be applied
to a depth profile.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

ArtifactKind = Literal["series", "intervals"]


class ValidationError(ValueError):
    """Raised when data violates the sciflow format invariants."""


UNIX_EPOCH = {"time_origin": "1970-01-01T00:00:00Z"}


def as_stamps(t) -> tuple[np.ndarray, bool]:
    """Stamps as float64; ``datetime64`` becomes seconds since the Unix epoch.

    Returns the converted array and whether a datetime conversion happened.
    """
    a = np.asarray(t)
    if np.issubdtype(a.dtype, np.datetime64):
        if np.any(np.isnat(a)):
            raise ValidationError("datetime stamps must not contain NaT")
        return a.astype("datetime64[ns]").astype(np.int64) / 1e9, True
    return a.astype(np.float64), False


def _default_unit(axis: str, unit: str | None) -> str | None:
    return "s" if unit is None and axis == "time" else unit


@dataclass(frozen=True)
class Series:
    stamps: np.ndarray
    data: np.ndarray
    name: str = ""
    units: str | None = None
    # Labels for every data axis; axes[0] names the parametric axis.
    axes: tuple[str, ...] = ()
    # Unit of the stamps (default "s" when the parametric axis is time).
    stamp_unit: str | None = None
    # Optional coordinates for the other axes, keyed by axis label.
    coords: dict[str, np.ndarray] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    kind: ArtifactKind = field(default="series", init=False)

    def __post_init__(self):
        stamps, was_datetime = as_stamps(self.stamps)
        if was_datetime:
            object.__setattr__(self, "meta", {**self.meta, **UNIX_EPOCH})
        data = np.asarray(self.data)
        if data.ndim == 0:
            raise ValidationError("data must have at least one (parametric) axis")
        if stamps.ndim != 1:
            raise ValidationError(f"stamps must be 1-D, got shape {stamps.shape}")
        if len(stamps) != data.shape[0]:
            raise ValidationError(f"stamps ({len(stamps)}) and data axis 0 ({data.shape[0]}) differ in length")
        if not np.all(np.isfinite(stamps)):
            raise ValidationError("stamps must be finite")
        if len(stamps) > 1 and np.any(np.diff(stamps) < 0):
            raise ValidationError("stamps must be non-decreasing")
        axes = tuple(self.axes) or ("time",) + tuple(f"dim{i}" for i in range(1, data.ndim))
        if len(axes) != data.ndim:
            raise ValidationError(f"{len(axes)} axis labels given for {data.ndim}-D data")
        if len(set(axes)) != len(axes):
            raise ValidationError(f"axis labels must be unique, got {axes}")
        coords = {k: np.asarray(v) for k, v in self.coords.items()}
        for k, v in coords.items():
            if k not in axes[1:]:
                raise ValidationError(f"coordinate {k!r} does not name a non-parametric axis of {axes}")
            if len(v) != data.shape[axes.index(k)]:
                raise ValidationError(f"coordinate {k!r} length does not match its axis")
        object.__setattr__(self, "stamps", stamps)
        object.__setattr__(self, "data", data)
        object.__setattr__(self, "axes", axes)
        object.__setattr__(self, "coords", coords)
        object.__setattr__(self, "stamp_unit", "s" if was_datetime else _default_unit(axes[0], self.stamp_unit))

    @classmethod
    def from_array(cls, stamps, data, parametric_axis: int = 0, **kw) -> "Series":
        """Build from data whose parametric axis may not be first (Collate's ``time_axis``)."""
        return cls(stamps, np.moveaxis(np.asarray(data), parametric_axis, 0), **kw)

    @classmethod
    def regular(cls, data, sample_rate: float, start: float = 0.0, parametric_axis: int = 0, **kw):
        """Evenly spaced stamps: ``start + i / sample_rate``."""
        data = np.moveaxis(np.asarray(data), parametric_axis, 0)
        return cls(start + np.arange(data.shape[0]) / sample_rate, data, **kw)

    @property
    def axis(self) -> str:
        """Name of the parametric axis."""
        return self.axes[0]

    @property
    def sample_shape(self) -> tuple[int, ...]:
        return self.data.shape[1:]

    def sample_period(self) -> float | None:
        if len(self.stamps) < 2:
            return None
        return float(np.median(np.diff(self.stamps)))

    def replace(self, **changes) -> "Series":
        """Copy with some fields changed (stamps, data, axes, coords, ...)."""
        fields = {k: getattr(self, k) for k in ("stamps", "data", "name", "units", "axes", "stamp_unit", "coords", "meta")}
        return Series(**{**fields, **changes})

    def summary(self) -> dict[str, Any]:
        dt = self.sample_period()
        return {
            "kind": self.kind,
            "name": self.name,
            "units": self.units,
            "parametric_axis": self.axis,
            "stamp_unit": self.stamp_unit,
            "shape": list(self.data.shape),
            "dtype": str(self.data.dtype),
            "axes": list(self.axes),
            "coords": {k: _coord_summary(v) for k, v in self.coords.items() if len(v)},
            "n_samples": len(self.stamps),
            "start": float(self.stamps[0]) if len(self.stamps) else None,
            "stop": float(self.stamps[-1]) if len(self.stamps) else None,
            "sample_rate_estimate": (1.0 / dt) if dt else None,
            "nan_fraction": float(np.mean(np.isnan(self.data))) if np.issubdtype(self.data.dtype, np.floating) else 0.0,
            "meta": self.meta,
        }


def _coord_summary(v: np.ndarray):
    if np.issubdtype(v.dtype, np.number):
        return {"first": float(v[0]), "last": float(v[-1]), "n": len(v)}
    return {"values": v[:10].tolist(), "n": len(v)}


@dataclass(frozen=True)
class IntervalSet:
    ons: np.ndarray
    offs: np.ndarray
    name: str = ""
    axis: str = "time"
    unit: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    kind: ArtifactKind = field(default="intervals", init=False)

    def __post_init__(self):
        ons, on_dt = as_stamps(np.atleast_1d(self.ons))
        offs, off_dt = as_stamps(np.atleast_1d(self.offs))
        if on_dt or off_dt:
            object.__setattr__(self, "meta", {**self.meta, **UNIX_EPOCH})
        if ons.ndim != 1 or ons.shape != offs.shape:
            raise ValidationError("ons and offs must be 1-D and the same length")
        if np.any(np.isnan(ons)) or np.any(np.isnan(offs)):
            raise ValidationError("interval bounds must not be NaN")
        if np.any(ons > offs):
            raise ValidationError("every interval must satisfy on <= off")
        if len(ons) > 1 and np.any(np.diff(ons) < 0):
            order = np.lexsort((offs, ons))
            ons, offs = ons[order], offs[order]
        object.__setattr__(self, "ons", ons)
        object.__setattr__(self, "offs", offs)
        object.__setattr__(self, "unit", "s" if (on_dt or off_dt) else _default_unit(self.axis, self.unit))

    @classmethod
    def from_times(cls, times, **kw) -> "IntervalSet":
        """Point events (on == off)."""
        t = np.asarray(times)
        return cls(t, t.copy(), **kw)

    def like(self, ons, offs, **changes) -> "IntervalSet":
        """New intervals on the same parametric axis and unit."""
        return IntervalSet(ons, offs, **{"name": self.name, "axis": self.axis, "unit": self.unit, "meta": self.meta, **changes})

    def __len__(self) -> int:
        return len(self.ons)

    def durations(self) -> np.ndarray:
        return self.offs - self.ons

    def summary(self) -> dict[str, Any]:
        d = self.durations()
        return {
            "kind": self.kind,
            "name": self.name,
            "parametric_axis": self.axis,
            "unit": self.unit,
            "n_intervals": len(self),
            "start": float(self.ons[0]) if len(self) else None,
            "stop": float(self.offs[-1]) if len(self) else None,
            "total_extent": float(d.sum()) if len(self) else 0.0,
            "median_extent": float(np.median(d)) if len(self) else None,
            "is_point_events": bool(len(self) and np.all(d == 0)),
            "meta": self.meta,
        }


Artifact = Series | IntervalSet


def axis_of(art: Artifact) -> tuple[str, str | None]:
    return (art.axis, art.stamp_unit) if isinstance(art, Series) else (art.axis, art.unit)


def check_same_axis(**arts: Artifact) -> None:
    """Raise unless every artifact lies on the same parametric axis with the same unit."""
    items = list(arts.items())
    name0, (axis0, unit0) = items[0][0], axis_of(items[0][1])
    for name, art in items[1:]:
        axis, unit = axis_of(art)
        if axis != axis0:
            raise ValidationError(
                f"{name0!r} is on the {axis0!r} axis but {name!r} is on {axis!r}; "
                "use `reparametrize` to change a series' parametric axis"
            )
        if unit0 is not None and unit is not None and unit != unit0:
            raise ValidationError(
                f"{name0!r} is in {unit0!r} but {name!r} is in {unit!r}; convert one with `rescale_stamps`"
            )
