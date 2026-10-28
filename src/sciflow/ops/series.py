"""Generalised, dimension-agnostic series operations.

Ported from the MATLAB ``generalizedAnalysisFunctions`` used in Barnes et al.
2026, plus the preprocessing steps repeated across Collate's acquisition
system examples (filtering, downsampling, tick-to-second conversion). Every
``Series`` keeps its parametric axis on axis 0, so none of these need a
``parametricAxis`` argument: they work unchanged on a 1-D trace, a
channels x time recording, a pixels x pixels x time stack, or a depth profile.

"Window", "cutoff" and similar parameters are in units of the parametric axis
(seconds for time, micrometres for depth, ...), unless an op says otherwise.

Fixes relative to the MATLAB versions:
- ``binEventDataToEvents`` / ``collate_event_data_to_events`` used every
  event's window on every iteration; ``event_histogram`` uses each event's own.
- ``collatePairwiseCorrContinuousDataToEvents`` correlated data1 with itself;
  ``pairwise_interval_correlation`` correlates data1 with data2.
- the correlation helpers' ``0.args.post_event_seconds`` exponent (a
  find-and-replace accident) is a square root again.
"""

from __future__ import annotations

import warnings
from typing import Literal

import numpy as np
from pydantic import Field, model_validator

from .. import intervals as iv
from ..registry import NoParams, Params, op
from ..types import IntervalSet, Series, ValidationError, check_same_axis

Statistic = Literal["mean", "median", "std", "sem", "sum", "min", "max"]


# -- helpers ---------------------------------------------------------------

def _as_float(x: np.ndarray) -> np.ndarray:
    return x if np.issubdtype(x.dtype, np.floating) else x.astype(np.float64)


def _match_index(stamps: np.ndarray, t: np.ndarray, match: str = "nearest") -> np.ndarray:
    if match == "next":
        # First sample at or after the event: the original MATLAB/Python behaviour.
        return np.searchsorted(stamps, t, side="left")
    i = np.clip(np.searchsorted(stamps, t), 1, len(stamps) - 1)
    left_closer = (t - stamps[i - 1]) <= (stamps[i] - t)
    return np.where(left_closer, i - 1, i)


def _require_period(series: Series) -> float:
    dt = series.sample_period()
    if not dt:
        raise ValidationError(f"series {series.name!r} needs at least two distinct samples")
    return dt


def _require_regular(series: Series, what: str) -> float:
    dt = _require_period(series)
    if len(series.stamps) > 2 and np.max(np.abs(np.diff(series.stamps) - dt)) > 1e-3 * dt:
        raise ValidationError(f"{what} needs evenly spaced samples; resample with `interpolate_to` first")
    return dt


def _to_samples(value: float, unit: str, dt: float) -> int:
    return int(round(value / dt)) if unit == "stamps" else int(round(value))


def _gather_windows(series: Series, t: np.ndarray, start: int, stop: int,
                    match: str = "nearest") -> tuple[np.ndarray, np.ndarray]:
    """Rows ``data[i + start : i + stop]`` around the sample matched to each position in ``t``.

    Windows that run off either end of the series are NaN-filled so the
    output stays aligned one-to-one with the events.
    """
    stamps, n = series.stamps, len(series.stamps)
    dt = _require_period(series)
    data = _as_float(series.data)
    if len(t) == 0:
        return np.empty((0, stop - start) + series.sample_shape), np.zeros(0, bool)
    ind = _match_index(stamps, t, match)
    valid = (ind + start >= 0) & (ind + stop <= n) & (t >= stamps[0] - dt) & (t <= stamps[-1] + dt)
    idx = np.clip(ind[:, None] + np.arange(start, stop)[None, :], 0, n - 1)
    out = data[idx]
    out[~valid] = np.nan
    return out, valid


def _reduce(x: np.ndarray, statistic: str, axis: int, nan_ignore: bool) -> np.ndarray:
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        if statistic == "sem":
            n = np.sum(~np.isnan(x), axis=axis) if nan_ignore else x.shape[axis]
            sd = np.nanstd(x, axis=axis, ddof=1) if nan_ignore else np.std(x, axis=axis, ddof=1)
            return sd / np.sqrt(n)
        fn = {
            "mean": (np.mean, np.nanmean), "median": (np.median, np.nanmedian),
            "std": (np.std, np.nanstd), "sum": (np.sum, np.nansum),
            "min": (np.min, np.nanmin), "max": (np.max, np.nanmax),
        }[statistic][int(nan_ignore)]
        kw = {"ddof": 1} if statistic == "std" else {}
        return fn(x, axis=axis, **kw)


def _circular_mean(x: np.ndarray, axis: int) -> np.ndarray:
    return np.mod(np.arctan2(np.mean(np.sin(x), axis), np.mean(np.cos(x), axis)), 2 * np.pi)


def _event_positions(events: IntervalSet, anchor: str) -> np.ndarray:
    return events.ons if anchor == "on" else events.offs


def _axis_index(series: Series, axis: str | int, allow_parametric: bool = False) -> int:
    if isinstance(axis, str):
        if axis not in series.axes:
            raise ValidationError(f"no axis {axis!r}; axes are {series.axes}")
        i = series.axes.index(axis)
    else:
        i = int(axis)
    if not (0 if allow_parametric else 1) <= i < series.data.ndim:
        raise ValidationError(f"axis {axis!r} must name a non-parametric axis of {series.axes}")
    return i


def _pearson(x: np.ndarray, y: np.ndarray, omitnan: bool) -> np.ndarray:
    """Column-wise Pearson r along axis 0."""
    mean = np.nanmean if omitnan else np.mean
    total = np.nansum if omitnan else np.sum
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        xs, ys = x - mean(x, axis=0), y - mean(y, axis=0)
        return total(xs * ys, axis=0) / np.sqrt(total(xs**2, axis=0) * total(ys**2, axis=0))


def _interval_slices(stamps: np.ndarray, intervals: IntervalSet):
    starts = np.searchsorted(stamps, intervals.ons, side="left")
    stops = np.searchsorted(stamps, intervals.offs, side="right")
    inside = (intervals.ons >= stamps[0]) & (intervals.offs <= stamps[-1]) & (stops - starts >= 2)
    return starts, stops, inside


def _same_stamps(a: Series, b: Series):
    check_same_axis(a=a, b=b)
    if a.stamps.shape != b.stamps.shape or not np.allclose(a.stamps, b.stamps, rtol=0, atol=1e-12):
        raise ValidationError(
            f"{a.name!r} and {b.name!r} are sampled at different stamps; "
            "resample one onto the other with `interpolate_to` first"
        )


def _epochs(series: Series, positions: np.ndarray, data: np.ndarray, lag: np.ndarray, **kw) -> Series:
    return Series(positions, data, name=series.name, units=kw.pop("units", series.units),
                  axes=(series.axis, "lag") + series.axes[1:], stamp_unit=series.stamp_unit,
                  coords={**series.coords, "lag": lag}, meta=kw.pop("meta"))


# -- selection -------------------------------------------------------------

@op("restrict", category="series", inputs={"series": "series", "intervals": "intervals"}, output="series")
def restrict(p: NoParams, series: Series, intervals: IntervalSet) -> Series:
    """Keep only the samples whose stamps fall inside the intervals (e.g. only running periods, or one cortical layer)."""
    check_same_axis(series=series, intervals=intervals)
    s = iv.normalize(intervals)
    stamps = series.stamps
    if len(s) == 0:
        keep = np.zeros(len(stamps), bool)
    else:
        k = np.searchsorted(s.ons, stamps, side="right") - 1
        keep = (k >= 0) & (stamps <= s.offs[np.clip(k, 0, None)])
    return series.replace(stamps=stamps[keep], data=series.data[keep])


class SelectParams(Params):
    axis: str | int = Field(description="Non-parametric axis to index, by label or position (>= 1).")
    indices: list[int] = Field(description="Indices to keep along that axis.")


@op("select", category="series", inputs={"series": "series"}, output="series", params=SelectParams)
def select(p: SelectParams, series: Series) -> Series:
    """Select channels / pixels / features along one non-parametric axis."""
    ax = _axis_index(series, p.axis)
    name = series.axes[ax]
    coords = dict(series.coords)
    if name in coords:
        coords[name] = coords[name][p.indices]
    return series.replace(data=np.take(series.data, p.indices, axis=ax), coords=coords)


class ReduceParams(Params):
    axis: str | int = Field(description="Non-parametric axis to collapse, by label or position (>= 1).")
    statistic: Statistic = "mean"
    nan_ignore: bool = True


@op("reduce_axis", category="series", inputs={"series": "series"}, output="series", params=ReduceParams)
def reduce_axis(p: ReduceParams, series: Series) -> Series:
    """Collapse a non-parametric axis with a statistic (e.g. average over channels or an ROI's pixels)."""
    ax = _axis_index(series, p.axis)
    removed = series.axes[ax]
    return series.replace(data=_reduce(_as_float(series.data), p.statistic, ax, p.nan_ignore),
                          axes=series.axes[:ax] + series.axes[ax + 1:],
                          coords={k: v for k, v in series.coords.items() if k != removed})


class EventsWhereParams(Params):
    values: list[float] = Field(description="Data values to keep, e.g. cluster ids.")
    channel: list[int] | None = Field(None, description="Index into the non-parametric axes if the series is not 1-D.")


@op("events_where", category="series", inputs={"series": "series"}, output="intervals", params=EventsWhereParams)
def events_where(p: EventsWhereParams, series: Series) -> IntervalSet:
    """Point events at the stamps where the data equals one of ``values``.

    For labelled event streams such as spike times paired with cluster ids
    (Kilosort output in Collate's OpenEphys example), this selects the spikes
    of particular units.
    """
    x = series.data if p.channel is None else series.data[(slice(None), *p.channel)]
    if x.ndim != 1:
        raise ValidationError(f"series has sample shape {series.sample_shape}; pass `channel` to select one signal")
    return IntervalSet.from_times(series.stamps[np.isin(x, p.values)], name=f"{series.name}:{p.values}",
                                  axis=series.axis, unit=series.stamp_unit)


# -- parametric axis ------------------------------------------------------

class ReparametrizeParams(Params):
    axis: str = Field(description="Axis whose coordinates become the new stamps (it must have numeric coords).")
    unit: str | None = Field(None, description="Unit of the new parametric axis, e.g. 'um' for depth.")


@op("reparametrize", category="series", inputs={"series": "series"}, output="series", params=ReparametrizeParams)
def reparametrize(p: ReparametrizeParams, series: Series) -> Series:
    """Make another axis the parametric one (e.g. turn a time x depth CSD into a depth x time series).

    The chosen axis' coordinates become the stamps and the old parametric
    axis becomes an ordinary axis whose coordinates are the old stamps. Every
    operation then works along the new axis: restrict to a cortical layer,
    average over depth bands, find peaks along frequency.
    """
    ax = _axis_index(series, p.axis)
    if p.axis not in series.coords:
        raise ValidationError(f"axis {p.axis!r} has no coordinates to use as stamps")
    new_stamps = np.asarray(series.coords[p.axis], dtype=float)
    data = np.moveaxis(series.data, ax, 0)
    if len(new_stamps) > 1 and np.all(np.diff(new_stamps) <= 0):
        new_stamps, data = new_stamps[::-1], data[::-1]  # e.g. depth listed top-down
    axes = (p.axis,) + tuple(a for i, a in enumerate(series.axes) if i != ax)
    coords = {k: v for k, v in series.coords.items() if k != p.axis}
    coords[series.axis] = series.stamps
    meta = {**series.meta, "coord_units": {**series.meta.get("coord_units", {}), series.axis: series.stamp_unit}}
    return series.replace(stamps=new_stamps, data=data, axes=axes, coords=coords, stamp_unit=p.unit, meta=meta)


class RescaleParams(Params):
    scale: float = Field(1.0, description="Multiply stamps by this (e.g. 1/30000 to turn sample numbers into seconds).")
    offset: float = Field(0.0, description="Then add this.")
    unit: str | None = Field(None, description="Unit of the rescaled stamps; omit to keep the current unit.")


@op("rescale_stamps", category="alignment", inputs={"target": "any"}, output="same:target", params=RescaleParams)
def rescale_stamps(p: RescaleParams, target):
    """Linearly rescale stamps: ``scale * stamp + offset`` (Collate's ``clock_ticks_per_second`` conversion)."""
    if p.scale <= 0:
        raise ValidationError("scale must be positive so stamps stay sorted")
    if isinstance(target, Series):
        return target.replace(stamps=target.stamps * p.scale + p.offset, stamp_unit=p.unit or target.stamp_unit)
    return target.like(target.ons * p.scale + p.offset, target.offs * p.scale + p.offset, unit=p.unit or target.unit)


# -- filtering and resampling ---------------------------------------------

class ButterParams(Params):
    btype: Literal["lowpass", "highpass", "bandpass", "bandstop"]
    cutoff: float | list[float] = Field(description="Cutoff frequency (cycles per unit of the parametric axis, Hz for time); "
                                                    "two values for bandpass / bandstop.")
    order: int = Field(2, ge=1, le=10)
    zero_phase: bool = Field(True, description="Filter forwards and backwards (sosfiltfilt) so peaks don't shift.")


@op("butter_filter", category="filtering", inputs={"series": "series"}, output="series", params=ButterParams)
def butter_filter(p: ButterParams, series: Series) -> Series:
    """Butterworth filter along the parametric axis (e.g. high-pass LFP, low-pass a fluorescence baseline)."""
    from scipy import signal

    fs = 1.0 / _require_regular(series, "butter_filter")
    sos = signal.butter(p.order, p.cutoff, btype=p.btype, fs=fs, output="sos")
    x = _as_float(series.data)
    y = signal.sosfiltfilt(sos, x, axis=0) if p.zero_phase else signal.sosfilt(sos, x, axis=0)
    return series.replace(data=y, meta={**series.meta, "filter": p.model_dump(mode="json")})


class MedianParams(Params):
    window: int = Field(ge=1, description="Window length in samples (odd).")

    @model_validator(mode="after")
    def _odd(self):
        if self.window % 2 == 0:
            raise ValueError("window must be odd")
        return self


@op("median_filter", category="filtering", inputs={"series": "series"}, output="series", params=MedianParams)
def median_filter(p: MedianParams, series: Series) -> Series:
    """Median filter along the parametric axis, e.g. to clean an analog sync line before thresholding."""
    from scipy import ndimage

    size = (p.window,) + (1,) * (series.data.ndim - 1)
    return series.replace(data=ndimage.median_filter(series.data, size=size, mode="nearest"))


class BinParams(Params):
    factor: int = Field(ge=1, description="Number of samples averaged into each bin.")
    axis: str | int = Field(0, description="Axis to bin: 0 / the parametric axis to downsample, or another axis "
                                           "(e.g. 'x' and 'y' to spatially bin an imaging stack).")
    statistic: Literal["mean", "circular_mean", "sum", "median"] = Field(
        "mean", description="circular_mean is for angles in radians, e.g. wheel phase.")
    keep_remainder: bool = Field(True, description="Average leftover samples into a final, smaller bin.")


@op("bin_average", category="filtering", inputs={"series": "series"}, output="series", params=BinParams)
def bin_average(p: BinParams, series: Series) -> Series:
    """Downsample by averaging consecutive samples along any axis (Collate's ``downsample_data``, for any dimensionality).

    Along the parametric axis, each bin's stamp is the mean stamp of its samples.
    """
    ax = _axis_index(series, p.axis, allow_parametric=True)
    x = np.moveaxis(_as_float(series.data), ax, 0)
    n, f = x.shape[0], p.factor
    n_full = n // f
    reduce = {
        "mean": lambda a, axis: np.mean(a, axis=axis), "sum": lambda a, axis: np.sum(a, axis=axis),
        "median": lambda a, axis: np.median(a, axis=axis), "circular_mean": _circular_mean,
    }[p.statistic]
    parts = [reduce(x[:n_full * f].reshape((n_full, f) + x.shape[1:]), 1)]
    rem = n - n_full * f
    if rem and p.keep_remainder:
        parts.append(reduce(x[n_full * f:], 0)[None])
    binned = np.moveaxis(np.concatenate(parts), 0, ax)

    def bin_coord(c):
        c = np.asarray(c, dtype=float)
        out = [c[:n_full * f].reshape(n_full, f).mean(1)]
        if rem and p.keep_remainder:
            out.append(c[n_full * f:].mean(keepdims=True))
        return np.concatenate(out)

    if ax == 0:
        return series.replace(stamps=bin_coord(series.stamps), data=binned)
    name = series.axes[ax]
    coords = dict(series.coords)
    if name in coords:
        coords[name] = bin_coord(coords[name])
    return series.replace(data=binned, coords=coords)


@op("interpolate_to", category="alignment", inputs={"series": "series", "target": "series"}, output="series")
def interpolate_to(p: NoParams, series: Series, target: Series) -> Series:
    """Linearly resample ``series`` onto ``target``'s stamps (NaN outside series' range).

    Use this to put two streams on a common clock before correlating them, or
    (after ``restrict``) to fill in samples that were dropped as artefacts.
    """
    check_same_axis(series=series, target=target)
    src, tgt = series.stamps, target.stamps
    x = _as_float(series.data).reshape(len(src), -1)
    if len(src) < 2:
        raise ValidationError("series needs at least two samples to interpolate")
    i = np.clip(np.searchsorted(src, tgt, side="right"), 1, len(src) - 1)
    s0, s1 = src[i - 1], src[i]
    with np.errstate(all="ignore"):
        w = np.where(s1 > s0, (tgt - s0) / (s1 - s0), 0.0)[:, None]
    out = x[i - 1] * (1 - w) + x[i] * w
    out[(tgt < src[0]) | (tgt > src[-1])] = np.nan
    return series.replace(stamps=tgt, data=out.reshape((len(tgt),) + series.sample_shape),
                          meta={**series.meta, "resampled_to": target.name})


class ArithmeticParams(Params):
    operation: Literal["add", "subtract", "multiply", "divide"]


@op("arithmetic", category="series", inputs={"a": "series", "b": "series"}, output="series", params=ArithmeticParams)
def arithmetic(p: ArithmeticParams, a: Series, b: Series) -> Series:
    """Element-wise a (op) b for two series on the same stamps; b's sample shape must broadcast to a's.

    For example, dF/F against a low-pass baseline: ``subtract`` the
    ``butter_filter`` output, then ``divide`` by it.
    """
    _same_stamps(a, b)
    fn = {"add": np.add, "subtract": np.subtract, "multiply": np.multiply, "divide": np.divide}[p.operation]
    try:
        with np.errstate(divide="ignore", invalid="ignore"):
            out = fn(_as_float(a.data), _as_float(b.data))
    except ValueError:
        raise ValidationError(f"sample shapes {a.sample_shape} and {b.sample_shape} don't broadcast") from None
    if out.shape != a.data.shape:
        raise ValidationError(f"result shape {out.shape} differs from a's {a.data.shape}")
    units = a.units if p.operation in ("add", "subtract") else None
    return a.replace(data=out, units=units, name=f"{p.operation}({a.name},{b.name})")


class StandardizeParams(Params):
    method: Literal["zscore", "demean", "minmax"] = "zscore"


@op("standardize", category="series", inputs={"series": "series"}, output="series", params=StandardizeParams)
def standardize(p: StandardizeParams, series: Series) -> Series:
    """Standardise each element along the parametric axis (z-score, mean-subtract or min-max), ignoring NaN."""
    x = _as_float(series.data)
    with np.errstate(all="ignore"):
        if p.method == "zscore":
            x = (x - _reduce(x, "mean", 0, True)) / _reduce(x, "std", 0, True)
        elif p.method == "demean":
            x = x - _reduce(x, "mean", 0, True)
        else:
            lo, hi = _reduce(x, "min", 0, True), _reduce(x, "max", 0, True)
            x = (x - lo) / (hi - lo)
    return series.replace(data=x, units=None if p.method != "demean" else series.units)


# -- peri-event ------------------------------------------------------------

class PeriEventParams(Params):
    pre: float = Field(ge=0, description="Length of the window before each event.")
    post: float = Field(gt=0, description="Length of the window after each event.")
    unit: Literal["stamps", "samples"] = Field("stamps", description="pre/post in units of the parametric axis (e.g. seconds) or in samples.")
    anchor: Literal["on", "off"] = Field("on", description="Align to interval onsets or offsets.")
    match: Literal["nearest", "next"] = Field(
        "nearest", description="Sample each event is aligned to: the nearest one, or the first at or after the event "
                               "(the original MATLAB behaviour)."
    )


@op("peri_event", category="peri-event", inputs={"series": "series", "events": "intervals"},
    output="series", params=PeriEventParams)
def peri_event(p: PeriEventParams, series: Series, events: IntervalSet) -> Series:
    """Cut a window of data around every event (MATLAB ``collateContinuousDataToEvents``).

    Returns epochs with shape ``(n_events, n_lags, *sample_shape)``: the
    stamps are the event positions and the new ``lag`` axis carries the
    offset relative to each event. Windows that run off the series are NaN.
    """
    check_same_axis(series=series, events=events)
    dt = _require_period(series)
    start, stop = -_to_samples(p.pre, p.unit, dt), _to_samples(p.post, p.unit, dt)
    t = _event_positions(events, p.anchor)
    out, valid = _gather_windows(series, t, start, stop, p.match)
    return _epochs(series, t, out, np.arange(start, stop) * dt,
                   meta={"epochs_of": series.name, "events": events.name, "n_valid": int(valid.sum())})


class PeriEventNormalizeParams(PeriEventParams):
    baseline_start: float = Field(description="Baseline start relative to the event (negative = before), in `unit`.")
    baseline_stop: float = Field(description="Baseline stop relative to the event (exclusive), in `unit`.")
    method: Literal["dff", "zscore", "subtract"] = Field(
        "dff", description="dff: (x - mu) / mu; zscore: (x - mu) / sigma; subtract: x - mu, using baseline statistics."
    )


@op("peri_event_normalize", category="peri-event", inputs={"series": "series", "events": "intervals"},
    output="series", params=PeriEventNormalizeParams)
def peri_event_normalize(p: PeriEventNormalizeParams, series: Series, events: IntervalSet) -> Series:
    """Peri-event epochs normalised to a per-event baseline (MATLAB ``collateNormalizeContinuousDataToEvents``)."""
    check_same_axis(series=series, events=events)
    dt = _require_period(series)
    to_s = lambda v: _to_samples(v, p.unit, dt)  # noqa: E731
    start, stop = -to_s(p.pre), to_s(p.post)
    b0, b1 = to_s(p.baseline_start), to_s(p.baseline_stop)
    if b1 <= b0:
        raise ValidationError("baseline_stop must be after baseline_start")
    t = _event_positions(events, p.anchor)
    x, valid = _gather_windows(series, t, start, stop, p.match)
    base, bvalid = _gather_windows(series, t, b0, b1, p.match)
    mu = _reduce(base, "mean", 1, True)[:, None]
    with np.errstate(all="ignore"):
        if p.method == "dff":
            x = (x - mu) / mu
        elif p.method == "zscore":
            x = (x - mu) / _reduce(base, "std", 1, True)[:, None]
        else:
            x = x - mu
    ok = valid & bvalid
    x[~ok] = np.nan
    return _epochs(series, t, x, np.arange(start, stop) * dt,
                   units=None if p.method != "subtract" else series.units,
                   meta={"epochs_of": series.name, "events": events.name, "normalization": p.method,
                         "n_valid": int(ok.sum())})


class EpochAverageParams(Params):
    statistic: Statistic = "mean"
    nan_ignore: bool = Field(True, description="Ignore NaN epochs (e.g. windows that ran off the series).")


@op("epoch_average", category="peri-event", inputs={"epochs": "series"}, output="series", params=EpochAverageParams)
def epoch_average(p: EpochAverageParams, epochs: Series) -> Series:
    """Average peri-event epochs into an event-triggered response.

    The result is itself a series whose parametric axis is ``lag`` (offset
    from the event), so it can be plotted, restricted or reduced like any other.
    """
    if len(epochs.axes) < 2 or epochs.axes[1] != "lag":
        raise ValidationError("epoch_average expects the output of peri_event / peri_event_normalize / event_histogram")
    data = _reduce(_as_float(epochs.data), p.statistic, 0, p.nan_ignore)
    n = len(epochs.stamps)
    n_valid = int(np.sum(~np.all(np.isnan(epochs.data.reshape(n, -1)), axis=1))) if n else 0
    return Series(epochs.coords["lag"], data, name=epochs.name, units=epochs.units,
                  axes=("lag",) + epochs.axes[2:], stamp_unit=epochs.stamp_unit,
                  coords={k: v for k, v in epochs.coords.items() if k != "lag"},
                  meta={**epochs.meta, "relative_to": epochs.axis, "statistic": p.statistic,
                        "n_epochs": n, "n_valid": n_valid})


class HistogramParams(Params):
    start: float = Field(description="Window start relative to each reference event (e.g. -2).")
    stop: float = Field(description="Window end relative to each reference event (e.g. 2).")
    n_bins: int = Field(gt=0)
    normalize: Literal["count", "rate"] = Field("count", description="rate divides counts by bin width (e.g. events/s).")


@op("event_histogram", category="peri-event", inputs={"events": "intervals", "reference": "intervals"},
    output="series", params=HistogramParams)
def event_histogram(p: HistogramParams, events: IntervalSet, reference: IntervalSet) -> Series:
    """Bin event positions around each reference event (peri-event time histogram, e.g. spikes around stimuli).

    Returns ``(n_reference, n_bins)`` epochs; feed into ``epoch_average`` for a PSTH.
    """
    check_same_axis(events=events, reference=reference)
    if p.stop <= p.start:
        raise ValidationError("stop must be greater than start")
    edges = np.linspace(p.start, p.stop, p.n_bins + 1)
    ev = np.sort(events.ons)
    ref = reference.ons
    counts = np.zeros((len(ref), p.n_bins))
    lo = np.searchsorted(ev, ref + p.start, side="left")
    hi = np.searchsorted(ev, ref + p.stop, side="right")
    for i in range(len(ref)):
        counts[i] = np.histogram(ev[lo[i]:hi[i]] - ref[i], bins=edges)[0]
    if p.normalize == "rate":
        counts /= np.diff(edges)
    return Series(ref, counts, name=events.name,
                  units=f"events/{reference.unit or 'unit'}" if p.normalize == "rate" else "count",
                  axes=(reference.axis, "lag"), stamp_unit=reference.unit,
                  coords={"lag": (edges[:-1] + edges[1:]) / 2},
                  meta={"events": events.name, "reference": reference.name})


# -- correlation -----------------------------------------------------------

class CorrParams(Params):
    omitnan: bool = False


@op("interval_correlation", category="correlation",
    inputs={"a": "series", "b": "series", "intervals": "intervals"}, output="series", params=CorrParams)
def interval_correlation(p: CorrParams, a: Series, b: Series, intervals: IntervalSet) -> Series:
    """Within each interval, correlate a and b element-wise (MATLAB ``collateCorrContinuousDataToEvents``).

    a and b must share stamps and sample shape. Output has one row per
    interval (stamped at its onset) and a's sample shape; intervals that fall
    outside the series are NaN.
    """
    _same_stamps(a, b)
    check_same_axis(a=a, intervals=intervals)
    if a.sample_shape != b.sample_shape:
        raise ValidationError(f"sample shapes differ: {a.sample_shape} vs {b.sample_shape}")
    x, y = _as_float(a.data).reshape(len(a.stamps), -1), _as_float(b.data).reshape(len(b.stamps), -1)
    starts, stops, inside = _interval_slices(a.stamps, intervals)
    out = np.full((len(intervals), x.shape[1]), np.nan)
    for i in np.flatnonzero(inside):
        out[i] = _pearson(x[starts[i]:stops[i]], y[starts[i]:stops[i]], p.omitnan)
    return Series(intervals.ons, out.reshape((len(intervals),) + a.sample_shape), units="r",
                  name=f"corr({a.name},{b.name})", axes=a.axes, stamp_unit=a.stamp_unit, coords=a.coords,
                  meta={"intervals": intervals.name})


@op("pairwise_interval_correlation", category="correlation",
    inputs={"a": "series", "b": "series", "intervals": "intervals"}, output="series", params=CorrParams)
def pairwise_interval_correlation(p: CorrParams, a: Series, b: Series, intervals: IntervalSet) -> Series:
    """Within each interval, correlate every element of a with every element of b.

    Output shape is ``(n_intervals, n_a, n_b)`` where n_a and n_b are the
    flattened sample sizes. With ``omitnan``, samples with a NaN in any
    element are dropped from that interval.
    """
    _same_stamps(a, b)
    check_same_axis(a=a, intervals=intervals)
    x, y = _as_float(a.data).reshape(len(a.stamps), -1), _as_float(b.data).reshape(len(b.stamps), -1)
    starts, stops, inside = _interval_slices(a.stamps, intervals)
    out = np.full((len(intervals), x.shape[1], y.shape[1]), np.nan)
    for i in np.flatnonzero(inside):
        xi, yi = x[starts[i]:stops[i]], y[starts[i]:stops[i]]
        if p.omitnan:
            ok = ~(np.isnan(xi).any(1) | np.isnan(yi).any(1))
            xi, yi = xi[ok], yi[ok]
        if len(xi) < 2:
            continue
        with np.errstate(all="ignore"):
            zx = (xi - xi.mean(0)) / xi.std(0)
            zy = (yi - yi.mean(0)) / yi.std(0)
            out[i] = zx.T @ zy / len(xi)
    return Series(intervals.ons, out, units="r", name=f"pairwise_corr({a.name},{b.name})",
                  axes=(a.axis, "a", "b"), stamp_unit=a.stamp_unit, meta={"intervals": intervals.name})


# -- clock alignment -------------------------------------------------------

def fit_clock(local: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, int, float]:
    """Collate's ``timebase_transform``: match pulse trains by cross-correlating
    inter-pulse intervals, then fit ``reference = slope * local + offset``."""
    from scipy import signal

    if len(local) < 3 or len(reference) < 3:
        raise ValidationError("need at least 3 clock pulses on each system")
    lags = signal.correlation_lags(len(reference) - 1, len(local) - 1, mode="full")
    corr = signal.correlate(np.diff(reference), np.diff(local), mode="full")
    lag = int(lags[np.argmax(corr)])
    ref, loc = (reference[lag:], local) if lag > 0 else (reference, local[-lag:])
    n = min(len(ref), len(loc))
    coef = np.polyfit(loc[:n], ref[:n], 1)
    resid = float(np.sqrt(np.mean((np.polyval(coef, loc[:n]) - ref[:n]) ** 2)))
    if coef[0] <= 0:
        raise ValidationError("clock fit has non-positive slope; check that the pulse trains match")
    return coef, lag, resid


@op("align_clock", category="alignment",
    inputs={"target": "any", "local_clock": "intervals", "reference_clock": "intervals"}, output="same:target")
def align_clock(p: NoParams, target, local_clock: IntervalSet, reference_clock: IntervalSet):
    """Map ``target`` from one acquisition system's clock onto a reference system's clock.

    ``local_clock`` and ``reference_clock`` are the shared sync pulses as
    recorded by each system (Collate's alignment streams). The local clock
    may be in raw ticks or frame numbers: the output takes the reference
    clock's unit. Works on series and interval sets; the fit is recorded in
    the output's metadata.
    """
    check_same_axis(target=target, local_clock=local_clock)
    coef, lag, resid = fit_clock(local_clock.ons, reference_clock.ons)
    info = {"clock_alignment": {"slope": float(coef[0]), "offset": float(coef[1]), "pulse_lag": lag,
                                "residual_rms": resid, "from_unit": local_clock.unit}}
    if isinstance(target, Series):
        return target.replace(stamps=np.polyval(coef, target.stamps), axes=(reference_clock.axis,) + target.axes[1:],
                              stamp_unit=reference_clock.unit, meta={**target.meta, **info})
    return target.like(np.polyval(coef, target.ons), np.polyval(coef, target.offs), axis=reference_clock.axis,
                       unit=reference_clock.unit, meta={**target.meta, **info})
