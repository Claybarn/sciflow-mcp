"""Import raw files into the sciflow format and export artifacts back out.

Supported inputs: ``.npy``, ``.npz``, ``.csv``, ``.mat`` (v5 via scipy;
v7.3/HDF5 via h5py if installed). Collate's ``Experiment.save_to_file``
layout (one struct per stream with ``timestamps`` / ``data`` /
``meta_data``) is read with ``stream=<name>``. CSV columns of ISO dates
become seconds since the Unix epoch.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np

from .types import IntervalSet, Series, ValidationError


def _load_mapping(path: Path, stream: str | None) -> dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as z:
            return {k: z[k] for k in z.files}
    if suffix == ".mat":
        try:
            from scipy.io import loadmat
            raw = loadmat(path, squeeze_me=True, struct_as_record=False)
            m = {k: v for k, v in raw.items() if not k.startswith("__")}
            if stream is not None:
                s = m[stream]
                s = s[0] if isinstance(s, np.ndarray) and s.dtype == object else s
                m = {f: getattr(s, f) for f in s._fieldnames}
            return m
        except NotImplementedError:  # v7.3 is HDF5
            try:
                import h5py
            except ImportError:
                raise ValidationError("MATLAB v7.3 files need h5py: `uv add h5py`") from None
            with h5py.File(path, "r") as f:
                g = f[stream] if stream else f
                # MATLAB writes column-major, so HDF5 sees every array transposed.
                return {k: np.asarray(g[k]).T for k in g.keys() if isinstance(g[k], h5py.Dataset)}
    raise ValidationError(f"unsupported container {suffix}")


def _csv_times(col: np.ndarray) -> np.ndarray:
    """Numeric times pass through; date strings (e.g. 2024-03-01) become datetime64."""
    if col.dtype.kind in "US":
        try:
            return np.array(col, dtype="datetime64[ns]")
        except ValueError:
            raise ValidationError(f"could not parse times such as {col[0]!r}; use ISO 8601 dates or numbers") from None
    return col.astype(float)


def _strip_strings(a: np.ndarray) -> np.ndarray:
    return np.char.strip(a) if a.dtype.kind == "U" else a


def _unwrap(a) -> np.ndarray:
    # Collate wraps arrays in a 1-element list before savemat.
    a = np.asarray(a)
    while a.dtype == object and a.size == 1:
        a = np.asarray(a.item())
    return a


def import_file(
    path: str | Path,
    kind: Literal["series", "intervals"] = "series",
    *,
    name: str | None = None,
    units: str | None = None,
    stream: str | None = None,
    data_key: str = "data",
    stamps_key: str | None = None,
    ons_key: str = "ons",
    offs_key: str = "offs",
    times_key: str | None = None,
    stamp_column: str | None = None,
    parametric_axis: int = 0,
    sample_rate: float | None = None,
    start: float = 0.0,
    ticks_per_unit: float | None = None,
    axes: list[str] | None = None,
    axis: str | None = None,
    unit: str | None = None,
) -> Series | IntervalSet:
    """Read one series or interval set from a file.

    ``stamps_key`` defaults to ``stamps`` or Collate's ``timestamps``.
    ``axes`` labels every data axis, the first naming the parametric axis;
    ``axis`` alone names just the parametric axis (e.g. ``depth``), and
    ``unit`` its unit. ``ticks_per_unit`` divides raw stamps, e.g. 30000 for
    sample numbers recorded at 30 kHz (Collate's ``clock_ticks_per_second``).
    """
    path = Path(path).expanduser()
    name = name or (stream or path.stem)
    suffix = path.suffix.lower()
    scale = (lambda t: t / ticks_per_unit) if ticks_per_unit else (lambda t: t)  # noqa: E731
    iv_kw = {"name": name, "axis": axis or "time", "unit": unit}

    if suffix == ".csv":
        table = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
        cols = list(table.dtype.names)
        if kind == "intervals":
            if times_key:
                return IntervalSet.from_times(scale(_csv_times(table[times_key])), **iv_kw)
            return IntervalSet(scale(_csv_times(table[ons_key])), scale(_csv_times(table[offs_key])), **iv_kw)
        scol = stamp_column or cols[0]
        data_cols = [c for c in cols if c != scol]
        data = np.column_stack([table[c].astype(float) for c in data_cols])
        multi = len(data_cols) > 1
        return Series(scale(_csv_times(table[scol])), data if multi else data[:, 0], name=name, units=units,
                      axes=(axis or "time", "channel") if multi else (axis or "time",), stamp_unit=unit,
                      coords={"channel": np.array(data_cols)} if multi else {})

    if suffix == ".npy":
        m = {"data": np.load(path, allow_pickle=False)}
        data_key = "data"
    else:
        m = _load_mapping(path, stream)

    if kind == "intervals":
        if times_key:
            return IntervalSet.from_times(scale(_unwrap(m[times_key]).ravel()), **iv_kw)
        if ons_key in m:
            return IntervalSet(scale(_unwrap(m[ons_key]).ravel()), scale(_unwrap(m[offs_key]).ravel()), **iv_kw)
        raise ValidationError(f"no {ons_key!r}/{offs_key!r} (or times_key) in {path.name}; keys: {sorted(m)}")

    if data_key not in m:
        raise ValidationError(f"no {data_key!r} in {path.name}; keys: {sorted(m)}")
    data = np.moveaxis(_unwrap(m[data_key]), parametric_axis, 0)
    key = stamps_key or next((k for k in ("stamps", "timestamps") if k in m), None)
    if key and key in m:
        stamps = scale(_unwrap(m[key]).ravel())
    elif sample_rate:
        stamps = start + np.arange(data.shape[0]) / sample_rate
    else:
        raise ValidationError(f"no stamps found in {path.name} (keys: {sorted(m)}); pass stamps_key or sample_rate")
    if axes is None and "axes" in m:  # written by export_artifact
        # MATLAB char matrices pad shorter strings with spaces.
        axes = [str(a).strip() for a in np.atleast_1d(_unwrap(m["axes"]))]
    if axes is None and axis:
        axes = [axis] + [f"dim{i}" for i in range(1, data.ndim)]
    if unit is None and "stamp_unit" in m:
        u = _unwrap(m["stamp_unit"])
        unit = str(u).strip() if u.size and str(u).strip() else None
    coords = {k.removeprefix("coord_"): _strip_strings(_unwrap(v).ravel()) for k, v in m.items() if k.startswith("coord_")}
    if axes is not None:
        coords = {k: v for k, v in coords.items() if k in axes[1:]}
    else:
        coords = {}
    return Series(stamps, data, name=name, units=units, axes=tuple(axes) if axes else (),
                  stamp_unit=unit, coords=coords)


def export_artifact(art: Series | IntervalSet, path: str | Path, fmt: Literal["npz", "mat", "csv"] | None = None) -> Path:
    """Write ``.npz`` / ``.mat`` (Collate struct layout) / ``.csv``.

    Series on the time axis are written with Collate's ``timestamps`` key;
    other parametric axes use ``stamps``. Axis labels, the stamp unit and
    coordinates travel with the data so the file re-imports losslessly.
    """
    path = Path(path).expanduser()
    fmt = fmt or path.suffix.lstrip(".").lower()
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(art, IntervalSet):
        arrays = {"ons": art.ons, "offs": art.offs, "axis": art.axis, "unit": art.unit or ""}
    else:
        key = "timestamps" if art.axis == "time" else "stamps"
        arrays = {key: art.stamps, "data": art.data, "axes": np.array(art.axes), "stamp_unit": art.stamp_unit or "",
                  **{f"coord_{k}": v for k, v in art.coords.items()}}
    if fmt == "npz":
        np.savez(path, **arrays)
    elif fmt == "mat":
        from scipy.io import savemat
        # Collate layout: one struct per stream, parametric axis first.
        savemat(path, {(art.name or "stream").replace(":", "_").replace("-", "_"): arrays}, do_compression=True)
    elif fmt == "csv":
        if isinstance(art, IntervalSet):
            np.savetxt(path, np.column_stack([art.ons, art.offs]), delimiter=",", header="ons,offs", comments="")
        else:
            if art.data.ndim > 2:
                raise ValidationError("csv export supports at most one non-parametric axis; use npz or mat")
            data = art.data.reshape(len(art.stamps), -1)
            labels = art.coords.get(art.axes[1]) if art.data.ndim == 2 else None
            names = [str(c) for c in labels] if labels is not None else [f"c{i}" for i in range(data.shape[1])]
            header = ",".join([art.axis] + names)
            np.savetxt(path, np.column_stack([art.stamps, data]), delimiter=",", header=header, comments="")
    else:
        raise ValidationError(f"unsupported export format {fmt!r}")
    return path


def from_collate(data_stream) -> Series:
    """Convert a ``collate.CollateClasses.DataStream`` (or ``GeneralizedData.Data``) in memory."""
    stamps = getattr(data_stream, "parametric_stamps", None)
    if stamps is None:
        stamps = data_stream.timestamps
    axis = getattr(data_stream, "parametric_axis", None)
    if axis is None:
        axis = getattr(data_stream, "time_axis", 0)
    return Series.from_array(stamps, data_stream.data, parametric_axis=axis,
                             name=getattr(data_stream, "name", ""), meta=dict(data_stream.meta_data or {}))


def export_table(art: Series | IntervalSet, path: str | Path, max_rows: int = 5_000_000) -> Path:
    """Write a long-format ("tidy") CSV: one row per data element.

    Each axis becomes a column holding its stamp / coordinate (or index), and
    attributes of stacked or grouped axes (subject, condition, n, ...) become
    extra columns. This is the shape R, pandas and statsmodels expect for
    mixed-effects models and ANOVAs.
    """
    import csv

    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(art, IntervalSet):
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["on", "off", "extent"])
            w.writerows(zip(art.ons.tolist(), art.offs.tolist(), art.durations().tolist()))
        return path
    if art.data.size > max_rows:
        raise ValidationError(f"{art.data.size} rows exceeds max_rows={max_rows}; reduce or select first")
    idx = np.indices(art.data.shape).reshape(art.data.ndim, -1)
    cols: dict[str, np.ndarray] = {art.axis: art.stamps[idx[0]]}
    attrs = art.meta.get("unit_attributes", {})
    for d, name in enumerate(art.axes[1:], start=1):
        coord = art.coords.get(name)
        cols[name] = coord[idx[d]] if coord is not None else idx[d]
        for attr, values in attrs.get(name, {}).items():
            if attr != name:
                cols[f"{attr}"] = np.asarray(values, dtype=object)[idx[d]]
    cols["value"] = art.data.reshape(-1)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(list(cols))
        w.writerows(zip(*(c.tolist() for c in cols.values())))
    return path
