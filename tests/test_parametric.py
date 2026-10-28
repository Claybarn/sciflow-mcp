"""Parametric axes other than time, and the preprocessing ops from Collate's examples."""

import numpy as np
import pytest

from sciflow import IntervalSet, Series, Store, ValidationError, io, run
from sciflow.registry import get_op


def call(op_name, inputs, **params):
    spec = get_op(op_name)
    return spec.fn(spec.params.model_validate(params), **inputs)


def csd(n_t=200, depths=np.arange(1000, -1, -50.0)):
    """time x depth, depth listed top-down as from a probe."""
    rng = np.random.default_rng(0)
    data = rng.normal(size=(n_t, len(depths)))
    data[:, depths < 300] += 5  # a 'deep layer' with high signal
    return Series.regular(data, 100.0, axes=("time", "depth"), coords={"depth": depths}, name="csd")


# -- axis bookkeeping ---------------------------------------------------------

def test_axis_mismatch_is_rejected():
    depth_profile = call("reparametrize", {"series": csd()}, axis="depth", unit="um")
    running = IntervalSet([0.1], [0.5], name="running")
    with pytest.raises(ValidationError, match="reparametrize"):
        call("restrict", {"series": depth_profile, "intervals": running})


def test_unit_mismatch_is_rejected():
    s = Series.regular(np.zeros(100), 1000.0, stamp_unit="ms")
    with pytest.raises(ValidationError, match="rescale_stamps"):
        call("peri_event", {"series": s, "events": IntervalSet.from_times([0.05])}, pre=0.01, post=0.01)


def test_reparametrize_and_restrict_to_layer():
    prof = call("reparametrize", {"series": csd()}, axis="depth", unit="um")
    assert prof.axes == ("depth", "time") and prof.stamp_unit == "um"
    assert np.all(np.diff(prof.stamps) > 0)  # flipped to ascending
    np.testing.assert_allclose(prof.coords["time"][:3], [0, 0.01, 0.02])
    layer = call("define_intervals", {}, ons=[0], offs=[250], axis="depth", unit="um", name="deep")
    deep = call("restrict", {"series": prof, "intervals": layer})
    assert deep.stamps.tolist() == [0, 50, 100, 150, 200, 250]
    mean_over_time = call("reduce_axis", {"series": deep}, axis="time")
    assert np.all(mean_over_time.data > 4)
    # round-trip back to time keeps the data intact
    back = call("reparametrize", {"series": prof}, axis="time", unit="s")
    np.testing.assert_allclose(back.data, csd().data[:, ::-1])


def test_lag_axis_window():
    s = Series.regular(np.arange(1000.0), 100.0)
    ep = call("peri_event", {"series": s, "events": IntervalSet.from_times([2.0, 5.0])}, pre=0.5, post=1.0)
    resp = call("epoch_average", {"epochs": ep})
    assert resp.axis == "lag" and resp.stamp_unit == "s"
    early = call("define_intervals", {}, ons=[0.0], offs=[0.2], axis="lag", unit="s")
    w = call("restrict", {"series": resp, "intervals": early})
    np.testing.assert_allclose(w.stamps, np.arange(0, 0.21, 0.01), atol=1e-9)


# -- Collate preprocessing ops -------------------------------------------------

def test_rescale_and_align_from_ticks():
    rng = np.random.default_rng(5)
    ref = np.cumsum(rng.uniform(0.8, 1.2, 60))                      # seconds
    ticks = np.round((ref - 3.0) * 30000.0)                          # same pulses as 30 kHz sample numbers
    local = IntervalSet.from_times(ticks, axis="time", unit="ticks")
    spikes = Series(ticks[5:15], np.arange(10), stamp_unit="ticks")
    out = call("align_clock", {"target": spikes, "local_clock": local,
                               "reference_clock": IntervalSet.from_times(ref)})
    assert out.stamp_unit == "s"
    np.testing.assert_allclose(out.stamps, ref[5:15], atol=1e-4)
    by_hand = call("rescale_stamps", {"target": spikes}, scale=1 / 30000.0, offset=3.0, unit="s")
    np.testing.assert_allclose(by_hand.stamps, ref[5:15], atol=1e-4)


def test_median_filter_midpoint_threshold_recovers_sync_pulses():
    t = np.arange(0, 10, 1e-3)
    clean = ((t % 1.0) < 0.1).astype(float) * 5
    noisy = clean.copy()
    noisy[np.random.default_rng(1).choice(len(t), 40, replace=False)] = 5  # single-sample glitches
    s = Series(t, noisy)
    raw = call("threshold_intervals", {"series": s}, threshold="midpoint")
    filt = call("threshold_intervals", {"series": call("median_filter", {"series": s}, window=7)}, threshold="midpoint")
    assert len(filt) == 10 and len(raw) > 10
    np.testing.assert_allclose(filt.ons, np.arange(10.0), atol=1e-9)


def test_butter_filter():
    t = np.arange(0, 4, 1e-3)
    slow, fast = np.sin(2 * np.pi * 2 * t), np.sin(2 * np.pi * 200 * t)
    s = Series(t, np.stack([slow + fast, slow], 1))
    lp = call("butter_filter", {"series": s}, btype="lowpass", cutoff=20, order=4)
    np.testing.assert_allclose(lp.data[500:-500, 0], slow[500:-500], atol=0.02)
    irregular = Series(np.sort(np.random.default_rng(0).uniform(0, 1, 100)), np.zeros(100))
    with pytest.raises(ValidationError, match="evenly spaced"):
        call("butter_filter", {"series": irregular}, btype="lowpass", cutoff=5)


def test_bin_average_any_axis():
    stack = Series.regular(np.arange(10 * 4 * 6, dtype=float).reshape(10, 4, 6), 10.0,
                           axes=("time", "y", "x"), coords={"x": np.arange(6.0)})
    down = call("bin_average", {"series": stack}, factor=4)
    assert down.data.shape == (3, 4, 6)                               # 2 full bins + remainder
    np.testing.assert_allclose(down.stamps, [0.15, 0.55, 0.85])
    np.testing.assert_allclose(down.data[0], stack.data[:4].mean(0))
    spatial = call("bin_average", {"series": stack}, factor=2, axis="x")
    assert spatial.data.shape == (10, 4, 3)
    np.testing.assert_allclose(spatial.coords["x"], [0.5, 2.5, 4.5])


def test_bin_average_circular_wraps():
    phase = Series.regular(np.array([6.2, 0.1, 6.2, 0.1]), 1.0)
    out = call("bin_average", {"series": phase}, factor=4, statistic="circular_mean")
    assert min(out.data[0], 2 * np.pi - out.data[0]) < 0.1             # ~0, not ~3.15


def test_events_where_selects_units():
    spikes = Series([0.1, 0.2, 0.3, 0.4, 0.5], np.array([3, 7, 3, 9, 7]), name="spikes")
    unit7 = call("events_where", {"series": spikes}, values=[7])
    assert unit7.ons.tolist() == [0.2, 0.5] and unit7.axis == "time"


def test_arithmetic_dff_against_baseline():
    # Miniscope-style dF/F: slowly drifting fluorescence plus a 1 s transient at t = 30 s.
    t = np.arange(0, 60, 0.05)
    f = 100 + 10 * np.sin(2 * np.pi * t / 120) + np.where((t > 30) & (t < 31), 20.0, 0.0)
    s = Series(t, np.stack([f, 2 * f], 1))
    base = call("butter_filter", {"series": s}, btype="lowpass", cutoff=0.05, order=1)
    dff = call("arithmetic", {"a": call("arithmetic", {"a": s, "b": base}, operation="subtract"), "b": base},
               operation="divide")
    assert dff.data.shape == s.data.shape
    peak = np.argmax(dff.data[:, 0])
    assert 30 < t[peak] < 31 and dff.data[peak, 0] == pytest.approx(dff.data[peak, 1])  # scale-invariant


# -- persistence and workflows ----------------------------------------------------

def test_non_time_axis_roundtrips_through_files(tmp_path):
    prof = call("reparametrize", {"series": csd()}, axis="depth", unit="um")
    for fmt in ("npz", "mat"):
        back = io.import_file(io.export_artifact(prof, tmp_path / f"p.{fmt}"), stream="csd" if fmt == "mat" else None)
        assert back.axes == ("depth", "time") and back.stamp_unit == "um"
        np.testing.assert_allclose(back.coords["time"], prof.coords["time"])
        np.testing.assert_allclose(back.data, prof.data)


def test_workflow_with_defined_intervals(tmp_path):
    store = Store(tmp_path / "s")
    cid = store.put(csd())
    rec = run({"name": "layer-power", "inputs": {"csd": cid}, "steps": [
        {"id": "profile", "op": "reparametrize", "inputs": {"series": "inputs.csd"}, "params": {"axis": "depth", "unit": "um"}},
        {"id": "deep", "op": "define_intervals", "params": {"ons": [0], "offs": [250], "axis": "depth", "unit": "um"}},
        {"id": "layer", "op": "restrict", "inputs": {"series": "steps.profile", "intervals": "steps.deep"}},
        {"id": "mean", "op": "reduce_axis", "inputs": {"series": "steps.layer"}, "params": {"axis": "time"}},
    ]}, store)
    out = store.get(rec["outputs"]["mean"])
    assert out.axis == "depth" and len(out.stamps) == 6
