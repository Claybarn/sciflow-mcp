import numpy as np
import pytest

from sciflow import IntervalSet, Series, ValidationError
from sciflow import intervals as iv
from sciflow.registry import get_op


def call(name, inputs, **params):
    spec = get_op(name)
    return spec.fn(spec.params.model_validate(params), **inputs)


def pairs(s):
    return list(zip(s.ons.tolist(), s.offs.tolist()))


# -- types -------------------------------------------------------------------

def test_timeseries_invariants():
    with pytest.raises(ValidationError):
        Series([0, 1, 2], np.zeros(4))
    with pytest.raises(ValidationError):
        Series([0, 2, 1], np.zeros(3))
    ts = Series.from_array(np.arange(5), np.zeros((3, 5)), parametric_axis=1)
    assert ts.data.shape == (5, 3) and ts.axes == ("time", "dim1")


def test_intervalset_sorts_and_validates():
    s = IntervalSet([3, 1], [4, 2])
    assert pairs(s) == [(1, 2), (3, 4)]
    with pytest.raises(ValidationError):
        IntervalSet([2], [1])


# -- interval algebra (through EventLogic) --------------------------------------

A = IntervalSet([0, 5, 20], [3, 10, 25])
B = IntervalSet([2, 8, 30], [6, 12, 31])


def test_union_intersection():
    assert pairs(iv.union(A, B)) == [(0, 12), (20, 25), (30, 31)]
    assert pairs(iv.intersection(A, B)) == [(2, 3), (5, 6), (8, 10)]


def test_complement_and_difference():
    assert pairs(iv.complement(A, domain=(0, 30))) == [(3, 5), (10, 20), (25, 30)]
    assert pairs(iv.difference(A, B)) == [(0, 2), (6, 8), (20, 25)]
    assert pairs(iv.complement(IntervalSet([], []))) == [(-np.inf, np.inf)]


def test_xor_matches_sampled_truth():
    # Compare against a dense boolean grid, ignoring zero-width boundary artefacts.
    x = iv.symmetric_difference(A, B)
    grid = np.arange(-1, 33, 0.01) + 0.005
    inside = lambda s: np.any((grid[:, None] >= s.ons) & (grid[:, None] <= s.offs), axis=1)  # noqa: E731
    np.testing.assert_array_equal(inside(x), inside(A) ^ inside(B))


def test_unsorted_overlapping_inputs_are_normalised():
    a = IntervalSet([0, 1], [10, 2])  # nested
    assert pairs(iv.intersection(a, IntervalSet([5], [6]))) == [(5, 6)]


def test_merge_and_filter():
    s = IntervalSet([0, 1.05, 3], [1, 2, 4])
    assert pairs(iv.merge_gaps(s, 0.1)) == [(0, 2), (3, 4)]
    assert pairs(iv.filter_duration(s, 0.96)) == [(0, 1), (3, 4)]


def test_select_contained():
    licks = IntervalSet.from_times([1, 4, 7, 21])
    out = call("select_contained", {"items": licks, "container": A})
    assert out.ons.tolist() == [1, 7, 21]


def test_threshold_intervals():
    t = np.arange(10.0)
    x = np.array([0, 1, 1, 0, 0, 1, 1, 1, 0, 1])
    out = call("threshold_intervals", {"series": Series(t, x)}, threshold=0.5)
    assert pairs(out) == [(1, 2), (5, 7), (9, 9)]
    with pytest.raises(ValidationError):
        call("threshold_intervals", {"series": Series(t, np.zeros((10, 2)))}, threshold=0.5)
    out = call("threshold_intervals", {"series": Series(t, np.stack([x, x], 1))}, threshold=0.5, channel=[1])
    assert len(out) == 3


# -- peri-event ---------------------------------------------------------------

def imaging_stack(n=1000, fs=100.0):
    rng = np.random.default_rng(0)
    return Series.regular(rng.normal(size=(n, 4, 5)), fs, axes=("time", "y", "x"))


def test_peri_event_any_dimensionality():
    s = imaging_stack()
    ev = IntervalSet.from_times([1.0, 5.0, 9.95, -3.0])
    ep = call("peri_event", {"series": s, "events": ev}, pre=0.1, post=0.2)
    assert ep.data.shape == (4, 30, 4, 5)
    assert ep.axes == ("time", "lag", "y", "x")
    np.testing.assert_allclose(ep.coords["lag"][[0, 10]], [-0.1, 0.0])
    # event at t=1.0 is sample 100: window is samples 90..119 (MATLAB ind-pre : ind+post-1)
    # events are sorted: [-3.0, 1.0, 5.0, 9.95]
    np.testing.assert_array_equal(ep.data[1], s.data[90:120])
    assert np.isnan(ep.data[0]).all() and np.isnan(ep.data[3]).all()  # before start / run off the end
    assert ep.meta["n_valid"] == 2


def test_epoch_average_is_event_relative_timeseries():
    s = imaging_stack()
    ev = IntervalSet.from_times([1.0, 2.0, 3.0, 50.0])
    ep = call("peri_event", {"series": s, "events": ev}, pre=0.1, post=0.2)
    avg = call("epoch_average", {"epochs": ep})
    assert avg.data.shape == (30, 4, 5) and avg.axes == ("lag", "y", "x")
    np.testing.assert_allclose(avg.stamps, ep.coords["lag"])
    expected = np.mean([s.data[i - 10:i + 20] for i in (100, 200, 300)], axis=0)
    np.testing.assert_allclose(avg.data, expected)
    assert avg.meta["n_valid"] == 3


def test_peri_event_normalize_zscore():
    s = imaging_stack()
    ev = IntervalSet.from_times([2.0])
    out = call("peri_event_normalize", {"series": s, "events": ev}, pre=0.1, post=0.1,
               baseline_start=-0.1, baseline_stop=0.0, method="zscore")
    base = s.data[190:200]
    np.testing.assert_allclose(out.data[0], (s.data[190:210] - base.mean(0)) / base.std(0, ddof=1))


def test_event_histogram_matches_brute_force():
    rng = np.random.default_rng(1)
    spikes = IntervalSet.from_times(np.sort(rng.uniform(0, 100, 2000)))
    stim = IntervalSet.from_times([10, 30, 30.5, 70])
    h = call("event_histogram", {"events": spikes, "reference": stim}, start=-1, stop=1, n_bins=8)
    for i, r in enumerate(stim.ons):
        np.testing.assert_array_equal(h.data[i], np.histogram(spikes.ons - r, bins=np.linspace(-1, 1, 9))[0])
    assert h.axes == ("time", "lag") and len(h.coords["lag"]) == 8


# -- correlation ----------------------------------------------------------------

def test_interval_correlation_matches_corrcoef():
    rng = np.random.default_rng(2)
    t = np.arange(500) / 50
    a = rng.normal(size=(500, 3))
    b = a + rng.normal(size=(500, 3))
    win = IntervalSet([1, 4, 20], [3, 6, 30])
    r = call("interval_correlation", {"a": Series(t, a), "b": Series(t, b), "intervals": win})
    assert r.data.shape == (3, 3)
    sl = (t >= 1) & (t <= 3)
    for c in range(3):
        np.testing.assert_allclose(r.data[0, c], np.corrcoef(a[sl, c], b[sl, c])[0, 1])
    assert np.isnan(r.data[2]).all()  # interval outside recording


def test_pairwise_correlation_uses_both_inputs():
    rng = np.random.default_rng(3)
    t = np.arange(300) / 30
    a, b = rng.normal(size=(300, 2)), rng.normal(size=(300, 3))
    r = call("pairwise_interval_correlation", {"a": Series(t, a), "b": Series(t, b),
                                               "intervals": IntervalSet([0], [9])})
    assert r.data.shape == (1, 2, 3)
    sl = t <= 9
    np.testing.assert_allclose(r.data[0, 1, 2], np.corrcoef(a[sl, 1], b[sl, 2])[0, 1])


def test_correlation_requires_common_clock():
    a = Series(np.arange(10.0), np.zeros(10))
    b = Series(np.arange(10.0) + 0.5, np.zeros(10))
    with pytest.raises(ValidationError, match="interpolate_to"):
        call("interval_correlation", {"a": a, "b": b, "intervals": IntervalSet([0], [5])})


# -- alignment ------------------------------------------------------------------

def test_interpolate_to():
    src = Series(np.arange(5.0), np.stack([np.arange(5.0), 2 * np.arange(5.0)], 1))
    tgt = Series(np.array([0.5, 3.25, 10.0]), np.zeros(3))
    out = call("interpolate_to", {"series": src, "target": tgt})
    np.testing.assert_allclose(out.data[:2], [[0.5, 1.0], [3.25, 6.5]])
    assert np.isnan(out.data[2]).all()


def test_align_clock_recovers_drift_and_offset():
    rng = np.random.default_rng(4)
    ref_pulses = np.cumsum(rng.uniform(0.5, 1.5, 200))  # jittered sync pulses on the reference system
    local_pulses = (ref_pulses - 12.0) / 1.0002          # same pulses on a drifting, offset clock
    # local system missed its first 5 pulses
    local = IntervalSet.from_times(local_pulses[5:])
    ref = IntervalSet.from_times(ref_pulses)
    target = Series(local_pulses[10:20], np.arange(10.0))
    out = call("align_clock", {"target": target, "local_clock": local, "reference_clock": ref})
    np.testing.assert_allclose(out.stamps, ref_pulses[10:20], atol=1e-9)
    assert out.meta["clock_alignment"]["pulse_lag"] == 5
    out_iv = call("align_clock", {"target": local, "local_clock": local, "reference_clock": ref})
    assert isinstance(out_iv, IntervalSet)


def test_reduce_and_select_by_axis_label():
    s = imaging_stack()
    roi = call("select", {"series": s}, axis="x", indices=[0, 1])
    assert roi.data.shape == (1000, 4, 2)
    m = call("reduce_axis", {"series": roi}, axis="y")
    assert m.axes == ("time", "x") and m.data.shape == (1000, 2)
    np.testing.assert_allclose(m.data, s.data[:, :, :2].mean(1))


def _reference_collate(ts, data, events, pre, post):
    """The original Python collateContinuousDataToEvents (time on axis 0), with its 0-based bound fixed."""
    out = np.full((len(events), pre + post) + data.shape[1:], np.nan)
    for i, e in enumerate(events):
        ind = np.searchsorted(ts, e)
        if ind - pre >= 0 and ind + post <= len(ts):
            out[i] = data[ind - pre:ind + post]
    return out


def test_peri_event_match_next_reproduces_original():
    s = imaging_stack()
    ev = np.array([1.004, 2.006, 7.5051])  # between samples, so nearest and next differ
    ep = call("peri_event", {"series": s, "events": IntervalSet.from_times(ev)},
              pre=10, post=20, unit="samples", match="next")
    np.testing.assert_array_equal(ep.data, _reference_collate(s.stamps, s.data, ev, 10, 20))
    nearest = call("peri_event", {"series": s, "events": IntervalSet.from_times(ev)}, pre=10, post=20, unit="samples")
    assert not np.array_equal(nearest.data[0], ep.data[0])  # 1.004 s: nearest is 1.00, next is 1.01


def test_datetime_timestamps():
    days = np.arange("2024-01-01", "2024-03-01", dtype="datetime64[D]")
    price = Series(days, np.arange(len(days), dtype=float), name="close")
    assert price.meta["time_origin"].startswith("1970") and price.sample_period() == 86400
    exdiv = IntervalSet.from_times(np.array(["2024-01-15", "2024-02-15"], dtype="datetime64[D]"))
    ep = call("peri_event", {"series": price, "events": exdiv}, pre=3 * 86400, post=5 * 86400)
    np.testing.assert_array_equal(ep.data[:, 3], [14, 45])  # the value on each ex-dividend day
