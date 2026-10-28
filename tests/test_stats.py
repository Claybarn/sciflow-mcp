import itertools

import numpy as np
import pytest

from sciflow import Series, ValidationError
from sciflow.ops.aggregate import StackParams, stack_sessions
from sciflow.registry import get_op


def call(op_name, inputs, **params):
    spec = get_op(op_name)
    return spec.fn(spec.params.model_validate(params), **inputs)


def stacked(n_subjects=6, sessions_per_cell=2, effect=1.5, subject_sd=3.0, seed=0, extra=None):
    """(lag 20, session, channel 3); condition b adds `effect` at lags 5-9 on channel 0 only.

    Large subject offsets make an unstratified analysis noisy; the within-subject design recovers the effect.
    """
    rng = np.random.default_rng(seed)
    lag = np.arange(20) * 0.05
    items, sessions, subjects, conds = [], [], [], []
    for s in range(n_subjects):
        offset = rng.normal(scale=subject_sd)
        for cond in ("a", "b"):
            for k in range(sessions_per_cell):
                x = offset + rng.normal(scale=0.5, size=(20, 3))
                if cond == "b":
                    x[5:10, 0] += effect
                items.append(Series(lag, x, axes=("lag", "channel")))
                sessions.append(f"m{s}-{cond}{k}")
                subjects.append(f"m{s}")
                conds.append(cond)
    for sid, subj, cond, x in extra or []:
        items.append(Series(lag, x, axes=("lag", "channel")))
        sessions.append(sid), subjects.append(subj), conds.append(cond)
    return stack_sessions(StackParams(sessions=sessions, attributes={"subject": subjects, "condition": conds}), items)


def stat(result, name):
    return result.data[:, list(result.coords["statistic"]).index(name)]


def test_permutation_finds_effect_with_fwer_control():
    r = call("permutation_test", {"series": stacked()}, condition="condition", reps=4000, seed=1)
    assert r.axes == ("lag", "statistic", "channel")
    p = stat(r, "p_corrected")
    assert np.all(p[5:10, 0] < 0.01)
    null = np.ones_like(p, dtype=bool)
    null[5:10, 0] = False
    assert np.mean(p[null] < 0.05) <= 0.05
    obs = stat(r, "observed")
    assert r.meta["test"]["statistic"] == "mean[a] - mean[b]"
    assert np.all(obs[5:10, 0] < -6)          # a - b summed over 6 subjects ~ -1.5 * 6 = -9


def test_stratification_matters():
    s = stacked(subject_sd=5.0, n_subjects=4, sessions_per_cell=1, effect=1.0)
    strat = call("permutation_test", {"series": s}, condition="condition", reps=3000, seed=2, correction="none")
    flat = call("permutation_test", {"series": s}, condition="condition", strata=None, reps=3000, seed=2, correction="none")
    assert np.median(stat(strat, "p")[5:10, 0]) < np.median(stat(flat, "p")[5:10, 0])


def test_permutation_p_matches_exact_enumeration():
    # 3 subjects x (2 a + 2 b): 6^3 = 216 equally likely within-subject relabelings.
    s = stacked(n_subjects=3, effect=1.0, seed=3)
    y = s.data[7, :, 0]                                  # one element, 12 observations
    attrs = s.meta["unit_attributes"]["session"]
    subj, cond = np.array(attrs["subject"]), np.array(attrs["condition"])

    def statistic(c):
        return sum(y[(subj == g) & (c == "a")].mean() - y[(subj == g) & (c == "b")].mean() for g in np.unique(subj))

    observed = statistic(cond)
    per_subject = [[np.where(subj == g)[0][list(ix)] for ix in itertools.combinations(range(4), 2)]
                   for g in np.unique(subj)]
    null = []
    for choice in itertools.product(*per_subject):
        c = np.full(len(y), "b")
        c[np.concatenate(choice)] = "a"
        null.append(statistic(c))
    null = np.array(null)
    exact = min(1.0, 2 * min(np.mean(null >= observed), np.mean(null <= observed)))
    r = call("permutation_test", {"series": s}, condition="condition", reps=20000, seed=4, correction="none")
    assert stat(r, "observed")[7, 0] == pytest.approx(observed)
    assert stat(r, "p")[7, 0] == pytest.approx(exact, abs=0.02)


def test_incomplete_strata_are_dropped_not_falsely_significant():
    # A subject observed only in condition a: in the raw permute implementation this makes every
    # statistic NaN and every p-value the minimum possible.
    lonely = [("m9-a0", "m9", "a", np.zeros((20, 3))), ("m9-a1", "m9", "a", np.ones((20, 3)))]
    r = call("permutation_test", {"series": stacked(extra=lonely)}, condition="condition", reps=2000, seed=5)
    assert r.meta["test"]["dropped_strata"] == ["m9"] and r.meta["test"]["n_strata"] == 6
    p = stat(r, "p")
    assert np.median(p[10:, 1:]) > 0.2


def test_nan_elements_get_nan_p():
    s = stacked()
    data = s.data.copy()
    data[0, 3, 2] = np.nan
    r = call("permutation_test", {"series": s.replace(data=data)}, condition="condition", reps=500)
    assert np.isnan(stat(r, "p")[0, 2]) and not np.isnan(stat(r, "p")[1, 2])


def test_permutation_is_deterministic_and_validates():
    s = stacked()
    a = call("permutation_test", {"series": s}, condition="condition", reps=500, seed=7)
    b = call("permutation_test", {"series": s}, condition="condition", reps=500, seed=7)
    np.testing.assert_array_equal(a.data, b.data)
    with pytest.raises(ValidationError, match="unknown attribute"):
        call("permutation_test", {"series": s}, condition="genotype", reps=500)


def test_mixed_model_estimates_effect():
    r = call("mixed_model", {"series": stacked(n_subjects=8)}, formula="value ~ condition", groups="subject")
    assert r.axes == ("lag", "term", "statistic", "channel")
    terms = list(r.coords["term"])
    assert terms == ["Intercept", "condition[T.b]"]
    est = r.data[:, 1, 0, :]                              # condition[T.b] estimate
    p = r.data[:, 1, 4, :]                                # Holm-corrected p
    np.testing.assert_allclose(est[5:10, 0], 1.5, atol=0.35)
    assert np.all(p[5:10, 0] < 0.001)
    assert np.mean(p[10:, 1:] < 0.05) < 0.1
    assert r.meta["test"]["n_groups"] == 8 and r.meta["test"]["failed_fits"] == 0
