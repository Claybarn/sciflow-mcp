"""Statistics over stacked sessions that use every observation of every subject.

Both operations take a series stacked by ``stack_sessions`` (one entry per
session along the ``session`` axis, with subject / condition / ... in
``meta["unit_attributes"]``) and test every remaining element (every lag x
channel, every pixel, ...) separately:

- ``permutation_test``: stratified permutation test from the Claybarn/permute
  fork (``multitest_stratified_permutationtest``). Condition labels are
  shuffled only within each stratum (subject), so repeated sessions of a
  subject are never treated as independent.
- ``mixed_model``: a linear mixed-effects model per element (statsmodels
  ``MixedLM``), with a random intercept (and optional slopes) per subject.

Results are series on the same parametric axis, with a ``statistic`` axis
(and ``term`` for the mixed model), so p-values can be restricted to a lag
window, plotted or exported like any other result.
"""

from __future__ import annotations

import warnings
from typing import Literal

import numpy as np
from pydantic import Field

from ..registry import Params, op
from ..types import Series, ValidationError
from .series import _as_float, _axis_index

Correction = Literal["none", "bonferroni", "holm", "fdr_bh"]


def _observations(series: Series, axis: str, max_tests: int):
    """(n_obs, n_tests) responses, the attribute table, and the shape of the tested elements."""
    ax = _axis_index(series, axis)
    attrs = series.meta.get("unit_attributes", {}).get(axis)
    if attrs is None:
        raise ValidationError(f"axis {axis!r} has no unit attributes; stack sessions with stack_sessions first")
    x = np.moveaxis(_as_float(series.data), ax, 0)
    test_shape = x.shape[1:]
    y = x.reshape(x.shape[0], -1)
    if y.shape[1] > max_tests:
        raise ValidationError(f"{y.shape[1]} tests exceeds max_tests={max_tests}; reduce, select or bin_average first")
    return y, attrs, test_shape, ax


def _correct(p: np.ndarray, method: str) -> np.ndarray:
    out = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    if method == "none" or not ok.any():
        out[ok] = p[ok]
        return out
    from statsmodels.stats.multitest import multipletests

    out[ok] = multipletests(p[ok], method=method)[1]
    return out


def _result(series: Series, ax: int, test_shape: tuple, stats: np.ndarray, labels: list[str],
            extra_axes: tuple[tuple[str, list[str]], ...] = (), meta: dict | None = None) -> Series:
    """Rebuild a series from per-test results: (*extra, n_stat, n_tests) -> (stamps, *extra, statistic, *rest)."""
    rest_axes = series.axes[:ax] + series.axes[ax + 1:]          # parametric axis first
    data = stats.reshape(stats.shape[:-1] + test_shape)           # (*extra, n_stat, stamps, *rest)
    data = np.moveaxis(data, -len(test_shape), 0)                 # (stamps, *extra, n_stat, *rest)
    names = tuple(n for n, _ in extra_axes) + ("statistic",)
    axes = (rest_axes[0],) + names + rest_axes[1:]
    coords = {k: v for k, v in series.coords.items() if k in axes}
    coords.update({n: np.array(v) for n, v in extra_axes})
    coords["statistic"] = np.array(labels)
    kept = {k: v for k, v in series.meta.items() if k not in ("unit_attributes", "n_valid")}
    return series.replace(data=data, axes=axes, coords=coords, units=None, meta={**kept, **(meta or {})})


# -- stratified permutation test -------------------------------------------

class PermutationParams(Params):
    axis: str = Field("session", description="Stacked axis holding the observations.")
    condition: str = Field(description="Attribute whose labels are permuted, e.g. 'condition'.")
    strata: str | None = Field("subject", description="Attribute to permute within (e.g. 'subject'); null = no strata.")
    alternative: Literal["two-sided", "greater", "less"] = "two-sided"
    reps: int = Field(10_000, ge=100, le=1_000_000)
    seed: int = Field(0, description="Random seed; part of the cache key, so results are reproducible.")
    correction: Literal["max", "none", "bonferroni", "holm", "fdr_bh"] = Field(
        "max", description="Multiple-comparison correction across tested elements. 'max' is the permutation "
                           "max-statistic (Westfall-Young) FWER correction on null-standardised statistics; "
                           "the others are applied to the per-test p-values.")
    max_tests: int = Field(20_000, description="Refuse more tests than this (the null distribution is reps x tests).")


@op("permutation_test", category="stats", inputs={"series": "series"}, output="series", params=PermutationParams)
def permutation_test(p: PermutationParams, series: Series) -> Series:
    """Stratified permutation test of a condition effect at every element (Claybarn/permute multitest).

    The statistic is the difference in condition means (two conditions) or
    their standard deviation (more), computed within each stratum and summed.
    Labels are permuted within strata, so multiple sessions per subject are
    used without assuming they are independent. Strata missing a condition
    carry no information under within-stratum permutation and are dropped
    (listed in the output's metadata); elements with missing data get NaN.

    Output adds a ``statistic`` axis: ``observed``, ``p``, ``p_corrected``.
    """
    from permute.multitest_stratified import multitest_stratified_permutationtest

    y, attrs, test_shape, ax = _observations(series, p.axis, p.max_tests)
    if y.shape[1] * p.reps > 2e8:
        raise ValidationError(f"{y.shape[1]} tests x {p.reps} reps needs too much memory for the null distribution; "
                              "lower reps or reduce the tested elements")
    for a in [p.condition] + ([p.strata] if p.strata else []):
        if a not in attrs:
            raise ValidationError(f"unknown attribute {a!r}; available: {sorted(attrs)}")
    cond = np.asarray(attrs[p.condition]).astype(str)
    strata = np.asarray(attrs[p.strata]).astype(str) if p.strata else np.zeros(len(cond), dtype=str)
    levels = np.unique(cond)
    if len(levels) < 2:
        raise ValidationError(f"{p.condition!r} has a single level {levels.tolist()}")
    if len(levels) > 2 and p.alternative != "greater":
        raise ValidationError("with more than two conditions the statistic is a spread, so use alternative='greater'")

    complete = [s for s in np.unique(strata) if set(cond[strata == s]) == set(levels)]
    dropped = sorted(set(np.unique(strata)) - set(complete))
    keep = np.isin(strata, complete)
    if not complete:
        raise ValidationError(f"no {p.strata!r} has observations in every {p.condition!r} level")
    y, cond, strata = y[keep], cond[keep], strata[keep]
    finite = ~np.any(np.isnan(y), axis=0)

    observed = np.full(y.shape[1], np.nan)
    pvals = np.full(y.shape[1], np.nan)
    p_corr = np.full(y.shape[1], np.nan)
    if finite.any():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            if p.strata:
                pv, tst, dist = multitest_stratified_permutationtest(
                    strata, cond, y[:, finite], alternative=p.alternative, reps=p.reps, seed=p.seed)
            else:
                # Unstratified: every observation exchangeable (multitest_core needs equal group sizes).
                from permute.multitest_core import multitest_two_sample

                a, b = y[cond == levels[0]][:, finite], y[cond == levels[1]][:, finite]
                if len(levels) != 2 or a.shape != b.shape:
                    raise ValidationError("unstratified tests need exactly two conditions with equal numbers of observations")
                pv, tst, dist = multitest_two_sample(a, b, reps=p.reps, alternative=p.alternative,
                                                     keep_dist=True, seed=p.seed)
        observed[finite], pvals[finite] = tst, np.clip(pv, 0, 1)
        if p.correction == "max":
            # Standardise by each test's null so channels on different scales compete fairly.
            mu, sd = dist.mean(0), dist.std(0)
            sd[sd == 0] = np.inf
            z_obs, z_null = (tst - mu) / sd, (dist - mu) / sd
            if p.alternative == "two-sided":
                hits = np.abs(z_null).max(1)[:, None] >= np.abs(z_obs)[None, :]
            elif p.alternative == "greater":
                hits = z_null.max(1)[:, None] >= z_obs[None, :]
            else:
                hits = z_null.min(1)[:, None] <= z_obs[None, :]
            p_corr[finite] = (1 + hits.sum(0)) / (p.reps + 1)
        else:
            p_corr = _correct(pvals, p.correction)
    order = sorted(levels.tolist())
    return _result(series, ax, test_shape, np.stack([observed, pvals, p_corr]), ["observed", "p", "p_corrected"],
                   meta={"test": {"kind": "stratified_permutation", "condition": p.condition, "levels": order,
                                  "statistic": f"mean[{order[0]}] - mean[{order[1]}]" if len(order) == 2 else "std of means",
                                  "strata": p.strata, "n_strata": len(complete), "dropped_strata": dropped,
                                  "n_observations": int(keep.sum()), "alternative": p.alternative, "reps": p.reps,
                                  "seed": p.seed, "correction": p.correction, "n_tests": int(finite.sum()),
                                  "implementation": "permute.multitest_stratified.multitest_stratified_permutationtest"
                                  if p.strata else "permute.multitest_core.multitest_two_sample"}})


# -- linear mixed-effects model ---------------------------------------------

class MixedModelParams(Params):
    axis: str = Field("session", description="Stacked axis holding the observations.")
    formula: str = Field("value ~ condition", description=(
        "Fixed effects in patsy / R syntax, with `value` as the response and stacked attributes as predictors, "
        "e.g. 'value ~ condition', 'value ~ condition * day', 'value ~ C(day)'."))
    groups: str = Field("subject", description="Attribute defining the random-effect groups.")
    re_formula: str | None = Field(None, description="Random-effects formula, e.g. '~day' for random slopes. "
                                                     "Default: a random intercept per group.")
    reml: bool = True
    correction: Correction = Field("holm", description="Correction across tested elements, applied per term.")
    max_tests: int = Field(5_000, description="Refuse more element-wise fits than this.")


@op("mixed_model", category="stats", inputs={"series": "series"}, output="series", params=MixedModelParams)
def mixed_model(p: MixedModelParams, series: Series) -> Series:
    """Fit a linear mixed-effects model at every element (statsmodels MixedLM), e.g. response ~ condition + (1 | subject).

    Output adds ``term`` (fixed-effect names, e.g. ``condition[T.sleep_dep]``)
    and ``statistic`` (``estimate``, ``se``, ``z``, ``p``, ``p_corrected``)
    axes. Elements whose fit fails or has missing data are NaN; the number of
    fits that failed or warned about convergence is in the metadata.
    """
    import pandas as pd
    import statsmodels.formula.api as smf
    from statsmodels.tools.sm_exceptions import ConvergenceWarning

    y, attrs, test_shape, ax = _observations(series, p.axis, p.max_tests)
    if p.groups not in attrs:
        raise ValidationError(f"unknown attribute {p.groups!r}; available: {sorted(attrs)}")
    df = pd.DataFrame({k: v for k, v in attrs.items()})
    terms: list[str] | None = None
    results: list[np.ndarray | None] = []
    failed = warned = 0
    for j in range(y.shape[1]):
        if np.any(np.isnan(y[:, j])):
            results.append(None)
            continue
        df["value"] = y[:, j]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                fit = smf.mixedlm(p.formula, df, groups=df[p.groups], re_formula=p.re_formula).fit(reml=p.reml)
            except Exception:
                failed += 1
                results.append(None)
                continue
        warned += any(issubclass(w.category, ConvergenceWarning) for w in caught)
        fe = fit.fe_params.index.tolist()
        if terms is None:
            terms = fe
        results.append(np.stack([fit.fe_params[terms], fit.bse_fe[terms], fit.tvalues[terms], fit.pvalues[terms]]))
    if terms is None:
        raise ValidationError(f"no element could be fitted with {p.formula!r} (failed: {failed})")
    stats = np.full((len(terms), 5, y.shape[1]), np.nan)
    for j, r in enumerate(results):
        if r is not None:
            stats[:, :4, j] = r.T
    for t in range(len(terms)):
        stats[t, 4] = _correct(stats[t, 3], p.correction)
    return _result(series, ax, test_shape, stats, ["estimate", "se", "z", "p", "p_corrected"],
                   extra_axes=(("term", terms),),
                   meta={"test": {"kind": "linear_mixed_model", "formula": p.formula, "groups": p.groups,
                                  "re_formula": p.re_formula, "reml": p.reml, "correction": p.correction,
                                  "n_observations": int(y.shape[0]), "n_groups": int(df[p.groups].nunique()),
                                  "n_tests": int(sum(r is not None for r in results)), "failed_fits": failed,
                                  "convergence_warnings": warned, "implementation": "statsmodels MixedLM"}})
