import numpy as np
import pandas as pd
import pytest
from example_data import LME4_FITS, Lme4Fit, fit_inputs, groups_of, synthetic

from mera.lmm import LmmFit, fit_lmm


def assert_same_fit(fit: LmmFit, expected: Lme4Fit, groups: list[np.ndarray]) -> None:
    # The same optimum: never worse than lme4's, and equal to optimiser precision.
    assert fit.deviance <= expected.deviance + 1e-8
    assert fit.deviance == pytest.approx(expected.deviance, abs=1e-6)
    np.testing.assert_allclose(fit.theta, expected.theta, atol=1e-5)
    assert fit.sigma == pytest.approx(expected.sigma, abs=1e-7)
    np.testing.assert_allclose(fit.beta, expected.beta, atol=1e-7)
    np.testing.assert_allclose(fit.beta_se, expected.beta_se, atol=1e-7)
    for got, want, levels, want_levels in zip(
        fit.ranef, expected.ranef, fit.levels, expected.levels
    ):
        np.testing.assert_array_equal(levels, want_levels)
        np.testing.assert_allclose(got, want, atol=1e-5)
    for got, want in zip(fit.condsd, expected.condsd):
        np.testing.assert_allclose(got, want, atol=1e-6)
    np.testing.assert_allclose(fit.fitted, expected.fitted(groups), atol=1e-5)


@pytest.mark.parametrize("data, intercept, site_term", LME4_FITS)
def test_matches_lme4(
    example_residuals: pd.DataFrame,
    lme4_fits: dict[tuple[str, bool, bool], Lme4Fit],
    data: str,
    intercept: bool,
    site_term: bool,
) -> None:
    y, groups = fit_inputs(example_residuals, data, site_term)
    fit = fit_lmm(y, groups, intercept=intercept)
    expected = lme4_fits[data, intercept, site_term]
    assert fit.converged
    assert_same_fit(fit, expected, groups)
    np.testing.assert_allclose(fit.resid, y - fit.fitted)
    # where lme4 ends exactly on the boundary (a singular fit), so must fit_lmm
    assert np.all(fit.theta[expected.theta == 0.0] == 0.0)
    for k in np.flatnonzero(fit.theta == 0.0):
        assert np.all(fit.ranef[k] == 0.0) and np.all(fit.condsd[k] == 0.0)


@pytest.mark.parametrize("seed", [45, 48])
def test_singular_fit_reports_exact_zero(
    example_residuals: pd.DataFrame, seed: int
) -> None:
    """With no event effect, lme4 returns theta_event == 0 exactly (isSingular) here."""
    y = synthetic(example_residuals, "no_event_effect", seed=seed)
    fit = fit_lmm(y, groups_of(example_residuals, True))
    assert fit.theta[0] == 0.0
    assert np.all(fit.ranef[0] == 0.0) and np.all(fit.condsd[0] == 0.0)


def test_doubly_singular_fit_reports_exact_zeros(
    example_residuals: pd.DataFrame,
) -> None:
    """Both variances are zero here, as lme4 also finds; COBYQA stops ~1e-16 (rounding)
    above the bound in theta^2, which must not come out as theta ~ 1e-8."""
    y = synthetic(example_residuals, "weak_effects", seed=138)
    fit = fit_lmm(y, groups_of(example_residuals, True))
    np.testing.assert_array_equal(fit.theta, 0.0)


def test_weak_effect_is_not_fitted_as_zero(example_residuals: pd.DataFrame) -> None:
    """theta_event = 0 is a stationary point of the criterion in theta, but here the
    criterion falls as theta_event grows (lme4 finds theta_event = 0.021), so the event
    term must stay in the fit."""
    df = example_residuals
    y = synthetic(df, "weak_effects", seed=133)
    fit = fit_lmm(y, groups_of(df, True), intercept=False)
    # theta_event = 0 is the site-only model
    site_only = fit_lmm(y, [df["stat_id"].to_numpy()], intercept=False)
    assert fit.theta[0] > 0.0
    assert fit.deviance < site_only.deviance - 1e-3


def balanced_design(
    n_events: int, n_sites: int, sd_event: float, sd_site: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A response with residual SD 1 where every site records every event exactly once,
    with the event and site label of each observation."""
    rng = np.random.default_rng(3)
    events = np.repeat(np.arange(n_events), n_sites)
    sites = np.tile(np.arange(n_sites), n_events)
    y = (
        0.2
        + sd_event * rng.normal(size=n_events)[events]
        + sd_site * rng.normal(size=n_sites)[sites]
        + rng.normal(size=events.size)
    )
    return y, events, sites


def anova_estimates(
    y: np.ndarray, n_events: int, n_sites: int
) -> tuple[np.ndarray, float]:
    """REML's (theta_event, theta_site) and sigma for :func:`balanced_design`, in closed
    form.

    For a balanced design, the REML estimates are the textbook two-way analysis of
    variance (ANOVA) ones, as long as those are positive (Searle, Casella & McCulloch
    1992, *Variance Components*, section 4.7): the residual variance is the residual mean
    square, and each factor's variance is its mean square less the residual one, over the
    number of observations per level.
    """
    table = y.reshape(n_events, n_sites)
    event_means = table.mean(axis=1, keepdims=True)
    site_means = table.mean(axis=0, keepdims=True)
    mean = table.mean()
    ms_event = n_sites * ((event_means - mean) ** 2).sum() / (n_events - 1)
    ms_site = n_events * ((site_means - mean) ** 2).sum() / (n_sites - 1)
    ms_residual = ((table - event_means - site_means + mean) ** 2).sum() / (
        (n_events - 1) * (n_sites - 1)
    )
    variance_ratios = np.array(
        [
            (ms_event / ms_residual - 1) / n_sites,
            (ms_site / ms_residual - 1) / n_events,
        ]
    )
    return np.sqrt(variance_ratios), float(np.sqrt(ms_residual))


@pytest.mark.parametrize("sd_event", [1.0, 30.0, 300.0])
@pytest.mark.parametrize("sd_site", [0.5, 200.0])
def test_matches_anova_on_a_balanced_design(sd_event: float, sd_site: float) -> None:
    """The optimiser must reach the optimum when a random effect is far larger than the
    residual. Optimising theta^2, it stopped 1.5% short at a ratio of 300, and still
    reported convergence.

    The tolerance allows for the rounding error of the criterion, which grows with the
    ratio: the fits are within 5e-5 of the ANOVA estimates here.
    """
    n_events, n_sites = 30, 12
    y, events, sites = balanced_design(n_events, n_sites, sd_event, sd_site)
    theta, sigma = anova_estimates(y, n_events, n_sites)
    fit = fit_lmm(y, [events, sites])
    assert fit.converged
    np.testing.assert_allclose(fit.theta, theta, rtol=1e-3)
    assert fit.sigma == pytest.approx(sigma, rel=1e-4)


@pytest.mark.parametrize("offset", [1e4, 1e6])
def test_a_large_mean_costs_no_precision(
    example_residuals: pd.DataFrame,
    lme4_fits: dict[tuple[str, bool, bool], Lme4Fit],
    offset: float,
) -> None:
    """Adding a constant to the response must move the intercept and nothing else.

    The engine finds the residual sum of squares by subtraction, which loses digits when
    the mean of the response is far larger than its spread: fitted as given, a response
    with a mean of 1e4 had theta out by 1e-4, and with 1e6 by 0.3, both reported as
    converged.
    """
    y, groups = fit_inputs(example_residuals, "PGA", True)
    expected = lme4_fits["PGA", True, True]
    fit = fit_lmm(y + offset, groups)
    assert fit.converged
    np.testing.assert_allclose(fit.theta, expected.theta, atol=1e-5)
    assert fit.sigma == pytest.approx(expected.sigma, abs=1e-6)
    assert fit.beta[0] - offset == pytest.approx(expected.beta[0], abs=1e-6)
    np.testing.assert_allclose(fit.beta_se, expected.beta_se, atol=1e-6)
    for got, want in zip(fit.ranef, expected.ranef):
        np.testing.assert_allclose(got, want, atol=1e-5)
    np.testing.assert_allclose(fit.fitted - offset, expected.fitted(groups), atol=1e-5)
    np.testing.assert_allclose(fit.resid, y - expected.fitted(groups), atol=1e-5)


def test_rejects_designs_lme4_rejects() -> None:
    """lme4 stops on these: a single level, or as many levels as observations."""
    y = np.array([0.3, -0.2, 0.5, 0.1, -0.4, 0.2])
    one_event = np.array(["e1"] * 6)
    three_events = np.array(["e1", "e1", "e2", "e2", "e3", "e3"])
    with pytest.raises(ValueError, match="more than one level"):
        fit_lmm(y, [one_event])
    with pytest.raises(ValueError, match="more than one level"):
        fit_lmm(y, [one_event, np.array(["s1", "s2"] * 3)])
    with pytest.raises(ValueError, match="fewer levels than observations"):
        fit_lmm(y, [three_events, np.arange(6)])


def test_rejects_missing_group_labels() -> None:
    """A missing event or site id must not become a level called "nan" or "None"."""
    y = np.array([0.3, -0.2, 0.5, 0.1])
    with pytest.raises(ValueError, match="missing"):
        fit_lmm(y, [np.array(["a", None, "b", "b"], dtype=object)])
    with pytest.raises(ValueError, match="missing"):
        fit_lmm(y, [np.array([1.0, np.nan, 2.0, 2.0])])


def test_rejects_invalid_input() -> None:
    y = np.array([1.0, 2.0, 3.0, 4.0])
    g = np.array(["a", "a", "b", "b"])
    with pytest.raises(ValueError, match="NaN"):
        fit_lmm(np.array([1.0, np.nan, 3.0, 4.0]), [g])
    with pytest.raises(ValueError, match="one label per observation"):
        fit_lmm(y, [g[:3]])
    with pytest.raises(ValueError, match="grouping factor"):
        fit_lmm(y, [])
    with pytest.raises(ValueError, match="one-dimensional"):
        fit_lmm(y.reshape(2, 2), [g])
    with pytest.raises(ValueError, match="constant"):
        fit_lmm(np.full(4, 2.0), [g])


def test_errors_note_what_was_being_fitted() -> None:
    g = np.array(["a", "a", "b", "b"])
    # one error from fit_lmm's own checks, one from the engine (a constant response)
    for y in [np.array([1.0, 2.0, 3.0]), np.full(4, 2.0)]:
        with pytest.raises(ValueError) as error:
            fit_lmm(y, [g], name="IM PGA")
        assert error.value.__notes__ == ["while fitting IM PGA"]
