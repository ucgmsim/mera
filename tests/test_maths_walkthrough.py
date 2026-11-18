"""The numpy walk-through of the mathematics, mera/example/maths_walkthrough.py, must
agree with mera itself: it is the documentation's account of what the engine computes."""

import numpy as np
import pytest
from example_data import add_covariate_trend, assert_matches_dense

from mera.example import maths_walkthrough as walkthrough
from mera.lmm import fit_lmm


@pytest.mark.parametrize("intercept", [True, False])
@pytest.mark.parametrize("site_term", [True, False])
@pytest.mark.parametrize("with_covariates", [False, True])
def test_walkthrough_matches_fit_lmm(
    intercept: bool, site_term: bool, with_covariates: bool
) -> None:
    """Every result of fit_lmm must equal that of the dense numpy implementation, which
    is independent of the Rust engine: with and without a bias, a site term and fixed
    effects beside the bias."""
    y, events, sites = walkthrough.make_example_data()
    x = np.ones((y.size, int(intercept)))
    covariates = None
    if with_covariates:
        y, _, covariates = add_covariate_trend(y, events, sites)
        x = np.column_stack([x, covariates])
    groups = [events, sites] if site_term else [events]
    z, level_factor, levels = walkthrough.indicator_matrix(groups)
    model = walkthrough.Model(y, x, z, level_factor)
    ours = walkthrough.estimates(walkthrough.find_theta(model), model)
    fit = fit_lmm(y, groups, intercept=intercept, covariates=covariates)

    assert fit.beta.shape == fit.beta_se.shape == (x.shape[1],)
    assert_matches_dense(fit, ours, levels, y)


@pytest.mark.parametrize("theta", [(1.0, 1.0), (0.0, 0.7), (0.4, 2.5)])
def test_engine_formula_matches_the_definition(theta: tuple[float, float]) -> None:
    """The criterion computed without V equals the one from its definition, with V."""
    y, events, sites = walkthrough.make_example_data()
    x = np.ones((y.size, 1))
    z, level_factor, _ = walkthrough.indicator_matrix([events, sites])
    model = walkthrough.Model(y, x, z, level_factor)
    assert walkthrough.evaluate(np.array(theta), model).deviance == pytest.approx(
        walkthrough.criterion_from_definition(np.array(theta), model), rel=1e-12
    )


def test_r2_is_the_penalised_residual_sum_of_squares() -> None:
    """r2 is the sum of squares of the remaining residuals plus that of u, where the
    random effects are b = Lambda u: so phi_w is more than the root-mean-square of the
    remaining residuals, as the documentation says."""
    y, events, sites = walkthrough.make_example_data()
    z, level_factor, _ = walkthrough.indicator_matrix([events, sites])
    model = walkthrough.Model(y, np.ones((y.size, 1)), z, level_factor)
    theta = walkthrough.find_theta(model)
    at = walkthrough.evaluate(theta, model)
    ours = walkthrough.estimates(theta, model)
    u = ours.ranef / at.lam
    sum_of_squares = (ours.resid**2).sum()
    assert at.r2 == pytest.approx(sum_of_squares + (u**2).sum(), rel=1e-10)
    assert ours.sigma > np.sqrt(sum_of_squares / at.dof)


def test_main_prints_the_comparison(capsys: pytest.CaptureFixture[str]) -> None:
    walkthrough.main()
    out = capsys.readouterr().out
    assert "REML criterion" in out and "Shrinkage" in out
    # The printed product of the shrinkage and the mean is the printed event term.
    product, term = (
        line.split()[-1]
        for line in out.splitlines()
        if "their product" in line or "the event term" in line
    )
    assert product == term
