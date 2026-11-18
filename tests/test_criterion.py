"""Checks of the Rust criterion: against brute force on small problems, and against
lme4's on the example data."""

import numpy as np
import pandas as pd
import pytest
from example_data import CRITERION_IMS, CRITERION_THETAS, LME4_REFERENCE, groups_of

from mera import _lmm


def brute_force_deviance(
    y: np.ndarray,
    codes: list[np.ndarray],
    x: np.ndarray,
    theta: np.ndarray,
    reml: bool,
) -> float:
    """lme4's criterion straight from V = I + sum_k theta_k^2 Z_k Z_k'.

    A code of -1 means that the observation has no level in that factor.
    """
    n = y.size
    v = np.eye(n)
    for t, c in zip(theta, codes):
        # Z_k Z_k' is 1 where two observations share a level of factor k
        v += t**2 * ((c[:, None] == c[None, :]) & (c[:, None] >= 0))
    log_det_v = np.linalg.slogdet(v)[1]
    p = x.shape[1]
    vi_x = np.linalg.solve(v, x)
    xvx = x.T @ vi_x
    beta = np.linalg.solve(xvx, vi_x.T @ y)
    r = y - x @ beta
    log_det_x = np.linalg.slogdet(xvx)[1]
    r2 = r @ np.linalg.solve(v, r)
    dof = n - p if reml else n
    return float(
        log_det_v
        + (log_det_x if reml else 0.0)
        + dof * (1 + np.log(2 * np.pi * r2 / dof))
    )


def small_problem(
    n_factors: int, p: int, partial: bool = False
) -> tuple[np.ndarray, list[np.ndarray], list[int], np.ndarray]:
    """A response, the codes and the number of levels of each factor, and x. With
    ``partial``, a third of the observations have no level (code -1) in each factor."""
    rng = np.random.default_rng(n_factors)
    n = 60
    levels = [9, 6, 4][:n_factors]
    codes = [rng.integers(0, m, n) for m in levels]
    if partial:
        codes = [np.where(rng.random(n) < 1 / 3, -1, c) for c in codes]
    y = (
        0.2
        # a term for each level; where the code is -1, the observation gets none
        + sum(
            np.where(c >= 0, 0.6 * rng.normal(size=m)[c], 0.0)
            for m, c in zip(levels, codes)
        )
        + rng.normal(size=n)
    )
    # p = 2 adds a covariate, so an x with several columns is checked too
    x = np.column_stack([np.ones(n), rng.normal(size=n)])[:, :p]
    return y, codes, levels, x


@pytest.mark.parametrize("reml", [True, False])
@pytest.mark.parametrize("p", [0, 1, 2])
@pytest.mark.parametrize("n_factors", [1, 2, 3])
@pytest.mark.parametrize("partial", [False, True])
def test_rust_deviance_matches_brute_force(
    n_factors: int, p: int, reml: bool, partial: bool
) -> None:
    """Also when some observations have no level in a factor (``partial``)."""
    y, codes, n_levels, x = small_problem(n_factors, p, partial)
    rust = _lmm.CrossedModel(codes, n_levels, y, x, reml=reml)
    for theta in [np.full(n_factors, 0.8), np.linspace(0.0, 1.5, n_factors)]:
        expected = brute_force_deviance(y, codes, x, theta, reml)
        assert rust.deviance(theta.tolist()) == pytest.approx(expected, rel=1e-11)


@pytest.mark.parametrize("intercept", [True, False])
@pytest.mark.parametrize("site_term", [True, False])
def test_rust_deviance_matches_lme4(
    example_residuals: pd.DataFrame, intercept: bool, site_term: bool
) -> None:
    df = example_residuals
    encoded = [
        np.unique(labels.astype(str), return_inverse=True)
        for labels in groups_of(df, site_term)
    ]
    codes = [code.astype(np.int64) for _, code in encoded]
    n_levels = [level.size for level, _ in encoded]
    x = np.ones((len(df), int(intercept)))
    lme4 = pd.read_csv(LME4_REFERENCE / "criterion.csv")
    rows = lme4[(lme4["intercept"] == intercept) & (lme4["site_term"] == site_term)]
    assert len(rows) == len(CRITERION_IMS) * len(CRITERION_THETAS)
    for im, values in rows.groupby("im"):
        rust = _lmm.CrossedModel(codes, n_levels, df[im].to_numpy(), x)
        for row in values.to_dict("records"):
            theta = [row["theta_event"], row["theta_site"]][: len(codes)]
            assert rust.deviance(theta) == pytest.approx(row["deviance"], rel=1e-11)


@pytest.mark.parametrize("reml", [True, False])
@pytest.mark.parametrize("p", [0, 2])
@pytest.mark.parametrize("n_factors", [1, 3])
@pytest.mark.parametrize("partial", [False, True])
def test_rust_gradient_matches_differences_of_the_criterion(
    n_factors: int, p: int, reml: bool, partial: bool
) -> None:
    """The gradient is the slope of the criterion in each theta squared, s_k. It must
    agree with a central difference of the criterion in s_k, the other thetas fixed."""
    y, codes, n_levels, x = small_problem(n_factors, p, partial)
    rust = _lmm.CrossedModel(codes, n_levels, y, x, reml=reml)
    for theta in [np.full(n_factors, 0.8), np.linspace(0.05, 1.5, n_factors)]:
        value, gradient = rust.deviance_and_gradient(theta.tolist())
        assert value == pytest.approx(rust.deviance(theta.tolist()), rel=1e-13)
        for k in range(n_factors):
            s = theta**2
            h = 1e-5 * s[k]
            up, down = s.copy(), s.copy()
            up[k] += h
            down[k] -= h
            difference = (
                rust.deviance(np.sqrt(up).tolist())
                - rust.deviance(np.sqrt(down).tolist())
            ) / (2 * h)
            assert gradient[k] == pytest.approx(difference, rel=1e-5, abs=1e-6)


def test_rust_gradient_at_zero_says_which_way_the_criterion_goes() -> None:
    """At theta_k = 0 the slope in theta_k is always zero, but the slope in theta_k
    squared is not: it is negative where a small positive theta_k is better than zero."""
    y, codes, n_levels, x = small_problem(2, 1)
    rust = _lmm.CrossedModel(codes, n_levels, y, x)
    at_zero = rust.deviance([0.0, 0.7])
    _, gradient = rust.deviance_and_gradient([0.0, 0.7])
    h = 1e-6
    difference = (rust.deviance([np.sqrt(h), 0.7]) - at_zero) / h
    assert gradient[0] == pytest.approx(difference, rel=1e-3)
    assert gradient[0] < 0  # these data have an effect of the first factor
