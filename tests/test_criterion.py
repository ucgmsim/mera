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
    """lme4's criterion straight from V = I + sum_k theta_k^2 Z_k Z_k'."""
    n = y.size
    v = np.eye(n)
    for t, c in zip(theta, codes):
        v += t**2 * (c[:, None] == c[None, :])
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
    n_factors: int, p: int
) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
    rng = np.random.default_rng(n_factors)
    n = 60
    levels = [9, 6, 4][:n_factors]
    codes = [rng.integers(0, m, n) for m in levels]
    y = (
        0.2
        + sum(0.6 * rng.normal(size=m)[c] for m, c in zip(levels, codes))
        + rng.normal(size=n)
    )
    # p = 2 adds a covariate, so an x with several columns is checked too
    x = np.column_stack([np.ones(n), rng.normal(size=n)])[:, :p]
    return y, codes, x


@pytest.mark.parametrize("reml", [True, False])
@pytest.mark.parametrize("p", [0, 1, 2])
@pytest.mark.parametrize("n_factors", [1, 2, 3])
def test_rust_deviance_matches_brute_force(n_factors: int, p: int, reml: bool) -> None:
    y, codes, x = small_problem(n_factors, p)
    rust = _lmm.CrossedModel(codes, y, x, reml=reml)
    for theta in [np.full(n_factors, 0.8), np.linspace(0.0, 1.5, n_factors)]:
        expected = brute_force_deviance(y, codes, x, theta, reml)
        assert rust.deviance(theta.tolist()) == pytest.approx(expected, rel=1e-11)


@pytest.mark.parametrize("intercept", [True, False])
@pytest.mark.parametrize("site_term", [True, False])
def test_rust_deviance_matches_lme4(
    example_residuals: pd.DataFrame, intercept: bool, site_term: bool
) -> None:
    df = example_residuals
    codes = [
        np.unique(labels.astype(str), return_inverse=True)[1].astype(np.int64)
        for labels in groups_of(df, site_term)
    ]
    x = np.ones((len(df), int(intercept)))
    lme4 = pd.read_csv(LME4_REFERENCE / "criterion.csv")
    rows = lme4[(lme4["intercept"] == intercept) & (lme4["site_term"] == site_term)]
    assert len(rows) == len(CRITERION_IMS) * len(CRITERION_THETAS)
    for im, values in rows.groupby("im"):
        rust = _lmm.CrossedModel(codes, df[im].to_numpy(), x)
        for row in values.to_dict("records"):
            theta = [row["theta_event"], row["theta_site"]][: len(codes)]
            assert rust.deviance(theta) == pytest.approx(row["deviance"], rel=1e-11)
