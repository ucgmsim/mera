"""What the Rust engine offers to Python.

``mera._lmm`` is a compiled extension module: its code is Rust (``crates/mera_lmm``, with
the Python bindings in ``src-rust/lib.rs``), so there is no Python source to read. This
stub file describes its classes and methods for editors and type checkers, and for
readers. :func:`mera.lmm.fit_lmm` is the only code in mera that uses it.

Typical use::

    model = CrossedModel(codes, n_levels, y, x)   # once per fit
    model.deviance_and_gradient([0.8, 1.2])  # the criterion at a trial theta and its
                                             # slope in each theta^2, about ten times
    model.deviance([0.8, 1.2])             # the criterion alone, for a search without it
    estimates = model.fit_at(best_theta)   # everything else, once
"""

from collections.abc import Sequence
from typing import TypedDict

import numpy as np
import numpy.typing as npt

class FitAt(TypedDict):
    """The estimates returned by :meth:`CrossedModel.fit_at`, as a dictionary."""

    deviance: float
    """The REML criterion (or -2 log-likelihood for ML)."""
    sigma: float
    """The residual standard deviation."""
    beta: npt.NDArray[np.float64]
    """The fixed effects, one per column of ``x``."""
    beta_se: npt.NDArray[np.float64]
    """The standard errors of ``beta``."""
    beta_cov: npt.NDArray[np.float64]
    """The covariance matrix of ``beta``, of shape (p, p). Its diagonal holds the squares
    of ``beta_se``."""
    ranef: npt.NDArray[np.float64]
    """The random effects: those of the first factor's levels (level 0, 1, ...), followed
    by the second factor's, and so on."""
    condsd: npt.NDArray[np.float64]
    """The conditional standard deviations of the random effects, in the order of
    ``ranef``."""

class CrossedModel:
    """A linear mixed model with crossed random intercepts, evaluated at any theta.

    Parameters
    ----------
    codes : sequence of int64 arrays
        One array per grouping factor (e.g. events, sites), with the level of each
        observation as an integer 0, 1, 2, ..., or -1 for an observation that has no
        level in that factor.
    n_levels : sequence of int
        The number of levels of each grouping factor. Every code must be below it.
    y : float64 array
        The response, one value per observation.
    x : float64 array of shape (n, p)
        The fixed-effects matrix: one row per observation and one column per fixed
        effect. A column of ones fits an intercept. With no columns (shape (n, 0)) no
        fixed effects are fitted.
    reml : bool, default True
        Restricted maximum likelihood (True) or maximum likelihood (False).

    Raises
    ------
    ValueError
        If the inputs are inconsistent or contain NaN or infinite values.

    Notes
    -----
    The calculations release Python's global interpreter lock (the GIL), so models that
    are evaluated in separate threads use separate processor cores. One model cannot be
    used by two threads at once: the second call raises a ``RuntimeError``.
    """

    def __init__(
        self,
        codes: Sequence[npt.NDArray[np.int64]],
        n_levels: Sequence[int],
        y: npt.NDArray[np.float64],
        x: npt.NDArray[np.float64],
        *,
        reml: bool = True,
    ) -> None: ...
    def deviance(self, theta: Sequence[float]) -> float:
        """The profiled REML criterion (ML: -2 log-likelihood) at ``theta``.

        ``theta`` has one value per grouping factor: the standard deviation of that
        factor's random effects divided by the residual standard deviation. Each must be
        zero or positive. The smaller the criterion, the better ``theta`` fits the data.
        """

    def deviance_and_gradient(
        self, theta: Sequence[float]
    ) -> tuple[float, npt.NDArray[np.float64]]:
        """The criterion at ``theta``, and its derivative with respect to the square of
        each theta.

        The derivative says how fast the criterion changes as each ``theta_k ** 2``
        grows, which tells an optimiser which way is downhill. It is taken with respect
        to the square because the slope with respect to ``theta_k`` itself is always
        zero at ``theta_k = 0``. Where a theta is below the engine's floor
        (``THETA_FLOOR`` in ``crates/mera_lmm/src/model.rs``, a ten-millionth), both
        results are those at the floor, which differ from the exact ones by less than
        rounding matters.
        """

    def fit_at(self, theta: Sequence[float]) -> FitAt:
        """Estimates, random effects and conditional standard deviations at ``theta``."""
