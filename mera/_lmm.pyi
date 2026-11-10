"""What the Rust engine offers to Python.

``mera._lmm`` is a compiled extension module: its code is Rust (``crates/mera_lmm``, with
the Python bindings in ``src-rust/lib.rs``), so there is no Python source to read. This
stub file describes its classes and methods for editors and type checkers, and for
readers. :func:`mera.lmm.fit_lmm` is the only code in mera that uses it.

Typical use::

    model = CrossedModel(codes, y, x)      # once per fit
    model.deviance([0.8, 1.2])             # the criterion at a trial theta, many times
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
        observation as an integer 0, 1, 2, ...
    y : float64 array
        The response, one value per observation.
    x : float64 array of shape (n, p), optional
        The fixed-effects matrix: one row per observation and one column per fixed
        effect. A column of ones fits an intercept. None fits no fixed effects.
    reml : bool, default True
        Restricted maximum likelihood (True) or maximum likelihood (False).

    Raises
    ------
    ValueError
        If the inputs are inconsistent or contain NaN or infinite values.
    """

    def __init__(
        self,
        codes: Sequence[npt.NDArray[np.int64]],
        y: npt.NDArray[np.float64],
        x: npt.NDArray[np.float64] | None = None,
        *,
        reml: bool = True,
    ) -> None: ...
    def deviance(self, theta: Sequence[float]) -> float:
        """The profiled REML criterion (ML: -2 log-likelihood) at ``theta``.

        ``theta`` has one value per grouping factor: the standard deviation of that
        factor's random effects divided by the residual standard deviation. Each must be
        zero or positive. The smaller the criterion, the better ``theta`` fits the data.
        """

    def fit_at(self, theta: Sequence[float]) -> FitAt:
        """Estimates, random effects and conditional standard deviations at ``theta``."""
