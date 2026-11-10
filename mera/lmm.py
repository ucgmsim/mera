"""Fits one linear mixed model: the statistics behind :func:`mera.run_mera`.

A linear mixed model with *crossed random intercepts* describes each observation as::

    y = fixed effects + (term of its level in factor 1) + ... + residual

For ground-motion residuals this is ``residual = bias + event term + site term +
remaining residual``. The *grouping factors* are the events and the sites, each event and
each site is a *level*, and *crossed* means that any event may be recorded at any site.
In the notation of the R package lme4, the model is ``y ~ 1 + (1|event) + (1|site)``,
where ``1`` stands for the bias and ``(1|event)`` for "a term for each event".

A word of warning about "residual". In this module, as in lme4, the *residual* is what
the model leaves over: mera's remaining residual. The response ``y`` that mera fits
happens to be a ground-motion residual itself, but here it is just called the response.

:func:`fit_lmm` estimates the bias, the standard deviation of each factor's terms, the
standard deviation of the residual and the terms themselves, by restricted maximum
likelihood (REML). The method is lme4's (Bates et al. 2015, *Journal of Statistical
Software* 67(1)), and has four steps:

1. Turn the labels of each grouping factor into integer codes.
2. Hand the data to the Rust engine (:mod:`mera._lmm`), which can then compute the REML
   criterion for any ``theta``. ``theta`` has one value per grouping factor: the standard
   deviation of that factor's terms divided by the standard deviation of the residual.
   The criterion measures how well a ``theta`` fits the data, smaller being better.
3. Search for the ``theta`` with the smallest criterion, using scipy's derivative-free
   optimiser COBYQA (lme4 uses the similar BOBYQA).
4. Ask the engine for all the estimates at that ``theta``.

Where to read more:

- ``docs/mixed_effects_maths.md`` derives the mathematics step by step.
- ``mera/example/maths_walkthrough.py`` repeats the calculation in plain numpy.
- ``docs/adding_models.md`` describes how to fit other models.
"""

from __future__ import annotations

import warnings
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.optimize import minimize

from mera import _lmm

# COBYQA shortens its steps as it closes in on the optimum, and stops once they are
# shorter than this: it is the precision to which the optimiser's variables are found.
FINAL_TR_RADIUS = 1e-8
# COBYQA gives up, and fit_lmm warns, after this many evaluations of the criterion. A fit
# normally takes 20 to 100.
MAX_EVALUATIONS = 1000


@dataclass(frozen=True)
class LmmFit:
    """Results of :func:`fit_lmm`.

    The names are lme4's. Where :func:`mera.run_mera` reports a quantity under a
    ground-motion name, that name is given in square brackets, for
    ``groups = [event ids, site ids]``.

    Attributes
    ----------
    levels : list of np.ndarray
        Sorted string labels of each grouping factor: ``levels[0]`` holds the distinct
        event ids and ``levels[1]`` the distinct site ids.
    theta : np.ndarray
        Random-effect standard deviations relative to ``sigma``, one per factor. A value
        of exactly 0 is a *singular fit*: the data show no variation between the levels
        of that factor beyond what the residual explains, so all its terms are 0.
    sigma : float
        Residual standard deviation [``phi_w``]. This is not the total standard
        deviation, which mera's outputs also call sigma.
    beta : np.ndarray
        Fixed effects: the intercept [``bias``] if one was fitted, otherwise empty.
    beta_se : np.ndarray
        Standard errors of ``beta`` [``bias_std_err``].
    ranef : list of np.ndarray
        Conditional modes (BLUPs) of each factor's random effects, aligned with
        ``levels``: ``ranef[0][i]`` is the term of event ``levels[0][i]`` [the event
        terms, then the site terms].
    condsd : list of np.ndarray
        Conditional standard deviations of the random effects (conditional on ``beta``):
        how uncertain each term in ``ranef`` is, given the data.
    fitted : np.ndarray
        Fitted values, fixed plus random effects, one per observation.
    resid : np.ndarray
        ``y - fitted`` [the remaining residuals].
    deviance : float
        REML criterion (or -2 log-likelihood for ML) at the optimum, as lme4 defines it.
    converged : bool
        Whether the optimiser reported convergence.
    n_evals : int
        Number of criterion evaluations by the optimiser.
    """

    levels: list[np.ndarray]
    theta: np.ndarray
    sigma: float
    beta: np.ndarray
    beta_se: np.ndarray
    ranef: list[np.ndarray]
    condsd: list[np.ndarray]
    fitted: np.ndarray
    resid: np.ndarray
    deviance: float
    converged: bool
    n_evals: int

    @property
    def sd(self) -> np.ndarray:
        """Random-effect standard deviations, one per factor [``tau``, ``phi_S2S``].

        Returns
        -------
        np.ndarray
            ``theta * sigma``.
        """
        return self.theta * self.sigma


def fit_lmm(
    y: npt.ArrayLike,
    groups: Sequence[npt.ArrayLike],
    intercept: bool = True,
    reml: bool = True,
    name: str | None = None,
) -> LmmFit:
    """Fits a linear mixed model with crossed random intercepts.

    Parameters
    ----------
    y : array-like
        Response, one value per observation.
    groups : sequence of array-like
        One array of labels per grouping factor (e.g. event ids, site ids), each with one
        label per observation. Labels are compared as strings.
    intercept : bool, default True
        Whether to fit a fixed intercept (mera's bias).
    reml : bool, default True
        Restricted maximum likelihood (as lme4's default) or maximum likelihood.
    name : str, optional
        What is being fitted (e.g. "IM PGA"), named in the warning if the optimiser does
        not converge and in a note on any error.

    Returns
    -------
    LmmFit
        The fitted model.

    Raises
    ------
    ValueError
        If the inputs are invalid (NaN or infinite values, missing group labels,
        mismatched lengths, no grouping factor, a constant response), or if a grouping
        factor has a single level or as many levels as observations (designs lme4 also
        refuses).
    RuntimeError
        If a matrix factorisation fails.

        Both errors are also how a response that the model explains exactly shows up,
        such as synthetic data made without noise. No residual is then left to estimate
        ``sigma`` from, and ``theta`` grows until the arithmetic fails. The message says
        so.

    Warns
    -----
    RuntimeWarning
        If the optimiser stops without converging. The fit is still returned, with
        ``converged`` False.

    Notes
    -----
    Any number of grouping factors can be given, not only events and sites: see
    ``docs/adding_models.md``.

    Labels are compared as strings, so ``1`` and ``"1"`` are the same level, while ``1``
    and ``1.0`` are different ones (``"1"`` and ``"1.0"``).

    With an intercept, the fit is done on ``y`` less its mean, and the mean is added
    back to the intercept. This changes no result, and keeps the fit accurate however
    large the mean of ``y`` is compared with its spread.

    The criterion does lose precision when a random-effect standard deviation is
    hundreds of times larger than the residual standard deviation: at a thousand times,
    ``theta`` is good to about four digits. Ground-motion residuals are nowhere near
    that. The same happens without an intercept when the mean of ``y`` is far from zero,
    because the random effects then have to account for it.

    Examples
    --------
    Three events, each recorded at the same three sites:

    >>> import numpy as np
    >>> from mera.lmm import fit_lmm
    >>> events = np.repeat(["e1", "e2", "e3"], 3)
    >>> sites = np.tile(["s1", "s2", "s3"], 3)
    >>> y = np.array([0.5, 0.9, 0.1, -0.2, 0.4, -0.5, 0.3, 0.6, -0.1])
    >>> fit = fit_lmm(y, [events, sites])
    >>> bias = fit.beta[0]
    >>> tau, phi_s2s = fit.sd
    >>> phi_w = fit.sigma
    >>> event_terms = dict(zip(fit.levels[0], fit.ranef[0]))
    >>> site_terms = dict(zip(fit.levels[1], fit.ranef[1]))
    """
    with _noted(name):
        # ---- 1. Check the inputs and turn the labels into codes ----
        y = np.asarray(y, dtype=float)
        if y.ndim != 1:
            raise ValueError(f"y must be one-dimensional, got shape {y.shape}")
        # For each grouping factor: its distinct labels (the levels), and the code of
        # each observation, which is the position of its label among the levels.
        levels, codes = [], []
        for k, group in enumerate(groups):
            labels = np.asarray(group)
            if labels.shape != y.shape:
                raise ValueError(
                    f"each grouping factor needs one label per observation ({y.size}), "
                    f"got shape {labels.shape}"
                )
            # Otherwise every missing id would become one shared level called "nan"/"None".
            if np.any(pd.isna(labels)):
                raise ValueError(
                    f"grouping factor {k} has missing labels (NaN or None); "
                    "drop or label those observations first"
                )
            # np.unique sorts the distinct labels and, with return_inverse, also gives
            # the position of each observation's label in that sorted list. For the
            # labels ["b", "a", "b"], level is ["a", "b"] and code is [1, 0, 1].
            level, code = np.unique(labels.astype(str), return_inverse=True)
            # As lme4: a single level cannot be separated from the intercept / residual,
            # and as many levels as observations cannot be separated from the residual.
            if level.size < 2:
                raise ValueError(
                    f"grouping factor {k} has only one level; each grouping factor "
                    "needs more than one level"
                )
            if level.size >= y.size:
                raise ValueError(
                    f"grouping factor {k} has {level.size} levels for {y.size} "
                    "observations; each grouping factor needs fewer levels than "
                    "observations"
                )
            levels.append(level)
            codes.append(code.astype(np.int64).ravel())

        # ---- 2. The fixed effects, and the engine ----
        # x is the fixed-effects matrix X: a row per observation and a column per fixed
        # effect. A column of ones is the intercept, because it adds the same constant
        # (beta[0], the bias) to every observation. Without an intercept x has no
        # columns. More columns would fit more fixed effects: see docs/adding_models.md.
        x = np.ones((y.size, int(intercept)))
        # With an intercept, the engine is given y less its mean, and the mean is added
        # back to the intercept afterwards. Moving y by a constant changes nothing else
        # in a model with an intercept, and it keeps the arithmetic accurate: the engine
        # finds a sum of squares by subtraction, which loses digits when the mean of y
        # is far larger than its spread.
        centre = y.mean() if intercept else 0.0
        # Creating the model reduces the data to sums per level, once. After that,
        # model.deviance(theta) is the criterion at any theta.
        model = _lmm.CrossedModel(codes, y - centre, x, reml=reml)
        n_factors = len(codes)

        # ---- 3. Search for the theta with the smallest criterion ----
        # COBYQA does not search over theta itself, but over omega = log(1 + theta^2)
        # for each factor, for two reasons (section 7 of docs/mixed_effects_maths.md).
        #
        # Near zero, omega is theta^2. The criterion depends on theta only through
        # theta^2, so plotted against theta it is always flat (its slope is zero) at
        # theta_k = 0. COBYQA can stop on that flat spot even when the criterion is
        # lower at a small positive theta_k. Plotted against theta^2 there is no flat
        # spot: if the best value is zero, the criterion rises from there, and COBYQA,
        # which never steps outside its bounds, stops on the bound (a singular fit,
        # theta_k = 0, as in lme4).
        #
        # For large theta, omega is 2 log(theta). COBYQA takes steps of one length in
        # every variable, which works if a step changes the criterion by a similar
        # amount wherever it is taken. That holds for log(theta). Against theta^2 the
        # criterion becomes so flat that, from theta of about 50, COBYQA stopped short of
        # the optimum and still reported convergence.
        result = minimize(
            lambda omega: model.deviance(_theta_from(omega).tolist()),
            np.full(n_factors, np.log(2.0)),  # theta = 1, where lme4 starts too
            method="COBYQA",
            bounds=[(0.0, None)] * n_factors,  # omega >= 0, that is theta >= 0
            options={"final_tr_radius": FINAL_TR_RADIUS, "maxfev": MAX_EVALUATIONS},
        )
        # COBYQA only reaches the bound up to its own rounding (~1e-16), which the square
        # root would turn into theta ~ 1e-8. Below the final trust-region radius, omega
        # is zero within the optimiser's resolution (and the criterion is linear in
        # omega there).
        theta = _theta_from(np.where(result.x < FINAL_TR_RADIUS, 0.0, result.x))
        if not result.success:
            subject = f" for {name}" if name is not None else ""
            warnings.warn(
                f"The mixed-effects optimiser did not converge{subject}: {result.message}",
                RuntimeWarning,
                stacklevel=2,
            )

        # ---- 4. The estimates at that theta ----
        out = model.fit_at(theta.tolist())
        beta = out["beta"]
        if intercept:
            beta[0] += centre
        # The engine returns the random effects of all factors in one array: those of
        # the first factor's levels, then the second's. Cut it back into one array per
        # factor, each aligned with that factor's levels.
        splits = np.cumsum([level.size for level in levels])[:-1]
        ranef = np.split(out["ranef"], splits)
        condsd = np.split(out["condsd"], splits)
        # Fitted values: the fixed effects plus, for each factor, the term of the
        # observation's level. b[code] looks up that term for every observation.
        fitted = x @ beta
        for b, code in zip(ranef, codes):
            fitted = fitted + b[code]
        return LmmFit(
            levels=levels,
            theta=theta,
            sigma=float(out["sigma"]),
            beta=beta,
            beta_se=out["beta_se"],
            ranef=ranef,
            condsd=condsd,
            fitted=fitted,
            resid=y - fitted,
            deviance=float(out["deviance"]),
            converged=bool(result.success),
            n_evals=int(result.nfev),
        )


def _theta_from(omega: np.ndarray) -> np.ndarray:
    """Converts the optimiser's variables to theta.

    Parameters
    ----------
    omega : np.ndarray
        ``log(1 + theta^2)`` for each grouping factor.

    Returns
    -------
    np.ndarray
        ``theta`` for each grouping factor.
    """
    # expm1(omega) is exp(omega) - 1, computed accurately for small omega.
    return np.sqrt(np.expm1(omega))


@contextmanager
def _noted(name: str | None) -> Generator[None]:
    """Notes ``name`` on any ValueError or RuntimeError raised in the block.

    The note is shown under the error's message, so that an error in one of many fits
    says which fit it was.

    Parameters
    ----------
    name : str or None
        What is being fitted; nothing is noted if None.

    Yields
    ------
    None
        Control, for the body of the ``with`` block.
    """
    try:
        yield
    except (ValueError, RuntimeError) as error:
        if name is not None:
            error.add_note(f"while fitting {name}")
        raise
