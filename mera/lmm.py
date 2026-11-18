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
3. Search for the ``theta`` with the smallest criterion. The engine also returns the
   slope of the criterion (its gradient), which tells the search which way is downhill,
   so that it needs about ten evaluations. (lme4 searches without the slope, with the
   optimiser BOBYQA, and takes 30 to 40 on the example data.)
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

# The settings of the search for theta, which _search carries out in up to three stages.
# The lengths are in the optimiser's variable omega = log(1 + theta^2), in which a small
# step is the same relative change of 1 + theta^2 anywhere.
#
# Stage 1 follows the slope of the criterion downhill (scipy's L-BFGS-B). It stops when
# a step lowers the criterion by less than this fraction of its value. By then it is
# close enough for stage 2 to finish. A much smaller fraction would keep it going until
# its steps changed the criterion by less than the rounding of the arithmetic, where it
# cannot tell downhill from uphill and wastes evaluations finding that out.
GRADIENT_SEARCH_FTOL = 1e-9
# Stage 2 finishes the search with Newton's method, which jumps to where the slope would
# be zero if it kept changing at its present rate. It measures that rate by shifting
# each omega by NEWTON_DIFFERENCE. The search has arrived when the jump is shorter than
# NEWTON_DONE. A jump longer than NEWTON_MAX_JUMP, or no arrival within NEWTON_MAX_STEPS
# jumps, means that stage 1 stopped too far from the optimum for this to work.
NEWTON_DIFFERENCE = 1e-5
NEWTON_DONE = 1e-6
NEWTON_MAX_JUMP = 0.1
NEWTON_MAX_STEPS = 3
# Stage 3 is used only if stage 2 cannot confirm the optimum: a search that compares
# values of the criterion and needs no slope (scipy's COBYQA), from where the others
# stopped. It starts with steps of FALLBACK_TR_RADIUS, shortens them as it closes in,
# and stops once they are shorter than FINAL_TR_RADIUS.
FALLBACK_TR_RADIUS = 0.1
FINAL_TR_RADIUS = 1e-8
# The search gives up, and fit_lmm warns, after this many evaluations of the criterion.
# A fit normally takes about 10.
MAX_EVALUATIONS = 1000
# The largest theta that the search considers: a random-effect standard deviation ten
# thousand times that of the residuals. A search that ends there is refused, because it
# was still going up: the model explains the response almost exactly, as it does for
# synthetic data made without noise. Nothing is then left to estimate the residual
# standard deviation from, and theta would grow until the arithmetic failed. (Well below
# the limit the criterion has already lost most of its precision: see the notes of
# fit_lmm.)
MAX_THETA = 1e4


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
        event ids and ``levels[1]`` the distinct site ids. They are the labels of the
        observations that are in the factor.
    theta : np.ndarray
        Random-effect standard deviations relative to ``sigma``, one per factor. A value
        of exactly 0 is a *singular fit*: the data show no variation between the levels
        of that factor beyond what the residual explains, so all its terms are 0.
    sigma : float
        Residual standard deviation [``phi_w``]. This is not the total standard
        deviation, which mera's outputs also call sigma.
    beta : np.ndarray
        Fixed effects: the intercept [``bias``] if one was fitted, then the coefficient
        of each covariate, in the order of their columns. Empty if there are neither.
    beta_se : np.ndarray
        Standard errors of ``beta`` [``bias_std_err``].
    beta_cov : np.ndarray
        Covariance matrix of ``beta`` (lme4's ``vcov``), with a row and a column per
        fixed effect. Its diagonal holds the squares of ``beta_se``. The other entries
        are needed for the standard error of a combination of coefficients: that of
        the difference of the first two covariates' is
        ``sqrt(c @ beta_cov @ c)`` with ``c = [0, 1, -1, 0, ...]``.
    ranef : list of np.ndarray
        Conditional modes (BLUPs) of each factor's random effects, aligned with
        ``levels``: ``ranef[0][i]`` is the term of event ``levels[0][i]`` [the event
        terms, then the site terms].
    condsd : list of np.ndarray
        Conditional standard deviations of the random effects (conditional on ``beta``):
        how uncertain each term in ``ranef`` is, given the data.
    fitted : np.ndarray
        Fitted values, fixed plus random effects, one per observation. The fixed part
        includes the covariates' trends.
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
    beta_cov: np.ndarray
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
    covariates: npt.ArrayLike | None = None,
    start: npt.ArrayLike | None = None,
) -> LmmFit:
    """Fits a linear mixed model with crossed random intercepts.

    Parameters
    ----------
    y : array-like
        Response, one value per observation.
    groups : sequence of array-like
        One array of labels per grouping factor (e.g. event ids, site ids), each with one
        label per observation. Labels are compared as strings. In a numpy masked array,
        the masked observations have no level in that factor and get no term from it:
        see :func:`split_by_class`.
    intercept : bool, default True
        Whether to fit a fixed intercept (mera's bias).
    reml : bool, default True
        Restricted maximum likelihood (as lme4's default) or maximum likelihood.
    name : str, optional
        What is being fitted (e.g. "IM PGA"), named in the warning if the optimiser does
        not converge and in a note on any error.
    covariates : array-like, optional
        More fixed effects: a column per covariate (such as magnitude, or the logarithm
        of Vs30) and a row per observation, or a single covariate as a vector. Each gets
        a coefficient in ``beta``, after the intercept.
    start : array-like, optional
        Where the search for ``theta`` starts: one value per grouping factor. The
        default is 1 for every factor, as in lme4. See the notes for what it is good
        for.

    Returns
    -------
    LmmFit
        The fitted model.

    Raises
    ------
    ValueError
        If the inputs are invalid (NaN or infinite values, missing group labels,
        mismatched lengths, no grouping factor, a constant response), if a grouping
        factor has a single level or as many levels as observations (designs lme4 also
        refuses), if the fixed effects are collinear (one is an exact combination of
        the others, as a covariate and the same covariate plus a constant are when there
        is an intercept), or if ``start`` is not one finite, non-negative value per
        grouping factor, none above the limit of the search.

        Also if the model explains the response exactly, or almost, as it does for
        synthetic data made without noise. No residual is then left to estimate
        ``sigma`` from, and ``theta`` grows until it meets the limit of the search
        (``MAX_THETA``, ten thousand). The message says so.
    RuntimeError
        If a matrix factorisation fails in the engine. This is not expected for inputs
        that pass the checks above.

    Warns
    -----
    RuntimeWarning
        If the optimiser stops without converging. The fit is still returned, with
        ``converged`` False.

    Notes
    -----
    Any number of grouping factors can be given, not only events and sites, and a
    grouping factor can be given a separate standard deviation for each class of its
    levels (:func:`split_by_class`): see ``docs/adding_models.md``.

    Labels are compared as strings, so ``1`` and ``"1"`` are the same level, while ``1``
    and ``1.0`` are different ones (``"1"`` and ``"1.0"``).

    The fit is done on an equivalent, better-conditioned form of the fixed effects
    (orthonormal combinations of their columns) and on ``y`` less its least-squares fit
    on them, and the estimates are turned back afterwards. This changes no result, and
    keeps the fit accurate however large the mean of ``y`` or of a covariate is compared
    with its spread, and however alike two covariates are.

    The search follows the slope of the criterion downhill, then confirms that it has
    arrived by Newton's method (see ``_search``). It needs about ten evaluations of the
    criterion, each with its slope, which costs two to three times as much as the
    criterion alone.

    The result does not depend on ``start``, to the precision of the search, and the
    number of evaluations depends on it only a little. From the ``theta`` of a fit of
    similar data, as in a bootstrap, the search takes as many evaluations as from the
    default, because most of them go on learning how the criterion curves, wherever it
    starts. Only a start at the optimum itself saves some (five evaluations against nine
    on the example data). A second fit from a different ``start`` is a way of checking
    the first: both should end at the same ``theta``.

    Fits can run in several threads at once (with
    ``concurrent.futures.ThreadPoolExecutor``, for example), because the engine releases
    Python's global interpreter lock while it calculates. With four threads, four fits
    of 150,000 records took about a third of the time they take one after another, and
    so did 200 fits of 5,000 records. ``docs/adding_models.md`` has an example.

    Subtracting a typical value from a covariate (magnitude less 6, say) makes the
    intercept the mean at that value rather than at zero, and changes nothing else.
    ``docs/adding_models.md`` has more advice on covariates, and on comparing models
    with and without one.

    The criterion does lose precision when a random-effect standard deviation is
    hundreds of times larger than the residual standard deviation: at a thousand times,
    ``theta`` is good to about four digits. Ground-motion residuals are nowhere near
    that, and beyond ten thousand times a fit is refused (see Raises). The same loss of
    precision happens when the mean of ``y`` is far from zero and no fixed effect
    can absorb it (with neither an intercept nor covariates, say), because the random
    effects then have to account for it.

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
            # A numpy masked array marks the observations that have no level in this
            # factor, as split_by_class makes them. in_factor is True for the others:
            # for every observation, if the labels are an ordinary array.
            in_factor = ~np.ma.getmaskarray(group)
            labels = np.ma.getdata(group)
            if labels.shape != y.shape:
                raise ValueError(
                    f"each grouping factor needs one label per observation ({y.size}), "
                    f"got shape {labels.shape}"
                )
            # level holds the distinct labels as strings, sorted, and code_in_factor
            # the position of each observation's label in that list. For the labels
            # ["b", "a", "b"], level is ["a", "b"] and code_in_factor is [1, 0, 1].
            level, code_in_factor = _encode_labels(labels[in_factor])
            # Otherwise every missing id would become one shared level called "nan"/"None".
            if np.any(code_in_factor < 0):
                raise ValueError(
                    f"grouping factor {k} has missing labels (NaN or None); "
                    "drop or label those observations first"
                )
            # An observation that is not in the factor keeps the code -1, "no level".
            code = np.full(y.size, -1, dtype=np.int64)
            code[in_factor] = code_in_factor
            # As lme4: a single level cannot be separated from the intercept / residual,
            # and as many levels as observations cannot be separated from the residual.
            if level.size < 2:
                found = "only one level" if level.size == 1 else "no levels"
                raise ValueError(
                    f"grouping factor {k} has {found}; each grouping factor needs "
                    "more than one level"
                )
            if level.size >= y.size:
                raise ValueError(
                    f"grouping factor {k} has {level.size} levels for {y.size} "
                    "observations; each grouping factor needs fewer levels than "
                    "observations"
                )
            levels.append(level)
            codes.append(code)

        # ---- 2. The fixed effects, and the engine ----
        # x is the fixed-effects matrix X: a row per observation and a column per fixed
        # effect. A column of ones is the intercept, because it adds the same constant
        # (beta[0], the bias) to every observation. Without an intercept x has no
        # columns. Each covariate is one more column, with its own coefficient in beta.
        x = np.ones((y.size, int(intercept)))
        if covariates is not None:
            extra = np.asarray(covariates, dtype=float)
            if extra.ndim == 1:
                extra = extra[:, np.newaxis]  # a single covariate, given as a vector
            if extra.ndim != 2:
                raise ValueError(
                    f"covariates must be at most two-dimensional, got shape {extra.shape}"
                )
            if extra.shape[0] != y.size:
                raise ValueError(
                    f"covariates need one row per observation ({y.size}), "
                    f"got shape {extra.shape}"
                )
            x = np.column_stack([x, extra])
        if not (np.all(np.isfinite(y)) and np.all(np.isfinite(x))):
            raise ValueError("y or the covariates contain NaN or infinite values")

        # The engine is not given x and y themselves, but a tidier pair that describes
        # the same model, so that its arithmetic stays accurate whatever the scales of
        # the covariates and of y.
        #
        # In place of x it gets q, whose columns are combinations of the columns of x
        # that have length 1 and are at right angles to each other ("orthonormal"). Any
        # trend that the columns of x can describe, the columns of q can describe as
        # well, so the fitted values are the same. But the columns of q cannot be nearly
        # alike, as a covariate with a large mean and the column of ones are, which is
        # what costs digits. numpy's QR decomposition finds q, and the triangular matrix
        # r that turns q back into x: x / lengths = q @ r.
        #
        # In place of y it gets what an ordinary regression of y on the fixed effects,
        # ignoring the groups, leaves over (y less its mean, with only an intercept).
        # The engine finds a sum of squares by subtraction, which loses digits when the
        # mean of y, or its trend with a covariate, is far larger than its spread.
        #
        # gamma denotes the coefficients of the columns of q, and beta those of x. They
        # describe the same trend, q @ gamma = x @ beta, so beta = from_gamma @ gamma
        # with the matrix from_gamma below.
        n_fixed = x.shape[1]
        gamma_start = np.zeros(n_fixed)  # the regression's part of gamma
        from_gamma = np.zeros((n_fixed, n_fixed))
        log_det_change = 0.0
        q = x
        if n_fixed > 0:
            # Each column is scaled to length 1 first, so that nothing below depends on
            # the units of the covariates.
            lengths = np.linalg.norm(x, axis=0)
            if np.all(lengths > 0.0):
                q, r = np.linalg.qr(x / lengths)
                # An entry on the diagonal of r is how much of a column of x is not a
                # combination of the columns before it.
                independent = np.abs(np.diag(r))
            else:
                independent = lengths
            # If a column is a combination of the others, to within rounding, their
            # coefficients cannot be told apart.
            if independent.min() <= max(x.shape) * np.finfo(float).eps:
                raise ValueError(
                    "the fixed effects are collinear: one column (the intercept or a "
                    "covariate) is a combination of the others"
                )
            from_gamma = np.linalg.inv(r) / lengths[:, np.newaxis]
            gamma_start = q.T @ y
            # log|X'V^-1X|, a term of the REML criterion, is larger for x than for q by
            # this constant. It does not depend on theta, so the search is unaffected.
            log_det_change = 2.0 * np.log(independent * lengths).sum()
        # Creating the model reduces the data to sums per level, once. After that,
        # model.deviance(theta) is the criterion at any theta.
        n_levels = [level.size for level in levels]
        model = _lmm.CrossedModel(codes, n_levels, y - q @ gamma_start, q, reml=reml)
        n_factors = len(codes)

        # ---- 3. Search for the theta with the smallest criterion ----
        # The search is not over theta itself, but over omega = log(1 + theta^2) for
        # each factor, for two reasons (section 7 of docs/mixed_effects_maths.md).
        #
        # Near zero, omega is theta^2. The criterion depends on theta only through
        # theta^2, so plotted against theta it is always flat (its slope is zero) at
        # theta_k = 0, and a search can stop on that flat spot even when the criterion
        # is lower at a small positive theta_k. Plotted against theta^2 there is no flat
        # spot: if the best value is zero, the criterion rises from there, and the
        # search, which never steps outside its bounds, stops on the bound (a singular
        # fit, theta_k = 0, as in lme4).
        #
        # For large theta, omega is 2 log(theta). A search works best if a step of one
        # length changes the criterion by a similar amount wherever it is taken. That
        # holds for log(theta). Against theta^2 the criterion becomes so flat that, from
        # theta of about 50, a search stopped short of the optimum.
        omega_max = float(np.log1p(MAX_THETA**2))  # the limit on theta, as an omega
        if start is None:
            omega_start = np.full(n_factors, np.log(2.0))  # theta = 1, as lme4 starts
        else:
            theta_start = np.asarray(start, dtype=float)
            if theta_start.shape != (n_factors,):
                raise ValueError(
                    f"start needs one value of theta per grouping factor ({n_factors}), "
                    f"got shape {theta_start.shape}"
                )
            in_range = (theta_start >= 0.0) & (theta_start <= MAX_THETA)
            if not np.all(np.isfinite(theta_start) & in_range):
                raise ValueError(
                    f"start must be finite, not negative and at most {MAX_THETA:g}, "
                    f"got {theta_start}"
                )
            omega_start = np.log1p(theta_start**2)
        omega, n_evals, failure = _search(model, omega_start, omega_max, n_levels)
        # A search that ends at the upper limit was stopped on its way up: see MAX_THETA.
        # (It ends within rounding of the limit, hence the small allowance.)
        at_limit = omega >= omega_max - 1e-4
        if np.any(at_limit):
            raise ValueError(
                f"theta reached its limit of {MAX_THETA:g} for grouping factor "
                f"{', '.join(map(str, np.flatnonzero(at_limit)))}: the grouping factors "
                "and fixed effects explain the response almost exactly, which leaves "
                "too little residual variation to fit the model (synthetic data made "
                "without noise does this)"
            )
        # An omega below the resolution of the search is zero: a theta below 1e-4 is not
        # told apart from none (lme4 draws the same line for a singular fit).
        theta = _theta_from(np.where(omega < FINAL_TR_RADIUS, 0.0, omega))
        if failure is not None:
            subject = f" for {name}" if name is not None else ""
            warnings.warn(
                f"The mixed-effects optimiser did not converge{subject}: {failure}",
                RuntimeWarning,
                stacklevel=2,
            )

        # ---- 4. The estimates at that theta ----
        out = model.fit_at(theta.tolist())
        # The engine's fixed effects are the coefficients of q, less the regression's
        # part. Turn them, and their covariance matrix, into those of x.
        gamma = out["beta"] + gamma_start
        beta = from_gamma @ gamma
        beta_cov = from_gamma @ out["beta_cov"] @ from_gamma.T
        # The engine returns the random effects of all factors in one array: those of
        # the first factor's levels, then the second's. Cut it back into one array per
        # factor, each aligned with that factor's levels.
        splits = np.cumsum([level.size for level in levels])[:-1]
        ranef = np.split(out["ranef"], splits)
        condsd = np.split(out["condsd"], splits)
        # Fitted values: the fixed effects plus, for each factor, the term of the
        # observation's level. b[code[has_level]] looks up that term for every
        # observation that has a level in the factor.
        fitted = q @ gamma  # the same as x @ beta, computed more accurately
        for b, code in zip(ranef, codes):
            has_level = code >= 0
            fitted[has_level] += b[code[has_level]]
        return LmmFit(
            levels=levels,
            theta=theta,
            sigma=float(out["sigma"]),
            beta=beta,
            beta_se=np.sqrt(np.diag(beta_cov)),
            beta_cov=beta_cov,
            ranef=ranef,
            condsd=condsd,
            fitted=fitted,
            resid=y - fitted,
            deviance=float(out["deviance"]) + (log_det_change if reml else 0.0),
            converged=failure is None,
            n_evals=n_evals,
        )


def split_by_class(
    labels: npt.ArrayLike, classes: npt.ArrayLike
) -> list[np.ma.MaskedArray]:
    """Splits a grouping factor into one factor per class of its levels.

    Fitted with one grouping factor for the sites, a model gives the site terms one
    standard deviation. Fitted with the factors that this function returns, it gives
    the site terms of each class (of site conditions, say) a standard deviation of
    their own::

        fit = fit_lmm(y, [events, *split_by_class(sites, site_class)])
        tau, *phi_s2s_of_each_class = fit.sd

    Each of the returned factors holds the labels of one class. The observations of the
    other classes are masked in it: they have no level in that factor, and get no term
    from it.

    Parameters
    ----------
    labels : array-like
        The labels of the grouping factor (e.g. the site ids), one per observation.
    classes : array-like
        The class of each observation's label (e.g. the class of its site), one per
        observation. Classes are compared as strings.

    Returns
    -------
    list of np.ma.MaskedArray
        One array of labels per class, in the sorted order of the classes, which is
        that of ``np.unique(np.asarray(classes).astype(str))``.

    Raises
    ------
    ValueError
        If ``classes`` does not have one class per label, or has missing values.

    Notes
    -----
    The classes usually also differ in their mean, which the model has to be told about
    separately, with an indicator covariate for every class but one: see
    ``docs/adding_models.md``.

    A class needs at least two levels, as every grouping factor does.

    Examples
    --------
    >>> import numpy as np
    >>> from mera.lmm import split_by_class
    >>> sites = np.array(["s1", "s2", "s3", "s1"])
    >>> site_class = np.array(["rock", "soil", "rock", "rock"])
    >>> rock, soil = split_by_class(sites, site_class)
    >>> rock.tolist()
    ['s1', None, 's3', 's1']
    >>> soil.tolist()
    [None, 's2', None, None]
    """
    labels = np.asarray(labels)
    classes = np.asarray(classes)
    if classes.shape != labels.shape:
        raise ValueError(
            f"classes needs one class per label ({labels.shape}), got shape "
            f"{classes.shape}"
        )
    if np.any(pd.isna(classes)):
        raise ValueError("classes has missing values (NaN or None)")
    classes = classes.astype(str)
    # In a masked array, True in the mask marks an entry as absent: here, the labels of
    # the observations that are not of this class.
    return [
        np.ma.masked_array(labels, mask=classes != name) for name in np.unique(classes)
    ]


def _encode_labels(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Finds the levels of a grouping factor and the code of each label.

    The result is that of ``np.unique(labels.astype(str), return_inverse=True)``, found
    faster where possible, and with missing labels set apart.

    Parameters
    ----------
    labels : np.ndarray
        The labels, one per observation.

    Returns
    -------
    levels : np.ndarray
        The distinct labels as strings, sorted.
    codes : np.ndarray
        For each label, the position of its string in ``levels``, as int64. A missing
        label (NaN or None) has the code -1.
    """
    # Turning every label into a string and sorting them all is the slow way: for a
    # million records it takes about a second. If equal labels always have equal
    # strings, it is enough to find the distinct labels (pd.factorize does this with a
    # hash table, without sorting) and to turn only those into strings and sort them.
    # That holds for whole numbers, booleans, dates and strings. It does not hold for
    # an array of mixed objects, where 1 and 1.0 are equal but "1" and "1.0" are not,
    # nor for floating-point numbers, where 0.0 and -0.0 are equal.
    is_strings = labels.dtype == object and pd.api.types.infer_dtype(labels) == "string"
    if is_strings or labels.dtype.kind in "iubUM":
        # first_codes numbers the distinct labels in the order they first appear (-1
        # for a missing label), and position[j] is where the j-th of them is in levels.
        first_codes, distinct = pd.factorize(labels)
        levels, position = np.unique(
            np.asarray(distinct).astype(str), return_inverse=True
        )
        codes = np.full(labels.size, -1, dtype=np.int64)
        present = first_codes >= 0
        codes[present] = position[first_codes[present]]
        return levels, codes
    present = ~pd.isna(labels)
    levels, codes_present = np.unique(labels[present].astype(str), return_inverse=True)
    codes = np.full(labels.size, -1, dtype=np.int64)
    codes[present] = codes_present
    return levels, codes


def _search(
    model: _lmm.CrossedModel,
    omega_start: np.ndarray,
    omega_max: float,
    n_levels: Sequence[int],
) -> tuple[np.ndarray, int, str | None]:
    """Searches for the omega with the smallest criterion.

    The search has up to three stages, whose settings are at the top of this file.
    The first follows the slope of the criterion downhill, and gets close to the
    optimum in few evaluations. The second jumps to the optimum by Newton's method,
    which also tells whether the first had arrived. Where it cannot confirm that, which
    is rare, a slower search that needs no slope finishes the work.

    Parameters
    ----------
    model : _lmm.CrossedModel
        The engine's model, which evaluates the criterion and its gradient.
    omega_start : np.ndarray
        Where to start: ``log(1 + theta^2)`` for each grouping factor.
    omega_max : float
        The upper limit of every omega. The lower limit is 0, where theta is 0.
    n_levels : sequence of int
        The number of levels of each grouping factor, which sets how sharply the
        criterion curves in each omega.

    Returns
    -------
    omega : np.ndarray
        The best omega found.
    n_evals : int
        The number of evaluations of the criterion, with or without its gradient.
    failure : str or None
        Why the search stopped without converging, None if it converged.
    """
    n_evals = 0
    bounds = [(0.0, omega_max)] * omega_start.size
    # the latest omega at which the gradient was evaluated, and that gradient
    latest: tuple[np.ndarray, np.ndarray] | None = None

    def value_and_gradient(omega: np.ndarray) -> tuple[float, np.ndarray]:
        """The criterion at omega and its slope in each omega.

        Parameters
        ----------
        omega : np.ndarray
            ``log(1 + theta^2)`` for each grouping factor.

        Returns
        -------
        value : float
            The criterion.
        gradient : np.ndarray
            Its derivative with respect to each omega.
        """
        nonlocal n_evals, latest
        n_evals += 1
        value, gradient = model.deviance_and_gradient(_theta_from(omega).tolist())
        # The engine's gradient is with respect to theta^2 = exp(omega) - 1, which
        # changes exp(omega) times as fast as omega.
        gradient = np.exp(omega) * gradient
        latest = (omega.copy(), gradient)
        return value, gradient

    # How much each omega is stretched for stage 1: see there.
    stretch = np.sqrt(np.asarray(n_levels, dtype=float))

    def stretched(v: np.ndarray) -> tuple[float, np.ndarray]:
        """The criterion and its slope, in the stretched variables of stage 1.

        Parameters
        ----------
        v : np.ndarray
            ``omega * stretch`` for each grouping factor.

        Returns
        -------
        value : float
            The criterion.
        gradient : np.ndarray
            Its derivative with respect to each stretched variable.
        """
        value, gradient = value_and_gradient(v / stretch)
        return value, gradient / stretch

    def value(omega: np.ndarray) -> float:
        """The criterion at omega.

        Parameters
        ----------
        omega : np.ndarray
            ``log(1 + theta^2)`` for each grouping factor.

        Returns
        -------
        float
            The criterion.
        """
        nonlocal n_evals
        n_evals += 1
        return model.deviance(_theta_from(omega).tolist())

    # ---- Stage 1: downhill along the slope ----
    # L-BFGS-B starts by assuming that the criterion curves equally in every variable,
    # and learns the real curvatures as it goes. The criterion curves in the omega of a
    # factor roughly in proportion to the number of its levels (each level adds its own
    # bit of information about the factor's standard deviation), so in omega itself the
    # curvatures can differ a thousandfold between events and sites. Stretching each
    # omega by the square root of its number of levels makes them alike from the start,
    # which saves the evaluations it would spend finding out.
    result = minimize(
        stretched,
        omega_start * stretch,
        jac=True,  # the function returns the gradient as well as the value
        method="L-BFGS-B",
        bounds=[(0.0, omega_max * factor) for factor in stretch],
        # gtol is a second stopping rule, on the size of the slope. It is set too small
        # to act, because the slope has no natural scale: it grows with the data.
        options={
            "ftol": GRADIENT_SEARCH_FTOL,
            "gtol": 1e-10,
            "maxfun": MAX_EVALUATIONS,
        },
    )
    # (clip: dividing by the stretch can leave an omega one rounding step past a bound)
    omega = np.clip(result.x / stretch, 0.0, omega_max)
    if n_evals >= MAX_EVALUATIONS:
        return omega, n_evals, "the maximum number of evaluations was reached"

    # ---- Stage 2: Newton's method ----
    for _ in range(NEWTON_MAX_STEPS):
        # The gradient at omega is usually the latest that was evaluated.
        if latest is not None and np.array_equal(latest[0], omega):
            gradient = latest[1]
        else:
            _, gradient = value_and_gradient(omega)
        # An omega on a bound stays there if the criterion rises away from the bound.
        # That is the case of a singular fit: omega = 0 with a positive slope.
        held = ((omega <= 0.0) & (gradient > 0.0)) | (
            (omega >= omega_max) & (gradient < 0.0)
        )
        free = np.flatnonzero(~held)
        if free.size == 0:
            # every omega is held on a bound, so there is nowhere better to go
            return omega, n_evals, None
        # The Hessian: how fast each slope changes with each of the free omegas, from
        # the change of the gradient when one omega is shifted a little.
        hessian = np.empty((free.size, free.size))
        for column, j in enumerate(free):
            shifted = omega.copy()
            shifted[j] += NEWTON_DIFFERENCE
            hessian[:, column] = (
                value_and_gradient(shifted)[1][free] - gradient[free]
            ) / NEWTON_DIFFERENCE
        hessian = (hessian + hessian.T) / 2  # it is symmetric, but for rounding
        # At a minimum the criterion curves upwards in every direction, which makes the
        # Hessian positive definite: the Cholesky factorisation fails if it is not.
        try:
            np.linalg.cholesky(hessian)
        except np.linalg.LinAlgError:
            break
        # The jump that would bring every free slope to zero.
        jump = -np.linalg.solve(hessian, gradient[free])
        landing = omega[free] + jump
        # A long jump, or one that would leave the bounds, is not trusted: the Hessian
        # only describes the criterion close to where it was measured.
        too_far = np.abs(jump).max() > NEWTON_MAX_JUMP
        if too_far or np.any(landing < 0.0) or np.any(landing > omega_max):
            break
        omega = omega.copy()
        omega[free] = landing
        # A jump this short means that the search had already arrived.
        if np.abs(jump).max() <= NEWTON_DONE:
            return omega, n_evals, None

    # ---- Stage 3: without the slope, from where the search stands ----
    # (maxfev: what is left of the allowance of evaluations)
    result = minimize(
        value,
        omega,
        method="COBYQA",
        bounds=bounds,
        options={
            "initial_tr_radius": FALLBACK_TR_RADIUS,
            "final_tr_radius": FINAL_TR_RADIUS,
            "maxfev": max(MAX_EVALUATIONS - n_evals, 1),
        },
    )
    return result.x, n_evals, None if result.success else str(result.message)


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
