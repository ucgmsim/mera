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
# The largest theta that the search considers: a random-effect standard deviation ten
# thousand times that of the residuals. A search that ends there is refused, because it
# was still going up: the model explains the response almost exactly, as it does for
# synthetic data made without noise. Nothing is then left to estimate the residual
# standard deviation from, and theta would grow until the arithmetic failed. (Well below
# the limit the criterion has already lost most of its precision: see the notes of
# fit_lmm.)
MAX_THETA = 1e4
# The length of COBYQA's first steps when fit_lmm is given a start, in the optimiser's
# variables (omega, below). Its default of 1 suits a search that begins anywhere; from a
# start that is already close, shorter steps waste fewer evaluations, and they lengthen
# by themselves if the start turns out to be far off.
WARM_START_TR_RADIUS = 0.1


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
        Where the search for ``theta`` starts: one value per grouping factor, for
        example the ``theta`` of an earlier fit of similar data. The default is 1 for
        every factor, as in lme4.

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

    The fit is done on ``y`` less its least-squares fit on the fixed effects (its mean,
    when the intercept is the only one), and the coefficients of that fit are added
    back to ``beta``. This changes no result, and keeps the fit accurate however large
    the mean of ``y``, or its trend with a covariate, is compared with its spread.

    The result does not depend on ``start``, to the precision of the search, but the
    number of evaluations does. When many similar datasets are fitted, as in a bootstrap,
    starting each from the ``theta`` of the fit of the full data saves evaluations:
    about 15% of them on the example data.

    Fits can run in several threads at once (with
    ``concurrent.futures.ThreadPoolExecutor``, for example), because the engine releases
    Python's global interpreter lock while it calculates. With four threads, four fits
    of 150,000 records took a third of the time they take one after another. Small fits
    gain less, because the search itself (scipy's COBYQA) is Python code.

    Covariates are used as given. Subtracting a typical value from each (magnitude less
    6, say) makes the intercept the mean at that value rather than at zero, changes
    nothing else, and keeps the arithmetic accurate. ``docs/adding_models.md`` has more
    advice on covariates, and on comparing models with and without one.

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

        # The engine is not given y itself, but y less its least-squares fit on the
        # fixed effects: what an ordinary regression of y on the columns of x, ignoring
        # the groups, leaves over. With only an intercept, that is y less its mean. The
        # coefficients of that regression (start_beta) are added back to beta after the
        # fit. This changes no result, because the model can express the same shift
        # through beta, and it keeps the arithmetic accurate: the engine finds a sum of
        # squares by subtraction, which loses digits when the mean of y, or its trend
        # with a covariate, is far larger than its spread.
        start_beta = np.zeros(x.shape[1])
        if x.shape[1] > 0:
            # Each column is scaled to length 1 first, so that neither the regression
            # nor the check below depends on the units of the covariates.
            lengths = np.linalg.norm(x, axis=0)
            rank = 0
            if np.all(lengths > 0.0):
                scaled_beta, _, rank, _ = np.linalg.lstsq(x / lengths, y, rcond=None)
                start_beta = scaled_beta / lengths
            # The rank is the number of columns that are not combinations of the
            # others. If it is less than the number of columns, their coefficients
            # cannot be told apart, and the engine would fail or return meaningless
            # coefficients with enormous standard errors, depending on rounding.
            if rank < x.shape[1]:
                raise ValueError(
                    "the fixed effects are collinear: one column (the intercept or a "
                    "covariate) is a combination of the others"
                )
        # Creating the model reduces the data to sums per level, once. After that,
        # model.deviance(theta) is the criterion at any theta.
        n_levels = [level.size for level in levels]
        model = _lmm.CrossedModel(codes, n_levels, y - x @ start_beta, x, reml=reml)
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
        options = {"final_tr_radius": FINAL_TR_RADIUS, "maxfev": MAX_EVALUATIONS}
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
            options["initial_tr_radius"] = WARM_START_TR_RADIUS
        result = minimize(
            lambda omega: model.deviance(_theta_from(omega).tolist()),
            omega_start,
            method="COBYQA",
            bounds=[(0.0, omega_max)] * n_factors,  # 0 <= theta <= MAX_THETA
            options=options,
        )
        # A search that ends at the upper limit was stopped on its way up: see MAX_THETA.
        # (It ends within rounding of the limit, hence the small allowance.)
        at_limit = result.x >= omega_max - 1e-4
        if np.any(at_limit):
            raise ValueError(
                f"theta reached its limit of {MAX_THETA:g} for grouping factor "
                f"{', '.join(map(str, np.flatnonzero(at_limit)))}: the grouping factors "
                "and fixed effects explain the response almost exactly, which leaves "
                "too little residual variation to fit the model (synthetic data made "
                "without noise does this)"
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
        beta = out["beta"] + start_beta
        # The engine returns the random effects of all factors in one array: those of
        # the first factor's levels, then the second's. Cut it back into one array per
        # factor, each aligned with that factor's levels.
        splits = np.cumsum([level.size for level in levels])[:-1]
        ranef = np.split(out["ranef"], splits)
        condsd = np.split(out["condsd"], splits)
        # Fitted values: the fixed effects plus, for each factor, the term of the
        # observation's level. b[code[has_level]] looks up that term for every
        # observation that has a level in the factor.
        fitted = x @ beta
        for b, code in zip(ranef, codes):
            has_level = code >= 0
            fitted[has_level] += b[code[has_level]]
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
