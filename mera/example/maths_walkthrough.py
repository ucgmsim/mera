"""The mathematics of a mera fit, step by step, in plain numpy.

mera's real engine is written in Rust and works with sparse matrices, so that it copes
with very large datasets. This script does the same calculation with ordinary (dense)
numpy arrays, in the same order and with the same names, so that every step can be read,
printed and experimented with. It is for learning and for checking only: it builds
matrices with a row for every record, which is far too slow for real datasets.

Read it alongside ``docs/mixed_effects_maths.md``, which explains the formulas. The
section numbers in the comments below refer to that document.

How to read this file: start at :func:`main`, at the bottom, which runs the steps in
order. The mathematics is in three short functions: :func:`evaluate` (the criterion at
one theta), :func:`find_theta` (the search for the best theta) and :func:`estimates` (the
results at the best theta). The small classes that carry data between them (``Model``,
``Evaluation`` and ``Estimates``) are defined just above :func:`main`.

Run it with::

    python -m mera.example.maths_walkthrough

It makes a small synthetic dataset with a known bias and known standard deviations, fits
it with the functions below, and prints the results next to those of mera itself.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.optimize import minimize

from mera.lmm import fit_lmm

# What the synthetic data are made with, and so what a fit should roughly recover.
TRUE_BIAS = 0.2
TRUE_TAU = 0.5  # standard deviation of the event terms
TRUE_PHI_S2S = 0.4  # standard deviation of the site terms
TRUE_PHI_W = 0.6  # standard deviation of the remaining residuals


def make_example_data(
    n_events: int = 40, n_sites: int = 15, sites_per_event: int = 8, seed: int = 1
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Makes synthetic residuals from the model that mera fits (section 1).

    Every event is recorded at ``sites_per_event`` sites picked at random, and each
    residual is the bias plus the term of its event, the term of its site and a remaining
    residual, all drawn from normal distributions.

    Parameters
    ----------
    n_events : int, default 40
        Number of events.
    n_sites : int, default 15
        Number of sites.
    sites_per_event : int, default 8
        Number of sites that record each event.
    seed : int, default 1
        Seed of the random number generator.

    Returns
    -------
    y : np.ndarray
        The residual of each record.
    events : np.ndarray
        The event id of each record.
    sites : np.ndarray
        The site id of each record.
    """
    rng = np.random.default_rng(seed)
    event_of_record = np.repeat(np.arange(n_events), sites_per_event)
    site_of_record = np.concatenate(
        [
            rng.choice(n_sites, size=sites_per_event, replace=False)
            for _ in range(n_events)
        ]
    )
    event_terms = TRUE_TAU * rng.normal(size=n_events)
    site_terms = TRUE_PHI_S2S * rng.normal(size=n_sites)
    remaining = TRUE_PHI_W * rng.normal(size=event_of_record.size)
    y = (
        TRUE_BIAS
        + event_terms[event_of_record]
        + site_terms[site_of_record]
        + remaining
    )
    events = np.array([f"event_{e:02d}" for e in event_of_record])
    sites = np.array([f"site_{s:02d}" for s in site_of_record])
    return y, events, sites


def indicator_matrix(
    groups: Sequence[npt.ArrayLike],
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """Builds Z, the matrix that says which levels each record belongs to (section 3).

    Z has a row per record and a column per level: first a column for each event, then
    a column for each site. A row has a 1 in the column of the record's event and a 1
    in the column of its site, and zeros elsewhere.

    Parameters
    ----------
    groups : sequence of array-like
        One array of labels per grouping factor (the event ids, the site ids), each
        with one label per record.

    Returns
    -------
    z : np.ndarray
        The indicator matrix, of shape (records, levels).
    level_factor : np.ndarray
        For each column of ``z``, the index of its grouping factor (0 for an event, 1
        for a site).
    levels : list of np.ndarray
        The sorted labels of each grouping factor, in the order of its columns.
    """
    blocks, level_factor, levels = [], [], []
    for k, group in enumerate(groups):
        # level holds the distinct labels, sorted, and code the position of each
        # record's label among them.
        level, code = np.unique(np.asarray(group).astype(str), return_inverse=True)
        block = np.zeros((code.size, level.size))
        # For every record i at once, put a 1 in row i, column code[i].
        block[np.arange(code.size), code] = 1.0
        blocks.append(block)
        level_factor.append(np.full(level.size, k))
        levels.append(level)
    return np.hstack(blocks), np.concatenate(level_factor), levels


def criterion_from_definition(theta: np.ndarray, model: Model) -> float:
    """The REML criterion straight from its definition (sections 4 and 5).

    This builds V, which has a row and a column for every record, and so is only
    possible for small datasets. :func:`evaluate` gets the same number without V.

    Parameters
    ----------
    theta : np.ndarray
        Standard deviation of each factor's terms divided by that of the remaining
        residuals.
    model : Model
        The fit in matrix form.

    Returns
    -------
    float
        The REML criterion at ``theta``.
    """
    y, x, z = model.y, model.x, model.z
    n, p = x.shape
    # The diagonal of Lambda: the theta of each level's factor.
    lam = theta[model.level_factor]
    # V = I + Z Lambda Lambda' Z'. Two records are correlated if they share an event
    # or a site. z * lam**2 multiplies each column of z by the square of its theta.
    v = np.eye(n) + (z * lam**2) @ z.T
    v_inv = np.linalg.inv(v)
    # Generalised least squares: the best beta when the records are correlated.
    xvx = x.T @ v_inv @ x
    beta = np.linalg.solve(xvx, x.T @ v_inv @ y)
    # The sum of squares of what beta leaves unexplained, allowing for the correlations.
    residual = y - x @ beta
    r2 = residual @ v_inv @ residual
    # slogdet returns the sign and the logarithm of a determinant: [1] is the logarithm.
    log_det_v = np.linalg.slogdet(v)[1]
    log_det_xvx = np.linalg.slogdet(xvx)[1]
    return float(
        log_det_v + log_det_xvx + (n - p) * (1 + np.log(2 * np.pi * r2 / (n - p)))
    )


def evaluate(theta: np.ndarray, model: Model) -> Evaluation:
    """Evaluates the model at one theta, the way mera's engine does (section 6).

    Parameters
    ----------
    theta : np.ndarray
        Standard deviation of each factor's terms divided by that of the remaining
        residuals.
    model : Model
        The fit in matrix form.

    Returns
    -------
    Evaluation
        The REML criterion and the quantities computed on the way to it, which
        :func:`estimates` reuses.
    """
    y, x, z = model.y, model.x, model.z
    n, p = x.shape
    q = z.shape[1]
    # The diagonal of Lambda: the theta of each level's factor.
    lam = theta[model.level_factor]

    # Step 1. A = Lambda' Z'Z Lambda + I: a row and a column per level, not per record.
    # Z'Z holds counts of records, and does not depend on theta. The next line
    # multiplies its entry in row i and column j by lam[i] and by lam[j].
    ztz = z.T @ z
    a = lam[:, None] * ztz * lam[None, :] + np.eye(q)
    # The Cholesky factorisation A = L L' gives log|A| from the diagonal of L.
    chol = np.linalg.cholesky(a)
    log_det_a = 2 * np.log(np.diag(chol)).sum()

    # Steps 2 to 4. M = [y X]' V^-1 [y X], without ever building V:
    # M = [y X]'[y X] - R' W, with R = Lambda' Z'[y X] and W the solution of A W = R.
    yx = np.column_stack([y, x])
    r = lam[:, None] * (z.T @ yx)
    w = np.linalg.solve(a, r)
    m = yx.T @ yx - r.T @ w

    # Step 5. The blocks of M give beta and the penalised residual sum of squares.
    xvx = m[1:, 1:]  # X' V^-1 X
    xvy = m[1:, 0]  # X' V^-1 y
    beta = np.linalg.solve(xvx, xvy)
    r2 = m[0, 0] - xvy @ beta

    # Step 6. The criterion.
    log_det_xvx = np.linalg.slogdet(xvx)[1]
    dof = n - p
    deviance = log_det_a + log_det_xvx + dof * (1 + np.log(2 * np.pi * r2 / dof))
    return Evaluation(
        deviance=float(deviance),
        lam=lam,
        a=a,
        w=w,
        xvx=xvx,
        beta=beta,
        r2=float(r2),
        dof=dof,
    )


def find_theta(model: Model) -> np.ndarray:
    """Searches for the theta with the smallest criterion (section 7).

    Parameters
    ----------
    model : Model
        The fit in matrix form.

    Returns
    -------
    np.ndarray
        The best theta, one value per grouping factor.
    """
    n_factors = model.level_factor.max() + 1
    # As in mera.lmm.fit_lmm, the optimiser searches over omega = log(1 + theta^2) for
    # each factor, starting from theta = 1 and never going below theta = 0.
    result = minimize(
        lambda omega: evaluate(_theta_from(omega), model).deviance,
        np.full(n_factors, np.log(2.0)),
        method="COBYQA",
        bounds=[(0.0, None)] * n_factors,
        options={"final_tr_radius": 1e-8},
    )
    # A value within the optimiser's precision of zero is zero.
    return _theta_from(np.where(result.x < 1e-8, 0.0, result.x))


def _theta_from(omega: np.ndarray) -> np.ndarray:
    """Converts the optimiser's variables, ``log(1 + theta^2)``, to theta.

    Parameters
    ----------
    omega : np.ndarray
        ``log(1 + theta^2)`` for each grouping factor.

    Returns
    -------
    np.ndarray
        ``theta`` for each grouping factor.
    """
    return np.sqrt(np.expm1(omega))


def estimates(theta: np.ndarray, model: Model) -> Estimates:
    """Everything that mera reports, at the best theta (section 8).

    Parameters
    ----------
    theta : np.ndarray
        The best theta, from :func:`find_theta`.
    model : Model
        The fit in matrix form.

    Returns
    -------
    Estimates
        The estimates.
    """
    at = evaluate(theta, model)
    lam, a, w, beta = at.lam, at.a, at.w, at.beta

    # The standard deviation of the remaining residuals (phi_w), and those of the
    # random effects (tau and phi_S2S), which theta gives relative to it.
    sigma = float(np.sqrt(at.r2 / at.dof))
    sd = theta * sigma

    # The random effects (the event terms, then the site terms):
    # u = A^-1 Lambda' Z'(y - X beta), and b = Lambda u.
    # Column 0 of w is A^-1 Lambda' Z'y and the others are A^-1 Lambda' Z'X.
    u = w[:, 0] - w[:, 1:] @ beta
    ranef = lam * u

    # Each record's fitted value: the bias plus its event term plus its site term.
    fitted = model.x @ beta + model.z @ ranef

    # The standard error of beta: sigma^2 (X'V^-1X)^-1 is its covariance matrix.
    beta_se = sigma * np.sqrt(np.diag(np.linalg.inv(at.xvx)))
    # The conditional standard deviations: sigma^2 Lambda A^-1 Lambda' is the
    # covariance matrix of the random effects given the data. The engine finds the
    # diagonal of A^-1 without inverting A.
    condsd = sigma * lam * np.sqrt(np.diag(np.linalg.inv(a)))

    return Estimates(
        theta=theta,
        sigma=sigma,
        sd=sd,
        beta=beta,
        beta_se=beta_se,
        ranef=ranef,
        condsd=condsd,
        fitted=fitted,
        resid=model.y - fitted,
        deviance=at.deviance,
    )


# ---- What the steps above pass to each other ----


@dataclass
class Model:
    """One fit in matrix form, ``y = X beta + Z b + e`` (section 3).

    Attributes
    ----------
    y : np.ndarray
        The residual of each record.
    x : np.ndarray
        X: a row per record and a column per fixed effect. A column of ones is the bias.
    z : np.ndarray
        Z, from :func:`indicator_matrix`: a row per record and a column per level.
    level_factor : np.ndarray
        The grouping factor of each column of ``z``: 0 for an event, 1 for a site.
    """

    y: np.ndarray
    x: np.ndarray
    z: np.ndarray
    level_factor: np.ndarray


@dataclass
class Evaluation:
    """The model at one theta, from :func:`evaluate`, named as in the Rust engine.

    Attributes
    ----------
    deviance : float
        The REML criterion.
    lam : np.ndarray
        The diagonal of Lambda: the theta of each level's factor.
    a : np.ndarray
        ``A = Lambda' Z'Z Lambda + I``, with a row and a column per level.
    w : np.ndarray
        ``W = A^-1 Lambda' Z'[y X]``.
    xvx : np.ndarray
        ``X' V^-1 X``.
    beta : np.ndarray
        The fixed effects (the bias).
    r2 : float
        The penalised residual sum of squares.
    dof : int
        The degrees of freedom: records less fixed effects.
    """

    deviance: float
    lam: np.ndarray
    a: np.ndarray
    w: np.ndarray
    xvx: np.ndarray
    beta: np.ndarray
    r2: float
    dof: int


@dataclass
class Estimates:
    """What mera reports for one fit, from :func:`estimates`, named as in ``LmmFit``.

    Attributes
    ----------
    theta : np.ndarray
        Standard deviation of each factor's terms, divided by ``sigma``.
    sigma : float
        Standard deviation of the remaining residuals (``phi_w``).
    sd : np.ndarray
        Standard deviation of each factor's terms (``tau``, ``phi_S2S``).
    beta : np.ndarray
        The fixed effects (the bias).
    beta_se : np.ndarray
        The standard errors of ``beta``.
    ranef : np.ndarray
        The random effects: the event terms, then the site terms.
    condsd : np.ndarray
        The conditional standard deviation of each random effect.
    fitted : np.ndarray
        The fitted value of each record.
    resid : np.ndarray
        The remaining residual of each record.
    deviance : float
        The REML criterion.
    """

    theta: np.ndarray
    sigma: float
    sd: np.ndarray
    beta: np.ndarray
    beta_se: np.ndarray
    ranef: np.ndarray
    condsd: np.ndarray
    fitted: np.ndarray
    resid: np.ndarray
    deviance: float


def main() -> None:
    """Fits the synthetic data step by step and compares the results with mera's."""
    y, events, sites = make_example_data()
    n_events, n_sites = np.unique(events).size, np.unique(sites).size
    print(f"{y.size} records of {n_events} events at {n_sites} sites")

    # Sections 1 to 3: the model in matrix form.
    x = np.ones((y.size, 1))  # a column of ones: the bias
    z, level_factor, levels = indicator_matrix([events, sites])
    model = Model(y, x, z, level_factor)
    print(f"X has shape {x.shape} and Z has shape {z.shape}")
    ones = np.flatnonzero(z[0]).tolist()
    print(f"The first record is {events[0]} at {sites[0]}: its row of Z has a 1 in")
    print(f"columns {ones}, the columns of that event and that site")

    # Sections 4 to 6: the criterion at a trial theta, from its definition and the way
    # the engine computes it.
    trial = np.array([1.0, 1.0])
    direct = criterion_from_definition(trial, model)
    engine = evaluate(trial, model).deviance
    print("\nREML criterion at theta = (1, 1)")
    print(f"  from the definition, with V:   {direct:.8f}")
    print(f"  as the engine computes it:     {engine:.8f}")

    # Section 7: the best theta. Section 8: everything else, at that theta.
    theta = find_theta(model)
    ours = estimates(theta, model)
    theirs = fit_lmm(y, [events, sites])

    print("\n                    true   this script        mera")
    rows = [
        ("bias", TRUE_BIAS, ours.beta[0], theirs.beta[0]),
        ("tau", TRUE_TAU, ours.sd[0], theirs.sd[0]),
        ("phi_S2S", TRUE_PHI_S2S, ours.sd[1], theirs.sd[1]),
        ("phi_w", TRUE_PHI_W, ours.sigma, theirs.sigma),
        ("bias_std_err", None, ours.beta_se[0], theirs.beta_se[0]),
        ("REML criterion", None, ours.deviance, theirs.deviance),
    ]
    for name, true, this_script, mera_value in rows:
        true_text = "" if true is None else f"{true:.3f}"
        print(f"  {name:<14} {true_text:>7} {this_script:13.6f} {mera_value:11.6f}")
    difference = np.abs(ours.ranef - np.concatenate(theirs.ranef)).max()
    print(f"  largest difference in an event or site term: {difference:.1e}")

    # Shrinkage (section 2). An event term is not simply the mean of what is left of
    # the event's residuals once the bias and the site terms are taken off: it is that
    # mean pulled towards zero, more strongly the fewer records the event has.
    event = levels[0][0]
    records = events == event
    n_e = records.sum()
    is_site = level_factor == 1
    site_terms = (z[:, is_site] @ ours.ranef[is_site])[records]
    mean_left = (y[records] - ours.beta[0] - site_terms).mean()
    tau, phi_w = ours.sd[0], ours.sigma
    shrinkage = n_e * tau**2 / (n_e * tau**2 + phi_w**2)
    product = shrinkage * mean_left
    print(f"\nShrinkage of the term of {event}, which has n_e = {n_e} records")
    print(f"  mean of (residual - bias - site term):           {mean_left:+.6f}")
    print(f"  shrinkage, n_e tau^2 / (n_e tau^2 + phi_w^2):     {shrinkage:.6f}")
    print(f"  their product:                                   {product:+.6f}")
    print(f"  the event term:                                  {ours.ranef[0]:+.6f}")
    print(f"  its conditional standard deviation:               {ours.condsd[0]:.6f}")


if __name__ == "__main__":
    main()
