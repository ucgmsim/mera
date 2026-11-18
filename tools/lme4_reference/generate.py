"""Regenerates tests/resources/lme4_reference with R's lme4, through pymer4.

Run by hand, outside CI, in the environment described in that directory's README.md.
Without arguments it rewrites every file there except the README. To rewrite only some,
name their groups, which are the keys of ``WRITERS`` at the end of this file::

    python tools/lme4_reference/generate.py covariate_fits
"""

import sys
import warnings
from collections.abc import Sequence
from pathlib import Path
from unittest.mock import patch

import numpy as np
import numpy.typing as npt
import pandas as pd
from pymer4.models import Lmer  # ty: ignore[unresolved-import]
from rpy2.robjects import FloatVector  # ty: ignore[unresolved-import]
from rpy2.robjects.packages import importr  # ty: ignore[unresolved-import]

import mera.analysis
from mera.lmm import LmmFit

sys.path.insert(0, str(Path(__file__).parents[2] / "tests"))
from example_data import (  # ty: ignore[unresolved-import]
    CLASS_FITS,
    CONFIGS,
    COVARIATE_FITS,
    CRITERION_IMS,
    CRITERION_THETAS,
    LME4_FITS,
    LME4_REFERENCE,
    SITE_CLASSES,
    class_inputs,
    covariate_inputs,
    fit_inputs,
    groups_of,
    load_example_residuals,
    reference_mask,
    run_example,
)

# lme4's default optimiser (nloptwrap's BOBYQA) stops on its relative tolerances, up to
# ~2e-4 short of the optimum in the random effects. With only a 1e-14 step tolerance it
# runs to the optimum.
CONTROL = (
    "optCtrl = list(ftol_abs = 0, ftol_rel = 0, xtol_abs = 1e-14, xtol_rel = 0, "
    "maxeval = 1e5)"
)
lme4 = importr("lme4")
base, stats = importr("base"), importr("stats")


def lmer(
    y: npt.ArrayLike,
    groups: Sequence[npt.ArrayLike],
    intercept: bool,
    reml: bool,
    covariates: npt.ArrayLike | None = None,
) -> Lmer:
    """lme4's fit of ``y ~ [1|0] + x0 + ... + (1|g0) + ...``, run to convergence.

    A grouping factor that leaves observations out (a masked array, as
    ``mera.lmm.split_by_class`` makes) is fitted as ``(0 + in_k | g_k)``: lme4 has no
    "no level", so those observations share a placeholder level, whose term is
    multiplied by the indicator ``in_k``, which is 0 for them.
    """
    data = pd.DataFrame({"y": np.asarray(y, dtype=float)})
    fixed = [str(int(intercept))]
    if covariates is not None:
        # one column per covariate, as for fit_lmm
        for j, column in enumerate(np.asarray(covariates, dtype=float).T):
            data[f"x{j}"] = column
            fixed.append(f"x{j}")
    terms = []
    for k, group in enumerate(groups):
        in_factor = ~np.ma.getmaskarray(group)
        labels = np.ma.getdata(group).astype(str)
        if in_factor.all():
            data[f"g{k}"] = labels
            terms.append(f"(1|g{k})")
        else:
            data[f"g{k}"] = np.where(in_factor, labels, "(not in this factor)")
            data[f"in{k}"] = in_factor.astype(float)
            terms.append(f"(0 + in{k}|g{k})")
    model = Lmer(f"y ~ {' + '.join(fixed)} + {' + '.join(terms)}", data=data)
    model.fit(summary=False, REML=reml, control=CONTROL)
    info = model.model_obj.do_slot("optinfo")
    message = info.rx2("message")[0]
    if info.rx2("conv").rx2("opt")[0] != 0 or "MAXEVAL" in message:
        raise RuntimeError(f"lme4 did not converge: {message}")
    return model


def internal_theta(model: Lmer, theta: list[float]) -> FloatVector:
    """``theta`` (in the order of the grouping factors) in lme4's own order."""
    names = lme4.getME(model.model_obj, "theta").names
    return FloatVector([theta[int(name[1 : name.index(".")])] for name in names])


def fit_lme4(
    y: npt.ArrayLike,
    groups: Sequence[npt.ArrayLike],
    intercept: bool = True,
    reml: bool = True,
    name: str | None = None,
    covariates: npt.ArrayLike | None = None,
) -> LmmFit:
    """What :func:`mera.lmm.fit_lmm` returns, from lme4. ``name`` is unused: here, lme4
    not converging is an error."""
    model = lmer(y, groups, intercept, reml, covariates)
    theta_r = lme4.getME(model.model_obj, "theta")
    # lme4 names each theta after its factor and term: "g0.(Intercept)", or "g1.in1".
    theta = np.array(
        [
            theta_r.rx2(next(n for n in theta_r.names if n.startswith(f"g{k}.")))[0]
            for k in range(len(groups))
        ]
    )
    # lme4's REMLcrit() is wrong without fixed effects (2.0.6); its deviance function isn't.
    deviance = lme4.getME(model.model_obj, "devfun")(theta_r)[0]
    effects = model.ranef_df
    # the levels of each factor: the labels of the observations that are in it
    levels = [
        np.unique(np.ma.getdata(group)[~np.ma.getmaskarray(group)].astype(str))
        for group in groups
    ]
    by_level = [
        effects[effects["grpvar"] == f"g{k}"].set_index("grp").loc[level]
        for k, level in enumerate(levels)
    ]
    # The rows of coefs follow the formula: the intercept, then the covariates.
    has_fixed_effects = intercept or covariates is not None
    coefs = (
        model.coefs if has_fixed_effects else pd.DataFrame(columns=["Estimate", "SE"])
    )
    # lme4's covariance matrix of the fixed effects, as an ordinary matrix
    beta_cov = (
        np.asarray(base.as_matrix(stats.vcov(model.model_obj)), dtype=float)
        if has_fixed_effects
        else np.zeros((0, 0))
    )
    fitted = np.asarray(model.fits, dtype=float)
    return LmmFit(
        levels=levels,
        theta=theta,
        sigma=float(model.ranef_var.loc["Residual", "Std"]),
        beta=coefs["Estimate"].to_numpy(dtype=float),
        beta_se=coefs["SE"].to_numpy(dtype=float),
        beta_cov=beta_cov,
        ranef=[b["condval"].to_numpy() for b in by_level],
        condsd=[b["condsd"].to_numpy() for b in by_level],
        fitted=fitted,
        resid=np.asarray(y, dtype=float) - fitted,
        deviance=float(deviance),
        converged=True,
        n_evals=int(model.model_obj.do_slot("optinfo").rx2("feval")[0]),
    )


def write_run_mera_outputs(res: pd.DataFrame) -> None:
    """no_mask/ and mask/: run_mera's own bookkeeping around lme4's fits."""
    ims = res.columns.drop(["event_id", "stat_id"]).tolist()
    for kind, mask in [("no_mask", None), ("mask", reference_mask(res))]:
        with patch.object(mera.analysis, "fit_lmm", fit_lme4):
            run_example(res, ims, mask).save(LME4_REFERENCE / kind)


def fit_row(
    fit: LmmFit, key: dict[str, str | bool], factors: list[str], fixed: list[str]
) -> dict[str, str | bool | float]:
    """A fit as a row of a fits file: theta by factor, then the fixed effects and their
    standard errors by name (the layout that ``example_data.load_lme4_reference``
    reads)."""
    return (
        key
        | {"deviance": fit.deviance}
        | {f"theta_{name}": value for name, value in zip(factors, fit.theta)}
        | {"sigma": fit.sigma}
        | {f"beta_{name}": value for name, value in zip(fixed, fit.beta)}
        | {f"se_{name}": value for name, value in zip(fixed, fit.beta_se)}
    )


def effects_of(
    fit: LmmFit, key: dict[str, str | bool], factors: list[str]
) -> pd.DataFrame:
    """The random effects of a fit and their conditional SDs, a row per level."""
    return pd.concat(
        pd.DataFrame(
            key | {"factor": factor, "level": levels, "ranef": ranef, "condsd": condsd}
        )
        for factor, levels, ranef, condsd in zip(
            factors, fit.levels, fit.ranef, fit.condsd
        )
    )


def write_fits(res: pd.DataFrame) -> None:
    """fits.csv and random_effects.csv: lme4's fits for LME4_FITS."""
    fits, effects = [], []
    for data, intercept, site_term in LME4_FITS:
        y, groups = fit_inputs(res, data, site_term)
        fit = fit_lme4(y, groups, intercept)
        key = {"data": data, "intercept": intercept, "site_term": site_term}
        factors = ["event", "site"][: len(groups)]
        fits.append(fit_row(fit, key, factors, ["intercept"] if intercept else []))
        effects.append(effects_of(fit, key, factors))
    # The first fit has every column, which fixes their order; a fit without a site
    # term or an intercept leaves those entries empty.
    pd.DataFrame(fits).to_csv(LME4_REFERENCE / "fits.csv", index=False)
    pd.concat(effects).to_csv(LME4_REFERENCE / "random_effects.csv", index=False)


def write_covariate_fits(res: pd.DataFrame) -> None:
    """covariate_fits.csv and covariate_random_effects.csv: lme4's fits for
    COVARIATE_FITS."""
    factors, fixed = ["event", "site"], ["intercept", "magnitude", "distance"]
    fits, effects = [], []
    for im, reml in COVARIATE_FITS:
        y, groups, covariates = covariate_inputs(res, im)
        fit = fit_lme4(y, groups, reml=reml, covariates=covariates)
        key = {"im": im, "reml": reml}
        fits.append(fit_row(fit, key, factors, fixed))
        effects.append(effects_of(fit, key, factors))
    pd.DataFrame(fits).to_csv(LME4_REFERENCE / "covariate_fits.csv", index=False)
    pd.concat(effects).to_csv(
        LME4_REFERENCE / "covariate_random_effects.csv", index=False
    )


def write_class_fits(res: pd.DataFrame) -> None:
    """class_fits.csv and class_random_effects.csv: lme4's fits for CLASS_FITS, which
    have a site standard deviation per class of site."""
    factors = ["event", *(f"site_{name}" for name in SITE_CLASSES)]
    # the first class is the reference: the others have an indicator covariate
    fixed = ["intercept", *SITE_CLASSES[1:]]
    fits, effects = [], []
    for im, reml in CLASS_FITS:
        y, groups, covariates = class_inputs(res, im)
        fit = fit_lme4(y, groups, reml=reml, covariates=covariates)
        key = {"im": im, "reml": reml}
        fits.append(fit_row(fit, key, factors, fixed))
        effects.append(effects_of(fit, key, factors))
    pd.DataFrame(fits).to_csv(LME4_REFERENCE / "class_fits.csv", index=False)
    pd.concat(effects).to_csv(LME4_REFERENCE / "class_random_effects.csv", index=False)


def write_criterion(res: pd.DataFrame) -> None:
    """criterion.csv: lme4's REML criterion at fixed theta."""
    rows = []
    for im in CRITERION_IMS:
        for intercept, site_term in CONFIGS:
            groups = groups_of(res, site_term)
            model = lmer(res[im].to_numpy(), groups, intercept, True)
            devfun = lme4.getME(model.model_obj, "devfun")
            for theta in CRITERION_THETAS:
                theta = list(theta[: len(groups)])
                rows.append(
                    {
                        "im": im,
                        "intercept": intercept,
                        "site_term": site_term,
                        "theta_event": theta[0],
                        "theta_site": theta[1] if site_term else np.nan,
                        "deviance": devfun(internal_theta(model, theta))[0],
                    }
                )
    pd.DataFrame(rows).to_csv(LME4_REFERENCE / "criterion.csv", index=False)


WRITERS = {
    "run_mera": write_run_mera_outputs,
    "fits": write_fits,
    "covariate_fits": write_covariate_fits,
    "class_fits": write_class_fits,
    "criterion": write_criterion,
}


def main(names: Sequence[str]) -> None:
    """Writes the named groups of files of LME4_REFERENCE, or all of them."""
    warnings.simplefilter("ignore", FutureWarning)  # pymer4 0.8 on pandas 2
    res = load_example_residuals()
    for name in names or WRITERS:
        WRITERS[name](res)


if __name__ == "__main__":
    main(sys.argv[1:])
