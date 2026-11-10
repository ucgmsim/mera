"""Regenerates tests/resources/lme4_reference with R's lme4, through pymer4.

Run by hand, outside CI, in the environment described in that directory's README.md.
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
    CONFIGS,
    CRITERION_IMS,
    CRITERION_THETAS,
    LME4_FITS,
    LME4_REFERENCE,
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


def lmer(
    y: npt.ArrayLike, groups: Sequence[npt.ArrayLike], intercept: bool, reml: bool
) -> Lmer:
    """lme4's fit of ``y ~ [1|0] + (1|g0) + ...``, run to convergence."""
    data = pd.DataFrame({"y": np.asarray(y, dtype=float)})
    for k, labels in enumerate(groups):
        data[f"g{k}"] = np.asarray(labels).astype(str)
    terms = " + ".join(f"(1|g{k})" for k in range(len(groups)))
    model = Lmer(f"y ~ {int(intercept)} + {terms}", data=data)
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
) -> LmmFit:
    """What :func:`mera.lmm.fit_lmm` returns, from lme4. ``name`` is unused: here, lme4
    not converging is an error."""
    model = lmer(y, groups, intercept, reml)
    theta_r = lme4.getME(model.model_obj, "theta")
    theta = np.array([theta_r.rx2(f"g{k}.(Intercept)")[0] for k in range(len(groups))])
    # lme4's REMLcrit() is wrong without fixed effects (2.0.6); its deviance function isn't.
    deviance = lme4.getME(model.model_obj, "devfun")(theta_r)[0]
    effects = model.ranef_df
    levels = [np.unique(np.asarray(labels).astype(str)) for labels in groups]
    by_level = [
        effects[effects["grpvar"] == f"g{k}"].set_index("grp").loc[level]
        for k, level in enumerate(levels)
    ]
    coefs = model.coefs if intercept else pd.DataFrame(columns=["Estimate", "SE"])
    fitted = np.asarray(model.fits, dtype=float)
    return LmmFit(
        levels=levels,
        theta=theta,
        sigma=float(model.ranef_var.loc["Residual", "Std"]),
        beta=coefs["Estimate"].to_numpy(dtype=float),
        beta_se=coefs["SE"].to_numpy(dtype=float),
        ranef=[b["condval"].to_numpy() for b in by_level],
        condsd=[b["condsd"].to_numpy() for b in by_level],
        fitted=fitted,
        resid=np.asarray(y, dtype=float) - fitted,
        deviance=float(deviance),
        converged=True,
        n_evals=int(model.model_obj.do_slot("optinfo").rx2("feval")[0]),
    )


def main() -> None:
    """Writes every file of LME4_REFERENCE except its README."""
    warnings.simplefilter("ignore", FutureWarning)  # pymer4 0.8 on pandas 2
    res = load_example_residuals()

    ims = res.columns.drop(["event_id", "stat_id"]).tolist()
    for kind, mask in [("no_mask", None), ("mask", reference_mask(res))]:
        # run_mera's own bookkeeping around lme4's fits
        with patch.object(mera.analysis, "fit_lmm", fit_lme4):
            run_example(res, ims, mask).save(LME4_REFERENCE / kind)

    fits, effects = [], []
    for data, intercept, site_term in LME4_FITS:
        y, groups = fit_inputs(res, data, site_term)
        fit = fit_lme4(y, groups, intercept)
        key = {"data": data, "intercept": intercept, "site_term": site_term}
        fits.append(
            key
            | {
                "deviance": fit.deviance,
                "theta_event": fit.theta[0],
                "theta_site": fit.theta[1] if site_term else np.nan,
                "sigma": fit.sigma,
                "beta": fit.beta[0] if intercept else np.nan,
                "beta_se": fit.beta_se[0] if intercept else np.nan,
            }
        )
        for factor, levels, ranef, condsd in zip(
            ["event", "site"], fit.levels, fit.ranef, fit.condsd
        ):
            columns = {"factor": factor, "level": levels, "ranef": ranef}
            effects.append(pd.DataFrame(key | columns | {"condsd": condsd}))
    pd.DataFrame(fits).to_csv(LME4_REFERENCE / "fits.csv", index=False)
    pd.concat(effects).to_csv(LME4_REFERENCE / "random_effects.csv", index=False)

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


if __name__ == "__main__":
    main()
