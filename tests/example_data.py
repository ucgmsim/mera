"""Example data and run settings shared by the tests and tools/lme4_reference."""

from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
import pandas as pd

import mera
from mera.example import maths_walkthrough as walkthrough
from mera.lmm import LmmFit, split_by_class

DATA_DIR = Path(__file__).parent.parent / "mera" / "example" / "resources"
MASK_CSV = Path(__file__).parent / "resources" / "mask.csv"
LME4_REFERENCE = Path(__file__).parent / "resources" / "lme4_reference"
EXTRA_IMS = ["PGA", "PGV", "CAV", "AI", "Ds575", "Ds595"]
# (intercept, site term)
CONFIGS = [(True, True), (True, False), (False, True), (False, False)]
BOUNDARY_KINDS = ["no_site_effect", "no_event_effect", "noise_only"]
SUBSET = "pSA_1.0_subset"
# The fit_lmm fits checked against lme4, as (data, intercept, site term): IMs in every
# configuration, pSA_1.0 on a random subset, and synthetic responses whose fits end on
# the boundary.
LME4_FITS = (
    [
        (im, intercept, site_term)
        for im in ["PGA", "AI", "pSA_0.1", "pSA_0.132194114847", "pSA_1.0"]
        for intercept, site_term in CONFIGS
    ]
    + [(SUBSET, True, True)]
    + [
        (kind, intercept, True)
        for kind in BOUNDARY_KINDS
        for intercept in [True, False]
    ]
)
# The fits with covariates checked against lme4, as (IM, REML or ML): see covariate_inputs.
COVARIATE_FITS = [("PGA", True), ("PGA", False)]
# The fits with a site standard deviation per class checked against lme4, as (IM, REML or
# ML): see class_inputs.
CLASS_FITS = [("PGA", True), ("PGA", False)]
# The criterion is also checked against lme4 at these (event, site) theta.
CRITERION_IMS = ["PGA", "pSA_1.0"]
CRITERION_THETAS = [(1.0, 1.0), (0.0, 0.7), (0.9, 0.0), (0.3, 2.0)]
# run_mera's outputs: the frames of MeraResults
OUTPUTS = [field.name for field in fields(mera.MeraResults)]


def load_example_residuals() -> pd.DataFrame:
    """ln(obs / sim) for every IM of the example data, with prefixed event and site ids."""
    obs = pd.read_csv(DATA_DIR / "im_obs.csv", index_col=0)
    sim = pd.read_csv(DATA_DIR / "im_sim.csv", index_col=0)
    pd.testing.assert_frame_equal(
        obs[["event_id", "stat_id"]], sim[["event_id", "stat_id"]]
    )
    ims = [
        im
        for im in np.intersect1d(obs.columns, sim.columns)
        if im.startswith("pSA") or im in EXTRA_IMS
    ]
    ids = pd.DataFrame(
        {
            "event_id": np.char.add("event_", obs["event_id"].to_numpy().astype(str)),
            "stat_id": np.char.add("stat_", obs["stat_id"].to_numpy().astype(str)),
        },
        index=obs.index,
    )
    # One concat rather than inserting columns, which fragments the frame.
    return pd.concat([np.log(obs[ims] / sim[ims]), ids], axis=1)


def groups_of(df: pd.DataFrame, site_term: bool) -> list[np.ndarray]:
    """The event ids, then the site ids if ``site_term``: the ``groups`` of fit_lmm."""
    return [df["event_id"].to_numpy()] + (
        [df["stat_id"].to_numpy()] if site_term else []
    )


def synthetic(df: pd.DataFrame, kind: str, seed: int = 42) -> np.ndarray:
    """A synthetic response for the events and sites of ``df``: with an event effect only
    (``kind`` "no_site_effect"), a site effect only ("no_event_effect"), neither
    ("noise_only"), or weak ones ("weak_effects")."""
    rng = np.random.default_rng(seed)
    events, ev = np.unique(df["event_id"], return_inverse=True)
    sites, st = np.unique(df["stat_id"], return_inverse=True)
    event = 0.4 * rng.normal(size=events.size)[ev]
    site = 0.35 * rng.normal(size=sites.size)[st]
    noise = 0.5 * rng.normal(size=len(df))
    effect = {
        "no_site_effect": event,
        "no_event_effect": site,
        "noise_only": 0.0,
        "weak_effects": (event + site) / 13,
    }
    return 0.1 + noise + effect[kind]


def fit_inputs(
    df: pd.DataFrame, data: str, site_term: bool
) -> tuple[np.ndarray, list[np.ndarray]]:
    """The response and grouping factors of a fit in LME4_FITS."""
    if data == SUBSET:
        # 60% of the records, which leaves many events and sites with a single record
        df = df[np.random.default_rng(1).random(len(df)) < 0.6]
        y = df["pSA_1.0"].to_numpy()
    else:
        y = synthetic(df, data) if data in BOUNDARY_KINDS else df[data].to_numpy()
    return y, groups_of(df, site_term)


def add_covariate_trend(
    y: np.ndarray, events: np.ndarray, sites: np.ndarray
) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
    """A response with a trend in two made-up covariates, its grouping factors
    (events, sites) and the covariates.

    One covariate is constant within an event, like magnitude, and one differs for
    every record, like distance. The trend is added to ``y``, so that the coefficients
    of the covariates are far from zero.
    """
    # covariate_fits.csv was made from these draws: keep the seed, the ranges and the
    # order of the two draws.
    rng = np.random.default_rng(7)
    event_levels, event_codes = np.unique(events, return_inverse=True)
    magnitude = rng.uniform(-1.5, 1.5, size=event_levels.size)[event_codes]
    distance = rng.normal(size=y.size)
    covariates = np.column_stack([magnitude, distance])
    return y + 0.3 * magnitude - 0.2 * distance, [events, sites], covariates


def covariate_inputs(
    df: pd.DataFrame, im: str
) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
    """The response, grouping factors and covariates of a fit in COVARIATE_FITS: an IM's
    residuals with the trend of :func:`add_covariate_trend`."""
    return add_covariate_trend(df[im].to_numpy(), *groups_of(df, True))


# The made-up classes of site of add_site_classes, in the order that split_by_class
# gives them (sorted by name).
SITE_CLASSES = ["high", "low", "mid"]


def add_site_classes(
    y: np.ndarray, sites: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A response whose site terms are more scattered in some classes of site than in
    others, the class of each record's site, and indicator covariates for the classes.

    Every site is put in one of SITE_CLASSES. To ``y`` are added an offset per class
    and a site term whose standard deviation depends on the class. The covariates are
    an indicator column for each class but the first, which is the reference.
    """
    # class_fits.csv was made from these draws: keep the seed and their order.
    rng = np.random.default_rng(11)
    site_levels, site_codes = np.unique(sites, return_inverse=True)
    # the same number of sites in each class, give or take one, in a shuffled order
    class_of_site = rng.permutation(np.arange(site_levels.size) % len(SITE_CLASSES))
    offsets = np.array([0.3, 0.0, -0.2])
    extra_sd = np.array([0.9, 0.0, 0.4])
    extra_term = extra_sd[class_of_site] * rng.normal(size=site_levels.size)
    record_class = class_of_site[site_codes]
    covariates = np.column_stack(
        [record_class == k for k in range(1, len(SITE_CLASSES))]
    ).astype(float)
    return (
        y + offsets[record_class] + extra_term[site_codes],
        np.array(SITE_CLASSES)[record_class],
        covariates,
    )


def assert_matches_dense(
    fit: LmmFit, dense: walkthrough.Estimates, levels: list[np.ndarray], y: np.ndarray
) -> None:
    """Every result of fit_lmm equals that of the dense numpy implementation, which is
    independent of the Rust engine. ``levels`` are those of the dense implementation."""
    assert fit.converged
    for level, expected in zip(levels, fit.levels):
        np.testing.assert_array_equal(level, expected)
    np.testing.assert_allclose(dense.deviance, fit.deviance, rtol=0, atol=1e-8)
    np.testing.assert_allclose(dense.theta, fit.theta, atol=1e-6)
    np.testing.assert_allclose(dense.sigma, fit.sigma, rtol=0, atol=1e-7)
    np.testing.assert_allclose(dense.sd, fit.sd, atol=1e-6)
    np.testing.assert_allclose(dense.beta, fit.beta, atol=1e-7)
    np.testing.assert_allclose(dense.beta_se, fit.beta_se, atol=1e-7)
    np.testing.assert_allclose(dense.ranef, np.concatenate(fit.ranef), atol=1e-6)
    np.testing.assert_allclose(dense.condsd, np.concatenate(fit.condsd), atol=1e-6)
    np.testing.assert_allclose(dense.fitted, fit.fitted, atol=1e-6)
    np.testing.assert_allclose(dense.resid, fit.resid, atol=1e-6)
    np.testing.assert_allclose(fit.resid, y - fit.fitted)


def class_inputs(
    df: pd.DataFrame, im: str
) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
    """The response, grouping factors and covariates of a fit in CLASS_FITS: an IM's
    residuals with the class effects of :func:`add_site_classes`, the events and the
    sites split into a factor per class, and the indicators of the classes."""
    events, sites = groups_of(df, True)
    y, site_class, covariates = add_site_classes(df[im].to_numpy(), sites)
    return y, [events, *split_by_class(sites, site_class)], covariates


@dataclass(frozen=True)
class Lme4Fit:
    """lme4's fit for an entry of LME4_FITS, COVARIATE_FITS or CLASS_FITS, laid out like
    :class:`mera.lmm.LmmFit`."""

    levels: list[np.ndarray]
    theta: np.ndarray
    sigma: float
    beta: np.ndarray
    beta_se: np.ndarray
    ranef: list[np.ndarray]
    condsd: list[np.ndarray]
    deviance: float

    def fitted(
        self, groups: list[np.ndarray], covariates: np.ndarray | None = None
    ) -> np.ndarray:
        """Fitted values of observations with these grouping-factor labels (and these
        covariates, for a fit that has them after its intercept). An observation that
        is masked in a factor gets no term from it."""
        fitted = np.full(len(groups[0]), self.beta[0] if self.beta.size else 0.0)
        if covariates is not None:
            fitted = fitted + covariates @ self.beta[1:]
        for levels, ranef, group in zip(self.levels, self.ranef, groups):
            in_factor = ~np.ma.getmaskarray(group)
            labels = np.ma.getdata(group)[in_factor].astype(str)
            fitted[in_factor] += ranef[np.searchsorted(levels, labels)]
        return fitted


def load_lme4_reference(prefix: str, key_columns: list[str]) -> dict[tuple, Lme4Fit]:
    """lme4's fits stored in ``<prefix>fits.csv`` and ``<prefix>random_effects.csv`` of
    LME4_REFERENCE, keyed by their values in ``key_columns``.

    The files of every group of fits have one layout. In the fits file, the columns
    named ``theta_...`` hold theta (one per grouping factor), those named ``beta_...``
    the fixed effects and those named ``se_...`` their standard errors, each in the
    order of the model; an entry is empty if the fit does not have it. In the random
    effects file, a fit's rows are grouped by factor, in the order of the factors.
    """
    fits = pd.read_csv(LME4_REFERENCE / f"{prefix}fits.csv")
    effects = pd.read_csv(LME4_REFERENCE / f"{prefix}random_effects.csv")
    effects = effects.groupby(key_columns)

    def values(row: dict, start: str) -> np.ndarray:
        named = [row[c] for c in fits.columns if c.startswith(start)]
        return np.array([value for value in named if not pd.isna(value)])

    out = {}
    for row in fits.to_dict("records"):
        key = tuple(row[c] for c in key_columns)
        factors = [g for _, g in effects.get_group(key).groupby("factor", sort=False)]
        out[key] = Lme4Fit(
            levels=[g["level"].to_numpy(dtype=str) for g in factors],
            theta=values(row, "theta_"),
            sigma=float(row["sigma"]),
            beta=values(row, "beta_"),
            beta_se=values(row, "se_"),
            ranef=[g["ranef"].to_numpy() for g in factors],
            condsd=[g["condsd"].to_numpy() for g in factors],
            deviance=float(row["deviance"]),
        )
    return out


def assert_matches_lme4(
    fit: LmmFit,
    expected: Lme4Fit,
    groups: list[np.ndarray],
    covariates: np.ndarray | None = None,
) -> None:
    """The fit is at lme4's optimum: never worse than lme4's criterion, and equal to it
    and to lme4's estimates to the precision of the search."""
    assert fit.deviance <= expected.deviance + 1e-8
    np.testing.assert_allclose(fit.deviance, expected.deviance, rtol=0, atol=1e-6)
    np.testing.assert_allclose(fit.theta, expected.theta, atol=1e-5)
    np.testing.assert_allclose(fit.sigma, expected.sigma, rtol=0, atol=1e-7)
    np.testing.assert_allclose(fit.beta, expected.beta, atol=1e-7)
    np.testing.assert_allclose(fit.beta_se, expected.beta_se, atol=1e-7)
    for got, want, levels, want_levels in zip(
        fit.ranef, expected.ranef, fit.levels, expected.levels
    ):
        np.testing.assert_array_equal(levels, want_levels)
        np.testing.assert_allclose(got, want, atol=1e-5)
    for got, want in zip(fit.condsd, expected.condsd):
        np.testing.assert_allclose(got, want, atol=1e-6)
    np.testing.assert_allclose(
        fit.fitted, expected.fitted(groups, covariates), atol=1e-5
    )


def reference_mask(res_df: pd.DataFrame) -> pd.DataFrame:
    """The mask of the masked references: mask.csv, less events and sites with fewer
    than three records."""
    return mera.mask_too_few_records(
        res_df,
        mask=pd.read_csv(MASK_CSV, index_col=0),
        event_cname="event_id",
        site_cname="stat_id",
        min_num_records_per_event=3,
        min_num_records_per_site=3,
    )


def run_example(
    res_df: pd.DataFrame,
    ims: list[str],
    mask: pd.DataFrame | None = None,
    n_procs: int = 1,
    **kwargs: bool,
) -> mera.MeraResults:
    """run_mera on (a subset of) the example residuals, quietly, as for the references."""
    return mera.run_mera(
        res_df,
        list(ims),
        "event_id",
        "stat_id",
        mask=mask,
        verbose=False,
        raise_warnings=False,
        n_procs=n_procs,
        **kwargs,
    )
