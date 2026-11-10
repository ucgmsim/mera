"""Example data and run settings shared by the tests and tools/lme4_reference."""

from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
import pandas as pd

import mera

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


@dataclass(frozen=True)
class Lme4Fit:
    """lme4's fit for an entry of LME4_FITS, laid out like :class:`mera.lmm.LmmFit`."""

    levels: list[np.ndarray]
    theta: np.ndarray
    sigma: float
    beta: np.ndarray
    beta_se: np.ndarray
    ranef: list[np.ndarray]
    condsd: list[np.ndarray]
    deviance: float

    def fitted(self, groups: list[np.ndarray]) -> np.ndarray:
        """Fitted values of observations with these grouping-factor labels."""
        fitted = np.full(len(groups[0]), self.beta[0] if self.beta.size else 0.0)
        for levels, ranef, labels in zip(self.levels, self.ranef, groups):
            fitted = fitted + ranef[np.searchsorted(levels, labels.astype(str))]
        return fitted


def load_lme4_fits() -> dict[tuple[str, bool, bool], Lme4Fit]:
    """lme4's fits for LME4_FITS, keyed like it, from LME4_REFERENCE."""
    effects = pd.read_csv(LME4_REFERENCE / "random_effects.csv").groupby(
        ["data", "intercept", "site_term"]
    )
    out = {}
    for row in pd.read_csv(LME4_REFERENCE / "fits.csv").to_dict("records"):
        key = (str(row["data"]), bool(row["intercept"]), bool(row["site_term"]))
        factors = [g for _, g in effects.get_group(key).groupby("factor", sort=False)]
        out[key] = Lme4Fit(
            levels=[g["level"].to_numpy(dtype=str) for g in factors],
            theta=np.array([row["theta_event"], row["theta_site"]][: len(factors)]),
            sigma=float(row["sigma"]),
            beta=np.array([row["beta"]] if key[1] else []),
            beta_se=np.array([row["beta_se"]] if key[1] else []),
            ranef=[g["ranef"].to_numpy() for g in factors],
            condsd=[g["condsd"].to_numpy() for g in factors],
            deviance=float(row["deviance"]),
        )
    return out


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
