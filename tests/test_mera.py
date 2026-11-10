import re
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from example_data import (
    LME4_REFERENCE,
    OUTPUTS,
    Lme4Fit,
    groups_of,
    reference_mask,
    run_example,
)

import mera
from mera.lmm import fit_lmm

# The references are lme4's values, run to convergence (resources/lme4_reference/
# README.md). They were within 2e-7 of mera's when they were made.
ATOL = 1e-6


@pytest.fixture(scope="module")
def ims(example_residuals: pd.DataFrame) -> list[str]:
    # Use every third period to speed up the test
    return [im for im in example_residuals.columns if im.startswith("pSA")][::3]


@pytest.fixture(scope="module")
def res_df(example_residuals: pd.DataFrame, ims: list[str]) -> pd.DataFrame:
    return example_residuals[[*ims, "event_id", "stat_id"]]


@pytest.fixture(scope="module", params=["mask", "no_mask"])
def kind(request: pytest.FixtureRequest) -> str:
    return request.param


@pytest.fixture(scope="module")
def mask(kind: str, res_df: pd.DataFrame) -> pd.DataFrame | None:
    return None if kind == "no_mask" else reference_mask(res_df)


@pytest.fixture(scope="module")
def expected(kind: str) -> mera.MeraResults:
    return mera.MeraResults.load(LME4_REFERENCE / kind)


def select(df: pd.DataFrame, name: str, ims: list[str]) -> pd.DataFrame:
    return df.loc[ims] if name == "bias_std_df" else df[ims]


def assert_matches_reference(
    result: mera.MeraResults, expected: mera.MeraResults, ims: list[str]
) -> None:
    for name in OUTPUTS:
        pd.testing.assert_frame_equal(
            select(getattr(result, name), name, ims),
            select(getattr(expected, name), name, ims),
            atol=ATOL,
        )


def test_mera(
    res_df: pd.DataFrame,
    ims: list[str],
    mask: pd.DataFrame | None,
    expected: mera.MeraResults,
) -> None:
    """run_mera's outputs match lme4's, without and with the mask."""
    assert_matches_reference(run_example(res_df, ims, mask), expected, ims)


def test_mera_in_parallel(
    res_df: pd.DataFrame,
    ims: list[str],
    mask: pd.DataFrame | None,
    expected: mera.MeraResults,
) -> None:
    # A few IMs, two per worker, check the dispatch and the order of the results;
    # test_mera checks the fits themselves.
    few = ims[:4]
    assert_matches_reference(run_example(res_df, few, mask, n_procs=2), expected, few)


@pytest.mark.parametrize(
    "assume_biased, compute_site_term", [(True, False), (False, True), (False, False)]
)
def test_other_configurations_match_lme4(
    example_residuals: pd.DataFrame,
    lme4_fits: dict[tuple[str, bool, bool], Lme4Fit],
    assume_biased: bool,
    compute_site_term: bool,
) -> None:
    df = example_residuals
    test_ims = ["PGA", "pSA_1.0"]
    result = run_example(
        df, test_ims, assume_biased=assume_biased, compute_site_term=compute_site_term
    )
    bias = result.bias_std_df
    assert ("phi_S2S" in bias.columns) == compute_site_term
    assert (result.site_res_df is not None) == compute_site_term
    groups = groups_of(df, compute_site_term)
    for im in test_ims:
        lme4 = lme4_fits[im, assume_biased, compute_site_term]
        assert bias.loc[im, "tau"] == pytest.approx(
            lme4.theta[0] * lme4.sigma, abs=1e-5
        )
        assert bias.loc[im, "phi_w"] == pytest.approx(lme4.sigma, abs=1e-6)
        if compute_site_term:
            assert bias.loc[im, "phi_S2S"] == pytest.approx(
                lme4.theta[1] * lme4.sigma, abs=1e-5
            )
        if assume_biased:
            assert bias.loc[im, "bias"] == pytest.approx(lme4.beta[0], abs=1e-6)
        else:
            assert np.isnan(bias.loc[im, "bias"])
            assert np.isnan(bias.loc[im, "bias_std_err"])
        np.testing.assert_allclose(
            result.event_res_df.loc[lme4.levels[0], im], lme4.ranef[0], atol=1e-5
        )
        assert result.fit_df is not None
        np.testing.assert_allclose(result.fit_df[im], lme4.fitted(groups), atol=1e-5)


def test_event_only_fit_needs_no_site_column(
    res_df: pd.DataFrame, ims: list[str]
) -> None:
    """Without a site term the site column is not read, so it may be missing."""
    im = ims[0]
    with_column = run_example(res_df, [im], compute_site_term=False)
    without_column = run_example(
        res_df.drop(columns="stat_id"), [im], compute_site_term=False
    )
    for name in OUTPUTS:
        expected = getattr(with_column, name)
        if expected is None:
            assert getattr(without_column, name) is None
        else:
            pd.testing.assert_frame_equal(getattr(without_column, name), expected)


def test_event_and_site_tables_are_in_natural_order() -> None:
    """The terms and their conditional standard deviations have the same rows in the
    same order, with "event_2" before "event_10"."""
    rng = np.random.default_rng(0)
    events = ["event_10", "event_2", "event_1"]
    sites = ["stat_2", "stat_10", "stat_1", "stat_3"]
    df = pd.DataFrame(
        [(event, site) for event in events for site in sites],
        columns=["event_id", "stat_id"],
    )
    df["PGA"] = rng.normal(size=len(df))
    result = run_example(df, ["PGA"])
    assert result.site_res_df is not None and result.site_cond_std_df is not None
    assert list(result.event_res_df.index) == ["event_1", "event_2", "event_10"]
    assert list(result.event_cond_std_df.index) == ["event_1", "event_2", "event_10"]
    assert list(result.site_res_df.index) == ["stat_1", "stat_2", "stat_3", "stat_10"]
    assert list(result.site_cond_std_df.index) == [
        "stat_1",
        "stat_2",
        "stat_3",
        "stat_10",
    ]


def test_nan_among_the_records_used_is_an_error(
    res_df: pd.DataFrame, ims: list[str]
) -> None:
    """A NaN residual stops the run, naming the IM, unless the mask leaves it out."""
    im = ims[0]
    with_nan = res_df[[im, "event_id", "stat_id"]].copy()
    with_nan.iloc[0, 0] = np.nan
    with pytest.raises(ValueError, match=re.escape(f"NaNs found in IM {im}")):
        run_example(with_nan, [im])
    mask = pd.DataFrame(True, index=with_nan.index, columns=[im])
    mask.iloc[0, 0] = False
    result = run_example(with_nan, [im], mask)
    assert result.rem_res_df is not None
    assert np.isnan(result.rem_res_df[im].iloc[0])
    assert result.rem_res_df[im].iloc[1:].notna().all()


def test_im_with_no_records_is_nan_and_the_others_are_fitted(
    res_df: pd.DataFrame, ims: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    """An IM that the mask leaves no records for is skipped with a warning: its results
    are NaN, and the other IMs are fitted as if it were not there."""
    empty, other = ims[0], ims[1]
    mask = pd.DataFrame(True, index=res_df.index, columns=[empty, other])
    mask[empty] = False
    result = run_example(res_df, [empty, other], mask)
    assert f"No data for IM {empty}" in capsys.readouterr().out
    alone = run_example(res_df, [other])
    for name in OUTPUTS:
        frame, expected = getattr(result, name), getattr(alone, name)
        if name == "bias_std_df":
            assert frame.loc[empty].isna().all()
            pd.testing.assert_series_equal(frame.loc[other], expected.loc[other])
        else:
            assert frame[empty].isna().all()
            pd.testing.assert_series_equal(frame[other], expected[other])


def test_overlapping_event_and_site_labels(
    res_df: pd.DataFrame, ims: list[str]
) -> None:
    """Event and site ids that share labels must still get separate terms."""
    im = ims[0]
    plain = res_df[[im]].copy()
    plain["event_id"] = res_df["event_id"].str.removeprefix("event_")
    plain["stat_id"] = res_df["stat_id"].str.removeprefix("stat_")
    assert set(plain["event_id"]) & set(plain["stat_id"])  # the labels do overlap
    prefixed = run_example(res_df, [im])
    overlapping = run_example(plain, [im])
    pd.testing.assert_frame_equal(overlapping.bias_std_df, prefixed.bias_std_df)
    np.testing.assert_allclose(
        overlapping.event_res_df[im].to_numpy(), prefixed.event_res_df[im].to_numpy()
    )
    assert overlapping.site_res_df is not None and prefixed.site_res_df is not None
    np.testing.assert_allclose(
        overlapping.site_res_df[im].to_numpy(), prefixed.site_res_df[im].to_numpy()
    )
    assert overlapping.fit_df is not None and prefixed.fit_df is not None
    np.testing.assert_allclose(overlapping.fit_df[im], prefixed.fit_df[im])


@pytest.mark.filterwarnings("error::pandas.errors.PerformanceWarning")
def test_many_ims_without_fragmentation_warnings() -> None:
    """With over 100 IMs, inserting columns one at a time into the results makes pandas
    warn that the frame is fragmented."""
    rng = np.random.default_rng(0)
    ims = [f"im_{i}" for i in range(101)]
    df = pd.DataFrame(rng.normal(size=(60, len(ims))), columns=ims)
    df["event_id"] = [f"event_{i % 10}" for i in range(len(df))]
    df["stat_id"] = [f"stat_{i % 7}" for i in range(len(df))]
    # Only the assembly of the results matters here, so one fit stands in for all 101
    # (which would take ~5 s).
    fit = fit_lmm(df[ims[0]].to_numpy(), groups_of(df, site_term=True))
    with patch("mera.analysis.fit_lmm", return_value=fit):
        result = run_example(df, ims)
    assert result.rem_res_df is not None
    assert list(result.rem_res_df.columns) == [*ims, "event_id", "stat_id"]


def test_non_convergence_warning_names_the_im(
    res_df: pd.DataFrame, ims: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("mera.lmm.MAX_EVALUATIONS", 3)
    message = f"did not converge for IM {ims[0]}"
    with pytest.warns(RuntimeWarning, match=re.escape(message)):
        run_example(res_df, [ims[0]])


def test_old_import_path_still_works() -> None:
    import importlib
    import sys

    sys.modules.pop("mera.mera_pymer4", None)
    with pytest.warns(DeprecationWarning):
        legacy = importlib.import_module("mera.mera_pymer4")
    assert legacy.run_mera is mera.run_mera
