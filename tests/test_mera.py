import pytest
import numpy as np
import pandas as pd
from pathlib import Path

import mera

MASK = pd.read_csv(Path(__file__).parent / "resources" / "mask.csv", index_col=0)


@pytest.fixture(scope="module")
def obs_sim_df():
    # Load the data
    data_dir = Path(__file__).parent.parent / "mera/example/resources"

    obs_df = pd.read_csv(data_dir / "im_obs.csv", index_col=0)
    sim_df = pd.read_csv(data_dir / "im_sim.csv", index_col=0)

    return obs_df, sim_df


@pytest.fixture(scope="module")
def ims(obs_sim_df: tuple[pd.DataFrame, pd.DataFrame]):
    obs_df, sim_df = obs_sim_df

    # Use every third period to speed up the test
    ims = [
        cur_im
        for cur_im in np.intersect1d(obs_df.columns, sim_df.columns)
        if cur_im.startswith("pSA")
    ][::3]

    return ims


@pytest.fixture(scope="module")
def res_df(obs_sim_df: tuple[pd.DataFrame, pd.DataFrame], ims: list[str]):
    obs_df, sim_df = obs_sim_df

    assert (
        np.all(obs_df.index == sim_df.index)
        and np.all(obs_df.event_id == sim_df.event_id)
        and np.all(obs_df.stat_id == sim_df.stat_id)
    )

    # ims = [cur_im for cur_im in np.intersect1d(obs_df.columns, sim_df.columns) if cur_im.startswith("pSA")][::3]

    # Compute the residual
    res_df = np.log(obs_df[ims] / sim_df[ims])

    # Add event id and station id columns
    res_df["event_id"] = obs_df.event_id
    res_df["stat_id"] = obs_df.stat_id

    # Ensure that the event and site ids are different (i.e. no common values)
    res_df["event_id"] = np.char.add("event_", res_df["event_id"].values.astype(str))
    res_df["stat_id"] = np.char.add("stat_", res_df["stat_id"].values.astype(str))

    return res_df


@pytest.fixture(scope="module", params=[True, False])
def mask(request: pytest.FixtureRequest, res_df: pd.DataFrame):
    if request.param:
        return mera.mask_too_few_records(
            res_df,
            mask=MASK,
            event_cname="event_id",
            site_cname="stat_id",
            min_num_records_per_event=3,
            min_num_records_per_site=3,
        )
    else:
        return None


@pytest.fixture(scope="module")
def expected(mask: pd.DataFrame | None):
    if mask is None:
        residuals_dir = Path(__file__).parent / "resources/no_mask"
    else:
        residuals_dir = Path(__file__).parent / "resources/mask"

    return mera.MeraResults.load(residuals_dir)


@pytest.mark.parametrize("n_procs", [1, 2])
def test_mera(
    res_df: pd.DataFrame,
    ims: list[str],
    expected: mera.MeraResults,
    mask: pd.DataFrame | None,
    n_procs: int,
):
    result = mera.run_mera(
        res_df,
        list(ims),
        "event_id",
        "stat_id",
        mask=mask,
        compute_site_term=True,
        verbose=True,
        raise_warnings=True,
        min_num_records_per_event=3,
        min_num_records_per_site=3,
        n_procs=n_procs,
    )

    atol = 1e-4
    pd.testing.assert_frame_equal(
        result.event_res_df[ims], expected.event_res_df[ims], atol=atol
    )
    pd.testing.assert_frame_equal(
        result.event_cond_std_df[ims], expected.event_cond_std_df[ims], atol=atol
    )
    pd.testing.assert_frame_equal(
        result.rem_res_df[ims], expected.rem_res_df[ims], atol=atol
    )
    pd.testing.assert_frame_equal(
        result.bias_std_df.loc[ims], expected.bias_std_df.loc[ims], atol=atol
    )
    pd.testing.assert_frame_equal(result.fit_df[ims], expected.fit_df[ims], atol=atol)
    pd.testing.assert_frame_equal(
        result.site_res_df[ims], expected.site_res_df[ims], atol=atol
    )
    pd.testing.assert_frame_equal(
        result.site_cond_std_df[ims], expected.site_cond_std_df[ims], atol=atol
    )
