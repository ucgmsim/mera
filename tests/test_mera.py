import pytest
import numpy as np
import pandas as pd
from pathlib import Path

import mera


@pytest.fixture(scope="module")
def res_df_and_ims():
    # Load the data
    data_dir = Path(__file__).parent.parent / "mera/example/resources"

    obs_df = pd.read_csv(data_dir / "im_obs.csv", index_col=0)
    sim_df = pd.read_csv(data_dir / "im_sim.csv", index_col=0)

    assert (
        np.all(obs_df.index == sim_df.index)
        and np.all(obs_df.event_id == sim_df.event_id)
        and np.all(obs_df.stat_id == sim_df.stat_id)
    )

    ims = [cur_im for cur_im in np.intersect1d(obs_df.columns, sim_df.columns) if cur_im.startswith("pSA")][::3]

    # Compute the residual
    res_df = np.log(obs_df[ims] / sim_df[ims])

    # Add event id and station id columns
    res_df["event_id"] = obs_df.event_id
    res_df["stat_id"] = obs_df.stat_id

    # Ensure that the event and site ids are different (i.e. no common values)
    res_df["event_id"] = np.char.add("event_", res_df["event_id"].values.astype(str))
    res_df["stat_id"] = np.char.add("stat_", res_df["stat_id"].values.astype(str))

    return res_df, ims

@pytest.fixture(scope="module")
def expected():
    residuals_dir = Path(__file__).parent / "residuals"

    return mera.MeraResults.load(residuals_dir)

def test_mera(res_df_and_ims: tuple[pd.DataFrame, list[str]], expected: mera.MeraResults):
    res_df, ims = res_df_and_ims

    result = mera.run_mera(
        res_df,
        list(ims),
        "event_id",
        "stat_id",
        compute_site_term=True,
        verbose=True,
        raise_warnings=True,
        min_num_records_per_event=3,
        min_num_records_per_site=3,
    )

    atol=1e-6
    pd.testing.assert_frame_equal(result.event_res_df[ims], expected.event_res_df[ims], atol=atol)
    pd.testing.assert_frame_equal(result.event_cond_std_df[ims], expected.event_cond_std_df[ims], atol=atol)
    pd.testing.assert_frame_equal(result.rem_res_df[ims], expected.rem_res_df[ims], atol=atol)
    pd.testing.assert_frame_equal(result.bias_std_df.loc[ims], expected.bias_std_df.loc[ims], atol=atol)
    pd.testing.assert_frame_equal(result.fit_df[ims], expected.fit_df[ims], atol=atol)
    pd.testing.assert_frame_equal(result.site_res_df[ims], expected.site_res_df[ims], atol=atol)
    pd.testing.assert_frame_equal(result.site_cond_std_df[ims], expected.site_cond_std_df[ims], atol=atol)

