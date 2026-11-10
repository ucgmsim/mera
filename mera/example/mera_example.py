"""Example: mixed-effects regression of observed vs simulated IMs for the bundled data.

Run it with::

    python -m mera.example.mera_example

The steps are:

1. load the observed and the simulated intensity measures (IMs), one row per record;
2. compute the residuals, ln(observed / simulated);
3. optionally, make a mask that leaves out the records of events and sites that have too
   few records;
4. run the mixed-effects regression for every IM;
5. save the results as CSV files in the ``residuals`` directory next to this script.

``README.md`` at the top of the repository describes the output tables.
"""

from pathlib import Path

import numpy as np
import pandas as pd

import mera

if __name__ == "__main__":
    # Load the data
    data_dir = Path(__file__).parent / "resources"
    output_dir = Path(__file__).parent / "residuals"

    output_dir.mkdir(exist_ok=True)

    obs_ffp = data_dir / "im_obs.csv"
    sim_ffp = data_dir / "im_sim.csv"

    obs_df = pd.read_csv(obs_ffp, index_col=0)
    sim_df = pd.read_csv(sim_ffp, index_col=0)

    # Sanity checking
    # Ensure the index matches
    assert (
        np.all(obs_df.index == sim_df.index)
        and np.all(obs_df.event_id == sim_df.event_id)
        and np.all(obs_df.stat_id == sim_df.stat_id)
    )

    # List of IMs of interest
    # Just using all shared pSA periods in the data files
    ims = [
        cur_im
        for cur_im in np.intersect1d(obs_df.columns, sim_df.columns)
        if cur_im.startswith("pSA")
        or cur_im in ["PGA", "CAV", "Ds575", "Ds595", "AI", "PGV"]
    ]

    # Compute the residual
    # ln(observed / simulated): positive where the simulation under-predicts
    res_df = np.log(obs_df[ims] / sim_df[ims])

    # Add event id and station id columns
    res_df["event_id"] = obs_df.event_id
    res_df["stat_id"] = obs_df.stat_id

    # Prefix the ids so that event and site ids are easy to tell apart in the outputs
    # (mera itself doesn't need them to differ)
    res_df["event_id"] = np.char.add("event_", res_df["event_id"].values.astype(str))
    res_df["stat_id"] = np.char.add("stat_", res_df["stat_id"].values.astype(str))

    # Optional: Define an initial mask.
    # A mask has the rows of res_df and a column per IM. True means "use this record for
    # this IM" and False "leave it out".
    # A random mask is generated for this example.
    # The fraction of True and False can be changed with the p = [] parameter.
    # (With p=[1.0, 0.0] every value is True, so nothing is left out at this stage.)
    mask = pd.DataFrame(
        np.random.choice([True, False], p=[1.0, 0.0], size=res_df[ims].shape),
        columns=res_df[ims].columns,
        index=res_df[ims].index,
    )

    # Optional: Mask out records without sufficient records per station/event.
    # An event or site with one or two records says little about its own term.
    mask = mera.mask_too_few_records(
        res_df,
        mask=mask,
        event_cname="event_id",
        site_cname="stat_id",
        min_num_records_per_event=3,
        min_num_records_per_site=3,
    )

    # Optional: compute site term (recommended)
    compute_site_term = True

    # # Run MER
    # This fits, for each IM: residual = bias + event term + site term + remaining residual
    results = mera.run_mera(
        res_df,
        list(ims),
        "event_id",
        "stat_id",
        compute_site_term=compute_site_term,
        mask=mask,
        verbose=True,
        raise_warnings=True,
        min_num_records_per_event=3,
        min_num_records_per_site=3,
    )

    # The bias and the standard deviations of each IM
    print(results.bias_std_df)

    # Save the results
    results.save(output_dir)
    print(f"Saved the results to {output_dir}")
