import numpy as np
import pandas as pd
from pathlib import Path

from mera import utils
from mera.mera_pymer4 import run_mera

# Load the data
# data_dir = Path(__file__).parent / "resources"
# output_dir = Path(__file__).parent / "residuals"

data_dir = Path("/home/arr65/src/Andrew_test_code/mera_R/data")
output_dir = Path("/home/arr65/src/Andrew_test_code/mera_R/data/mera_output")
output_dir.mkdir(exist_ok=True)

stations_ffp = data_dir / "stations.csv"
events_ffp = data_dir / "events.csv"
obs_ffp = data_dir / "im_obs.csv"
sim_ffp = data_dir / "im_sim.csv"

stations_df = pd.read_csv(stations_ffp, index_col=0)
events_df = pd.read_csv(events_ffp, index_col=0)

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
res_df = np.log(obs_df[ims] / sim_df[ims])

# Add event id and station id columns
res_df["event_id"] = obs_df.event_id
res_df["stat_id"] = obs_df.stat_id

# Ensure that the event and site ids are different (i.e. no common values)
res_df["event_id"] = np.char.add("event_", res_df["event_id"].values.astype(str))
res_df["stat_id"] = np.char.add("stat_", res_df["stat_id"].values.astype(str))

# Optional: Define an initial mask.
# A random mask is generated for this example.
# The fraction of True and False can be changed with the p = [] parameter.
mask = pd.DataFrame(
    np.random.choice([True, False], p=[1.0, 0.0], size=res_df[ims].shape),
    columns=res_df[ims].columns,
    index=res_df[ims].index,
)

# Optional: Mask out records without sufficient records per station/event.
mask = utils.mask_too_few_records(
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
(
    event_res_df,
    site_res_df,
    rem_res_df,
    bias_std_df,
    event_standard_err_df,
    stat_standard_err_df,
) = run_mera(
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

# Save the results
event_res_df.to_csv(output_dir / "event_residuals.csv", index_label="event_id")
rem_res_df.to_csv(output_dir / "remaining_residuals.csv", index_label="gm_id")
bias_std_df.to_csv(output_dir / "bias_std.csv", index_label="IM")
event_standard_err_df.to_csv(
    output_dir / "event_standard_error.csv", index_label="event_id"
)
if compute_site_term:
    site_res_df.to_csv(output_dir / "site_residuals.csv", index_label="stat_id")

    stat_standard_err_df.to_csv(
        output_dir / "station_standard_error.csv", index_label="stat_id"
    )
