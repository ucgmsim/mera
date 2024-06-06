from typing import Optional
import pandas as pd
import numpy as np


def mask_too_few_records(
    residual_df: pd.DataFrame,
    ims: list,
    event_id: Optional[str] = "event_id",
    stat_id: Optional[str] = "stat_id",
    min_num_records_per_event: Optional[int] = 3,
    min_num_records_per_station: Optional[int] = 3,
) -> pd.DataFrame:
    """
    Mask records if there are too few records per event or per station to be useful.

    Parameters
    ----------
    residual_df: dataframe
        Residual dataframe, has to contain all
        specified IMs (as columns) along with
         columns for event and site id
    ims: list of strings
        IMs for which to run to mixed effects
        regression analysis.
    event_id : str, default = 'event_id'
        The first column to consider. The default value is 'event_id', which contains the event ids.
    stat_id : str, default = 'stat_id'
        The second column to consider. The default value is 'stat_id', which contains the station ids.
    min_num_records_per_event : int, default = 3
        The minimum number of records required in `event_id` to keep the records.
    min_num_records_per_station : int, default = 3
        The minimum number of records required in `stat_id` to keep the records.

    Returns
    -------
    final_mask2D : pd.DataFrame
        The final version of the mask (after iterating) in the form of a 2D pd.DataFrame with the same shape as
        residual_df. True values indicate records to keep.
    """

    # As we are masking records based on two related properties (event_id and station_id), we need to iteratively
    # construct the mask. For example, if on the first pass, we mask out all records associated with a particular event_id,
    # there may then be too few remaining records associated with a stat_id, so they need to be masked out too.
    # Therefore, we iterate until the mask is not changed by an additional iteration.

    iteration_counter = 0
    residual_df_prev = residual_df.copy()

    while True:
        iteration_counter += 1

        # Use transform on the groupby object to create a mask of the same length as the original DataFrame.
        drop_mask = (
            residual_df.groupby(event_id).transform("count")[ims]
            < min_num_records_per_event
        ) | (
            residual_df.groupby(stat_id).transform("count")[ims]
            < min_num_records_per_station
        )

        # Update residual_df by setting masked records to np.nan
        residual_df[drop_mask] = np.nan

        # If residual_df is the same before and after the masking operation, break the loop
        if residual_df.equals(residual_df_prev):
            break

        # Update residual_df_prev for the next iteration
        residual_df_prev = residual_df.copy()

    return_mask = residual_df.notnull()

    num_masked_records = return_mask[ims].size - np.sum(np.sum(return_mask[ims]))

    print(
        f"Masked {100*num_masked_records/residual_df.size:.2f}% of the records "
        f"(required {iteration_counter} iterations)."
    )

    return return_mask
