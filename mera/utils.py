from typing import Optional
import pandas as pd


def mask_too_few_records(
    residual_df: pd.DataFrame,
    min_num_records_per_event: Optional[int] = 3,
    min_num_records_per_station: Optional[int] = 3,
    verbose: Optional[bool] = False,
) -> pd.DataFrame:
    """
    Mask records if there are too few records per event or per station to be useful.

    Parameters
    ----------
    residual_df : pd.DataFrame
        The residual DataFrame to mask.
    min_num_records_per_event : int, optional
        The minimum number of records per event required to keep the records.
    min_num_records_per_station : int, optional
        The minimum number of records per station required to keep the records.
    verbose: bool, optional
        If True, print the station and event ids that were masked and the
        number of records they were associated with.

    Returns
    -------
    final_mask2D : pd.DataFrame
        A 2D mask DataFrame with the same shape as residual_df. True values indicate records to keep.
    """

    final_mask2D = residual_df.notnull()

    # Only need to consider one column as all columns in final_mask2D will be the same
    final_mask1D = final_mask2D.iloc[:, 0]

    # Iterate until the DataFrame before and after the masking operation are the same
    iteration_counter = 0
    # Create a copy of the original DataFrame to compare with the masked DataFrame.
    residual_df_prev = residual_df.copy()
    original_residual_df = residual_df.copy()

    masked_records_list = []

    while True:
        iteration_counter += 1

        # Use transform on the groupby object to create a mask of the same length as the original DataFrame.
        # Then use iloc to get the first column as all columns are the same so we only need one.
        mask_stat_id = (
            residual_df.groupby("stat_id").transform("count").iloc[:, 0]
            >= min_num_records_per_station
        )
        mask_event_id = (
            residual_df.groupby("event_id").transform("count").iloc[:, 0]
            >= min_num_records_per_event
        )

        # Combine the station and event masks
        mask = mask_stat_id & mask_event_id

        # Update the final mask. final_mask1D and mask may have different lengths if several iterations are required
        # so we use the index of the original DataFrame to update the final mask.
        final_mask1D[residual_df.index] = mask

        masked_records_list.append(residual_df[~mask])

        residual_df = residual_df[mask]

        # If the DataFrame before and after the masking operation are the same, break the loop
        if residual_df.equals(residual_df_prev):
            break

        # Update residual_df_prev for the next iteration
        residual_df_prev = residual_df.copy()

    # Copy final_mask1D to all columns in final_mask2D
    for col in final_mask2D.columns:
        final_mask2D[col] = final_mask1D

    masked_records_df = pd.concat(masked_records_list)

    if verbose:

        fully_masked_event_id_with_count = pd.merge(
            original_residual_df["event_id"].value_counts().reset_index(),
            masked_records_df["event_id"].value_counts().reset_index(),
            how="inner",
            on=["event_id", "count"],
        )

        fully_masked_stat_id_with_count = pd.merge(
            original_residual_df["stat_id"].value_counts().reset_index(),
            masked_records_df["stat_id"].value_counts().reset_index(),
            how="inner",
            on=["stat_id", "count"],
        )

        for row in fully_masked_event_id_with_count.itertuples():
            print(f"Fully masked {row.event_id} (had {row.count} station records).")

        for row in fully_masked_stat_id_with_count.itertuples():
            print(f"Fully masked {row.stat_id} (had recorded {row.count} events).")

    print(
        f"Masked {original_residual_df.shape[0] - residual_df.shape[0]} records "
        f"({(original_residual_df.shape[0] - residual_df.shape[0])*(100/original_residual_df.shape[0]):.2f}%). "
        f"Required {iteration_counter} iterations."
    )
    return final_mask2D
