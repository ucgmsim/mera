from typing import Optional
import pandas as pd


def mask_too_few_records(
    residual_df: pd.DataFrame,
    column_name1: Optional[str] = "event_id",
    column_name2: Optional[str] = "stat_id",
    min_num_records_per_column1: Optional[int] = 3,
    min_num_records_per_column2: Optional[int] = 3,
    verbose: Optional[bool] = False,
) -> pd.DataFrame:
    """
    Mask records if there are too few records per event or per station to be useful.

    Parameters
    ----------
    residual_df : pd.DataFrame
        The residual DataFrame to mask.
    column_name1 : str, default = 'event_id'
        The first column to consider. The default value is 'event_id', which contains the event ids.
    column_name2 : str, default = 'stat_id'
        The second column to consider. The default value is 'stat_id', which contains the station ids.
    min_num_records_per_column1 : int, default = 3
        The minimum number of records required in `column_name1` to keep the records.
    min_num_records_per_column2 : int, default = 3
        The minimum number of records required in `column_name2` to keep the records.
    verbose: bool, default = False
        If True, print the station and event ids that were masked and the number of records they were associated with.

    Returns
    -------
    final_mask2D : pd.DataFrame
        The final version of the mask (after iterating) in the form of a 2D pd.DataFrame with the same shape as
        residual_df. True values indicate records to keep.
    """

    final_mask2D = residual_df.notnull()

    # We only need to consider one column as all columns in final_mask2D will be the same (we are masking rows).
    final_mask1D = final_mask2D.iloc[:, 0]

    # As we are masking records based on two related columns, we need to iteratively construct the mask.
    # For example, if on the first pass, we mask out all records associated with a particular event_id,
    # there may then be too few remaining records associated with a stat_id, so they need to be masked out too.
    # Therefore, we iterate until the mask is not changed by an additional iteration.

    iteration_counter = 0
    original_residual_df = residual_df.copy()
    residual_df_prev = residual_df.copy()

    masked_records_list = []

    while True:
        iteration_counter += 1

        # Use transform on the groupby object to create a mask of the same length as the original DataFrame.
        # Then use iloc to get the first column as all columns are the same, so we only need one.

        mask_column_name1 = (
            residual_df.groupby(column_name1).transform("count").iloc[:, 0]
            >= min_num_records_per_column1
        )
        mask_column_name2 = (
            residual_df.groupby(column_name2).transform("count").iloc[:, 0]
            >= min_num_records_per_column2
        )

        # Combine the two masks
        mask = mask_column_name1 & mask_column_name2

        # Update final_mask1D. As final_mask1D and mask may have different lengths after several iterations,
        # we need to explicity specify the indices of final_mask1D to modify.
        # These indices are retained by residual_df after masking operations.
        final_mask1D[residual_df.index] = mask

        # Keep track of the records that were masked
        masked_records_list.append(residual_df[~mask])

        # Update residual_df by applying the new mask
        residual_df = residual_df[mask]

        # If residual_df is the same before and after the masking operation, break the loop
        if residual_df.equals(residual_df_prev):
            break

        # Update residual_df_prev for the next iteration
        residual_df_prev = residual_df.copy()

    # Copy final_mask1D to all columns in final_mask2D
    for col in final_mask2D.columns:
        final_mask2D[col] = final_mask1D

    masked_records_df = pd.concat(masked_records_list)

    if verbose:
        # Find the records that were fully masked by checking if the number of records in original_residual_df and
        # masked_records_df are the same. If they are the same, they are selected with an inner merge operation.
        fully_masked_event_id_with_count = pd.merge(
            original_residual_df[column_name1].value_counts().reset_index(),
            masked_records_df[column_name1].value_counts().reset_index(),
            how="inner",
            on=[column_name1, "count"],
        )

        fully_masked_stat_id_with_count = pd.merge(
            original_residual_df[column_name2].value_counts().reset_index(),
            masked_records_df[column_name2].value_counts().reset_index(),
            how="inner",
            on=[column_name2, "count"],
        )

        for row in fully_masked_event_id_with_count.itertuples():
            print(f"Fully masked {row.event_id} (had {row.count} station records).")

        for row in fully_masked_stat_id_with_count.itertuples():
            print(f"Fully masked {row.stat_id} (had recorded {row.count} events).")

    print(
        f"Masked {original_residual_df.shape[0] - residual_df.shape[0]} records "
        f"({(original_residual_df.shape[0] - residual_df.shape[0])*(100/original_residual_df.shape[0]):.2f}% "
        f"of the total). Required {iteration_counter} iterations."
    )
    return final_mask2D
