from typing import Optional

import pandas as pd


def mask_too_few_records(
    residual_df: pd.DataFrame,
    event_cname: str = "event_id",
    site_cname: str = "stat_id",
    mask: Optional[pd.DataFrame] = None,
    min_num_records_per_event: int = 3,
    min_num_records_per_site: int = 3,
) -> pd.DataFrame:
    """
    Creates a boolean mask (or modifies one passed in) that removes records without sufficient records per station/event.
    Note: This is done independently per IM.

    Parameters
    ----------
    residual_df: DataFrame
        Residual DataFrame, has to contain all
        specified IMs (as columns) along with
         columns for event and site id.
    mask: DataFrame, optional
        Mask DataFrame including the same IM columns as residual_df but not including the site and event id columns
        (the shape of the mask DataFrame should be the same as residual_df[ims]).
    event_cname: string, default = "event_id"
        Name of the column that contains the event ids.
    site_cname: string, default = "stat_id"
        Name of the column that contains the site ids.
    min_num_records_per_event : int, default = 3
        The minimum number of records per event required to keep the records.
    min_num_records_per_site : int, default = 3
        The minimum number of records per site required to keep the records.

    Returns
    -------
    mask: pd.DataFrame
        Mask DataFrame with the same shape as the given residual DataFrame.
    """
    # making a copy to ensure that the original DataFrame is not modified
    residual_df = residual_df.copy()
    mask = mask.copy() if mask is not None else None

    ims = residual_df.columns.drop([event_cname, site_cname])

    if mask is None:
        # This mask must always have the event_cname and site_cname columns as all True
        # so residual_df[mask] includes those columns for groupby (below)
        mask = residual_df[ims].notnull()

    # To mask records based on the number of records per station or event, an iterative method is needed.
    # For example, if all records of a particular event are masked, there may then be an insufficient
    # number of records from an affected site, so they need to be masked out too.
    # Therefore, this code iterates until the mask is not changed by an additional iteration.

    iteration_counter = 0

    num_false_in_initial_mask = (~mask[ims]).values.sum()
    num_false_in_previous_mask = num_false_in_initial_mask

    # Add the event and site id columns to the mask so that the masked DataFrame will also have those columns for groupby
    mask[event_cname] = True
    mask[site_cname] = True

    while True:
        assert all(
            mask[event_cname] == True
        ), "event_cname column in mask must be all True so that residual_df[mask] includes those columns for groupby"
        assert all(
            mask[site_cname] == True
        ), "site_cname column in mask must be all True so that residual_df[mask] includes those columns for groupby"

        iteration_counter += 1

        # Use transform on the groupby object to create a mask of the same shape as the original DataFrame.
        drop_mask = (
            residual_df[mask]
            .groupby(event_cname, observed=True)
            .transform("count")[ims]
            < min_num_records_per_event
        ) | (
            residual_df[mask].groupby(site_cname, observed=True).transform("count")[ims]
            < min_num_records_per_site
        )

        mask[drop_mask] = False
        num_false_in_current_mask = (~mask[ims]).values.sum()

        # If the number of False values in the mask does not change, break the loop
        if num_false_in_current_mask == num_false_in_previous_mask:
            break

        num_false_in_previous_mask = num_false_in_current_mask

    if num_false_in_initial_mask == 0:
        print(
            f"Masked {num_false_in_current_mask} records "
            f"({100*num_false_in_current_mask/mask[ims].size:.2f}%). "
            f"Masking required {iteration_counter} iterations."
        )

    if num_false_in_initial_mask > 0:
        print(
            f"Masked an additional {num_false_in_current_mask-num_false_in_initial_mask} records "
            f"({100*(num_false_in_current_mask-num_false_in_initial_mask)/mask[ims].size:.2f}%). "
            f"Masking required {iteration_counter} iterations."
        )

    return mask[ims]
