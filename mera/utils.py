import pandas as pd
from typing import Optional


def generate_insufficient_records_warning_str(
    residual_df_per_im: pd.DataFrame, group_col: str, min_num_group_warn: int
) -> list[str]:
    """
    Generates a warning string for records with an insufficient
    number of records per site or per event.

    Parameters
    ----------
    residual_df_per_im: pd.DataFrame
        The residual DataFrame for a single IM.
    group_col: str
        The column name that contains the record ids to check
        (either event or site ids).
    min_num_group_warn: int
           The minimum number of records per site or event required
           to not trigger a warning.
    Returns
    -------
    warn_str_lines: list of strings
        A list of warning strings for records with an insufficient
        number of records per site or per event.

    """
    count = residual_df_per_im.groupby(group_col).count().iloc[:, 0]
    mask = count < min_num_group_warn

    warn_str_lines = [
        f"{count.loc[mask].index[x]} has only {count.loc[mask].iloc[x]} records "
        f"(recommended minimum is {min_num_group_warn})"
        for x in range(len(count.loc[mask]))
    ]

    return warn_str_lines


def mask_too_few_records(
    residual_dataframe: pd.DataFrame,
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
    residual_dataframe: DataFrame
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
    residual_df = residual_dataframe.copy()

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

    # Add the event and site id columns to the mask so that the masked DataFrame will also have those columns for groupby
    mask[event_cname] = True
    mask[site_cname] = True
    print()
    num_false_in_previous_mask = (~mask[ims]).values.sum()

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
            residual_df[mask].groupby(event_cname).transform("count")[ims]
            < min_num_records_per_event
        ) | (
            residual_df[mask].groupby(site_cname).transform("count")[ims]
            < min_num_records_per_site
        )

        mask[drop_mask] = False
        num_false_in_current_mask = (~mask[ims]).values.sum()

        # If the number of False values in the mask does not change, break the loop
        if num_false_in_current_mask == num_false_in_previous_mask:
            break

        num_false_in_previous_mask = num_false_in_current_mask

    print(
        f"Masked {100*num_false_in_current_mask/mask[ims].size:.2f}% of the records "
        f"(required {iteration_counter} iterations)."
    )
    return mask[ims]
