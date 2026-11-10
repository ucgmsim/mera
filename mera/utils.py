"""Helpers for preparing residual data for the mixed-effects regression."""

import pandas as pd


def mask_too_few_records(
    residual_df: pd.DataFrame,
    event_cname: str = "event_id",
    site_cname: str = "stat_id",
    mask: pd.DataFrame | None = None,
    min_num_records_per_event: int = 3,
    min_num_records_per_site: int = 3,
) -> pd.DataFrame:
    """
    Creates a boolean mask (or modifies one passed in) that removes
    records without sufficient records per station/event.
    Note: This is done independently per IM.

    The mask has a row per record and a column per IM. True means that the record is
    used for that IM, and False that it is left out. Pass it to run_mera as ``mask``.

    An event with a single record says little about its event term, because the term
    cannot be told apart from that record's remaining residual; the same holds for
    sites. This function leaves out the records of events and sites that have fewer
    records than the minimum. Leaving records out can take another event or site below
    its minimum, so the counting is repeated until nothing changes.

    It prints how many records it masked.

    Parameters
    ----------
    residual_df : DataFrame
        Residual DataFrame, has to contain all
        specified IMs (as columns) along with
         columns for event and site id.
    event_cname : str, default = "event_id"
        Name of the column that contains the event ids.
    site_cname : str, default = "stat_id"
        Name of the column that contains the site ids.
    mask : DataFrame, optional
        Mask DataFrame including the same IM columns as residual_df but not including the site and event id columns
        (the shape of the mask DataFrame should be the same as residual_df[ims]).
        Records that it marks False stay left out. It should be False wherever
        residual_df is NaN, because run_mera refuses NaN values among the records
        that it uses. If no mask is given, every value that is not NaN starts as
        True.
    min_num_records_per_event : int, default = 3
        The minimum number of records per event required to keep the records.
    min_num_records_per_site : int, default = 3
        The minimum number of records per site required to keep the records.

    Returns
    -------
    mask: pd.DataFrame
        Mask DataFrame with the rows of the given residual DataFrame and a column
        per IM (without the event and site id columns).

    Examples
    --------
    >>> mask = mera.mask_too_few_records(residual_df, "event_id", "stat_id")
    >>> results = mera.run_mera(residual_df, ims, "event_id", "stat_id", mask=mask)
    """
    # making a copy to ensure that the original DataFrame is not modified
    residual_df = residual_df.copy()
    mask = mask.copy() if mask is not None else None

    ims = residual_df.columns.drop([event_cname, site_cname])

    if mask is None:
        # Start by using every value that is present
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
        assert all(mask[event_cname] == True), (
            "event_cname column in mask must be all True so that residual_df[mask] includes those columns for groupby"
        )
        assert all(mask[site_cname] == True), (
            "site_cname column in mask must be all True so that residual_df[mask] includes those columns for groupby"
        )

        iteration_counter += 1

        # Use transform on the groupby object to create a mask of the same shape as the original DataFrame.
        # residual_df[mask] is residual_df with every masked-out value replaced by NaN.
        # "count" counts the values that are not NaN, so for each record and IM, the
        # transform gives the number of records still in use for that record's event
        # (first half) and for its site (second half). drop_mask is True where either
        # count is below its minimum.
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
            f"({100 * num_false_in_current_mask / mask[ims].size:.2f}%). "
            f"Masking required {iteration_counter} iterations."
        )

    if num_false_in_initial_mask > 0:
        print(
            f"Masked an additional {num_false_in_current_mask - num_false_in_initial_mask} records "
            f"({100 * (num_false_in_current_mask - num_false_in_initial_mask) / mask[ims].size:.2f}%). "
            f"Masking required {iteration_counter} iterations."
        )

    return mask[ims]
