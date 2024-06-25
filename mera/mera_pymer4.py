import natsort
import numpy as np
import pandas as pd
from typing import List, Optional, NamedTuple

from pymer4.models import Lmer


class MeraResults(NamedTuple):
    """
    The results of the mixed-effects regression analysis.

    Attributes
    ----------
    event_res_df: dataframe
        Contains the random effect for each event (rows) and IM (columns)
    event_cond_std_df: dataframe
        Contains the conditional standard deviations
        for each event (rows) and IM (columns) (similar to a standard error)
    rem_res_df: dataframe
        Contains the leftover residuals for
        each record (rows) and IM (columns)
    bias_std_df: dataframe
        Contains bias, between-event sigma (tau),
        between-site sigma (phi_S2S) (only when compute_site_term is True),
        remaining residual sigma (phi_w) and total sigma (sigma) (columns)
        per IM (rows)
    fit_df: dataframe
        Contains the fitted values for each record (rows) and IM (columns)
    site_res_df: dataframe
        Contains the random effect for each site (rows) and IM (columns)
        Note: Only returned if compute_site_term is True
    site_cond_std_df: dataframe
        Contains the conditional standard deviations
        for each site (rows) and IM (columns) (similar to a standard error)
        Note: Only returned if compute_site_term is True
    """

    event_res_df: pd.DataFrame
    event_cond_std_df: pd.DataFrame
    rem_res_df: pd.DataFrame
    bias_std_df: pd.DataFrame
    fit_df: pd.DataFrame
    site_res_df: Optional[pd.DataFrame]
    site_cond_std_df: Optional[pd.DataFrame]


def run_mera(
    residual_df: pd.DataFrame,
    ims: List[str],
    event_cname: str,
    site_cname: str,
    assume_biased: bool = True,
    compute_site_term: bool = True,
    mask: pd.DataFrame = Optional[pd.DataFrame],
    verbose: bool = True,
    raise_warnings: bool = True,
    min_num_records_per_event: int = 3,
    min_num_records_per_site: int = 3,
) -> MeraResults:
    """
    Runs mixed effects regression analysis for the given
    residual dataframe using pymer4 package

    Parameters
    ----------
    residual_df: dataframe
        Residual dataframe, has to contain all
        specified IMs (as columns) along with
         columns for event and site id
    ims: list of strings
        IMs for which to run to mixed effects
        regression analysis.

        Note: Has to be a list, can't be a numpy array!
    event_cname: string
        Name of the column that contains the event ids
    site_cname: string
        Name of the column that contains the site ids
        Has no effect if compute_site_term is False
    assume_biased: bool, default = True
        If true then the model fits a bias term, removing
        any model bias (wrt. given dataset)

        Recommended to have this enabled, otherwise any model
        bias (wrt. given dataset) will affect random effect
        terms.
    compute_site_term: bool, default = True
        If true then the model fits a site term
    mask: dataframe or None, default = None
        Mask dataframe the size of the residual dataframe
        which selects which values are being used per IM
        for the lmer model. If None then all values are used.
    verbose: bool, default = True
        If true then prints the progress of the analysis
    raise_warnings: bool, default = True
        If true, generates a warning string for records with an insufficient number of records per site or per event.
    min_num_records_per_event : int, default = 3
        The minimum number of records per event required to keep the records.
    min_num_records_per_site : int, default = 3
        The minimum number of records per site required to keep the records.

    Returns
    -------
    MeraResults:
        Results of the mixed-effects regression analysis
    """

    # Result dataframes
    event_res_df = pd.DataFrame(
        index=np.unique(residual_df[event_cname].values.astype(str)),
        columns=ims,
        dtype=float,
    )

    if compute_site_term:
        site_cond_std_df = pd.DataFrame(
            index=natsort.natsorted(
                np.unique(residual_df[site_cname].values.astype(str))
            ),
            columns=ims,
            dtype=float,
        )

    event_cond_std_df = pd.DataFrame(
        index=natsort.natsorted(np.unique(residual_df[event_cname].values.astype(str))),
        columns=ims,
        dtype=float,
    )

    rem_res_df = pd.DataFrame(index=residual_df.index.values, columns=ims, dtype=float)

    fit_df = pd.DataFrame(index=residual_df.index.values, columns=ims, dtype=float)

    bias_std_df = pd.DataFrame(
        index=ims,
        columns=["bias", "bias_std_err", "tau", "phi_S2S", "phi_w", "sigma"],
        dtype=float,
    )

    random_effects_columns = [event_cname]
    if compute_site_term:
        site_res_df = pd.DataFrame(
            index=np.unique(residual_df[site_cname].values.astype(str)), columns=ims
        )
        random_effects_columns.append(site_cname)

    for cur_ix, cur_im in enumerate(ims):
        if verbose:
            print(f"Processing IM {cur_im}, {cur_ix + 1}/{len(ims)}")

        # Filter on the mask if given
        cur_columns = [cur_im] + random_effects_columns
        cur_residual_df = (
            residual_df[cur_columns]
            if mask is None
            else residual_df[cur_columns].loc[mask[cur_im]]
        )

        if raise_warnings:
            count_per_event = cur_residual_df.groupby(event_cname).count()[cur_im]
            count_per_site = (
                cur_residual_df.groupby(site_cname).count()[cur_im]
                if compute_site_term
                else pd.Series()
            )

            warning_counts = pd.concat(
                [
                    count_per_event[count_per_event < min_num_records_per_event],
                    count_per_site[count_per_site < min_num_records_per_site],
                ],
                axis=0,
            )

            for label, count in warning_counts.items():
                print(f"Warning: For IM {cur_im}, {label} has only {count} records.")

        # Check for nans
        if cur_residual_df[cur_im].isna().sum() > 0:
            raise ValueError(f"NaNs found in IM {cur_im}")

        # Create and fit the model
        if len(cur_residual_df) > 0:
            # With site-term
            if compute_site_term:
                cur_model = Lmer(
                    f"{cur_im} ~ {'1' if assume_biased else '0'} + (1|{event_cname}) + (1|{site_cname})",
                    data=cur_residual_df,
                )
                cur_model.fit(summary=False)

                # Get the site and event random effects
                # (Ensure we extract the right dataframe with correct indexes)
                event_index = (
                    0
                    if np.any(cur_model.ranef[0].index.isin(event_res_df.index))
                    else 1
                )
                site_index = (
                    1 if np.any(cur_model.ranef[1].index.isin(site_res_df.index)) else 0
                )
                event_re = cur_model.ranef[event_index].iloc[:, 0]
                site_re = cur_model.ranef[site_index].iloc[:, 0]

                # Get site term and standard deviation
                site_res_df.loc[site_re.index, cur_im] = site_re
                bias_std_df.loc[cur_im, "phi_S2S"] = cur_model.ranef_var.loc[
                    site_cname, "Std"
                ]

            # Without site-term
            else:
                cur_model = Lmer(
                    f"{cur_im} ~ {'1' if assume_biased else '0'} + (1|{event_cname})",
                    data=cur_residual_df,
                )
                cur_model.fit(summary=False)

                # Get the event term
                # (Ensure we extract the right dataframe with correct indexes)
                event_re = cur_model.ranef.iloc[:, 0]

            # Get event and remaining terms
            event_res_df.loc[event_re.index, cur_im] = event_re
            if mask is None:
                rem_res_df[cur_im] = cur_model.residuals
                fit_df[cur_im] = cur_model.fits
            else:
                rem_res_df.loc[mask[cur_im], cur_im] = cur_model.residuals
                fit_df.loc[mask[cur_im], cur_im] = cur_model.fits

            # Get bias
            if assume_biased:
                bias_std_df.loc[cur_im, "bias"] = cur_model.coefs.iloc[0, 0]
                bias_std_df.loc[cur_im, "bias_std_err"] = cur_model.coefs["SE"].iloc[0]

            # Get standard deviations
            bias_std_df.loc[cur_im, "tau"] = cur_model.ranef_var.loc[event_cname, "Std"]
            bias_std_df.loc[cur_im, "phi_w"] = cur_model.ranef_var.loc[
                "Residual", "Std"
            ]

            # set the index of ranef_df so it matches those of the output DataFrames
            cur_model.ranef_df.set_index("grp", inplace=True)
            cur_model.ranef_df.index.rename(None, inplace=True)

            event_cond_std_df[cur_im] = cur_model.ranef_df["condsd"]

            if compute_site_term:
                site_cond_std_df[cur_im] = cur_model.ranef_df["condsd"]

        else:
            print("WARNING: No data for IM, skipping...")

    # Compute total sigma and return
    if compute_site_term:
        bias_std_df["sigma"] = (
            bias_std_df["tau"] ** 2
            + bias_std_df["phi_S2S"] ** 2
            + bias_std_df["phi_w"] ** 2
        ) ** (1 / 2)
        return MeraResults(
            event_res_df,
            event_cond_std_df,
            rem_res_df,
            bias_std_df,
            fit_df,
            site_res_df,
            site_cond_std_df,
        )
    else:
        bias_std_df = bias_std_df.drop(columns=["phi_S2S"])
        bias_std_df["sigma"] = (
            bias_std_df["tau"] ** 2 + bias_std_df["phi_w"] ** 2
        ) ** (1 / 2)
        return MeraResults(
            event_res_df,
            event_cond_std_df,
            rem_res_df,
            bias_std_df,
            fit_df,
            None,
            None,
        )
