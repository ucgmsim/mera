"""Mixed-effects regression analysis (MERA) of ground-motion residuals.

This is the module that most users need. It provides:

- :func:`run_mera`, which fits the model to every intensity measure (IM) in a table of
  residuals;
- :class:`MeraResults`, which holds the tables that :func:`run_mera` returns, and saves
  and loads them.

The model is fitted separately to each IM, and splits every residual into four parts::

    residual = bias + event term + site term + remaining residual

The bias is the same for every record, an event term is shared by the records of one
earthquake, and a site term by the records at one site. The model also estimates the
standard deviations of the event terms (``tau``), of the site terms (``phi_S2S``) and of
the remaining residuals (``phi_w``). ``README.md`` describes the parts and the outputs,
and ``docs/mixed_effects_maths.md`` the method.

How the work is divided:

- :func:`run_mera` loops over the IMs, optionally in parallel, and assembles the tables.
- :func:`_run_im_mera` prepares the records of one IM, fits them and labels the results.
- :func:`mera.lmm.fit_lmm` does the statistics of one fit.
- :mod:`mera._lmm`, which is written in Rust, does the heavy arithmetic.
"""

import multiprocessing as mp
from dataclasses import dataclass
from pathlib import Path

import natsort
import numpy as np
import pandas as pd

from .lmm import fit_lmm


@dataclass
class MeraResults:
    """
    The results of the mixed-effects regression analysis.

    Every table has one column per IM, except ``bias_std_df``, which has one row per IM.
    A value is NaN where it does not exist: for a record that the mask left out of an
    IM's fit, or for an event or site with no records in that fit.

    Attributes
    ----------
    event_res_df : dataframe
        Contains the random effect for each event (rows) and IM (columns):
        the event term, or between-event residual
    event_cond_std_df : dataframe
        Contains the conditional standard deviations
        for each event (rows) and IM (columns) (similar to a standard error):
        how uncertain each event term is. It is small for an event with many records
    rem_res_df : dataframe
        Contains the leftover residuals for
        each record (rows) and IM (columns): the residual less the bias, the event
        term and the site term. Its last columns are the event ids and, if a site
        term was fitted, the site ids of the records
    bias_std_df : dataframe
        Contains bias, its standard error (bias_std_err), between-event
        standard deviation (tau), between-site standard deviation (phi_S2S)
        (only when compute_site_term is True), standard deviation of the remaining
        residuals (phi_w) and total standard deviation (sigma) (columns)
        per IM (rows). The total is the square root of the sum of the squares
        of the other standard deviations. The bias and its standard error are NaN
        if no bias was fitted
    fit_df : dataframe
        Contains the fitted values for each record (rows) and IM (columns): the
        bias plus the event term plus the site term of the record
    site_res_df : dataframe
        Contains the random effect for each site (rows) and IM (columns):
        the site term, or site-to-site residual
        Note: Only returned if compute_site_term is True
    site_cond_std_df : dataframe
        Contains the conditional standard deviations
        for each site (rows) and IM (columns) (similar to a standard error)
        Note: Only returned if compute_site_term is True

    Notes
    -----
    The rows of the event and site tables are labelled with the ids as strings, even if
    the ids were numbers. (Tables read back by ``load`` from CSV files have numeric ids
    as numbers again, because pandas reads them that way.)

    Those rows are in natural order ("event_2" before "event_10"), the same in the
    table of terms and in the table of their conditional standard deviations. The
    versions of mera that used R sorted ``event_res_df`` and ``site_res_df`` as plain
    text instead ("event_10" before "event_2"). Matching rows by their labels rather
    than by their positions, for example with ``.loc``, works with results of either.
    """

    event_res_df: pd.DataFrame
    event_cond_std_df: pd.DataFrame
    rem_res_df: pd.DataFrame | None
    bias_std_df: pd.DataFrame
    fit_df: pd.DataFrame | None
    site_res_df: pd.DataFrame | None
    site_cond_std_df: pd.DataFrame | None

    def save(self, output_dir: Path, save_fit: bool = True, save_rem: bool = True):
        """
        Saves the MeraResults to the given directory

        Each table is written to a CSV file named after its attribute, such as
        ``event_res_df.csv``.

        Parameters
        ----------
        output_dir : Path
            Directory to save the MeraResults files, which must exist
        save_fit : bool, default = True
            Whether to save the fitted values (fit_df)
        save_rem : bool, default = True
            Whether to save the remaining residuals (rem_res_df)
        """
        self.event_res_df.to_csv(output_dir / "event_res_df.csv")
        self.event_cond_std_df.to_csv(output_dir / "event_cond_std_df.csv")
        self.bias_std_df.to_csv(output_dir / "bias_std_df.csv")
        if save_rem and self.rem_res_df is not None:
            self.rem_res_df.to_csv(output_dir / "rem_res_df.csv")
        if save_fit and self.fit_df is not None:
            self.fit_df.to_csv(output_dir / "fit_df.csv")

        if self.site_res_df is not None:
            self.site_res_df.to_csv(output_dir / "site_res_df.csv")
        if self.site_cond_std_df is not None:
            self.site_cond_std_df.to_csv(output_dir / "site_cond_std_df.csv")

    @classmethod
    def load(cls, data_dir: Path):
        """
        Loads the MeraResults from the given directory

        The tables that were not saved (the fitted values, the remaining residuals or
        the site tables) are None.

        Parameters
        ----------
        data_dir : Path
            Directory containing the MeraResults files

        Returns
        -------
        MeraResults
            Loaded MeraResults
        """
        return cls(
            pd.read_csv(data_dir / "event_res_df.csv", index_col=0),
            pd.read_csv(data_dir / "event_cond_std_df.csv", index_col=0),
            (
                pd.read_csv(data_dir / "rem_res_df.csv", index_col=0)
                if (data_dir / "rem_res_df.csv").exists()
                else None
            ),
            pd.read_csv(data_dir / "bias_std_df.csv", index_col=0),
            (
                pd.read_csv(data_dir / "fit_df.csv", index_col=0)
                if (data_dir / "fit_df.csv").exists()
                else None
            ),
            (
                pd.read_csv(data_dir / "site_res_df.csv", index_col=0)
                if (data_dir / "site_res_df.csv").exists()
                else None
            ),
            (
                pd.read_csv(data_dir / "site_cond_std_df.csv", index_col=0)
                if (data_dir / "site_cond_std_df.csv").exists()
                else None
            ),
        )

    def save_to_parquet(
        self, output_dir: Path, save_fit: bool = True, save_rem: bool = True
    ):
        """
        Saves the MeraResults to the given directory in parquet format

        Parquet files are smaller and faster to read than CSV files. Writing and
        reading them needs the ``pyarrow`` package (``pip install pyarrow``).

        Parameters
        ----------
        output_dir : Path
            Directory to save the MeraResults files, which must exist
        save_fit : bool, default = True
            Whether to save the fitted values (fit_df)
        save_rem : bool, default = True
            Whether to save the remaining residuals (rem_res_df)
        """
        self.event_res_df.to_parquet(output_dir / "event_res_df.parquet")
        self.event_cond_std_df.to_parquet(output_dir / "event_cond_std_df.parquet")
        self.bias_std_df.to_parquet(output_dir / "bias_std_df.parquet")
        if save_rem and self.rem_res_df is not None:
            self.rem_res_df.to_parquet(output_dir / "rem_res_df.parquet")
        if save_fit and self.fit_df is not None:
            self.fit_df.to_parquet(output_dir / "fit_df.parquet")

        if self.site_res_df is not None:
            self.site_res_df.to_parquet(output_dir / "site_res_df.parquet")
        if self.site_cond_std_df is not None:
            self.site_cond_std_df.to_parquet(output_dir / "site_cond_std_df.parquet")

    @classmethod
    def load_from_parquet(cls, data_dir: Path):
        """
        Loads the MeraResults from the given directory in parquet format

        Parameters
        ----------
        data_dir : Path
            Directory containing the MeraResults files

        Returns
        -------
        MeraResults
            Loaded MeraResults
        """
        return cls(
            pd.read_parquet(data_dir / "event_res_df.parquet"),
            pd.read_parquet(data_dir / "event_cond_std_df.parquet"),
            (
                pd.read_parquet(data_dir / "rem_res_df.parquet")
                if (data_dir / "rem_res_df.parquet").exists()
                else None
            ),
            pd.read_parquet(data_dir / "bias_std_df.parquet"),
            (
                pd.read_parquet(data_dir / "fit_df.parquet")
                if (data_dir / "fit_df.parquet").exists()
                else None
            ),
            (
                pd.read_parquet(data_dir / "site_res_df.parquet")
                if (data_dir / "site_res_df.parquet").exists()
                else None
            ),
            (
                pd.read_parquet(data_dir / "site_cond_std_df.parquet")
                if (data_dir / "site_cond_std_df.parquet").exists()
                else None
            ),
        )


def run_mera(
    residual_df: pd.DataFrame,
    ims: list[str],
    event_cname: str,
    site_cname: str,
    assume_biased: bool = True,
    compute_site_term: bool = True,
    mask: pd.DataFrame | None = None,
    verbose: bool = True,
    raise_warnings: bool = True,
    min_num_records_per_event: int = 3,
    min_num_records_per_site: int = 3,
    n_procs: int = 1,
) -> MeraResults:
    """
    Runs mixed effects regression analysis for the given
    residual dataframe, fitting each IM by restricted maximum likelihood (REML)

    For each IM, the model is::

        residual = bias + event term + site term + remaining residual

    or, as the R package lme4 writes it, ``im ~ 1 + (1|event) + (1|site)``, where ``1``
    stands for the bias and ``(1|event)`` for "a term for each event". The bias is
    dropped if ``assume_biased`` is False, and the site term if ``compute_site_term`` is
    False.

    Parameters
    ----------
    residual_df : dataframe
        Residual dataframe, has to contain all
        specified IMs (as columns) along with
         columns for event and site id.
        It has one row per record. A residual is usually
        ln(observed) - ln(predicted)
    ims : list of strings
        IMs for which to run to mixed effects
        regression analysis.
    event_cname : str
        Name of the column that contains the event ids
    site_cname : str
        Name of the column that contains the site ids
        Has no effect if compute_site_term is False,
        when the dataframe does not need that column
    assume_biased : bool, default = True
        If true then the model fits a bias term, removing
        any model bias (wrt. given dataset)

        Recommended to have this enabled, otherwise any model
        bias (wrt. given dataset) will affect random effect
        terms.
    compute_site_term : bool, default = True
        If true then the model fits a site term
    mask : dataframe or None, default = None
        Mask dataframe the size of the residual dataframe
        which selects which values are being used per IM
        for the mixed-effects fit. If None then all values are used.
        Masked out record & IM combinations will be NaN in the output.
        It holds True or False, with the rows of residual_df and a column per IM:
        True means that the record is used for that IM.
        See mask_too_few_records for making one
    verbose : bool, default = True
        If true then prints the progress of the analysis
    raise_warnings : bool, default = True
        If true, prints a warning for each event with fewer records than
        min_num_records_per_event and each site with fewer records than
        min_num_records_per_site. Those records are still used in the fit
    min_num_records_per_event : int, default = 3
        The number of records below which an event is warned about.
        Only used if raise_warnings is True. It does not remove any records:
        for that, pass a mask made by mask_too_few_records
    min_num_records_per_site : int, default = 3
        The number of records below which a site is warned about.
        Only used if raise_warnings is True. It does not remove any records
    n_procs : int, default = 1
        Number of processors to use. Each IM is fitted on one processor,
        so more than one only helps with several IMs. See the notes

    Returns
    -------
    MeraResults
        Results of the mixed-effects regression analysis

    Raises
    ------
    ValueError
        If an IM has NaN values among the records used, or if the records cannot be
        fitted: for example if they belong to a single event, or every record belongs
        to an event of its own. The error names the IM.
    RuntimeError
        If the events and sites explain an IM's residuals exactly, as in synthetic data
        made without noise. (Some such data raise a ValueError instead.)

    Warns
    -----
    RuntimeWarning
        If the optimiser does not converge for an IM. Its results are still returned.

    Notes
    -----
    A standard deviation of exactly zero (``tau`` or ``phi_S2S``) is a *singular fit*:
    the records show no variation between events (or sites) beyond what the remaining
    residuals explain. All the terms of that kind are then zero.

    An IM for which the mask leaves no records at all is skipped, with a printed
    warning, and all its results are NaN.

    With ``n_procs`` greater than 1, the IMs are fitted in separate worker processes.
    On Windows and macOS, and on Linux from Python 3.14, each worker starts by importing
    the script that called ``run_mera``. That script must therefore do its work under
    ``if __name__ == "__main__":``, as ``mera/example/mera_example.py`` does. Without
    it, every worker would run the whole script again and the fits fail to start.

    Examples
    --------
    >>> import mera
    >>> results = mera.run_mera(residual_df, ["PGA", "pSA_1.0"], "event_id", "stat_id")
    >>> results.bias_std_df  # the bias and the standard deviations of each IM
    >>> results.event_res_df["PGA"]  # the event terms for PGA
    >>> results.save(output_dir)
    """
    # The columns holding the grouping factors of the model: the events, and the sites
    # if a site term is fitted.
    random_effects_columns = [event_cname]
    if compute_site_term:
        random_effects_columns.append(site_cname)

    # One task per IM: the tuple of arguments of one call to _run_im_mera. A task
    # holds only that IM's columns, because in parallel each task is copied to a worker
    # process.
    tasks = (
        (
            residual_df[[cur_im, *random_effects_columns]],
            cur_im,
            event_cname,
            site_cname if compute_site_term else None,
            assume_biased,
            min_num_records_per_event,
            min_num_records_per_site,
            random_effects_columns,
            None if mask is None else mask[[cur_im]],
            raise_warnings,
            verbose,
        )
        for cur_im in ims
    )
    if n_procs == 1:
        results = [_run_im_mera(*task) for task in tasks]
    else:
        # starmap calls _run_im_mera(*task) for each task in the worker processes, and
        # returns the results in the order of the tasks.
        with mp.Pool(processes=n_procs) as pool:
            results = pool.starmap(_run_im_mera, tasks)

    # Unpack results
    # Each result holds one IM's series: put them side by side, one column per IM.
    event_res_df = pd.concat(
        [cur_result["event_res_series"] for cur_result in results], axis=1
    )
    event_cond_std_df = pd.concat(
        [cur_result["event_cond_std_series"] for cur_result in results], axis=1
    )
    event_cond_std_df.columns = ims
    rem_res_df = pd.concat([cur_result["rem_res_df"] for cur_result in results], axis=1)
    # The id columns in one concat: inserting columns into a frame with a block per IM
    # makes pandas warn that it is fragmented (from 101 IMs).
    rem_res_df = pd.concat(
        [rem_res_df, residual_df.loc[rem_res_df.index, random_effects_columns]], axis=1
    )
    bias_std_df = pd.DataFrame(
        [cur_result["bias_std_series"] for cur_result in results], index=pd.Index(ims)
    )
    fit_df = pd.concat([cur_result["fit_series"] for cur_result in results], axis=1)
    if compute_site_term:
        site_res_df = pd.concat(
            [cur_result["site_res_series"] for cur_result in results], axis=1
        )
        site_cond_std_df = pd.concat(
            [cur_result["site_cond_std_series"] for cur_result in results], axis=1
        )
        site_cond_std_df.columns = ims

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


def _run_im_mera(
    residual_df: pd.DataFrame,
    im: str,
    event_cname: str,
    site_cname: str | None,
    assume_biased: bool,
    min_num_records_per_event: int,
    min_num_records_per_site: int,
    random_effects_columns: list[str],
    mask: pd.DataFrame | None,
    raise_warnings: bool,
    verbose: bool,
):
    """
    Runs the mixed-effects regression for a single IM.

    This selects the records of the IM, fits them with fit_lmm, and writes the
    estimates into series labelled by event, by site or by record. Each series has the
    same labels for every IM, whichever records its fit used, so that run_mera can put
    the series of all the IMs side by side.

    Parameters
    ----------
    residual_df : pd.DataFrame
        Residual dataframe with the IM and the event and site id columns
    im : str
        IM to fit
    event_cname : str
        Name of the column that contains the event ids
    site_cname : str or None
        Name of the column that contains the site ids, None to fit no site term
    assume_biased : bool
        Whether to fit a bias term
    min_num_records_per_event : int
        Events with fewer records than this are reported when raise_warnings is True
    min_num_records_per_site : int
        Sites with fewer records than this are reported when raise_warnings is True
    random_effects_columns : list of str
        Columns holding the grouping factors (event, and site if fitted)
    mask : pd.DataFrame or None
        Mask selecting the records used per IM, None to use all records
    raise_warnings : bool
        Whether to print warnings for events and sites with too few records
    verbose : bool
        Whether to print progress

    Returns
    -------
    dict
        The per-IM result series and dataframes, keyed as unpacked by run_mera.
        They are all NaN if the IM has no data.
    """
    if verbose:
        print(f"Processing {im}")

    # Filter on the mask if given
    # cur_residual_df holds the records used for this IM's fit: all of them, or those
    # that the mask marks True.
    cur_columns = [im] + random_effects_columns
    cur_residual_df = (
        residual_df[cur_columns]
        if mask is None
        else residual_df[cur_columns].loc[mask[im]]
    )

    # Check for nans
    if cur_residual_df[im].isna().sum() > 0:
        raise ValueError(f"NaNs found in IM {im}")

    # Result series
    # These start as NaN and are filled in, in place, after the fit. They are indexed
    # by every event, site or record of residual_df, not only those of this fit, so
    # that they are the same for every IM. The ids become strings, as fit_lmm compares
    # them, and are put in natural order ("event_2" before "event_10").
    event_ids = natsort.natsorted(
        np.unique(residual_df[event_cname].to_numpy().astype(str))
    )
    event_res_series = pd.Series(index=event_ids, dtype=float, name=im)
    event_cond_std_series = pd.Series(index=event_ids, dtype=float)
    bias_std_series = pd.Series(
        index=["bias", "bias_std_err", "tau", "phi_S2S", "phi_w", "sigma"],
        data=np.nan,
        dtype=float,
        name=im,
    )
    rem_res_df = pd.DataFrame(
        index=residual_df.index.values, columns=pd.Index([im]), dtype=float
    )
    fit_series = pd.Series(index=residual_df.index.values, dtype=float, name=im)
    site_res_series, site_cond_std_series = None, None
    if site_cname is not None:
        site_ids = natsort.natsorted(
            np.unique(residual_df[site_cname].to_numpy().astype(str))
        )
        site_res_series = pd.Series(index=site_ids, dtype=float, name=im)
        site_cond_std_series = pd.Series(index=site_ids, dtype=float)

    result = {
        "event_res_series": event_res_series,
        "event_cond_std_series": event_cond_std_series,
        "rem_res_df": rem_res_df,
        "bias_std_series": bias_std_series,
        "fit_series": fit_series,
        "site_res_series": site_res_series,
        "site_cond_std_series": site_cond_std_series,
    }

    # Nothing to fit if the mask leaves no records: the results stay NaN
    if len(cur_residual_df) == 0:
        print(f"WARNING: No data for IM {im}, skipping...")
        return result

    # Generate warnings for insufficient records per event or site
    if raise_warnings:
        count_per_event = cur_residual_df.groupby(event_cname, observed=True).count()[
            im
        ]
        count_per_site = (
            cur_residual_df.groupby(site_cname, observed=True).count()[im]
            if site_cname is not None
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
            print(f"Warning: For IM {im}, {label} has only {count} records.")

    # Fit the mixed-effects model
    # groups holds the event id of each record used and, if a site term is fitted,
    # the site id of each: the grouping factors of the model.
    groups = [cur_residual_df[c].to_numpy() for c in random_effects_columns]
    fit = fit_lmm(
        cur_residual_df[im].to_numpy(), groups, intercept=assume_biased, name=f"IM {im}"
    )

    # Event (and site) terms, their conditional standard deviations,
    # and the standard deviations of the random effects
    # Index 0 of each list in fit is the first grouping factor, the events, and index 1
    # the sites. fit.levels[0] holds the ids of the events in the fit and fit.ranef[0]
    # their terms in the same order, so .loc writes each term to the row of its event.
    # Events with no records in this fit stay NaN.
    event_res_series.loc[fit.levels[0]] = fit.ranef[0]
    event_cond_std_series.loc[fit.levels[0]] = fit.condsd[0]
    bias_std_series.loc["tau"] = fit.sd[0]
    if site_res_series is not None and site_cond_std_series is not None:
        site_res_series.loc[fit.levels[1]] = fit.ranef[1]
        site_cond_std_series.loc[fit.levels[1]] = fit.condsd[1]
        bias_std_series.loc["phi_S2S"] = fit.sd[1]
    bias_std_series.loc["phi_w"] = fit.sigma

    # Remaining residuals and fitted values
    # fit.resid and fit.fitted have one value per record used, in the order of the
    # records, so with a mask they go to the rows that the mask marks True.
    if mask is None:
        rem_res_df[im] = fit.resid
        fit_series[:] = fit.fitted
    else:
        rem_res_df.loc[mask[im], im] = fit.resid
        fit_series.loc[mask[im]] = fit.fitted

    # Bias
    if assume_biased:
        bias_std_series.loc["bias"] = fit.beta[0]
        bias_std_series.loc["bias_std_err"] = fit.beta_se[0]

    if verbose:
        print(f"Finished processing {im}")

    return result
