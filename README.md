# MERA: Mixed Effects Regression Analysis

MERA analyses ground-motion residuals with a mixed-effects regression. For each intensity
measure (IM), it splits every residual into four parts:

```
residual = bias + event term + site term + remaining residual
```

- the **bias** is the average misfit of the predictions, the same for every record;
- the **event term** (between-event residual) is shared by all records of one earthquake;
- the **site term** (site-to-site residual) is shared by all records at one site;
- the **remaining residual** is what is left for the individual record.

It returns the bias, the event and site terms with their uncertainties, and the standard
deviation of each part: between-event (`tau`), between-site (`phi_S2S`), remaining
(`phi_w`) and total (`sigma`).

In the notation of the R package [lme4](https://github.com/lme4/lme4), the model is
`residual ~ 1 + (1|event) + (1|site)`, where `1` stands for the bias and `(1|event)` for
"a term for each event". It is fitted by restricted maximum likelihood (REML) with lme4's
method. The numerical core is written in Rust; R is not required.

Contents: [Installation](#installation) · [Quick start](#quick-start) ·
[Inputs](#inputs) · [Outputs](#outputs) · [Things to know](#things-to-know) ·
[Understanding the method](#understanding-the-method) ·
[How the code is organised](#how-the-code-is-organised) ·
[Fitting a different model](#fitting-a-different-model) · [Development](#development)

## Installation

Releases are published to PyPI as `ucgmsim-mera`, with wheels for Linux, macOS and
Windows that need no Rust toolchain:

```
pip install ucgmsim-mera
```

Installing from source needs a [Rust toolchain](https://rustup.rs):

```
pip install git+https://github.com/ucgmsim/mera.git
```

Either way, the package is imported as `mera`.

## Quick start

```python
from pathlib import Path

import numpy as np
import pandas as pd

import mera

# Observed and predicted IMs: one row per record, with the event and site of each
obs = pd.read_csv("im_obs.csv", index_col=0)
sim = pd.read_csv("im_sim.csv", index_col=0)

# The residuals, with the id columns alongside
ims = ["PGA", "pSA_1.0"]
res_df = np.log(obs[ims] / sim[ims])
res_df["event_id"] = obs["event_id"]
res_df["stat_id"] = obs["stat_id"]

results = mera.run_mera(res_df, ims, "event_id", "stat_id")

print(results.bias_std_df)
#            bias  bias_std_err     tau  phi_S2S   phi_w   sigma
# PGA     -0.0569        0.0389  0.4072   0.4460  0.5135  0.7928
# pSA_1.0 -0.2907        0.0385  0.3570   0.4683  0.4478  0.7398

output_dir = Path("mera_results")
output_dir.mkdir(exist_ok=True)
results.save(output_dir)  # one CSV file per table
```

The numbers shown are for the example data in `mera/example/resources`.
[`mera/example/mera_example.py`](https://github.com/ucgmsim/mera/blob/main/mera/example/mera_example.py)
is a complete script for those data, including a mask.

## Inputs

`run_mera` takes a pandas DataFrame with one row per record. It has a column of residuals
for each IM, a column of event ids and a column of site ids:

```
          PGA  pSA_1.0  event_id  stat_id
gm_id
5      -1.014   -0.728         1        5
23      0.264    0.677         1       23
26     -0.593   -0.758         1       26
```

The ids can be numbers or text, and the columns can have any names: `run_mera` is told
which ones they are. A residual is usually ln(observed) − ln(predicted).

A fit cannot use missing values. To leave records out of some IMs, pass a **mask**: a
DataFrame of True and False with the same rows and a column per IM, where True means "use
this record for this IM". `mera.mask_too_few_records` makes one that leaves out the
records of events and sites that have too few records to say much about their terms:

```python
mask = mera.mask_too_few_records(
    res_df,
    "event_id",
    "stat_id",
    min_num_records_per_event=3,
    min_num_records_per_site=3,
)
results = mera.run_mera(res_df, ims, "event_id", "stat_id", mask=mask)
```

The other arguments are described in `run_mera`'s docstring (`help(mera.run_mera)`).

## Outputs

`run_mera` returns a `MeraResults`, which holds these tables as DataFrames:

| Table | Rows | Columns | Contents |
|---|---|---|---|
| `bias_std_df` | IMs | `bias`, `bias_std_err`, `tau`, `phi_S2S`, `phi_w`, `sigma` | the bias with its standard error, and the standard deviations |
| `event_res_df` | events | IMs | the event terms |
| `event_cond_std_df` | events | IMs | the conditional standard deviation of each event term: how uncertain it is |
| `site_res_df` | sites | IMs | the site terms |
| `site_cond_std_df` | sites | IMs | the conditional standard deviation of each site term |
| `rem_res_df` | records | IMs, then the id columns | the remaining residuals |
| `fit_df` | records | IMs | the fitted values: bias + event term + site term |

`results.save(dir)` writes each to `<name>.csv`, `results.save_to_parquet(dir)` to
`<name>.parquet`, and `MeraResults.load(dir)` and `MeraResults.load_from_parquet(dir)`
read them back. `dir` is a `pathlib.Path` of a directory that exists.

A value is NaN where it does not exist: for a record that the mask left out, or for an
event or site with no records in that IM's fit. An IM for which the mask leaves no
records at all is skipped with a warning, and is NaN throughout.

## Things to know

- **The event and site tables are labelled by id, as text, in natural order** (`"2"`
  before `"10"`). The versions of MERA that used R sorted `event_res_df` and
  `site_res_df` as plain text instead (`"10"` before `"2"`), so when comparing with
  results saved by them, match rows by label (`.loc`, `.reindex`) rather than by
  position. (Tables loaded back from CSV files have numeric ids as numbers again, because
  pandas reads them that way.)
- **`phi_w` is not the root-mean-square of `rem_res_df`**, which is smaller: the estimate
  allows for the scatter that the event and site terms have absorbed.
- **`sigma` in `bias_std_df` is the total standard deviation**,
  √(`tau`² + `phi_S2S`² + `phi_w`²).
- **A standard deviation of exactly zero** (`tau` or `phi_S2S`) is a *singular fit*: the
  records show no more variation between events (or sites) than the remaining residuals
  would produce by chance. All the terms of that kind are then zero. It is a legitimate
  result, most common with few events or sites.
- **`min_num_records_per_event` and `min_num_records_per_site` only print warnings** in
  `run_mera`. To leave those records out, pass a mask from `mask_too_few_records`.
- **Parallel runs need a guarded script.** `run_mera(..., n_procs=4)` fits the IMs in
  separate processes. On Windows and macOS, and on Linux from Python 3.14, the calling
  script must do its work under `if __name__ == "__main__":`, as the example script does.
- **A convergence warning** names the IM whose optimiser stopped early. Its results are
  still returned, and should be treated with caution.
- **Data without scatter cannot be fitted.** If the events and sites explain the residuals
  exactly, as in synthetic data made without noise, there is no remaining residual to
  estimate `phi_w` from, and `run_mera` stops with an error that says so.

## Understanding the method

For readers with basic Python and mathematics, and no Rust:

- The [introduction to mixed-effects models](https://github.com/ucgmsim/mera/blob/main/wiki/Home.md)
  (the project wiki) explains fixed and random effects and how to interpret them.
- [The mathematics of MERA, step by step](https://github.com/ucgmsim/mera/blob/main/docs/mixed_effects_maths.md)
  follows one fit from the model to the reported numbers.
- [`mera/example/maths_walkthrough.py`](https://github.com/ucgmsim/mera/blob/main/mera/example/maths_walkthrough.py)
  does the whole calculation in plain numpy, and prints its results next to MERA's:
  `python -m mera.example.maths_walkthrough`.
- [Reading the Rust engine](https://github.com/ucgmsim/mera/blob/main/crates/mera_lmm/README.md)
  explains the Rust code for Python readers.

## How the code is organised

```
mera/                        the Python package
  __init__.py                what `import mera` provides
  analysis.py                run_mera and MeraResults: one fit per IM, masks, output tables
  lmm.py                     fit_lmm: the statistics of one fit, and the search for the best fit
  utils.py                   mask_too_few_records
  _lmm.pyi                   describes the Rust engine to Python (the engine itself is compiled)
  mera_pymer4.py             the old import path, kept so that existing scripts still run
  example/
    mera_example.py          a complete example on the bundled data
    maths_walkthrough.py     the mathematics in plain numpy, for learning and checking
    resources/               example data: observed and simulated IMs
crates/mera_lmm/             the numerical engine, in Rust
  src/design.rs              reduces the data to counts and sums
  src/factor.rs              sparse Cholesky factorisation
  src/model.rs               the REML criterion and the estimates
src-rust/lib.rs              the bridge that makes the engine importable as mera._lmm
tests/                       Python tests; tests/resources/lme4_reference holds R's lme4 results
tools/lme4_reference/        regenerates those reference results (needs R; run by hand)
docs/                        the guides linked above
wiki/                        source of the GitHub wiki
```

A call to `run_mera` goes down through three layers:

```
mera.run_mera                 (analysis.py)  loops over the IMs and builds the tables
  └─ _run_im_mera             (analysis.py)  selects one IM's records and labels its results
      └─ fit_lmm              (lmm.py)       encodes the labels and searches for the best fit
          └─ _lmm.CrossedModel   (Rust)      evaluates the criterion, then the estimates
```

Every function has a docstring, and the code is commented step by step.

## Fitting a different model

[Fitting a different model](https://github.com/ucgmsim/mera/blob/main/docs/adding_models.md)
describes what else can be fitted and where in the source to change it:

- without changing any code: no bias, no site term, extra or nested grouping factors
  (regions, networks), maximum likelihood;
- with small changes to the Python: extra terms in `run_mera`'s tables, and fixed effects
  such as a trend with magnitude or distance;
- what would need the Rust engine changed, such as random slopes.

## Development

```
uv sync --all-extras --dev   # builds the Rust extension in place
uvx lefthook install         # git hooks running the shared ucgmsim CI checks locally
uv run pytest                # Python tests, including checks against lme4's values
cargo test                   # Rust tests
```

CI is the team's shared [meta-ci-action](https://github.com/ucgmsim/meta-ci-action):
ruff, deptry, npdlint, pytest with coverage, ty, and cargo fmt/clippy/test.

The tests compare mera's results with lme4's, stored in
`tests/resources/lme4_reference`. These come from R and are regenerated by hand, only
when `run_mera`'s outputs or the cases the tests check change; see the README there.

Until November 2026, MERA fitted its models with R's lme4, through pymer4. It moved to its
own Rust engine so that it installs with pip alone, without R, and stays fast on large
datasets. [Pull request 14](https://github.com/ucgmsim/mera/pull/14) records the
alternatives that were considered, and the accuracy and speed comparisons.
