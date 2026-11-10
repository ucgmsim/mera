# Fitting a different model

`mera.run_mera` fits one model:

```
residual = bias + event term + site term + remaining residual
```

This guide describes what else can be fitted, and where in the source code to make each
change. It has three parts, in order of effort:

1. [What works without changing any code](#1-without-changing-any-code)
2. [What needs a small change to the Python](#2-with-small-changes-to-the-python)
3. [What would need the Rust engine changed](#3-what-would-need-the-rust-engine-changed)

followed by [how to check a new model](#checking-a-new-model).
[`mixed_effects_maths.md`](mixed_effects_maths.md) explains the terms used here.

## What a model is made of

Every model that MERA's engine can fit has the same three ingredients:

| Ingredient | In `run_mera`'s model | In the code |
|---|---|---|
| A **response**: one number per record | the residual of one IM | `y` |
| **Fixed effects**: constants estimated directly, each multiplying a column of numbers | the bias, multiplying a column of ones | the columns of `x` |
| **Grouping factors**: labels that put the records into groups, each group getting its own term | the events and the sites | `groups` |

In the notation of the R package lme4, `run_mera`'s model is
`residual ~ 1 + (1|event) + (1|site)`: `1` is the bias, and `(1|event)` reads "a term for
each event".

| To fit ... | See | Needs |
|---|---|---|
| no bias, or no site term | [1.1](#11-drop-the-bias-or-the-site-term) | an argument of `run_mera` |
| a term for a third grouping (region, network, source zone, ...) | [1.2](#12-more-grouping-factors) | a call to `fit_lmm` |
| a grouping inside another (stations within networks) | [1.3](#13-a-grouping-inside-another) | a call to `fit_lmm` |
| maximum likelihood instead of REML | [1.4](#14-maximum-likelihood-and-comparing-models) | an argument of `fit_lmm` |
| an extra grouping reported in `run_mera`'s tables | [2.1](#21-report-another-grouping-factor-from-run_mera) | Python edits |
| a trend with magnitude, distance, Vs30, ... | [2.2](#22-add-a-fixed-effect) | Python edits |
| terms that vary with a covariate, unequal residual scatter, ... | [3](#3-what-would-need-the-rust-engine-changed) | Rust changes, or other software |

## 1. Without changing any code

### 1.1 Drop the bias or the site term

```python
results = mera.run_mera(res_df, ims, "event_id", "stat_id", assume_biased=False)
results = mera.run_mera(res_df, ims, "event_id", "stat_id", compute_site_term=False)
```

Without the bias, the model is `residual ~ 0 + (1|event) + (1|site)`. Any overall offset
of the residuals then ends up in the event and site terms, which is rarely what is wanted.

### 1.2 More grouping factors

`run_mera` is a wrapper that prepares each IM's records and tabulates the results. The
fitting is done by `mera.lmm.fit_lmm`, which takes any number of grouping factors. To
give each region its own term as well:

```python
from mera.lmm import fit_lmm

fit = fit_lmm(
    res_df["PGA"].to_numpy(),
    [
        res_df["event_id"].to_numpy(),
        res_df["stat_id"].to_numpy(),
        res_df["region"].to_numpy(),
    ],
)
```

This fits `PGA ~ 1 + (1|event) + (1|site) + (1|region)`. The result is an `LmmFit`
(documented in `mera/lmm.py`), whose lists follow the order of the grouping factors: index
0 is the events, 1 the sites and 2 the regions.

```python
tau, phi_s2s, sd_region = fit.sd          # standard deviation of each factor's terms
phi_w = fit.sigma                         # standard deviation of the remaining residuals
bias, bias_std_err = fit.beta[0], fit.beta_se[0]
region_terms = dict(zip(fit.levels[2], fit.ranef[2]))    # {"north": 0.19, ...}
region_term_sds = dict(zip(fit.levels[2], fit.condsd[2]))
remaining = fit.resid                     # one per record, in the order of the input
```

Three things to keep in mind:

- The records must have no NaN values. `run_mera` applies the mask before it calls
  `fit_lmm`; here, select the rows yourself.
- A grouping factor needs at least two levels, and fewer levels than there are records.
- A standard deviation is poorly determined by a handful of levels. With three regions,
  `sd_region` is estimated from three numbers, and is often exactly zero (a *singular
  fit*).

### 1.3 A grouping inside another

Stations within networks, or sites within basins, are *nested*: each station belongs to
one network. No special treatment is needed, as long as the labels of the inner grouping
are unique across the outer one. If two networks both have a station called `"001"`, join
the labels first:

```python
# "NZ/001", "GN/001"; astype(str) in case the ids are numbers
station = res_df["network"].astype(str) + "/" + res_df["station"].astype(str)
fit = fit_lmm(y, [res_df["event_id"], res_df["network"], station])
```

### 1.4 Maximum likelihood, and comparing models

`fit_lmm(..., reml=False)` fits by maximum likelihood (ML) instead of REML. REML gives
better estimates of the standard deviations and is the right choice for reporting them.
ML is needed for one thing: comparing two models that differ in their *fixed* effects, as
in [2.2](#22-add-a-fixed-effect).

`fit.deviance` is the criterion that the fit minimised, and for a given kind of fit the
smaller it is the better the model describes the data. It can show whether an addition to
a model is worth having, by a *likelihood-ratio test*, as long as like is compared with
like:

- **An added grouping factor**, with the fixed effects unchanged. The deviance never
  rises when a grouping factor is added, with REML or with ML. The question is whether
  it falls by enough.
- **An added fixed effect.** Fit both models with `reml=False`. The REML criterion of two
  models with different fixed effects cannot be compared: it even changes with the units
  of a covariate, and usually *rises* when one is added. The ML deviance never rises when
  a fixed effect is added.

In both cases, a fall of more than 3.84 for one addition is significant at the 5% level
(the fall is compared with a chi-squared distribution with one degree of freedom). For a
grouping factor that threshold is on the cautious side, because a standard deviation
cannot be negative. The
[lme4 book](https://people.math.ethz.ch/~maechler/MEMo-pages/lMMwR.pdf) discusses these
tests.

## 2. With small changes to the Python

These changes are made in a copy of the source code, not in an installed package:

```
git clone https://github.com/ucgmsim/mera.git
cd mera
uv sync --all-extras --dev      # or: pip install -e .
```

Installing from source compiles the Rust engine, so it needs a
[Rust toolchain](https://rustup.rs), even though none of the changes below touch Rust.
`uv run pytest` then runs the tests.

### 2.1 Report another grouping factor from `run_mera`

`fit_lmm` already fits it ([1.2](#12-more-grouping-factors)); what `run_mera` adds is the
loop over IMs, the mask, and the output tables. All the edits are in
`mera/analysis.py`, and each follows the pattern of the site term, so searching the file
for `site` finds every place:

1. **`run_mera`'s arguments.** Add the name of the new column, for example
   `region_cname: str | None = None`.
2. **`random_effects_columns`**, at the top of `run_mera`. Append the new column. The
   order of this list is the order of the grouping factors, so the new factor is index 2
   of `fit.levels`, `fit.ranef`, `fit.condsd` and `fit.sd`. If the site term is switched
   off (`compute_site_term=False`), it is index 1 instead: look its position up with
   `random_effects_columns.index(region_cname)` rather than writing 2.
3. **The `tasks` tuple** in `run_mera` and **the arguments of `_run_im_mera`**. Pass the
   new column name through, as `site_cname` is.
4. **"Result series" in `_run_im_mera`.** Create a series for the new terms and one for
   their conditional standard deviations, as for the sites, and add a name for the new
   standard deviation to the index of `bias_std_series`.
5. **After the call to `fit_lmm`** in `_run_im_mera`. Fill them in from `fit.levels[2]`,
   `fit.ranef[2]`, `fit.condsd[2]` and `fit.sd[2]`, and add them to the dictionary that
   the function returns.
6. **"Unpack results" in `run_mera`.** Join the new series into tables, and include the
   new standard deviation in the total `sigma`. `run_mera` ends in two branches, with and
   without the site term, and both need the change. When the new factor is not fitted,
   drop its column from `bias_std_df`, as the second branch does for `phi_S2S`.
7. **`MeraResults`.** Add the two tables as attributes, and to `save`, `load`,
   `save_to_parquet` and `load_from_parquet`. `MeraResults` is created by position in
   `run_mera` and in the two `load` methods, so put the new attributes last, with a
   default of `None`, and existing code keeps working.

One more thing to watch: `mask_too_few_records` treats every column of the residual table
other than the event and site ids as an IM. Give it only those columns, for example
`res_df[[*ims, "event_id", "stat_id"]]`, when the table has a region column too.

### 2.2 Add a fixed effect

Suppose the residuals trend with magnitude, and the model should be

```
residual = bias + slope × (magnitude − 6) + event term + site term + remaining residual
```

or `residual ~ 1 + magnitude + (1|event) + (1|site)`. The slope is a second fixed effect.
The engine fits any number of them: all it needs is one more column in the fixed-effects
matrix `x`.

`fit_lmm` builds `x` in its step 2, with only the column of ones:

```python
x = np.ones((y.size, int(intercept)))
```

To accept more columns, add an argument, for example
`covariates: npt.ArrayLike | None = None`, and append it to `x`:

```python
x = np.ones((y.size, int(intercept)))
if covariates is not None:
    x = np.column_stack([x, np.asarray(covariates, dtype=float)])
```

Nothing else in `fit_lmm` has to change. `fit.beta` and `fit.beta_se` then have one entry
per column of `x`: the bias first, then a coefficient for each covariate. The fitted
values (`x @ beta` plus the random effects) include the trend, so the event and site terms
and the remaining residuals are what is left after it.

```python
fit = fit_lmm(y, [events, sites], covariates=magnitude - 6.0)
bias, slope = fit.beta
bias_std_err, slope_std_err = fit.beta_se
```

For `run_mera` to do this for every IM, three places in `mera/analysis.py` change:

1. the `tasks` tuple in `run_mera` selects the columns that each fit receives
   (`residual_df[[cur_im, *random_effects_columns]]`): add the covariate columns;
2. `cur_columns` at the top of `_run_im_mera` selects them again, before the mask is
   applied: add them there too, so that the covariates are masked with the IM, and pass
   `cur_residual_df[covariate_columns].to_numpy()` to `fit_lmm`;
3. the bias is stored in `bias_std_series` near the end of `_run_im_mera`: store
   `fit.beta[1:]` and `fit.beta_se[1:]` there as well, under names of your choice, and add
   those names to the index of `bias_std_series`.

As in 2.1, keep covariate columns out of the table given to `mask_too_few_records`.

Advice on covariates:

- Subtract a typical value, as in `magnitude - 6.0`. The bias is then the average residual
  at magnitude 6 rather than at magnitude 0, and the arithmetic is more accurate.
- A covariate must vary, and must not be an exact combination of the other columns
  (magnitude and magnitude + 1, with a bias, is one). Otherwise the fit either stops with
  an error that says the fixed effects are collinear, or returns meaningless coefficients
  with enormous standard errors.
- A covariate that is constant within each event, such as magnitude, competes with the
  event terms: its coefficient is determined by the differences between events only.
- To judge whether the trend is real, compare the coefficient with its standard error, or
  compare the deviances of fits with and without it, both made with `reml=False`
  ([1.4](#14-maximum-likelihood-and-comparing-models)).

## 3. What would need the Rust engine changed

The engine in `crates/mera_lmm` is fast because it is specialised. It assumes that:

1. every random effect is a single number per level that is *added* to all the records of
   that level (a "random intercept");
2. each grouping factor has one standard deviation, and its terms are independent of each
   other and of the other factors' terms;
3. the remaining residuals all have the same standard deviation;
4. everything is normally distributed.

These models break one of the assumptions:

| Model | Example | Breaks |
|---|---|---|
| Random slopes | each site has its own scaling with magnitude or distance, `(1 + magnitude \| site)` | 1 and 2 |
| Correlated terms | site terms that are similar for nearby sites | 2 |
| Unequal residual scatter | φ<sub>w</sub> that depends on magnitude or distance | 3 |
| Other distributions | a yes/no response, or counts | 4 |

Where the assumptions are built in, for anyone who wants to generalise the engine:

- `Design::new` in `design.rs` builds ZᵀZ as *counts* of records, which is assumption 1.
  With a random slope, Z holds values of the covariate instead of ones.
- `Design::fill_a` in `design.rs` and `lambda` in `CrossedModel::profile` (`model.rs`)
  scale each level by the one θ of its factor, which is assumption 2. In general, each
  level has a small block of θ values (three for an intercept and a slope that are
  correlated), as sections 2.2 and 2.3 of Bates et al. (2015) describe.
- The formulas for the criterion in `model.rs` are for assumptions 3 and 4.

The sparse factorisation in `factor.rs` is general and would not need to change.

For a one-off analysis with such a model, it is much less work to use software that
already supports it, such as lme4 in R (`lmer(y ~ 1 + (1 | event) + (1 + magnitude |
site))`), or `statsmodels.MixedLM` in Python for the simpler cases.

## Checking a new model

A changed model deserves a check before its numbers are trusted. In order of effort:

**Fit made-up data.** Generate residuals from the model with known standard deviations and
see whether the fit recovers them, within the scatter expected from the amount of data.
`make_example_data` in `mera/example/maths_walkthrough.py` shows how.

**Compare with the dense numpy version.** The functions in
`mera/example/maths_walkthrough.py` fit the same family of models with ordinary matrices,
for any `x` and any number of grouping factors. They are slow, but independent of the Rust
engine, and fine for a few hundred records:

```python
import numpy as np

from mera.example import maths_walkthrough as walkthrough
from mera.lmm import fit_lmm

groups = [events, sites, regions]
x = np.ones((y.size, 1))  # the bias; add columns here for more fixed effects
z, level_factor, levels = walkthrough.indicator_matrix(groups)
model = walkthrough.Model(y, x, z, level_factor)
dense = walkthrough.estimates(walkthrough.find_theta(model), model)

fit = fit_lmm(y, groups)
assert np.allclose(dense.sd, fit.sd, atol=1e-6)
assert np.allclose(dense.ranef, np.concatenate(fit.ranef), atol=1e-6)
```

**Compare with lme4.** `tools/lme4_reference/generate.py` fits models with R's lme4 and
returns them in the layout of `fit_lmm`'s results (its `fit_lme4` function). It is how
the reference values of the test suite were made;
`tests/resources/lme4_reference/README.md` describes the R setup. `fit_lme4` handles any
number of grouping factors, but builds its formula with an intercept only: for a model
with covariates ([2.2](#22-add-a-fixed-effect)), add them to the data frame and the
formula in its `lmer` function.

**Add a test.** `tests/test_lmm.py` has examples of each kind: against lme4's values,
against closed-form answers for balanced data, and of the errors for invalid input.
