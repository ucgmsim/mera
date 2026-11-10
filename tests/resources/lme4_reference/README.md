# lme4 reference values

Values from R's lme4 that the tests compare mera against.

| Files | Contents | Used by |
|---|---|---|
| `no_mask/`, `mask/` | `run_mera`'s outputs for all 118 IMs of the example data (bias, standard deviations, random effects and their conditional SDs, fitted values and remaining residuals), without and with the reference mask | `test_mera.py::test_mera`, `test_mera_in_parallel` |
| `fits.csv`, `random_effects.csv` | lme4's fits for `LME4_FITS` in `tests/example_data.py`: the REML criterion, theta, sigma and the intercept with its standard error; per level, the random effect and its conditional SD | `test_lmm.py::test_matches_lme4`, `test_mera.py::test_other_configurations_match_lme4` |
| `criterion.csv` | lme4's REML criterion at the fixed `CRITERION_THETAS` for `CRITERION_IMS`, in all four configurations | `test_criterion.py::test_rust_deviance_matches_lme4` |

## How they were made

On 2026-11-10, by `tools/lme4_reference/generate.py`, with R 4.6.1, lme4 2.0-6,
lmerTest 3.2-1, rpy2 3.6.9 and pymer4 0.8.2 from ucgmsim's fork (branch
`get_ranef_cond_std`, commit 7b3a7c4).

- The fits use lme4's default optimiser (nloptwrap's BOBYQA) with every stopping
  tolerance at zero except `xtol_abs = 1e-14`, so they run to the optimum. With its
  default tolerances lme4 stops up to ~2e-4 short in the random effects, which is why
  the earlier values here, from the pymer4 version of mera, needed tolerances of 1e-4 to
  5e-4.
- The criterion comes from lme4's deviance function (`getME(model, "devfun")`):
  `REMLcrit()` is wrong for models without fixed effects in lme4 2.0.6.
- The `run_mera` outputs come from mera's own `run_mera` with each fit done by lme4, so
  their layout is exactly mera's. On 2026-11-11 the rows of `event_res_df.csv` and
  `site_res_df.csv` were put in natural order, as `run_mera` now returns them; whole
  lines were moved, so no value changed.

When they were made, they agreed with mera to within 5e-11 in the criterion at the
optimum (6e-15 relative at fixed theta), 3e-7 in theta, 8e-9 in sigma and the intercept
with its standard error, and 2e-7 in the random effects, conditional SDs, fitted values
and remaining residuals. Where mera reports a singular fit (theta exactly 0), lme4
stops within 3e-7 of 0 and sometimes exactly on it.

## Regenerating

This needs R with lme4 and lmerTest, and a Python environment with this repository's
mera (built with a Rust toolchain), rpy2 and the pymer4 fork, which needs pandas < 3. From
the repository root:

```
mkdir -p ~/R/lme4
Rscript -e 'install.packages(c("lme4", "lmerTest"), lib = "~/R/lme4", repos = "https://cloud.r-project.org")'
uv venv /tmp/lme4-env --python 3.12
uv pip install --python /tmp/lme4-env/bin/python . rpy2 "pandas<3" \
    "pymer4 @ git+https://github.com/ucgmsim/pymer4.git@get_ranef_cond_std"
R_LIBS=~/R/lme4 /tmp/lme4-env/bin/python tools/lme4_reference/generate.py
```

It takes about two minutes and rewrites every file here except this README; update the
versions, date and agreement above to match.
