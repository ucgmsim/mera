## Mixed Effects Regression Analysis

Convience wrapper for running mixed-effects regression analysis for ground motion prediction models.

Uses the python wrapper package [pymer4](http://eshinjolly.com/pymer4/index.html), to make use of the R-package `lme4`.
Note that a forked version of the wrapper is used, https://github.com/ucgmsim/pymer4/tree/get_ranef_cond_std (installed with the package)

----
### Installation

**Install dependencies**
Requires R and the R-package `lme4`.

**Install package**
Run `pip install ucgmsim-mera`

---
### Usage

```python
import mera

# See docstring for inputs and results
event_res_df, site_res_df, rem_res_df, bias_std_df = mera.run_mera(....)
```







