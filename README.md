## Mixed Effects Regression Analysis

This package contains two different implementations for running mixed-effects regression analysis for ground motion prediction models with 2 random effects, event and site.

The only difference between the two implementations is the underlying packages they use.

---- 

### Installation

Navigate to the `Ancillary-Tools` directory and run `pip install -e ./mera`

Note: Dependencies have to be installed manually depending on which implementation is used

-----

### Implementation - statsmodel 
This one is based on the python [statsmodel](https://www.statsmodels.org/stable/index.html) package. If the number of records is small then I recommend using this method as it has less dependencies. However, once the number of records becomes large (> 100k) memory becomes an issue as it creates a design matrix for each random effect of shape [n_records, n_unique_group_values].

Dependencies: `pip install statsmodels`

Usage:
```python
from mera.mera_statsmodel import run_mera

# See docstring for inputs and results
event_res_df, site_res_df, rem_res_df, bias_std_df = run_mera(....)
```


### Implementation - pymer4
This implementation uses R and the R-package `lme4` accessed via the python wrapper package [pymer4](http://eshinjolly.com/pymer4/index.html). This implementation is recommended for large datasets with a large number of events/sites as it does not suffer from the statsmodel memory limitation.

Dependencies: See http://eshinjolly.com/pymer4/installation.html for installation

Usage:
```python
from mera.mera_pymer4 import run_mera

# See docstring for inputs and results
event_res_df, site_res_df, rem_res_df, bias_std_df = run_mera(....)
```


