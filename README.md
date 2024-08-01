## Mixed Effects Regression Analysis

This package contains two different implementations for running mixed-effects regression analysis for ground motion prediction models with 2 random effects, event and site.

The only difference between the two implementations is the underlying packages they use.

----
### Installation

Install dependencies (see requirements.txt)

Run `pip install git+https://github.com/ucgmsim/mera.git`
-----


### Implementation - pymer4
This implementation uses R and the R-package `lme4` accessed via the python wrapper package [pymer4](http://eshinjolly.com/pymer4/index.html). This implementation is recommended for large datasets with a large number of events/sites as it does not suffer from the statsmodel memory limitation.

Dependencies: See http://eshinjolly.com/pymer4/installation.html for installation

Usage:
```python
import mera

# See docstring for inputs and results
event_res_df, site_res_df, rem_res_df, bias_std_df = mera.run_mera(....)
```







