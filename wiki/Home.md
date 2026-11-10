## Introduction to mixed effects models

Mixed effects models relate an observed response, such as an intensity measure (IM) (or its residual), 
to covariates, such as earthquake and site name. Interestingly, the covariates of mixed effects models can be 
categorical, meaning that they are labels rather than numerical values. In our case, we use the event and (optionally) 
site names as the labels for the categorical variables.

Mixed effects models are so named because they include both fixed effects and random effects. Fixed effects capture the 
overall trend of the observed response, and would be parameters such as the 
intercept, $c$, and the slope, $m$, if fitting a function such as $f(x) = m \times x + c$. In our case, we do not have a 
numerical independent variable like $x$, so we cannot estimate a simple slope like $m$. However, we can 
(optionally) fit the "intercept", which represents an overall _bias_ affecting all observations. Random 
effects contribute additional variation to the model to help explain features that are not captured by the fixed 
effects of the model. This is illustrated by Fig. 1, which shows a mixed effects regression on several groups of data.
Here, the slope remains constant and the intercept is allowed to vary. Fitting the intercept as a fixed effect produces 
the black line labelled $\mu_{\textrm{group}}$ that passes through the approximate center of all data points. 
Adding a random effect for each group improves the fit for each group, as indicated by the other colored lines. 
In our case, each event and (optionally) site contributes a random effect to the model.

![Fig. 1: Example mixed effects regression where the slope remains constant and the intercept is allowed to vary. 
The black line represents the result of fitting the intercept as a fixed effect. The other lines represent the
result of adding a random effect for each group. Figure adopted from pythonfordatascience.org at this 
[link.](https://www.pythonfordatascience.org/mixed-effects-regression-python/)](images/variable_intercept.png)
*Fig. 1: Example mixed effects regression where the slope remains constant and the intercept is allowed to vary. 
The black line represents the result of fitting the intercept as a fixed effect. The other lines represent the
result of adding a random effect for each group. Figure adopted from pythonfordatascience.org at this 
[link.](https://www.pythonfordatascience.org/mixed-effects-regression-python/)*

## Interpreting random effects

Formally, there are subtle differences between fixed and random effects. For example, fixed effects are referred to as 
model parameters and can be directly estimated from the fit, along with corresponding standard errors. Conversely, 
random effects are formally referred to as _variance components_, and cannot formally have best fit values or standard 
errors. However, in practice, we can treat a random effect’s _Best Linear Unbiased Predictor (BLUP)_, which is also 
known as its _conditional mode_, as something like a best fit value. Similarly, a random effect's 
_conditional standard deviation_ can be regarded as something like a standard error. The word _conditional_ is used to 
emphasize that these values are conditional on the observed data, with the fixed effects and the standard deviations 
of the model held at their fitted values. For more information about the terminology and mathematics of random effects, please refer to 
Chapter 1 of [Bates 2018](https://people.math.ethz.ch/~maechler/MEMo-pages/lMMwR.pdf). The MERA package provides a 
_BLUP_ or _conditional mode_ and a _conditional standard deviation_ for each event and (optionally) 
site that are considered as random effects.

## Details of MERA's implementation

MERA fits each IM's model with its own implementation of the profiled restricted maximum
likelihood (REML) method of the R package [lme4](https://github.com/lme4/lme4)
(Bates et al., 2015, *Journal of Statistical Software* 67(1)).

- **Numerical core.** It is written in Rust, using the [faer](https://crates.io/crates/faer)
  sparse Cholesky factorisation, and is called from Python. R, lme4 and rpy2 are no longer
  needed.
- **Optimiser.** As in lme4, the ratios of the random-effect standard deviations to the
  residual standard deviation are optimised with a derivative-free trust-region method.
- **Validation.** It was validated against lme4 2.0.6. The variance components, bias,
  random effects (BLUPs, or conditional modes) and their conditional standard deviations
  agree to within lme4's own convergence tolerance. MERA's REML optimum was at least as
  good as lme4's for every IM tested.

For more detail:

- [The mathematics of MERA, step by step](https://github.com/ucgmsim/mera/blob/main/docs/mixed_effects_maths.md)
  follows one fit from the model to the reported numbers, for readers with basic
  mathematics.
- [`mera/example/maths_walkthrough.py`](https://github.com/ucgmsim/mera/blob/main/mera/example/maths_walkthrough.py)
  does the same calculation in plain numpy.
- [Fitting a different model](https://github.com/ucgmsim/mera/blob/main/docs/adding_models.md)
  describes what else can be fitted, and where to change the code.
- [Reading the Rust engine](https://github.com/ucgmsim/mera/blob/main/crates/mera_lmm/README.md)
  explains the Rust code for Python readers.

lme4, and therefore MERA, provides standard errors for fixed-effects parameters such as
the overall bias, but not for random effects. Other statistics software may claim to
provide standard errors for random effects, but these numbers are likely unreliable
(please see Chapter 1 of [Bates 2018](https://people.math.ethz.ch/~maechler/MEMo-pages/lMMwR.pdf)
for more information). The conditional standard deviations that MERA reports for each
event and site are conditional on the estimated bias, exactly as lme4's
`ranef(condVar = TRUE)`.
