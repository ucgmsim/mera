## Introduction to mixed effects models

Mixed effects models relate an observed response, such as an intensity measure (IM), to covariates, such as earthquake 
ID and site ID. Interestingly, the covariates of mixed effects models can be categorical, meaning that they are labels 
rather than numerical values. Mixed effects models are so named because they include both fixed effects and random 
effects. Fixed effects capture the overall trend of the observed response, and would be parameters such as the 
intercept, `c`, and the slope, `m`, if fitting a function such as `f(x) = m*x + c`. In our case, we do not have a 
numerical independent axis like `x`, so we do not have a simple slope like `m` that can be estimated. However, we can 
(optionally) fit the "intercept", which represents an overall bias affecting all observations. Random 
effects contribute additional variation to the model to help explain features that are not captured by the fixed effects 
of the model. In our case, each event and (optionally) site contributes a random effect to the model.

## Interpreting random effects

Formally, there are subtle differences between fixed and random effects. For example, fixed effects are referred to as 
model parameters and can be directly estimated from the fit, along with corresponding standard errors. Random effects 
are formally referred to as "variance components", and cannot formally have best fit values or standard errors. 
However, in practice, we can treat a random effect’s Best Linear Unbiased Predictor (BLUP) as something like a best 
fit value, and its conditional standard deviation as something like a standard error. For more information about 
this distinction, please refer to Chapter 1 of 
[Bates 2018](https://people.math.ethz.ch/~maechler/MEMo-pages/lMMwR.pdf). 
The MERA package provides a BLUP and conditional standard deviation for each event and (optionally) site that are 
considered as random effects.

## Details of MERA's implementation

The Mixed Effects Regression Analysis (MERA) package is Python code that utilizes the 
robust [lme4](https://github.com/lme4/lme4) library in 
the [R programming language](https://www.r-project.org/about.html). It achieves this with 
the [Pymer4](https://github.com/ejolly/pymer4) package which uses the [rpy2](https://github.com/rpy2/rpy2)
library to establish an interface between Python and R.

R is an open source programming language that is often used for robust statistical analyses. 
The primary developer of R’s `lme4` package is a statistician called Douglas Bates who took great care to ensure that 
the `lme4` package is extremely statistically robust. As a consequence, `lme4` explicitly provides standard errors for 
fixed effects parameters like the “intercept” or overall bias, but does not provide standard errors for random effects. 
Other statistics software may claim to provide standard errors for random effects, but these numbers are likely 
unreliable 
(please see Chapter 1 of [Bates 2018](https://people.math.ethz.ch/~maechler/MEMo-pages/lMMwR.pdf) for more information).
