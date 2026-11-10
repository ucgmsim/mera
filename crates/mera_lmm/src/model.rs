//! Step 3 of a fit: the criterion that the optimiser minimises, and the results at the
//! optimum.
//!
//! This file holds the statistics. `design.rs` prepared the sums and `factor.rs` supplies
//! the linear algebra; here they are combined into lme4's profiled REML criterion
//! ([`CrossedModel::deviance`]) and into the estimates that mera reports
//! ([`CrossedModel::fit_at`]). `docs/mixed_effects_maths.md`, at the top of the repository,
//! derives the formulas. The summary below is enough to follow the code.
//!
//! # The model
//!
//! ```text
//!     y = X beta + Z b + e
//! ```
//!
//! - `y`: the n observations (the residuals of one intensity measure);
//! - `X beta`: the fixed effects. For mera, `X` is a column of ones and `beta` is the bias;
//! - `Z b`: the random effects. `b` holds one term per level (event or site), and `Z`
//!   picks out the terms of each observation (see `design.rs`). The terms of factor `k`
//!   are normally distributed with mean 0 and standard deviation `sd_k`;
//! - `e`: what is left over, normally distributed with mean 0 and standard deviation
//!   `sigma`.
//!
//! # theta, and why only theta is searched for
//!
//! Following lme4, the standard deviations of the random effects are measured relative to
//! sigma: `theta_k = sd_k / sigma`. Write `b = Lambda u`, where `Lambda` is the diagonal
//! matrix holding each level's theta and every element of `u` has standard deviation
//! `sigma`. Then the covariance matrix of `y` is `sigma^2 V`, with
//! `V = I + Z Lambda Lambda' Z'`.
//!
//! For any given theta, the best `beta` and `sigma` have formulas (below). So the search
//! for the best fit is over theta alone, which is two numbers for events and sites, and
//! the function to minimise is the *profiled* criterion:
//!
//! ```text
//!     deviance(theta) = log|A| + log|X'V^-1X| + (n - p) * (1 + log(2 pi r2 / (n - p)))
//! ```
//!
//! where `A = Lambda' Z'Z Lambda + I`, `p` is the number of fixed effects and `r2` is the
//! penalised residual sum of squares defined below. Smaller is better. This is the REML
//! criterion;
//! for maximum likelihood (ML) the middle term is dropped and `n - p` becomes `n`.
//!
//! # How it is computed
//!
//! Everything comes from one small matrix, with `1 + p` rows and columns (2 x 2 when a
//! bias is fitted):
//!
//! ```text
//!     M = [y X]' V^-1 [y X] = [ y'V^-1y   y'V^-1X ]
//!                             [ X'V^-1y   X'V^-1X ]
//! ```
//!
//! `V` has a row and a column for every observation, and is never built. The Woodbury
//! matrix identity gives `V^-1 = I - Z Lambda A^-1 Lambda' Z'`, so that
//!
//! ```text
//!     M = [y X]'[y X] - R' A^-1 R        with R = Lambda' Z'[y X]
//! ```
//!
//! `[y X]'[y X]` and `Z'[y X]` were computed once, in `design.rs`, and `A^-1 R` is a solve
//! with the Cholesky factor of `A`, from `factor.rs`. Then
//!
//! ```text
//!     beta    = (X'V^-1X)^-1 X'V^-1y           generalised least squares
//!     r2      = y'V^-1y - (X'V^-1y)' beta      what is left of y'V^-1y once beta is fitted
//!     sigma^2 = r2 / (n - p)
//! ```
//!
//! `r2` equals the sum of the squares of the residuals `e` plus the sum of the squares of
//! `u`, which is why lme4 calls it the *penalised* residual sum of squares.
//!
//! # Names in the code
//!
//! ```text
//!     values        the entries of A, on the sparsity pattern of Z'Z
//!     lambda        the diagonal of Lambda: the theta of each level's factor
//!     r             R = Lambda' Z'[y X]
//!     w             W = A^-1 R
//!     m             M
//!     xvx           the Cholesky factorisation of X'V^-1X (the lower-right block of M)
//!     xvy           X'V^-1y (the first column of M, below its top entry)
//!     r2            the penalised residual sum of squares
//!     dof           the degrees of freedom: n - p for REML, n for ML
//! ```
//!
//! # A limit on precision
//!
//! `r2` is found by subtraction: the total sum of squares less what the model explains.
//! When the model explains nearly everything, the two are nearly equal and digits are
//! lost: of the 16 or so significant digits of a floating-point number, about
//! `log10(y'y / r2)` go. For ground-motion residuals that is about one digit. It starts
//! to matter when the mean of the response, or a random-effect standard deviation, is
//! hundreds of times larger than sigma: at a thousand times, theta is good to about four
//! digits. `mera.lmm.fit_lmm` removes the first of these causes: when an intercept is
//! fitted, it subtracts the mean of the response before it calls this crate.

use faer::linalg::solvers::{DenseSolveCore, Llt, Solve};
use faer::{Col, Mat, MatRef, Side};

use crate::LmmError;
use crate::design::Design;
use crate::factor::{Factor, FactorKind, dense_log_det};

/// `ln(2 pi)`.
const LOG_2PI: f64 = 1.837_877_066_409_345_5;

/// The model evaluated at one `theta`, normally the optimum.
#[derive(Debug, Clone)]
pub struct Fit {
    /// REML criterion (or -2 log-likelihood for ML), lme4's definition.
    pub deviance: f64,
    /// Residual standard deviation (mera's `phi_w`).
    pub sigma: f64,
    /// Fixed-effect estimates (mera's bias, when one is fitted).
    pub beta: Vec<f64>,
    /// Standard errors of `beta`, `sqrt(diag(vcov))`.
    pub beta_se: Vec<f64>,
    /// Conditional modes (BLUPs) of the random effects, factors stacked in `Design` order:
    /// mera's event terms, followed by its site terms.
    pub ranef: Vec<f64>,
    /// Conditional standard deviations of the random effects, conditional on `beta`.
    pub condsd: Vec<f64>,
}

/// The profiled pieces shared by [`CrossedModel::deviance`] and [`CrossedModel::fit_at`].
struct Profile {
    deviance: f64,
    beta: Col<f64>,
    /// The diagonal of `Lambda`: each level's theta.
    lambda: Col<f64>,
    /// Cholesky factorisation of `X'V^-1X`.
    xvx: Llt<f64>,
    r2: f64,
    /// `A^-1 Lambda' Z' [y X]`.
    w: Mat<f64>,
}

/// A linear mixed model with crossed random intercepts that can be evaluated at any theta.
///
/// Python creates one of these per fit (`mera._lmm.CrossedModel`), calls
/// [`CrossedModel::deviance`] many times while the optimiser searches for the best theta,
/// and then calls [`CrossedModel::fit_at`] once, at that theta.
pub struct CrossedModel {
    design: Design,
    reml: bool,
    /// Values of `A = Lambda' Z'Z Lambda + I` on the pattern of `design`.
    values: Vec<f64>,
    factor: Factor,
}

impl CrossedModel {
    /// Analyses the sparsity pattern once; `reml` selects REML (true) or ML.
    pub fn new(design: Design, reml: bool) -> Result<Self, LmmError> {
        Self::with_factor_kind(design, reml, FactorKind::Auto)
    }

    /// As [`CrossedModel::new`], with the layout of the factor given rather than left to
    /// faer. The tests force the supernodal layout, which faer picks only for designs
    /// larger than theirs.
    fn with_factor_kind(design: Design, reml: bool, kind: FactorKind) -> Result<Self, LmmError> {
        // `1 + design.p` is the number of columns of [y X], which are solved for together.
        let factor = Factor::new(&design.col_ptr, &design.row_idx, 1 + design.p, kind)?;
        let values = vec![0.0; design.ztz.len()];
        Ok(Self {
            design,
            reml,
            values,
            factor,
        })
    }

    /// The profiled REML criterion (ML: -2 log-likelihood) at `theta`.
    ///
    /// This is the function that the optimiser minimises. `theta` has one value per
    /// grouping factor, each zero or positive.
    pub fn deviance(&mut self, theta: &[f64]) -> Result<f64, LmmError> {
        Ok(self.profile(theta)?.deviance)
    }

    /// Everything mera reports, at `theta`.
    pub fn fit_at(&mut self, theta: &[f64]) -> Result<Fit, LmmError> {
        let profile = self.profile(theta)?;
        // `profile` has just factorised A at this theta, so the factor is the right one.
        let inverse_diagonal = self.factor.inverse_diagonal();
        let p = self.design.p;
        let sigma2 = profile.r2 / self.degrees_of_freedom();
        let sigma = sigma2.sqrt();
        let (w, lambda) = (&profile.w, &profile.lambda);
        // The random effects. Column 0 of W is A^-1 Lambda' Z'y and the other columns are
        // A^-1 Lambda' Z'X, so this line is
        // u = A^-1 Lambda' Z'(y - X beta);  b = Lambda u
        // (`subcols(1, p)` is "p columns, starting at column 1": all but column 0.)
        let ranef = lambda.as_diagonal() * (w.col(0) - w.subcols(1, p) * &profile.beta);
        // vcov(beta) = sigma^2 (X'V^-1X)^-1
        let xvx_inverse = profile.xvx.inverse();
        let beta_se = (0..p)
            .map(|a| (sigma2 * xvx_inverse[(a, a)]).sqrt())
            .collect();
        // var(b | y, beta) = sigma^2 Lambda A^-1 Lambda'
        // Only its diagonal is needed: the variance of level i is
        // sigma^2 * lambda_i^2 * (A^-1)[i, i], and condsd is its square root.
        let condsd = lambda
            .iter()
            .zip(inverse_diagonal)
            .map(|(lambda_i, a_ii)| lambda_i * sigma * a_ii.sqrt())
            .collect();
        Ok(Fit {
            deviance: profile.deviance,
            sigma,
            beta: profile.beta.iter().copied().collect(),
            beta_se,
            ranef: ranef.iter().copied().collect(),
            condsd,
        })
    }

    /// What `r2` is divided by to estimate `sigma^2`.
    ///
    /// REML uses `n - p` rather than `n` because estimating the `p` fixed effects uses up
    /// `p` degrees of freedom, for the same reason that a sample variance divides by
    /// `n - 1`.
    fn degrees_of_freedom(&self) -> f64 {
        if self.reml {
            (self.design.n - self.design.p) as f64
        } else {
            self.design.n as f64
        }
    }

    /// Evaluates the model at `theta`: the criterion, and the pieces of it that
    /// [`CrossedModel::fit_at`] reuses. The module documentation at the top of this file
    /// gives the formulas.
    fn profile(&mut self, theta: &[f64]) -> Result<Profile, LmmError> {
        if theta.len() != self.design.n_factors() {
            return Err(LmmError::InvalidInput(format!(
                "theta has {} values but the model has {} grouping factors",
                theta.len(),
                self.design.n_factors()
            )));
        }
        if theta.iter().any(|t| !t.is_finite() || *t < 0.0) {
            return Err(LmmError::InvalidInput(format!(
                "theta must be finite and non-negative, got {theta:?}"
            )));
        }
        // ---- A at this theta, its Cholesky factor and log|A| ----
        let d = &self.design;
        d.fill_a(theta, &mut self.values);
        self.factor
            .factorize(&d.col_ptr, &d.row_idx, &self.values)?;
        let log_det_a = self.factor.log_det();

        // ---- M = [y X]' V^-1 [y X] ----
        // q levels, p fixed effects, and nc columns in [y X].
        let (q, p, nc) = (d.q(), d.p, 1 + d.p);
        let lambda = Col::<f64>::from_fn(q, |i| theta[d.level_factor[i]]);
        // R = Lambda' Z'[y X]: each row of Z'[y X] scaled by its level's theta.
        let r = lambda.as_diagonal() * MatRef::from_column_major_slice(&d.zt_yx, q, nc);
        // W = A^-1 R
        let mut w = r.clone();
        self.factor.solve_in_place(w.as_mut());
        // m = [y X]' V^-1 [y X] = [y X]'[y X] - R' A^-1 R, with V = I + Z Lambda Lambda' Z'
        let m = MatRef::from_column_major_slice(&d.yx_yx, nc, nc) - r.transpose() * &w;

        // ---- beta, the penalised residual sum of squares and the criterion ----
        // beta = (X'V^-1X)^-1 X'V^-1y; with no fixed effects all of these are empty.
        // `submatrix(1, 1, p, p)` is the block of p rows and p columns that starts at
        // row 1 and column 1, X'V^-1X, and `.llt(...)` is its dense Cholesky
        // factorisation. `subrows(1, p)` is likewise "p rows, starting at row 1".
        let xvx = m.submatrix(1, 1, p, p).llt(Side::Lower).map_err(|_| {
            LmmError::Factorisation(
                "X'V^-1X is not positive definite: either the fixed effects are collinear, \
                 or the grouping factors explain the response exactly, leaving no residual \
                 variation"
                    .into(),
            )
        })?;
        let xvy = m.col(0).subrows(1, p);
        let beta = xvx.solve(xvy);
        let r2 = m[(0, 0)] - xvy.transpose() * &beta;
        let log_det_xvx = dense_log_det(&xvx);
        // A (numerically) zero penalised residual sum of squares leaves nothing to
        // estimate sigma from, and its logarithm below would be minus infinity.
        if r2.is_nan() || r2 <= 64.0 * f64::EPSILON * m[(0, 0)].abs() {
            return Err(LmmError::InvalidInput(
                "the residual sum of squares is zero: either the response is constant, or \
                 the fixed effects and grouping factors explain it exactly"
                    .into(),
            ));
        }
        let dof = self.degrees_of_freedom();
        let deviance = log_det_a
            + if self.reml { log_det_xvx } else { 0.0 }
            + dof * (1.0 + LOG_2PI + (r2 / dof).ln());
        Ok(Profile {
            deviance,
            lambda,
            beta,
            xvx,
            r2,
            w,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::{Lcg, assert_close};

    #[test]
    fn supernodal_layout_gives_the_same_fits() {
        // faer picks the supernodal layout only for designs larger than this one; the fits,
        // conditional SDs included, must not depend on the layout.
        let mut rng = Lcg::new(7);
        let n = 300;
        let events = (0..n).map(|_| rng.below(40)).collect();
        let sites = (0..n).map(|_| rng.below(25)).collect();
        let y: Vec<f64> = (0..n).map(|_| rng.normal()).collect();
        let design = Design::new(&[events, sites], &y, &vec![1.0; n], 1).unwrap();
        let mut models = [FactorKind::Simplicial, FactorKind::Supernodal]
            .map(|kind| CrossedModel::with_factor_kind(design.clone(), true, kind).unwrap());
        assert!(models[0].factor.is_simplicial() && !models[1].factor.is_simplicial());
        for theta in [[0.8, 1.3], [0.2, 0.6]] {
            let [expected, fit] = models.each_mut().map(|m| m.fit_at(&theta).unwrap());
            assert_close(fit.deviance, expected.deviance, 1e-12, "deviance");
            assert_close(fit.sigma, expected.sigma, 1e-12, "sigma");
            for (what, values, expected_values) in [
                ("beta", &fit.beta, &expected.beta),
                ("beta_se", &fit.beta_se, &expected.beta_se),
                ("ranef", &fit.ranef, &expected.ranef),
                ("condsd", &fit.condsd, &expected.condsd),
            ] {
                for (&a, &b) in values.iter().zip(expected_values) {
                    assert_close(a, b, 1e-10, what);
                }
            }
        }
    }
}
