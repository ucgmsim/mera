//! Step 3 of a fit: the criterion that the optimiser minimises, and the results at the
//! optimum.
//!
//! This file holds the statistics. `design.rs` prepared the sums and `factor.rs` supplies
//! the linear algebra; here they are combined into lme4's profiled REML criterion
//! ([`CrossedModel::deviance`]), its slope ([`CrossedModel::deviance_and_gradient`]) and
//! the estimates that mera reports ([`CrossedModel::fit_at`]).
//! `docs/mixed_effects_maths.md`, at the top of the repository, derives the formulas. The
//! summary below is enough to follow the code.
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
//! # The slope of the criterion
//!
//! The search for the best theta is much shorter if it is also told how fast the
//! criterion changes as each theta changes: its slope, or *gradient*. That has a formula
//! too, built from the same pieces and from entries of `A^-1`. It is given, term by term,
//! at [`CrossedModel::deviance_and_gradient`].
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
//! hundreds of times larger than sigma: at a thousand times, about six digits go. The
//! slope of the criterion loses them too, which leaves theta good to about eight digits
//! when the search follows the slope. (A search that only compares values of the
//! criterion does far worse, about four digits, because near its minimum the criterion
//! hardly changes with theta.)
//!
//! `mera.lmm.fit_lmm` removes the first of these causes: before it calls this crate, it
//! subtracts from the response its least-squares fit on the fixed effects (its mean,
//! when the intercept is the only one). It also hands over the fixed effects as
//! orthonormal columns, so that `X'V^-1X` is well conditioned whatever their scales.

use faer::linalg::solvers::{DenseSolveCore, Llt, Solve};
use faer::{Col, Mat, MatRef, Side};

use crate::LmmError;
use crate::design::Design;
use crate::error::invalid;
use crate::factor::{Factor, FactorKind, dense_log_det};

/// `ln(2 pi)`.
const LOG_2PI: f64 = 1.837_877_066_409_345_5;

/// The smallest theta at which the gradient is evaluated: see
/// [`CrossedModel::deviance_and_gradient`].
const THETA_FLOOR: f64 = 1e-7;

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
    /// Covariance matrix of `beta` (lme4's `vcov`), `p x p`, row by row. Its diagonal holds
    /// the squares of `beta_se`; the other entries say how the estimates vary together,
    /// which the standard error of a difference between two of them needs.
    pub beta_cov: Vec<f64>,
    /// Conditional modes (BLUPs) of the random effects, factors stacked in `Design` order:
    /// mera's event terms, followed by its site terms.
    pub ranef: Vec<f64>,
    /// Conditional standard deviations of the random effects, conditional on `beta`.
    pub condsd: Vec<f64>,
}

/// The profiled pieces shared by [`CrossedModel::deviance`],
/// [`CrossedModel::deviance_and_gradient`] and [`CrossedModel::fit_at`].
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

impl Profile {
    /// `u = A^-1 Lambda' Z'(y - X beta)`: the random effects on the common scale on which
    /// each has standard deviation sigma. The random effects themselves are
    /// `b = Lambda u`.
    ///
    /// Column 0 of `W` is `A^-1 Lambda' Z'y` and the other columns are
    /// `A^-1 Lambda' Z'X`, so `u` is column 0 less the other columns times `beta`.
    /// (`subcols(1, p)` is "p columns, starting at column 1": all but column 0.)
    fn u(&self) -> Col<f64> {
        let p = self.beta.nrows();
        self.w.col(0) - self.w.subcols(1, p) * &self.beta
    }
}

/// A linear mixed model with crossed random intercepts that can be evaluated at any theta.
///
/// Python creates one of these per fit (`mera._lmm.CrossedModel`), calls
/// [`CrossedModel::deviance_and_gradient`] at each theta that the optimiser tries while
/// it searches for the best one, and then calls [`CrossedModel::fit_at`] once, at that
/// theta.
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
        Self::with_factor_kind(design, reml, FactorKind::Supernodal)
    }

    /// As [`CrossedModel::new`], with the layout of the factor given. Fits use the
    /// supernodal layout; the tests also check the simplicial one.
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
    /// grouping factor, each zero or positive. The optimiser normally asks for the
    /// gradient too ([`CrossedModel::deviance_and_gradient`]), and calls this only if it
    /// has to fall back on a search that does not use it.
    pub fn deviance(&mut self, theta: &[f64]) -> Result<f64, LmmError> {
        Ok(self.profile(theta)?.deviance)
    }

    /// The criterion at `theta`, and its derivative with respect to the *square* of each
    /// theta: how fast the criterion changes as each `theta_k^2` grows.
    ///
    /// With the gradient, an optimiser knows which way is downhill and roughly how far,
    /// and needs far fewer evaluations than one that only sees values of the criterion.
    ///
    /// # Why theta squared
    ///
    /// The criterion depends on `theta_k` only through its square, so its slope with
    /// respect to `theta_k` itself is always zero at `theta_k = 0`: useless for deciding
    /// whether zero is the best value. The slope with respect to `theta_k^2` is not.
    ///
    /// # The formula
    ///
    /// For factor `k`, summing over its levels `i`:
    ///
    /// ```text
    /// d(deviance) / d(theta_k^2) = sum_i [ (1 - S_ii) - c_i - dof * u_i^2 / r2 ] / theta_k^2
    /// ```
    ///
    /// - `S = A^-1`, and `1 - S_ii` comes from the change of `log|A|`. It is computed as
    ///   `sum_j S_ij (A - I)_ji`, which is the same number without the subtraction, which
    ///   would lose all its digits when `S_ii` is close to 1. Only entries of `S` where
    ///   `A` has entries are needed, and selected inversion provides those.
    /// - `c_i = w_i' (X'V^-1X)^-1 w_i`, with `w_i` row `i` of `A^-1 Lambda' Z'X`, comes
    ///   from the change of `log|X'V^-1X|`. It is left out for ML.
    /// - `u = A^-1 Lambda' Z'(y - X beta)`, and the last term comes from the change of
    ///   the penalised residual sum of squares `r2`. Divided by `theta_k`, `u_i` is the
    ///   weighted residual sum of level `i`; multiplied by it, its random effect.
    ///
    /// The square brackets shrink in proportion to `theta_k^2` as `theta_k` goes to
    /// zero, so the quotient stays finite. To compute it there, the whole evaluation is
    /// made at `theta_k = THETA_FLOOR` in place of any smaller value. That changes the
    /// criterion by less than a part in ten thousand million, and `theta = 0` itself is
    /// still reported exactly by the caller.
    pub fn deviance_and_gradient(&mut self, theta: &[f64]) -> Result<(f64, Vec<f64>), LmmError> {
        // theta is checked here, before the floor is applied: taking the larger of a
        // negative or not-a-number theta and the floor would turn it into the floor,
        // and the check inside `profile` would then have nothing to object to.
        self.check_theta(theta)?;
        // Any theta below THETA_FLOOR is replaced by it, as explained above. (The new
        // list takes the name `theta` from here on.)
        let theta: Vec<f64> = theta.iter().map(|t| t.max(THETA_FLOOR)).collect();
        // The criterion itself, and the pieces that it shares with the gradient: W,
        // beta, r2 and X'V^-1X ("How it is computed", at the top of this file).
        let profile = self.profile(&theta)?;
        let d = &self.design; // a shorter name
        let (q, p) = (d.q(), d.p);
        // bracket[k]: the sum of the square brackets over the levels of factor k
        let mut bracket = vec![0.0; d.n_factors()];

        // ---- sum_j S_ij (A - I)_ji ----
        // `profile` has just factorised A at this theta. For each stored entry of the
        // lower triangle of A, (A - I) is theta_row * theta_column * Z'Z there. An entry
        // on the diagonal belongs to one level; an entry below it stands for two entries
        // of the symmetric matrix, (row, c) and (c, row), one for each of its two levels.
        let s = self.factor.inverse_on_pattern();
        for c in 0..q {
            let factor_c = d.level_factor[c];
            // Walk down column c: its entries of S, their rows, and Z'Z there. `zip`
            // pairs two lists, so three of them give a pair inside a pair, which the
            // `for` line below takes apart again. (A range is used up by slicing with
            // it, so `.clone()` copies it for the first two slices.)
            let entries = d.col_ptr[c]..d.col_ptr[c + 1];
            let column = s[entries.clone()]
                .iter()
                .zip(&d.row_idx[entries.clone()])
                .zip(&d.ztz[entries]);
            for ((&s_entry, &row), &ztz) in column {
                let factor_row = d.level_factor[row];
                let term = s_entry * theta[factor_c] * theta[factor_row] * ztz;
                bracket[factor_c] += term;
                if row != c {
                    bracket[factor_row] += term;
                }
            }
        }

        // ---- c_i and dof * u_i^2 / r2 ----
        let u = profile.u();
        // n - p for REML, n for ML
        let dof = self.degrees_of_freedom();
        // c_i = w_i' (X'V^-1X)^-1 w_i, where w_i is row i of `wx`: the columns of W that
        // belong to X. Row i of `solved` is w_i' (X'V^-1X)^-1, so c_i is the sum of the
        // products of that row with w_i, entry by entry. (In numpy:
        // `solved = wx @ xvx_inverse` and `c = (solved * wx).sum(axis=1)`.)
        let wx = profile.w.subcols(1, p);
        let solved = wx * profile.xvx.inverse();
        for i in 0..q {
            // c_i is left out for ML, whose criterion has no log|X'V^-1X| term.
            let c = if self.reml {
                (0..p).map(|a| solved[(i, a)] * wx[(i, a)]).sum::<f64>()
            } else {
                0.0
            };
            bracket[d.level_factor[i]] -= c + dof * u[i] * u[i] / profile.r2;
        }
        // Divide the sum of each factor by its theta squared, as in the formula.
        let gradient = bracket
            .iter()
            .zip(&theta)
            .map(|(sum, t)| sum / (t * t))
            .collect();
        Ok((profile.deviance, gradient))
    }

    /// Checks that `theta` has one finite, non-negative value per grouping factor.
    fn check_theta(&self, theta: &[f64]) -> Result<(), LmmError> {
        if theta.len() != self.design.n_factors() {
            return Err(invalid(format!(
                "theta has {} values but the model has {} grouping factors",
                theta.len(),
                self.design.n_factors()
            )));
        }
        if theta.iter().any(|t| !t.is_finite() || *t < 0.0) {
            return Err(invalid(format!(
                "theta must be finite and non-negative, got {theta:?}"
            )));
        }
        Ok(())
    }

    /// Everything mera reports, at `theta`.
    pub fn fit_at(&mut self, theta: &[f64]) -> Result<Fit, LmmError> {
        let profile = self.profile(theta)?;
        // `profile` has just factorised A at this theta, so the factor is the right one.
        let inverse_diagonal = self.factor.inverse_diagonal();
        let p = self.design.p;
        let sigma2 = profile.r2 / self.degrees_of_freedom();
        let sigma = sigma2.sqrt();
        let lambda = &profile.lambda;
        // The random effects: b = Lambda u, each u times the theta of its level's factor.
        let ranef = lambda.as_diagonal() * profile.u();
        // vcov(beta) = sigma^2 (X'V^-1X)^-1
        let xvx_inverse = profile.xvx.inverse();
        let beta_se = (0..p)
            .map(|a| (sigma2 * xvx_inverse[(a, a)]).sqrt())
            .collect();
        // every pair (a, b) of fixed effects, row by row
        let beta_cov = (0..p)
            .flat_map(|a| (0..p).map(move |b| (a, b)))
            .map(|(a, b)| sigma2 * xvx_inverse[(a, b)])
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
            beta_cov,
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
    /// [`CrossedModel::deviance_and_gradient`] and [`CrossedModel::fit_at`] reuse. The
    /// module documentation at the top of this file gives the formulas.
    fn profile(&mut self, theta: &[f64]) -> Result<Profile, LmmError> {
        self.check_theta(theta)?;
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
            return Err(invalid(
                "the residual sum of squares is zero: either the response is constant, or \
                 the fixed effects and grouping factors explain it exactly",
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
    use crate::testing::{Lcg, two_factor_codes};
    use approx::{assert_abs_diff_eq, assert_relative_eq};
    use proptest::prelude::*;

    /// Checks that two models of the same data, one in each layout of the factor, give
    /// the same criterion, slope and estimates at `theta`: numbers that are compared as
    /// a ratio to within `relative`, the others to within `absolute`.
    fn check_layouts_agree(
        models: &mut [CrossedModel; 2],
        theta: &[f64],
        relative: f64,
        absolute: f64,
    ) {
        let [(value, gradient), (expected_value, expected_gradient)] = models
            .each_mut()
            .map(|m| m.deviance_and_gradient(theta).unwrap());
        assert_relative_eq!(value, expected_value, max_relative = relative);
        assert_abs_diff_eq!(gradient[..], expected_gradient[..], epsilon = absolute);
        let [fit, expected] = models.each_mut().map(|m| m.fit_at(theta).unwrap());
        assert_relative_eq!(fit.deviance, expected.deviance, max_relative = relative);
        assert_relative_eq!(fit.sigma, expected.sigma, max_relative = relative);
        for (values, expected_values) in [
            (&fit.beta, &expected.beta),
            (&fit.beta_se, &expected.beta_se),
            (&fit.beta_cov, &expected.beta_cov),
            (&fit.ranef, &expected.ranef),
            (&fit.condsd, &expected.condsd),
        ] {
            assert_abs_diff_eq!(values[..], expected_values[..], epsilon = absolute);
        }
    }

    #[test]
    fn supernodal_layout_gives_the_same_fits() {
        // Fits use the supernodal layout, and faer falls back on the simplicial one when
        // L is diagonal. The fits, conditional SDs included, must not depend on which.
        // the same data in both layouts: `random_model` always makes the same design
        let mut models = [FactorKind::Simplicial, FactorKind::Supernodal]
            .map(|kind| random_model(1, true, kind));
        assert!(models[0].factor.is_simplicial() && !models[1].factor.is_simplicial());
        for theta in [[0.8, 1.3], [0.2, 0.6]] {
            check_layouts_agree(&mut models, &theta, 1e-12, 1e-10);
        }
    }

    proptest! {
        // A property test (see `testing.rs`): at each run of the tests, 64 new random
        // designs. (proptest's usual 256 would take several seconds in a debug build.)
        #![proptest_config(ProptestConfig::with_cases(64))]

        /// Whatever the design, theta, number of fixed effects and criterion (REML or
        /// ML), the two layouts of the factor give the same results. `seed` sets the
        /// random response and covariate.
        #[test]
        fn both_layouts_give_the_same_results_for_any_design(
            (events, sites) in two_factor_codes(40, 25, 20..=300),
            theta in [0.0f64..4.0, 0.0f64..4.0],
            p in 0usize..=2,
            reml in any::<bool>(),
            seed in any::<u64>(),
        ) {
            let mut models = [FactorKind::Simplicial, FactorKind::Supernodal].map(|kind| {
                let (events, sites) = (events.clone(), sites.clone());
                model_of(events, sites, p, reml, kind, &mut Lcg::new(seed))
            });
            check_layouts_agree(&mut models, &theta, 1e-9, 1e-7);
        }
    }

    /// A model with two grouping factors and `p` fixed effects (an intercept, then a
    /// covariate), on random data.
    fn random_model(p: usize, reml: bool, kind: FactorKind) -> CrossedModel {
        let mut rng = Lcg::new(11);
        let n = 240;
        let events = (0..n).map(|_| rng.below(30)).collect();
        let sites = (0..n).map(|_| rng.below(18)).collect();
        model_of(events, sites, p, reml, kind, &mut rng)
    }

    /// A model with the given event and site of each record and `p` fixed effects (an
    /// intercept, then a covariate). The response and the covariate are drawn from `rng`.
    fn model_of(
        events: Vec<usize>,
        sites: Vec<usize>,
        p: usize,
        reml: bool,
        kind: FactorKind,
        rng: &mut Lcg,
    ) -> CrossedModel {
        let n = events.len();
        let y: Vec<f64> = (0..n).map(|_| rng.normal()).collect();
        let covariate: Vec<f64> = (0..n).map(|_| rng.normal()).collect();
        // x is column-major: the column of ones, then the covariate
        let x: Vec<f64> = std::iter::repeat_n(1.0, n)
            .chain(covariate)
            .take(n * p)
            .collect();
        let design = Design::from_full_codes(&[events, sites], &y, &x, p).unwrap();
        CrossedModel::with_factor_kind(design, reml, kind).unwrap()
    }

    #[test]
    fn the_gradient_matches_differences_of_the_criterion() {
        // The gradient is with respect to theta squared, s. A central difference of the
        // criterion in s_k, with the other factor's theta fixed, must agree with it.
        for (p, reml) in [(0, true), (1, true), (2, true), (2, false)] {
            for kind in [FactorKind::Simplicial, FactorKind::Supernodal] {
                let mut model = random_model(p, reml, kind);
                for theta in [[0.8, 1.3], [0.05, 2.0], [3.0, 0.4]] {
                    let (value, gradient) = model.deviance_and_gradient(&theta).unwrap();
                    assert_relative_eq!(
                        value,
                        model.deviance(&theta).unwrap(),
                        max_relative = 1e-12
                    );
                    for k in 0..2 {
                        let s = theta[k] * theta[k];
                        let h = 1e-5 * s;
                        let mut at = |s_k: f64| {
                            let mut shifted = theta;
                            shifted[k] = s_k.sqrt();
                            model.deviance(&shifted).unwrap()
                        };
                        let difference = (at(s + h) - at(s - h)) / (2.0 * h);
                        assert_relative_eq!(
                            gradient[k],
                            difference,
                            max_relative = 1e-5,
                            epsilon = 1e-6
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn the_gradient_is_finite_where_theta_is_zero() {
        // With respect to theta the slope is always zero at theta = 0. With respect to
        // theta squared it is not, and it must match a one-sided difference there.
        let mut model = random_model(1, true, FactorKind::Simplicial);
        let (_, gradient) = model.deviance_and_gradient(&[0.0, 0.7]).unwrap();
        let h: f64 = 1e-6;
        let at_zero = model.deviance(&[0.0, 0.7]).unwrap();
        let difference = (model.deviance(&[h.sqrt(), 0.7]).unwrap() - at_zero) / h;
        assert!(gradient[0].is_finite() && gradient[0] != 0.0);
        assert_relative_eq!(gradient[0], difference, max_relative = 1e-3);
        // invalid theta is still rejected
        assert!(model.deviance_and_gradient(&[-1.0, 0.7]).is_err());
        assert!(model.deviance_and_gradient(&[f64::NAN, 0.7]).is_err());
    }

    #[test]
    fn the_standard_errors_are_the_diagonal_of_the_covariance() {
        // Two fixed effects: an intercept and a covariate.
        let mut rng = Lcg::new(3);
        let n = 120;
        let events = (0..n).map(|_| rng.below(15)).collect();
        let y: Vec<f64> = (0..n).map(|_| rng.normal()).collect();
        let covariate = (0..n).map(|_| rng.normal());
        // x is column-major: the column of ones, then the covariate
        let x: Vec<f64> = std::iter::repeat_n(1.0, n).chain(covariate).collect();
        let design = Design::from_full_codes(&[events], &y, &x, 2).unwrap();
        let fit = CrossedModel::new(design, true)
            .unwrap()
            .fit_at(&[0.7])
            .unwrap();
        // beta_cov is 2 x 2, row by row: [var(b0), cov, cov, var(b1)]
        assert_eq!(fit.beta_cov.len(), 4);
        assert_relative_eq!(fit.beta_cov[1], fit.beta_cov[2], max_relative = 1e-12);
        assert!(fit.beta_cov[1] != 0.0);
        for a in 0..2 {
            assert_relative_eq!(
                fit.beta_cov[3 * a].sqrt(),
                fit.beta_se[a],
                max_relative = 1e-12
            );
        }
    }
}
