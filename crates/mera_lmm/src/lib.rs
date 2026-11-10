//! The numerical core of mera: fits a linear mixed model whose random effects are crossed
//! random intercepts, such as `residual ~ 1 + (1|event) + (1|site)`.
//!
//! # If you do not know Rust
//!
//! `README.md` in this crate's directory explains the Rust syntax used here in terms of
//! Python. The mathematics is derived step by step in `docs/mixed_effects_maths.md`, at
//! the top of the repository, and `mera/example/maths_walkthrough.py` does the same
//! calculation in numpy, with the same names.
//!
//! # What this crate does
//!
//! The formulation is lme4's (Bates et al. 2015, *Journal of Statistical Software*
//! 67(1)). Each observation is the sum of the fixed effects (for mera, the bias), one
//! random effect per grouping factor (the term of its event and the term of its site) and
//! a residual. In matrix form this is `y = X beta + Z b + e`, and the top of `model.rs`
//! explains each symbol. A fit is described by `theta`, which has one value per grouping
//! factor: the standard deviation of that factor's random effects divided by `sigma`,
//! the standard deviation of the residual.
//!
//! For a given `theta`, this crate computes the REML criterion: a number that measures how
//! well that `theta` fits the data, smaller being better. Python's `mera.lmm.fit_lmm`
//! searches for the `theta` with the smallest criterion, then asks this crate for the
//! estimates at that `theta`. The search itself is not in this crate.
//!
//! # Where to find things
//!
//! The files are the three steps of an evaluation, in order:
//!
//! 1. `design.rs`: sums over the data that do not depend on `theta`, computed once.
//! 2. `factor.rs`: the sparse Cholesky factorisation of `A = Lambda' Z'Z Lambda + I`, and
//!    the log-determinant, solves and diagonal of `A^-1` that are read from it.
//! 3. `model.rs`: the criterion and the estimates ([`CrossedModel`]).
//!
//! `error.rs` defines the errors and `testing.rs` has helpers for the tests. The Python
//! bindings are a separate, small crate: `src-rust/lib.rs`, at the top of the repository.
//!
//! # Why it copes with large datasets
//!
//! - For a given `theta`, `beta` and `sigma` have formulas (they are *profiled out*), so
//!   only `theta` has to be searched for: two numbers for events and sites.
//! - The data are reduced to sums per level once, so an evaluation does not go through the
//!   records again.
//! - `A` is factorised with a sparse Cholesky whose symbolic analysis is computed once.
//! - The conditional variances need `diag(A^-1)`, which comes from a selected inversion,
//!   so `A^-1` is never formed.

// The modules (files) of this crate. `pub` ones can be used from outside the crate.
pub mod design;
mod error;
pub mod factor;
pub mod model;
// Only compiled for `cargo test`.
#[cfg(test)]
mod testing;

// Re-exported so that users can write `mera_lmm::CrossedModel` instead of
// `mera_lmm::model::CrossedModel`.
pub use design::Design;
pub use error::LmmError;
pub use model::{CrossedModel, Fit};
