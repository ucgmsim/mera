/// Errors raised while building or evaluating a model.
///
/// Rust has no exceptions: a function that can fail returns a `Result`, which is either
/// `Ok(value)` or `Err(error)`. This is the error type of every such function in this
/// crate. The Python bindings (`src-rust/lib.rs`) turn `InvalidInput` into a Python
/// `ValueError` and `Factorisation` into a `RuntimeError`.
///
/// The `#[error("...")]` lines give the text of each message, with `{0}` standing for the
/// string that the variant carries.
#[derive(Debug, thiserror::Error)]
pub enum LmmError {
    /// The inputs are inconsistent, not finite, or degenerate.
    #[error("invalid input: {0}")]
    InvalidInput(String),
    /// A Cholesky factorisation failed.
    #[error("Cholesky factorisation failed: {0}")]
    Factorisation(String),
}
