//! Python bindings for `mera_lmm`, exposed as `mera._lmm`.
//!
//! This file is the bridge between Python and Rust, and contains no mathematics. It
//! converts numpy arrays to Rust vectors, calls the `mera_lmm` crate (`crates/mera_lmm`),
//! and converts the results and errors back. Installing mera compiles it into the
//! extension module `mera/_lmm.*.so` (`.pyd` on Windows), which Python imports like any
//! other module. `mera/_lmm.pyi` describes that module to editors and type checkers, and
//! is the place to read what it offers without reading Rust.
//!
//! The lines starting with `#[` are instructions to PyO3, the library that generates the
//! glue code:
//!
//! - `#[pyclass]` turns a Rust struct into a Python class;
//! - `#[pymethods]` makes the functions of the `impl` block below it the methods of that
//!   class, and `#[new]` marks the one that runs as `__init__`;
//! - `#[pymodule]` defines the module itself.

use mera_lmm::{CrossedModel, Design, LmmError};
use numpy::{IntoPyArray, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Converts the engine's errors to Python exceptions: invalid input becomes a
/// `ValueError`, and a failed factorisation a `RuntimeError`.
fn to_py_err(error: LmmError) -> PyErr {
    match error {
        LmmError::InvalidInput(_) => PyValueError::new_err(error.to_string()),
        LmmError::Factorisation(_) => PyRuntimeError::new_err(error.to_string()),
    }
}

/// A linear mixed model with crossed random intercepts, evaluated at any theta.
///
/// `codes` holds one int64 array of level codes (0, 1, ...) per grouping factor, `y` the
/// response and `x` the optional n x p fixed-effects matrix.
//
// `unsendable`: a model must be used on the thread that created it. mera fits each
// intensity measure in one thread (or one process), so this is no restriction.
#[pyclass(module = "mera._lmm", name = "CrossedModel", unsendable)]
struct PyCrossedModel {
    inner: CrossedModel,
}

#[pymethods]
impl PyCrossedModel {
    #[new]
    #[pyo3(signature = (codes, y, x=None, *, reml=true))]
    fn new(
        codes: Vec<PyReadonlyArray1<i64>>,
        y: PyReadonlyArray1<f64>,
        x: Option<PyReadonlyArray2<f64>>,
        reml: bool,
    ) -> PyResult<Self> {
        // Copy each array of codes into a Rust vector, checking that none is negative
        // (the codes are used as indexes).
        let codes = codes
            .iter()
            .map(|c| {
                c.as_array()
                    .iter()
                    .map(|&v| {
                        usize::try_from(v)
                            .map_err(|_| PyValueError::new_err("level codes must be non-negative"))
                    })
                    .collect::<PyResult<Vec<usize>>>()
            })
            .collect::<PyResult<Vec<_>>>()?;
        let y: Vec<f64> = y.as_array().iter().copied().collect();
        // The engine takes x as one flat list, column after column, with p its number of
        // columns. Without x, there are no fixed effects.
        let (x, p) = match &x {
            Some(x) => {
                let x = x.as_array();
                // iterating the transpose visits x in column-major order
                (x.t().iter().copied().collect::<Vec<f64>>(), x.ncols())
            }
            None => (Vec::new(), 0),
        };
        let design = Design::new(&codes, &y, &x, p).map_err(to_py_err)?;
        let inner = CrossedModel::new(design, reml).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    /// The profiled REML criterion (ML: -2 log-likelihood) at `theta`.
    fn deviance(&mut self, theta: Vec<f64>) -> PyResult<f64> {
        self.inner.deviance(&theta).map_err(to_py_err)
    }

    /// Estimates, random effects and conditional SDs at `theta`.
    //
    // `py` is a handle to the Python interpreter, which PyO3 supplies: from Python the
    // call is just `model.fit_at(theta)`.
    fn fit_at<'py>(&mut self, py: Python<'py>, theta: Vec<f64>) -> PyResult<Bound<'py, PyDict>> {
        let fit = self.inner.fit_at(&theta).map_err(to_py_err)?;
        // Return a Python dict, with the lists converted to numpy arrays.
        let out = PyDict::new(py);
        out.set_item("deviance", fit.deviance)?;
        out.set_item("sigma", fit.sigma)?;
        out.set_item("beta", fit.beta.into_pyarray(py))?;
        out.set_item("beta_se", fit.beta_se.into_pyarray(py))?;
        out.set_item("ranef", fit.ranef.into_pyarray(py))?;
        out.set_item("condsd", fit.condsd.into_pyarray(py))?;
        Ok(out)
    }
}

/// The module `mera._lmm`, with its one class.
#[pymodule]
fn _lmm(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyCrossedModel>()?;
    Ok(())
}
