# mera_lmm: the numerical engine

This crate ("crate" is Rust's word for a package) is the part of MERA that does the heavy
arithmetic. Python hands it the data of one fit; it evaluates the REML criterion for any
trial value of θ, and returns the estimates at the best one. The search for the best θ
stays in Python (`mera/lmm.py`).

This README is for readers who know some Python and no Rust. It covers:

- [why this part is in Rust](#why-rust);
- [a route through the files](#a-route-through-the-files);
- [how to read the Rust](#reading-rust-if-you-know-python), construct by construct;
- [how to build and test it](#building-and-testing).

The mathematics is explained in
[`docs/mixed_effects_maths.md`](../../docs/mixed_effects_maths.md). If you only want to
understand *what* is computed,
[`mera/example/maths_walkthrough.py`](../../mera/example/maths_walkthrough.py) does the
same calculation in numpy with the same variable names, and is much shorter. This crate
adds one thing to it: sparse matrices, so that a million records are feasible.

## Why Rust

A fit needs the Cholesky factorisation of a large sparse matrix, repeated for every trial
θ, and then a "selected inversion" of it. The usual Python route to sparse factorisations
(CHOLMOD, through scikit-sparse) cannot be installed with `pip` alone and has no selected
inversion. Rust, with the [faer](https://crates.io/crates/faer) linear-algebra library,
compiles into the wheel that `pip install ucgmsim-mera` downloads, so users need neither
R nor a compiler.

## A route through the files

Read them in this order. Each starts with a long comment that explains its step.

| File | What it does | Read it for |
|---|---|---|
| `src/lib.rs` | lists the other files and what the crate is for | orientation |
| `src/design.rs` | reduces the data to counts and sums, once per fit | how the data are stored; a worked example with five records |
| `src/factor.rs` | the sparse Cholesky factorisation, and what is read from it | the linear algebra |
| `src/model.rs` | the REML criterion and the estimates | the statistics; start here if short of time |
| `src/error.rs` | the two kinds of error | |
| `src/testing.rs` | helpers used only by the tests | |
| `../../src-rust/lib.rs` | the bridge to Python: `mera._lmm` | how numpy arrays reach Rust |

One evaluation of the criterion runs through them like this:

```
Python: model.deviance(theta)
  └─ src-rust/lib.rs      PyCrossedModel::deviance     converts the arguments
      └─ model.rs         CrossedModel::deviance, which calls CrossedModel::profile
          ├─ design.rs    Design::fill_a               A for this theta
          ├─ factor.rs    Factor::factorize            A = L L'
          ├─ factor.rs    Factor::log_det              log|A|
          ├─ factor.rs    Factor::solve_in_place       A^-1 R
          └─ model.rs     the rest of profile          beta, r2 and the criterion
```

`CrossedModel::fit_at` runs the same steps once more at the best θ, and then
`Factor::inverse_diagonal` for the conditional standard deviations.

## Reading Rust if you know Python

Rust looks busier than Python because every value has a fixed type, written out or worked
out by the compiler, and the compiler checks how data are shared. The logic underneath is
ordinary loops and arithmetic. These are the constructs that appear in this crate and in
the bridge to Python:

### Values and types

| Rust | Python equivalent, or meaning |
|---|---|
| `let n = y.len();` | `n = len(y)`. The value cannot be changed afterwards |
| `let mut sum = 0.0;` | the same, but `mut` allows it to be changed later |
| `f64` | `float` |
| `usize` | a non-negative `int`, used for sizes and positions |
| `bool`, `true` | `bool`, `True` |
| `Vec<f64>` | a list of floats (like a one-dimensional numpy array) |
| `vec![0.0; n]` | `[0.0] * n` |
| `&[f64]` | a read-only view of a list of floats that belongs to someone else |
| `&mut [f64]` | a view through which the list may be changed |
| `(usize, usize)` | a tuple of two integers |
| `pair.0`, `pair.1` | `pair[0]`, `pair[1]`: the elements of a tuple |
| `0usize`, `0.0f64` | the numbers 0 and 0.0 with their types attached |
| `()` | "nothing", like `None` as a return value |
| `x as f64` | `float(x)` |
| `a..b` | `range(a, b)`; in `values[a..b]`, the slice `values[a:b]` |
| `Option<T>`: `Some(x)` or `None` | a value that may be missing, like `x` or `None` |
| `if let Some(x) = maybe { ... }` | `if maybe is not None:`, with its value called `x` inside the braces |
| `maybe.map_or(0, \|m\| m + 1)` | `0 if maybe is None else maybe + 1` |
| `v.push(x)`, `v.extend(other)` | `v.append(x)`, `v.extend(other)` |

Matrices are stored as flat lists in *column-major* order: all of column 0, then all of
column 1. Entry (r, c) of a matrix with `n` rows is at position `r + n * c`.

### Functions, structs and methods

```rust
pub fn q(&self) -> usize {
    self.level_factor.len()
}
```

- `fn` is `def`. Argument types follow a colon, and the return type follows `->`.
- The last expression of a function, written without a semicolon, is its return value.
  `return` is only used to leave early.
- `pub` makes something usable from outside its file. `pub(crate)` limits that to this
  crate.
- A `struct` is a class's data, and the `impl` block that follows holds its methods.
  `&self` is Python's `self`; `&mut self` is a method that changes the object.
- `Self` is the struct's own type, and `Self { n, p, ... }` creates one.
- `Design::new(...)` calls a function that belongs to `Design` (like a class method).
  By convention `new` is the constructor.
- `enum` defines a type that is one of a few named alternatives, and `match` chooses
  what to do for each, like `if`/`elif` on which one it is. An alternative can carry a
  value: the arm `SymbolicCholeskyRaw::Simplicial(s) => ...` reads "if it is the
  simplicial alternative, call what it carries `s` and do this".
- `<'_>` and `<'py>` in a type, as in `MatMut<'_, f64>`, are *lifetimes*: notes for the
  compiler about how long a view stays valid. They can be ignored when reading.

### Errors

Rust has no exceptions. A function that can fail returns `Result<T, LmmError>`, which is
either `Ok(value)` or `Err(error)`.

| Rust | Meaning |
|---|---|
| `return Err(invalid("..."));` | like `raise ValueError("...")` |
| `Ok(Self { ... })` | return normally |
| `Ok(())` | return normally, with no value |
| `something()?` | if `something()` failed, return its error from this function now; otherwise carry on with its value. This is how errors travel up, like an uncaught exception |
| `.map_err(\|e\| ...)` | replace the error with a more helpful one |
| `.unwrap()` | "this cannot fail; crash if it does". Only in tests |

`src-rust/lib.rs` turns an `Err` into a Python exception.

### Loops and iterators

| Rust | Python equivalent |
|---|---|
| `for r in 0..n { ... }` | `for r in range(n):` |
| `for (i, &row) in rows.iter().enumerate()` | `for i, row in enumerate(rows):` |
| `a.iter().zip(b)` | `zip(a, b)` |
| `\|x\| x + 1` | `lambda x: x + 1` (a "closure") |
| `v.iter().map(\|x\| x * 2.0).collect()` | `[x * 2.0 for x in v]` |
| `v.iter().any(\|x\| !x.is_finite())` | `any(not math.isfinite(x) for x in v)` |
| `(0..n).rev()` | `reversed(range(n))` |
| `v.iter().map(\|x\| x.ln()).sum()` | `sum(math.log(x) for x in v)` |
| `while let Some(run) = runs.next_if(...) { ... }` | keep taking items from an iterator while a condition holds |

`&` in front of a loop variable, and `*value += 1.0`, are bookkeeping about views: `*value`
means "the number that this view points at". They can be read as if they were not there.

### Comments, attributes and macros

| Rust | Meaning |
|---|---|
| `// ...` | a comment |
| `/// ...` | documentation of the item below it, like a docstring |
| `//! ...` | documentation of the file it is in |
| `#[derive(Debug, Clone)]` | generate the code to print and to copy this struct |
| `#[cfg(test)] mod tests { ... }` | unit tests, compiled only by `cargo test` |
| `#[test]` | marks a test function, like a function named `test_...` for pytest |
| `format!("{n} values")` | `f"{n} values"`. A name ending in `!` is a macro, used like a function |
| `assert_eq!(a, b)` | `assert a == b` |
| `use crate::LmmError;` | `from <this crate> import LmmError` |
| `Col::<f64>::from_fn(...)`, `.sum::<f64>()` | the part in `::<...>` names a type that the compiler could not work out |

Names from the `faer` library, such as `Mat`, `Col`, `MatRef`, `Llt` and `Par::Seq`, are
dense matrices, column vectors, read-only views of matrices, Cholesky factorisations and
"run on one thread".

## Building and testing

Installing MERA from source builds this crate, so a
[Rust toolchain](https://rustup.rs) is needed for development:

```
uv sync --all-extras --dev   # builds the engine and installs it as mera._lmm
cargo test                   # the Rust unit tests, at the bottom of each file
uv run pytest                # the Python tests, which exercise the engine through mera
```

After editing a `.rs` file, `uv run ...` rebuilds the engine automatically. CI also runs
`cargo fmt --check` (formatting) and `cargo clippy` (lints).

The Rust tests check the layout of the stored matrices and the factorisation against
dense calculations. Whether the *statistics* are right is checked from Python: against
R's lme4 (`tests/resources/lme4_reference`), against a brute-force calculation
(`tests/test_criterion.py`), against closed-form answers (`tests/test_lmm.py`), and
against the numpy walkthrough (`tests/test_maths_walkthrough.py`).
