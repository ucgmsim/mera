//! Step 2 of a fit: the Cholesky factorisation of `A`, and what is read from it.
//!
//! # What a Cholesky factorisation is, and why the fit needs one
//!
//! `A = Lambda' Z'Z Lambda + I` (filled in by `Design::fill_a`) is a square, symmetric
//! matrix with one row and one column per level (event or site). The fit needs the
//! determinant of `A` and products with its inverse, and computes neither directly.
//! Instead `A` is *factorised* as
//!
//! ```text
//!     A = L L'
//! ```
//!
//! where `L` is lower triangular (zero above its diagonal): a kind of square root of `A`.
//! Only a *positive definite* matrix has one. That is the matrix version of a positive
//! number: `x'Ax` is positive for every non-zero vector `x`. `A` always is, thanks to the
//! `+ I` in its definition. With `L` in hand:
//!
//! - the log-determinant is `log|A| = 2 * (sum of the logs of the diagonal of L)`
//!   ([`Factor::log_det`]);
//! - solving `A x = b` for `x` takes two cheap passes, one with `L` and one with `L'`,
//!   because equations with a triangular matrix can be solved one unknown at a time
//!   ([`Factor::solve_in_place`]);
//! - the diagonal of the inverse of `A`, which the conditional standard deviations need,
//!   can be computed without forming the whole inverse ([`Factor::inverse_diagonal`]);
//! - so can the entries of the inverse where `A` itself has entries, which the slope of
//!   the criterion needs ([`Factor::inverse_on_pattern`]).
//!
//! # Keeping it sparse
//!
//! `A` is sparse (mostly zeros; see `design.rs`), and the factorisation is only fast if
//! `L` is sparse too. `L` usually has some *fill-in*: entries that are zero in `A` but not
//! in `L`. How much depends on the order in which the levels are numbered. So before
//! factorising, the levels are renumbered by a permutation `P` chosen to keep the fill-in
//! small (the "AMD" ordering, for approximate minimum degree), and what is factorised is
//! `P A P' = L L'`. Renumbering does not change the determinant, and the solutions and
//! the diagonal of the inverse are put back in the original order.
//!
//! The work is split in two, as lme4 does with the CHOLMOD library:
//!
//! - the *symbolic* analysis works out the ordering and where the non-zeros of `L` will
//!   be. It depends only on where the non-zeros of `A` are, which is the same for every
//!   theta, so it is done once per fit ([`Factor::new`]);
//! - the *numeric* factorisation computes the values of `L` for one theta, and is
//!   repeated at every evaluation of the criterion ([`Factor::factorize`]).
//!
//! # Two layouts of `L`
//!
//! faer, the linear-algebra library used here, can store `L` in two ways:
//!
//! - *simplicial*: column by column, in the compressed sparse column form described in
//!   `design.rs`;
//! - *supernodal*: neighbouring columns with the same pattern of non-zeros are grouped
//!   into a "supernode" and stored together as one dense block. The arithmetic on a
//!   block is done by dense matrix routines, which a processor runs several times
//!   faster than the same arithmetic one entry at a time.
//!
//! Both hold the same `L`, so the results do not depend on the layout. Left to itself,
//! faer picks the one it expects to factorise faster. Fits ask for the supernodal layout
//! instead ([`FactorKind::Supernodal`]). On designs from a few hundred levels to a
//! hundred thousand, an evaluation of the criterion, which is mostly the factorisation,
//! was between a sixth slower and twice as fast with it. An evaluation with the
//! gradient, which also needs a selected inversion, was always faster, by up to three
//! times.
//! faer still returns a simplicial `L` when `L` is diagonal, which is the case of a
//! single grouping factor, because there is then nothing to group. So the functions
//! below that read `L` directly ([`Factor::log_det`] and selected inversion) have a
//! branch for each layout.

use faer::dyn_stack::{MemBuffer, MemStack, StackReq};
use faer::linalg::cholesky::llt::factor::{LltParams, LltRegularization};
use faer::linalg::matmul::matmul;
use faer::linalg::matmul::triangular::{BlockStructure, matmul as triangular_matmul};
use faer::linalg::solvers::Llt;
use faer::linalg::triangular_inverse::invert_lower_triangular;
use faer::sparse::linalg::SupernodalThreshold;
use faer::sparse::linalg::cholesky::simplicial::SymbolicSimplicialCholesky;
use faer::sparse::linalg::cholesky::supernodal::{SupernodalLltRef, SymbolicSupernodalCholesky};
use faer::sparse::linalg::cholesky::{
    CholeskySymbolicParams, LltRef, SymbolicCholesky, SymbolicCholeskyRaw, SymmetricOrdering,
    factorize_symbolic_cholesky,
};
use faer::sparse::{SparseColMatRef, SymbolicSparseColMatRef};
use faer::{Accum, Conj, MatMut, MatRef, Par, Side, Spec};

use crate::LmmError;

/// Numeric layout of the factor (see "Two layouts of `L`" at the top of this file).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FactorKind {
    /// Supernodal, unless `L` is diagonal: what fits use.
    Supernodal,
    /// Always simplicial (used by tests, to check that layout on matrices of any shape).
    Simplicial,
}

/// A reusable sparse Cholesky factorisation for one sparsity pattern.
///
/// Create it once with [`Factor::new`], then call [`Factor::factorize`] each time the
/// values of the matrix change, and read the results with the other methods.
pub struct Factor {
    /// The symbolic analysis: the ordering `P` and where the non-zeros of `L` are.
    symbolic: SymbolicCholesky<usize>,
    /// The values of `L` from the latest call to [`Factor::factorize`].
    l_values: Vec<f64>,
    /// Scratch memory that faer uses while factorising and solving, allocated once.
    workspace: MemBuffer,
    /// Where to read the diagonal of the inverse from the result of selected inversion
    /// (see [`Factor::inverse_diagonal`]): one position per level, in the order of the
    /// matrix given to [`Factor::new`].
    diagonal_slots: Vec<usize>,
    /// The same for each stored entry of the matrix given to [`Factor::new`], in the
    /// order in which they were given (see [`Factor::inverse_on_pattern`]).
    a_slots: Vec<usize>,
}

impl Factor {
    /// Analyses the pattern of a symmetric matrix given as its lower triangle in CSC form.
    /// `rhs_ncols` is the largest number of right-hand sides passed to [`Factor::solve_in_place`].
    ///
    /// This is the symbolic analysis: it looks only at where the non-zeros are (`col_ptr`
    /// and `row_idx`), not at their values.
    pub fn new(
        col_ptr: &[usize],
        row_idx: &[usize],
        rhs_ncols: usize,
        kind: FactorKind,
    ) -> Result<Self, LmmError> {
        let n = col_ptr.len() - 1;
        // The pattern as faer wants it: an n x n matrix, described by where each column
        // starts and the row of each entry. (`None` is for an option that is not used.)
        let pattern = SymbolicSparseColMatRef::new_checked(n, n, col_ptr, None, row_idx);
        // Which layout to use; every other setting keeps its default value.
        let params = CholeskySymbolicParams {
            supernodal_flop_ratio_threshold: match kind {
                FactorKind::Supernodal => SupernodalThreshold::FORCE_SUPERNODAL,
                FactorKind::Simplicial => SupernodalThreshold::FORCE_SIMPLICIAL,
            },
            ..Default::default()
        };
        // `Side::Lower`: the pattern holds the lower triangle of the symmetric matrix.
        // `SymmetricOrdering::Amd`: renumber the levels with the AMD ordering.
        let symbolic =
            factorize_symbolic_cholesky(pattern, Side::Lower, SymmetricOrdering::Amd, params)
                .map_err(|e| LmmError::Factorisation(format!("symbolic analysis failed: {e:?}")))?;
        // Reserve enough scratch memory for the larger of the two jobs that need it.
        // `Par::Seq` means "run on one thread", here and below.
        let workspace = MemBuffer::new(StackReq::or(
            symbolic
                .factorize_numeric_llt_scratch::<f64>(Par::Seq, Spec::<LltParams, f64>::default()),
            symbolic.solve_in_place_scratch::<f64>(rhs_ncols, Par::Seq),
        ));
        let l_values = vec![0.0; symbolic.len_val()];

        // ---- Where the wanted entries of the selected inverse will be, found once ----
        // Selected inversion (see `Factor::inverse_diagonal`) returns a list of numbers
        // laid out exactly as the values of L are. Two kinds of entry are read from it:
        // the diagonal, and the entries where A has entries. Their positions in the list
        // depend only on the pattern, so they are worked out here rather than at every
        // evaluation.
        //
        // The factorisation renumbers the levels. `renumbered(i)` is the number that
        // level `i` of the matrix given here has in the renumbered matrix, P A P'.
        // (`symbolic.perm()` is the renumbering, if there is one, and `.arrays().1` is
        // the list that maps old numbers to new ones.)
        let old_to_new = symbolic.perm().map(|perm| perm.arrays().1);
        let renumbered = |i: usize| old_to_new.map_or(i, |new| new[i]);
        // The error if L has no entry where one is expected. It cannot happen: L always
        // has its diagonal, and is non-zero wherever A is.
        let missing = || LmmError::Factorisation("an entry of A is missing from L".into());
        // (`.ok_or_else(missing)?` below reads: if no position was found, stop and
        // return that error.)
        let mut diagonal_slots = Vec::with_capacity(n);
        for i in 0..n {
            let j = renumbered(i);
            diagonal_slots.push(slot_in_l(symbolic.raw(), j, j).ok_or_else(missing)?);
        }
        let mut a_slots = Vec::with_capacity(row_idx.len());
        for c in 0..n {
            for &r in &row_idx[col_ptr[c]..col_ptr[c + 1]] {
                // Entry (r, c) of A is entry (i, j) of the renumbered matrix. Only its
                // lower triangle is stored, where the row is the larger of the two
                // numbers and the column the smaller. The matrix is symmetric, so the
                // entry (j, i) holds the same value.
                let (i, j) = (renumbered(r), renumbered(c));
                let slot = slot_in_l(symbolic.raw(), i.max(j), i.min(j)).ok_or_else(missing)?;
                a_slots.push(slot);
            }
        }
        Ok(Self {
            symbolic,
            l_values,
            workspace,
            diagonal_slots,
            a_slots,
        })
    }

    /// Dimension of the factorised matrix (the number of levels, `q`).
    pub fn dim(&self) -> usize {
        self.symbolic.nrows()
    }

    /// Whether the numeric factor is simplicial.
    pub fn is_simplicial(&self) -> bool {
        matches!(self.symbolic.raw(), SymbolicCholeskyRaw::Simplicial(_))
    }

    /// Numerically factorises the matrix with the analysed pattern and these values.
    ///
    /// This computes the values of `L` and keeps them in `self.l_values`, replacing those
    /// of the previous call. It fails only if the matrix is not positive definite. In
    /// exact arithmetic `A = Lambda' Z'Z Lambda + I` always is, but once theta passes
    /// about ten million the `+ I` is lost to rounding. The optimiser only drives theta
    /// that far when the grouping factors explain the response exactly.
    pub fn factorize(
        &mut self,
        col_ptr: &[usize],
        row_idx: &[usize],
        values: &[f64],
    ) -> Result<(), LmmError> {
        let n = self.dim();
        let pattern = SymbolicSparseColMatRef::new_checked(n, n, col_ptr, None, row_idx);
        // The arguments, in order: where to write the values of L (the output); the
        // matrix to factorise (the pattern with its values); that its lower triangle is
        // given; no adjustment of the matrix; one thread; the scratch memory; default
        // settings. `.map(|_| ())` then discards what faer returns (a view of the factor
        // that it has just written), so that only success or failure is passed on.
        self.symbolic
            .factorize_numeric_llt(
                &mut self.l_values,
                SparseColMatRef::new(pattern, values),
                Side::Lower,
                LltRegularization::default(),
                Par::Seq,
                MemStack::new(&mut self.workspace),
                Spec::default(),
            )
            .map(|_| ())
            .map_err(|e| {
                LmmError::Factorisation(format!(
                    "the matrix is not positive definite ({e:?}): theta has grown without \
                     limit, which happens when the grouping factors explain the response \
                     exactly, leaving no residual variation"
                ))
            })
    }

    /// `log |A|`, from the diagonal of `L`.
    ///
    /// The determinant of a triangular matrix is the product of its diagonal, so
    /// `|A| = |L| |L'| = (product of the diagonal of L)^2`. Taking logs turns the product
    /// into a sum, which also avoids overflow: the determinant itself is astronomically
    /// large.
    pub fn log_det(&self) -> f64 {
        let l = &self.l_values;
        let sum_log_diagonal: f64 = match self.symbolic.raw() {
            SymbolicCholeskyRaw::Simplicial(s) => {
                // Each column of L is stored with its diagonal entry first.
                let col_ptr = s.col_ptr();
                (0..s.nrows()).map(|j| l[col_ptr[j]].ln()).sum()
            }
            SymbolicCholeskyRaw::Supernodal(s) => {
                // Each supernode is a dense block of columns of L, its diagonal block on top.
                let llt = SupernodalLltRef::<usize, f64>::new(s, l);
                (0..s.n_supernodes())
                    .map(|si| {
                        let block = llt.supernode(si).val();
                        (0..block.ncols()).map(|k| block[(k, k)].ln()).sum::<f64>()
                    })
                    .sum()
            }
        };
        2.0 * sum_log_diagonal
    }

    /// Overwrites `rhs` with `A^-1 rhs`.
    ///
    /// That is, it solves `A x = rhs` for `x`, for every column of `rhs` at once, and
    /// stores `x` where `rhs` was. `A^-1` is never formed.
    pub fn solve_in_place(&mut self, rhs: MatMut<'_, f64>) {
        // Pair the symbolic analysis with the values of L, then solve. `Conj::No`: do not
        // take complex conjugates, since these are real numbers.
        LltRef::<usize, f64>::new(&self.symbolic, &self.l_values).solve_in_place_with_conj(
            Conj::No,
            rhs,
            Par::Seq,
            MemStack::new(&mut self.workspace),
        );
    }

    /// `diag(A^-1)` in the original (unpermuted) order, by selected inversion.
    ///
    /// Why not simply invert `A`? The inverse of a sparse matrix is dense: with 30,000
    /// levels it would hold 900 million numbers, when only its 30,000 diagonal entries are
    /// wanted. *Selected inversion* computes only the entries of the inverse that lie where
    /// `L` has non-zeros. The diagonal is among them, and the work is comparable to the
    /// factorisation itself.
    ///
    /// Selected inversion is the most intricate part of the crate: the functions
    /// `simplicial_selected_inverse` and `supernodal_selected_inverse` below, one for
    /// each layout of `L`. It can be read as a black box that returns entries of the
    /// inverse. The test `selected_inverse_matches_dense_inverse` at the bottom of this
    /// file checks those entries against an ordinary dense inverse.
    pub fn inverse_diagonal(&self) -> Vec<f64> {
        let z = self.selected_inverse();
        // Pick out the diagonal entries. `diagonal_slots` says where each one is, and
        // already allows for the renumbering of the levels.
        self.diagonal_slots.iter().map(|&slot| z[slot]).collect()
    }

    /// The entries of `A^-1` where `A` itself has entries: one value for each stored entry
    /// of the lower triangle of `A`, in the order in which [`Factor::new`] was given them.
    ///
    /// These are among the entries that selected inversion computes (see
    /// [`Factor::inverse_diagonal`]), because `L` has a non-zero wherever `A` does. The
    /// gradient of the criterion needs them (`CrossedModel::deviance_and_gradient`).
    pub fn inverse_on_pattern(&self) -> Vec<f64> {
        let z = self.selected_inverse();
        self.a_slots.iter().map(|&slot| z[slot]).collect()
    }

    /// The selected inverse: the entries of `Z = (P A P')^-1` where `L` has non-zeros.
    /// (This `Z` is not the `Z` of the model: the name is the convention for this
    /// algorithm.)
    ///
    /// It is returned as a list laid out exactly as the values of `L` are: `z[t]` is the
    /// entry of `Z` in the row and column in which `self.l_values[t]` is the entry of `L`.
    /// The rows and columns are those of the renumbered matrix `P A P'`.
    fn selected_inverse(&self) -> Vec<f64> {
        match self.symbolic.raw() {
            SymbolicCholeskyRaw::Simplicial(s) => simplicial_selected_inverse(s, &self.l_values),
            SymbolicCholeskyRaw::Supernodal(s) => supernodal_selected_inverse(s, &self.l_values),
        }
    }
}

/// Where the entry in row `row` and column `column` of `L` is in the list of its values,
/// or `None` if `L` has no entry there.
///
/// `row` must not be smaller than `column`, since `L` is lower triangular, and both are
/// numbers of the renumbered matrix `P A P'`. `raw` is the pattern of `L` in whichever of
/// the two layouts it has.
fn slot_in_l(raw: &SymbolicCholeskyRaw<usize>, row: usize, column: usize) -> Option<usize> {
    match raw {
        SymbolicCholeskyRaw::Simplicial(s) => {
            // Column by column, as described in `design.rs`: the entries of this column
            // are those from `start` up to where the next column starts. Their rows are
            // sorted, so `binary_search` finds the place of `row` among them by
            // bisection. (`.ok()?` reads: if it is not there, return `None` now.)
            let start = s.col_ptr()[column];
            let rows = &s.row_idx()[start..s.col_ptr()[column + 1]];
            let place = rows.binary_search(&row).ok()?;
            Some(start + place)
        }
        SymbolicCholeskyRaw::Supernodal(s) => {
            // Supernode `node` groups the columns `begin[node]` up to, but not
            // including, `end[node]`. The one that holds `column` is the first that ends
            // after it. `partition_point` finds it by bisection: it counts the
            // supernodes that end at or before `column`, which all come first.
            let (begin, end) = (s.supernode_begin(), s.supernode_end());
            let node = end.partition_point(|&e| e <= column);
            let width = end[node] - begin[node];
            // `below`: the rows that the columns of this supernode share below them
            let below = s.supernode(node).pattern();
            // The values of the supernode are one dense block, stored column after
            // column. Each column has `width` entries for the rows of the supernode's
            // own columns, then one for each row of `below`.
            let height = width + below.len();
            let row_in_block = if row < end[node] {
                row - begin[node]
            } else {
                width + below.binary_search(&row).ok()?
            };
            let column_in_block = column - begin[node];
            // `col_ptr_for_val()[node]` is where the block starts in the list of values.
            Some(s.col_ptr_for_val()[node] + column_in_block * height + row_in_block)
        }
    }
}

/// The selected inverse of a simplicial factor: `Z = (P A P')^-1` on the pattern of `L`,
/// in the layout of `l` (see `Factor::selected_inverse`).
///
/// Since `P A P' = L L'`, we have `Z L = (L')^-1`, which is upper triangular with
/// `1 / L[j, j]` on its diagonal. Reading that equation one entry at a time gives
/// Takahashi's recurrences (Takahashi, Fagan & Chin 1973; Erisman & Tinney 1975). For
/// column `j` with below-diagonal rows `R_j`:
///
/// ```text
/// Z[i, j] = -(1 / L[j, j]) * sum over k in R_j of Z[i, k] L[k, j]      for i in R_j
/// Z[j, j] = (1 / L[j, j] - sum over k in R_j of Z[k, j] L[k, j]) / L[j, j]
/// ```
///
/// Each column only needs entries of `Z` from later columns, so the columns are computed
/// from the last to the first, and `A^-1` is never formed.
fn simplicial_selected_inverse(s: &SymbolicSimplicialCholesky<usize>, l: &[f64]) -> Vec<f64> {
    // Where the non-zeros of L are, column by column, as described in `design.rs`.
    let (col_ptr, row_idx) = (s.col_ptr(), s.row_idx());
    let n = s.nrows();
    // z holds Z on the pattern of L: z[t] is the entry of Z where l[t] is the entry of L.
    let mut z = vec![0.0; l.len()];
    // The sums of the recurrences for the column being computed, one per row of R_j.
    // (The list is created once and reused for every column.)
    let mut acc: Vec<f64> = Vec::new();
    // position[row]: the place of `row` among the rows R_j of the column being computed,
    // `None` if it is not one of them
    let mut position: Vec<Option<usize>> = vec![None; n];
    for j in (0..n).rev() {
        let (start, end) = (col_ptr[j], col_ptr[j + 1]);
        let l_jj = l[start]; // rows are sorted, so the diagonal comes first
        let rows = &row_idx[start + 1..end];
        let l_col = &l[start + 1..end];
        // empty the list of sums, and refill it with a zero per row
        acc.clear();
        acc.resize(rows.len(), 0.0);
        for (slot, &row) in rows.iter().enumerate() {
            position[row] = Some(slot);
        }
        // acc[a] = sum_b Z[rows[a], rows[b]] L[rows[b], j]. Every Z[rows[a], k] with
        // k = rows[b] lives in column k of Z (its diagonal, or a row below it).
        for (b, &k) in rows.iter().enumerate() {
            let (k_start, k_end) = (col_ptr[k], col_ptr[k + 1]);
            // the diagonal entry Z[k, k]
            acc[b] += z[k_start] * l_col[b];
            // the entries below it: column k is walked, and the rows in R_j picked out
            for t in k_start + 1..k_end {
                if let Some(a) = position[row_idx[t]] {
                    // Z is symmetric, so the stored entry Z[rows[a], k] is also
                    // Z[k, rows[a]]: it contributes to both sums.
                    acc[a] += z[t] * l_col[b];
                    acc[b] += z[t] * l_col[a];
                }
            }
        }
        for &row in rows {
            position[row] = None;
        }
        // The two recurrences: first the entries below the diagonal, then the diagonal.
        let mut sum = 0.0;
        for (a, value) in acc.iter().enumerate() {
            let z_aj = -value / l_jj;
            z[start + 1 + a] = z_aj;
            sum += z_aj * l_col[a];
        }
        z[start] = (1.0 / l_jj - sum) / l_jj;
    }
    z
}

/// The selected inverse of a supernodal factor: `Z = (P A P')^-1` on the pattern of `L`,
/// in the layout of `l_values` (see `Factor::selected_inverse`).
///
/// This is the calculation of `simplicial_selected_inverse`, done for all the columns of
/// a supernode at once. A supernode `J` is a run of consecutive columns of `L` that have
/// the same rows below them, `R`. Its part of `L` is two dense blocks:
///
/// ```text
///         columns J
///        [  L_JJ  ]   rows J: a square block, zero above its diagonal
///        [  L_RJ  ]   rows R: the rows below, which all the columns of J share
/// ```
///
/// The equation `Z L = (L')^-1`, read for the columns `J`, gives
///
/// ```text
/// Y    = L_RJ L_JJ^-1
/// Z_RJ = -Z_RR Y
/// Z_JJ = (L_JJ L_JJ')^-1 - Y' Z_RJ
/// ```
///
/// where `Z_RR` holds the entries of `Z` for every pair of rows of `R`. Those belong to
/// later supernodes, which is why the supernodes are taken from the last to the first.
/// These are products of ordinary dense matrices, which faer does several times faster
/// than the same arithmetic done one entry at a time.
fn supernodal_selected_inverse(
    s: &SymbolicSupernodalCholesky<usize>,
    l_values: &[f64],
) -> Vec<f64> {
    let n = s.nrows();
    // `llt` pairs the pattern of L with its values, so that faer can hand out the dense
    // block of each supernode (as in `Factor::log_det`).
    let llt = SupernodalLltRef::<usize, f64>::new(s, l_values);
    // Supernode j groups the columns begin[j] up to, but not including, end[j]. Its
    // values are the entries val_ptr[j] up to val_ptr[j + 1] of `l_values`, and its
    // part of Z is kept in the same places of `z_values`.
    let (begin, end, val_ptr) = (s.supernode_begin(), s.supernode_end(), s.col_ptr_for_val());
    // supernode_of[c]: the supernode that holds column c.
    // (`.fill(j)` sets every entry of that part of the list to j.)
    let mut supernode_of = vec![0usize; n];
    for j in 0..s.n_supernodes() {
        supernode_of[begin[j]..end[j]].fill(j);
    }
    // Z is kept in the layout of L: for each supernode, a dense block with a column per
    // column of the supernode. The first rows of the block are Z_JJ (only its lower
    // triangle is filled in, since it is symmetric) and the rest are Z_RJ, in the order
    // of R.
    let mut z_values = vec![0.0; l_values.len()];
    // position[row]: the place of `row` among the rows below the supernode that is being
    // read from. It is filled in just before it is used.
    let mut position = vec![0usize; n];
    // Working space for L_JJ^-1, Y and Z_RR. The three lists are created once and reused
    // for every supernode, rather than asking for memory each time.
    let (mut inverse_buffer, mut y_buffer, mut z_rr_buffer) = (Vec::new(), Vec::new(), Vec::new());
    for j in (0..s.n_supernodes()).rev() {
        let node = llt.supernode(j);
        // rows: R, the rows below the supernode. m: how many of them there are.
        let rows = node.pattern();
        let m = rows.len();
        // The values of the supernode as a matrix, without copying them: a column for
        // each column of J, with the rows of J first and the rows R after them.
        // (`MatRef` is faer's read-only view of a matrix, and `MatMut`, below, a view
        // that can be written to.)
        let l_block = node.val();
        // width: the number of columns of J
        let width = l_block.ncols();
        // Split the block after its first `width` rows: L_JJ, then L_RJ.
        let (l_jj, l_rj) = l_block.split_at_row(width);

        // ---- L_JJ^-1 and Y = L_RJ L_JJ^-1 ----
        // Empty the working list and refill it with zeros, view it as a square matrix,
        // and write the inverse of L_JJ into it. The inverse of a lower triangular
        // matrix is lower triangular too. (`from_column_major_slice_mut` views a flat
        // list as a matrix with the given numbers of rows and columns, stored column
        // after column. `.as_mut()` lends a matrix to a function to write to, and
        // `.as_ref()` to read from.)
        inverse_buffer.clear();
        inverse_buffer.resize(width * width, 0.0);
        let mut inverse = MatMut::from_column_major_slice_mut(&mut inverse_buffer, width, width);
        invert_lower_triangular(inverse.as_mut(), l_jj, Par::Seq);
        // The same for Y, which has m rows and `width` columns.
        y_buffer.clear();
        y_buffer.resize(m * width, 0.0);
        let mut y = MatMut::from_column_major_slice_mut(&mut y_buffer, m, width);
        // faer's matrix products take, in this order: the matrix that receives the
        // result; whether the result replaces what is there or is added to it; the two
        // matrices to multiply; a number that multiplies their product; and `Par::Seq`,
        // "on one thread". In `triangular_matmul` each matrix is followed by the part
        // of it to use: all of it (`Rectangular`), or only its lower or upper triangle,
        // the rest being zeros that are skipped.
        //
        // This call is Y = 1.0 * L_RJ * L_JJ^-1.
        triangular_matmul(
            y.as_mut(),
            BlockStructure::Rectangular,
            Accum::Replace,
            l_rj,
            BlockStructure::Rectangular,
            inverse.as_ref(),
            BlockStructure::TriangularLower,
            1.0,
            Par::Seq,
        );

        // ---- Z_RR, gathered from the later supernodes ----
        // Z_RR has m rows and m columns: its entry (a, b) is Z[rows[a], rows[b]], kept
        // at place a + m * b of the flat list. Each column rows[b] of Z belongs to a
        // later supernode, k. Neighbouring rows of R often belong to the same one, so
        // the outer loop finds k, and the inner loop deals with every row of R that is
        // a column of k before moving on.
        z_rr_buffer.clear();
        z_rr_buffer.resize(m * m, 0.0);
        let mut b = 0;
        while b < m {
            let k = supernode_of[rows[b]];
            let (k_start, k_width) = (begin[k], end[k] - begin[k]);
            // Note where each of the rows below k is among them.
            let k_rows = s.supernode(k).pattern();
            for (t, &row) in k_rows.iter().enumerate() {
                position[row] = t;
            }
            // The block of Z for supernode k, computed in an earlier pass of the loop
            // over j: `k_height` entries for each of its columns.
            let k_height = k_width + k_rows.len();
            let z_k = &z_values[val_ptr[k]..val_ptr[k + 1]];
            while b < m && rows[b] < end[k] {
                // the column of the block that is column rows[b] of Z
                let column = rows[b] - k_start;
                let z_column = &z_k[column * k_height..(column + 1) * k_height];
                for a in b..m {
                    // Z[rows[a], rows[b]]. Row rows[a] is either a column of k too, in
                    // which case the entry is in the first rows of the block, or one
                    // of the rows below k. (It is always one or the other: two rows of
                    // one column of L share an entry.)
                    let value = if rows[a] < end[k] {
                        z_column[rows[a] - k_start]
                    } else {
                        z_column[k_width + position[rows[a]]]
                    };
                    // Z is symmetric, so the entry (b, a) is the same number.
                    z_rr_buffer[a + m * b] = value;
                    z_rr_buffer[b + m * a] = value;
                }
                b += 1;
            }
        }
        let z_rr = MatRef::from_column_major_slice(&z_rr_buffer, m, m);

        // ---- Z_RJ = -Z_RR Y, and Z_JJ = (L_JJ L_JJ')^-1 - Y' Z_RJ ----
        // The block of Z for this supernode, viewed as a matrix that can be written to,
        // and split into its first `width` rows (Z_JJ) and the rest (Z_RJ).
        let z_block = MatMut::from_column_major_slice_mut(
            &mut z_values[val_ptr[j]..val_ptr[j + 1]],
            width + m,
            width,
        );
        let (mut z_jj, mut z_rj) = z_block.split_at_row_mut(width);
        // Z_RJ = -1.0 * Z_RR * Y. (`matmul` is the product for matrices that are used
        // in full, so it takes no parts.)
        matmul(
            z_rj.as_mut(),
            Accum::Replace,
            z_rr,
            y.as_ref(),
            -1.0,
            Par::Seq,
        );
        // From here on the inverse of L_JJ is only read.
        let inverse = inverse.as_ref();
        // Z_JJ = (L_JJ L_JJ')^-1, which is (L_JJ^-1)' L_JJ^-1: an upper triangular
        // matrix times a lower triangular one. Only the lower triangle of the result
        // is computed, because Z_JJ is symmetric and only that triangle is read.
        triangular_matmul(
            z_jj.as_mut(),
            BlockStructure::TriangularLower,
            Accum::Replace,
            inverse.transpose(),
            BlockStructure::TriangularUpper,
            inverse,
            BlockStructure::TriangularLower,
            1.0,
            Par::Seq,
        );
        // Z_JJ = Z_JJ - Y' Z_RJ: the product is added to what is there, times -1.0.
        triangular_matmul(
            z_jj.as_mut(),
            BlockStructure::TriangularLower,
            Accum::Add,
            y.as_ref().transpose(),
            BlockStructure::Rectangular,
            z_rj.as_ref(),
            BlockStructure::Rectangular,
            -1.0,
            Par::Seq,
        );
    }
    z_values
}

/// `log |A|` from a dense Cholesky factorisation of `A`.
///
/// Used for the small dense matrix `X'V^-1X` in `model.rs`, and by the tests.
pub(crate) fn dense_log_det(llt: &Llt<f64>) -> f64 {
    let l = llt.L();
    2.0 * (0..l.nrows()).map(|i| l[(i, i)].ln()).sum::<f64>()
}

#[cfg(test)]
mod tests {
    // Each test builds `A` for a random design twice, as the sparse matrix that `Factor`
    // takes and as an ordinary dense matrix, and checks that `Factor` gives the same
    // answers as faer's dense routines, in both layouts.
    use super::*;
    use crate::design::Design;
    use crate::testing::{Lcg, two_factor_codes};
    use approx::{assert_abs_diff_eq, assert_relative_eq};
    use faer::Mat;
    use faer::linalg::solvers::{DenseSolveCore, Solve};
    use proptest::prelude::*;

    /// `A = Lambda Z'Z Lambda + I` for a random two-factor design: CSC lower triangle and dense.
    fn random_matrix(
        seed: u64,
        n: usize,
        events: usize,
        sites: usize,
    ) -> (Design, Vec<f64>, Mat<f64>) {
        let mut rng = Lcg::new(seed);
        let ev: Vec<usize> = (0..n).map(|_| rng.below(events)).collect();
        let st: Vec<usize> = (0..n).map(|_| rng.below(sites)).collect();
        matrix_of(ev, st, [0.8, 1.3])
    }

    /// `A = Lambda Z'Z Lambda + I` for the design with the given event and site of each
    /// record, at the given theta: CSC lower triangle and dense.
    fn matrix_of(ev: Vec<usize>, st: Vec<usize>, theta: [f64; 2]) -> (Design, Vec<f64>, Mat<f64>) {
        let n = ev.len();
        let design = Design::from_full_codes(&[ev, st], &vec![0.0; n], &[], 0).unwrap();
        let mut values = vec![0.0; design.ztz.len()];
        design.fill_a(&theta, &mut values);
        let q = design.q();
        let mut dense = Mat::<f64>::zeros(q, q);
        for c in 0..q {
            let entries = design.col_ptr[c]..design.col_ptr[c + 1];
            for (&r, &value) in design.row_idx[entries.clone()].iter().zip(&values[entries]) {
                dense[(r, c)] = value;
                dense[(c, r)] = value;
            }
        }
        (design, values, dense)
    }

    /// A factor of the given layout for the pattern of `design`, factorised with `values`.
    fn factorised(design: &Design, values: &[f64], rhs_ncols: usize, kind: FactorKind) -> Factor {
        let mut factor = Factor::new(&design.col_ptr, &design.row_idx, rhs_ncols, kind).unwrap();
        factor
            .factorize(&design.col_ptr, &design.row_idx, values)
            .unwrap();
        factor
    }

    fn check_log_det_and_solve(kind: FactorKind) {
        for seed in 0..4 {
            let (design, values, dense) = random_matrix(seed, 120, 15, 11);
            let q = design.q();
            let mut factor = factorised(&design, &values, 2, kind);
            let llt = dense.llt(Side::Lower).unwrap();
            assert_relative_eq!(factor.log_det(), dense_log_det(&llt), max_relative = 1e-12);

            let mut rng = Lcg::new(100 + seed);
            let b = Mat::<f64>::from_fn(q, 2, |_, _| rng.normal());
            let expected = llt.solve(&b);
            let mut rhs = b.clone();
            factor.solve_in_place(rhs.as_mut());
            for c in 0..2 {
                for i in 0..q {
                    assert_abs_diff_eq!(rhs[(i, c)], expected[(i, c)], epsilon = 1e-10);
                }
            }
        }
    }

    #[test]
    fn log_det_and_solve_simplicial() {
        check_log_det_and_solve(FactorKind::Simplicial);
    }

    #[test]
    fn log_det_and_solve_supernodal() {
        check_log_det_and_solve(FactorKind::Supernodal);
    }

    /// The shapes of design that the tests of selected inversion use, as `(observations,
    /// events, sites)`. In the first, nearly every event meets every site, and `L` is
    /// almost one dense block. In the second, few do: `L` has many supernodes, and a
    /// supernode reads entries of `Z` from several later ones.
    const SHAPES: [(usize, usize, usize); 2] = [(150, 20, 9), (600, 150, 80)];

    /// The number of supernodes of `L`, 0 if it is simplicial.
    fn n_supernodes(factor: &Factor) -> usize {
        match factor.symbolic.raw() {
            SymbolicCholeskyRaw::Supernodal(s) => s.n_supernodes(),
            SymbolicCholeskyRaw::Simplicial(_) => 0,
        }
    }

    /// Checks the entries that are read from the selected inverse (the diagonal, and the
    /// entries where `A` has entries) against a dense inverse of `A`, to within `epsilon`.
    fn check_selected_inverse(
        design: &Design,
        values: &[f64],
        dense: &Mat<f64>,
        kind: FactorKind,
        epsilon: f64,
    ) {
        let factor = factorised(design, values, 1, kind);
        let expected = dense.llt(Side::Lower).unwrap().inverse();
        // The diagonal, in the order of the levels.
        let diagonal = factor.inverse_diagonal();
        for i in 0..design.q() {
            assert_abs_diff_eq!(diagonal[i], expected[(i, i)], epsilon = epsilon);
        }
        // One value for each stored entry of A, in the order of its columns and rows.
        let on_pattern = factor.inverse_on_pattern();
        assert_eq!(on_pattern.len(), design.row_idx.len());
        for c in 0..design.q() {
            let entries = design.col_ptr[c]..design.col_ptr[c + 1];
            let column = on_pattern[entries.clone()]
                .iter()
                .zip(&design.row_idx[entries]);
            for (&value, &row) in column {
                assert_abs_diff_eq!(value, expected[(row, c)], epsilon = epsilon);
            }
        }
    }

    #[test]
    fn selected_inverse_matches_dense_inverse() {
        for kind in [FactorKind::Simplicial, FactorKind::Supernodal] {
            for (n, events, sites) in SHAPES {
                for seed in 0..4 {
                    let (design, values, dense) = random_matrix(seed, n, events, sites);
                    let factor = factorised(&design, &values, 1, kind);
                    assert_eq!(factor.is_simplicial(), kind == FactorKind::Simplicial);
                    check_selected_inverse(&design, &values, &dense, kind, 1e-12);
                }
            }
        }
    }

    proptest! {
        // A property test (see `testing.rs`): at each run of the tests, 64 new random
        // designs. (proptest's usual 256 would take several seconds in a debug build.)
        #![proptest_config(ProptestConfig::with_cases(64))]

        /// Whatever the design and theta, both layouts give the entries of the dense
        /// inverse. The fixed designs of the test above make sure that the supernodal
        /// code is reached in full; these look for a design that nobody thought of.
        #[test]
        fn selected_inverse_matches_dense_inverse_for_any_design(
            (events, sites) in two_factor_codes(60, 40, 1..=400),
            theta in [0.05f64..5.0, 0.05f64..5.0],
        ) {
            let (design, values, dense) = matrix_of(events, sites, theta);
            for kind in [FactorKind::Simplicial, FactorKind::Supernodal] {
                check_selected_inverse(&design, &values, &dense, kind, 1e-9);
            }
        }
    }

    #[test]
    fn the_sparse_shape_has_many_supernodes() {
        // What the test above relies on to reach every part of the supernodal code.
        let (n, events, sites) = SHAPES[1];
        let (design, _, _) = random_matrix(0, n, events, sites);
        let factor =
            Factor::new(&design.col_ptr, &design.row_idx, 1, FactorKind::Supernodal).unwrap();
        assert!(n_supernodes(&factor) > 20);
    }

    #[test]
    fn a_diagonal_matrix_is_factorised_in_the_simplicial_layout() {
        // One grouping factor: A is diagonal, with 1 + theta^2 * (records of the level).
        // faer has nothing to group into supernodes, and the simplicial code is used
        // whatever was asked for.
        let codes: Vec<usize> = (0..40).map(|i| i % 7).collect();
        let design = Design::from_full_codes(&[codes], &[0.0; 40], &[], 0).unwrap();
        let mut values = vec![0.0; design.ztz.len()];
        design.fill_a(&[0.8], &mut values);
        let factor = factorised(&design, &values, 1, FactorKind::Supernodal);
        assert!(factor.is_simplicial());
        let log_det: f64 = values.iter().map(|a| a.ln()).sum();
        assert_relative_eq!(factor.log_det(), log_det, max_relative = 1e-12);
        for (computed, a) in factor.inverse_diagonal().iter().zip(&values) {
            assert_relative_eq!(*computed, 1.0 / a, max_relative = 1e-12);
        }
        for (computed, a) in factor.inverse_on_pattern().iter().zip(&values) {
            assert_relative_eq!(*computed, 1.0 / a, max_relative = 1e-12);
        }
    }
}
