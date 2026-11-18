//! Step 2 of a fit: the Cholesky factorisation of `A`, and the three things read from it.
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
//!   can be computed without forming the whole inverse ([`Factor::inverse_diagonal`]).
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
    /// Where the non-zeros of a supernodal `L` are when its columns are listed one by
    /// one: `(col_ptr, row_idx)`, as in `design.rs`. The selected inverse is handed out
    /// in that layout. A simplicial `L` is stored that way already, and this is `None`.
    columns: Option<(Vec<usize>, Vec<usize>)>,
    /// For each stored entry of the matrix given to [`Factor::new`], where the same entry
    /// of the selected inverse is kept (see [`Factor::inverse_on_pattern`]).
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

        // ---- What selected inversion needs to know about the pattern, found once ----
        let columns = match symbolic.raw() {
            SymbolicCholeskyRaw::Simplicial(_) => None,
            SymbolicCholeskyRaw::Supernodal(s) => {
                let (l_col_ptr, l_row_idx, _) = supernodal_columns(s, &l_values);
                Some((l_col_ptr, l_row_idx))
            }
        };
        let (l_col_ptr, l_row_idx) = match (symbolic.raw(), &columns) {
            (SymbolicCholeskyRaw::Simplicial(s), _) => (s.col_ptr(), s.row_idx()),
            (_, Some((l_col_ptr, l_row_idx))) => (&l_col_ptr[..], &l_row_idx[..]),
            // a supernodal factor always has its columns, from the lines above
            (SymbolicCholeskyRaw::Supernodal(_), None) => (&[][..], &[][..]),
        };
        // inverse[i] is the number that level i of the original matrix has in the
        // renumbered one, P A P'.
        let inverse = symbolic.perm().map(|perm| perm.arrays().1);
        let renumbered = |i: usize| inverse.map_or(i, |inv| inv[i]);
        let mut a_slots = Vec::with_capacity(row_idx.len());
        for c in 0..n {
            for &r in &row_idx[col_ptr[c]..col_ptr[c + 1]] {
                // Entry (r, c) of A is entry (row, column) of the renumbered matrix,
                // taken from its lower triangle: the column is the smaller number.
                let (i, j) = (renumbered(r), renumbered(c));
                let (column, row) = (i.min(j), i.max(j));
                // The rows of a column of L are sorted, so the row is found by
                // bisection. It is there, because L is non-zero wherever A is.
                let rows = &l_row_idx[l_col_ptr[column]..l_col_ptr[column + 1]];
                let slot = rows.binary_search(&row).map_err(|_| {
                    LmmError::Factorisation("an entry of A is missing from L".into())
                })?;
                a_slots.push(l_col_ptr[column] + slot);
            }
        }
        Ok(Self {
            symbolic,
            l_values,
            workspace,
            columns,
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
    /// Write `Z = (P A P')^-1`. (This `Z` is not the `Z` of the model: the name is the
    /// convention for this algorithm.) Since `P A P' = L L'`, we have `Z L = (L')^-1`,
    /// which is upper triangular with `1 / L[j, j]` on its diagonal. Reading that equation
    /// one entry at a time gives Takahashi's recurrences (Takahashi, Fagan & Chin 1973;
    /// Erisman & Tinney 1975). For column `j` with below-diagonal rows `R_j`:
    ///
    /// ```text
    /// Z[i, j] = -(1 / L[j, j]) * sum over k in R_j of Z[i, k] L[k, j]      for i in R_j
    /// Z[j, j] = (1 / L[j, j] - sum over k in R_j of Z[k, j] L[k, j]) / L[j, j]
    /// ```
    ///
    /// Each column only needs entries of `Z` from later columns, so the columns are
    /// computed from the last to the first, and `A^-1` is never formed. For a supernodal
    /// `L`, the recurrences are applied to all the columns of a supernode at once
    /// (`supernodal_selected_inverse`, below).
    ///
    /// Selected inversion is the most intricate part of the crate, and it can be read as
    /// a black box that returns entries of the inverse: the tests
    /// `inverse_diagonal_matches_dense_inverse` and
    /// `inverse_on_pattern_matches_dense_inverse` at the bottom of this file check them
    /// against an ordinary dense inverse.
    pub fn inverse_diagonal(&self) -> Vec<f64> {
        self.with_selected_inverse(|col_ptr, _, z| {
            // Undo the fill-reducing permutation, if there is one. Level `i` of the
            // renumbered matrix is level `forward[i]` of the original, so the diagonal
            // entry of column `i` of Z (the first entry of the column, z[col_ptr[i]])
            // belongs at `forward[i]`.
            let n = self.dim();
            let forward = self.symbolic.perm().map(|perm| perm.arrays().0);
            let mut diagonal = vec![0.0; n];
            for (i, &start) in col_ptr[..n].iter().enumerate() {
                diagonal[forward.map_or(i, |f| f[i])] = z[start];
            }
            diagonal
        })
    }

    /// The entries of `A^-1` where `A` itself has entries: one value for each stored entry
    /// of the lower triangle of `A`, in the order in which [`Factor::new`] was given them.
    ///
    /// These are among the entries that selected inversion computes (see
    /// [`Factor::inverse_diagonal`]), because `L` has a non-zero wherever `A` does. The
    /// gradient of the criterion needs them (`CrossedModel::deviance_and_gradient`).
    pub fn inverse_on_pattern(&self) -> Vec<f64> {
        self.with_selected_inverse(|_, _, z| self.a_slots.iter().map(|&slot| z[slot]).collect())
    }

    /// Computes the selected inverse `Z` (see [`Factor::inverse_diagonal`]) and hands it
    /// to `read` together with the layout of `L`: `read(col_ptr, row_idx, z)`, where
    /// `z[t]` is the entry of `Z` in the row `row_idx[t]` of its column, and column `j`
    /// owns the entries from `col_ptr[j]` up to `col_ptr[j + 1]`, its diagonal first.
    /// The rows and columns are numbered as in the renumbered matrix `P A P'`.
    fn with_selected_inverse<T>(&self, read: impl FnOnce(&[usize], &[usize], &[f64]) -> T) -> T {
        // A supernodal factor is inverted a supernode at a time, with dense matrix
        // products. `z` comes back in the column layout that `Factor::new` worked out.
        let (col_ptr, row_idx, l): (&[usize], &[usize], &[f64]) =
            match (self.symbolic.raw(), &self.columns) {
                (SymbolicCholeskyRaw::Supernodal(s), Some((col_ptr, row_idx))) => {
                    let z = supernodal_selected_inverse(s, &self.l_values);
                    return read(col_ptr, row_idx, &z);
                }
                // not reached: a supernodal factor always has its columns
                (SymbolicCholeskyRaw::Supernodal(_), None) => return read(&[0], &[], &[]),
                // A simplicial factor is stored column by column, which is how the
                // recurrences below read it.
                (SymbolicCholeskyRaw::Simplicial(s), _) => {
                    (s.col_ptr(), s.row_idx(), &self.l_values)
                }
            };
        let n = self.dim();
        // z holds Z on the pattern of L: z[t] is the entry of Z where l[t] is the entry of L.
        let mut z = vec![0.0; l.len()];
        let mut acc: Vec<f64> = Vec::new();
        // position[row] = slot of `row` in the current column's pattern, usize::MAX if absent
        let mut position = vec![usize::MAX; n];
        for j in (0..n).rev() {
            let (start, end) = (col_ptr[j], col_ptr[j + 1]);
            let l_jj = l[start]; // rows are sorted, so the diagonal comes first
            let rows = &row_idx[start + 1..end];
            let l_col = &l[start + 1..end];
            acc.clear();
            acc.resize(rows.len(), 0.0);
            for (slot, &row) in rows.iter().enumerate() {
                position[row] = slot;
            }
            // acc[a] = sum_b Z[rows[a], rows[b]] L[rows[b], j]. Every Z[rows[a], k] with
            // k = rows[b] lives in column k of Z (its diagonal, or a row below it).
            for (b, &k) in rows.iter().enumerate() {
                let (k_start, k_end) = (col_ptr[k], col_ptr[k + 1]);
                // the diagonal entry Z[k, k]
                acc[b] += z[k_start] * l_col[b];
                // the entries below it: column k is walked, and the rows in R_j picked out
                for t in k_start + 1..k_end {
                    let a = position[row_idx[t]];
                    if a != usize::MAX {
                        // Z is symmetric, so the stored entry Z[rows[a], k] is also
                        // Z[k, rows[a]]: it contributes to both sums.
                        acc[a] += z[t] * l_col[b];
                        acc[b] += z[t] * l_col[a];
                    }
                }
            }
            for &row in rows {
                position[row] = usize::MAX;
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
        read(col_ptr, row_idx, &z)
    }
}

/// The selected inverse of a supernodal factor, computed a supernode at a time and
/// returned in the column layout of [`supernodal_columns`].
///
/// A supernode `J` is a run of consecutive columns of `L` that have the same rows below
/// them, `R`. Its part of `L` is two dense blocks: `L_JJ`, lower triangular, for the rows
/// of `J` itself, and `L_RJ` for the rows `R`. Takahashi's recurrences (see
/// [`Factor::inverse_diagonal`]), written for all the columns of `J` at once, are
///
/// ```text
/// Y    = L_RJ L_JJ^-1
/// Z_RJ = -Z_RR Y
/// Z_JJ = (L_JJ L_JJ')^-1 - Y' Z_RJ
/// ```
///
/// where `Z_RR` holds the entries of `Z` for every pair of rows of `R`. Those belong to
/// later supernodes, which is why the supernodes are taken from the last to the first.
/// The products are ordinary dense matrix products, which a processor does several
/// times faster than the same arithmetic one entry at a time.
fn supernodal_selected_inverse(
    s: &SymbolicSupernodalCholesky<usize>,
    l_values: &[f64],
) -> Vec<f64> {
    let n = s.nrows();
    let (begin, end, val_ptr) = (s.supernode_begin(), s.supernode_end(), s.col_ptr_for_val());
    // the supernode of every column
    let mut supernode_of = vec![0usize; n];
    for j in 0..s.n_supernodes() {
        supernode_of[begin[j]..end[j]].fill(j);
    }
    // Z is kept in the layout of L: for each supernode, a dense block with a column per
    // column of the supernode. Its first rows are Z_JJ (its lower triangle only, since it
    // is symmetric) and the rest Z_RJ, in the order of R.
    let mut z_values = vec![0.0; l_values.len()];
    // position[row]: where `row` is among the rows R of the supernode being read from
    let mut position = vec![0usize; n];
    // scratch for L_JJ^-1, Y and Z_RR, reused from supernode to supernode
    let (mut inverse_buffer, mut y_buffer, mut z_rr_buffer) = (Vec::new(), Vec::new(), Vec::new());
    for j in (0..s.n_supernodes()).rev() {
        let width = end[j] - begin[j];
        let rows = s.supernode(j).pattern();
        let m = rows.len();
        let l_block = MatRef::from_column_major_slice(
            &l_values[val_ptr[j]..val_ptr[j + 1]],
            width + m,
            width,
        );
        let (l_jj, l_rj) = (l_block.subrows(0, width), l_block.subrows(width, m));

        // ---- L_JJ^-1 and Y = L_RJ L_JJ^-1 ----
        // L_JJ and its inverse are lower triangular. The products below are told which
        // of their matrices are triangular, so that they skip the zeros.
        inverse_buffer.clear();
        inverse_buffer.resize(width * width, 0.0);
        let mut inverse = MatMut::from_column_major_slice_mut(&mut inverse_buffer, width, width);
        invert_lower_triangular(inverse.as_mut(), l_jj, Par::Seq);
        y_buffer.clear();
        y_buffer.resize(m * width, 0.0);
        let mut y = MatMut::from_column_major_slice_mut(&mut y_buffer, m, width);
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
        z_rr_buffer.clear();
        z_rr_buffer.resize(m * m, 0.0);
        let mut b = 0;
        while b < m {
            // The supernode k that holds column rows[b], and its block of Z.
            let k = supernode_of[rows[b]];
            let (k_start, k_width) = (begin[k], end[k] - begin[k]);
            let k_rows = s.supernode(k).pattern();
            for (t, &row) in k_rows.iter().enumerate() {
                position[row] = t;
            }
            let k_height = k_width + k_rows.len();
            let z_k = &z_values[val_ptr[k]..val_ptr[k + 1]];
            // Every row of R from rows[b] on that is a column of k is handled now.
            while b < m && rows[b] < end[k] {
                let column = rows[b] - k_start;
                let z_column = &z_k[column * k_height..(column + 1) * k_height];
                for a in b..m {
                    // Z[rows[a], rows[b]]: row rows[a] is either a column of k too, in
                    // the first rows of its block, or one of the rows below k. (It is
                    // always one or the other: two rows of one column of L share an
                    // entry.)
                    let value = if rows[a] < end[k] {
                        z_column[rows[a] - k_start]
                    } else {
                        z_column[k_width + position[rows[a]]]
                    };
                    z_rr_buffer[a + m * b] = value;
                    z_rr_buffer[b + m * a] = value;
                }
                b += 1;
            }
        }
        let z_rr = MatRef::from_column_major_slice(&z_rr_buffer, m, m);

        // ---- Z_RJ = -Z_RR Y, and Z_JJ = (L_JJ L_JJ')^-1 - Y' Z_RJ ----
        let z_block = MatMut::from_column_major_slice_mut(
            &mut z_values[val_ptr[j]..val_ptr[j + 1]],
            width + m,
            width,
        );
        let (mut z_jj, mut z_rj) = z_block.split_at_row_mut(width);
        matmul(
            z_rj.as_mut(),
            Accum::Replace,
            z_rr,
            y.as_ref(),
            -1.0,
            Par::Seq,
        );
        // (L_JJ L_JJ')^-1 = (L_JJ^-1)' L_JJ^-1: an upper triangular matrix times a lower
        // one. Only the lower triangle of the result is computed.
        let inverse = inverse.as_ref();
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
    supernodal_columns(s, &z_values).2
}

/// The columns of a supernodal `L` in compressed sparse column form, each with its diagonal
/// first, as a simplicial factor stores them.
///
/// Returns `(col_ptr, row_idx, values)`, laid out as described in `design.rs`.
fn supernodal_columns(
    s: &SymbolicSupernodalCholesky<usize>,
    l_values: &[f64],
) -> (Vec<usize>, Vec<usize>, Vec<f64>) {
    let llt = SupernodalLltRef::<usize, f64>::new(s, l_values);
    let (mut col_ptr, mut row_idx, mut values) = (vec![0], Vec::new(), Vec::new());
    for si in 0..s.n_supernodes() {
        // A dense block of consecutive columns: the diagonal block, then the rows of the
        // supernode's pattern.
        let node = llt.supernode(si);
        let (start, block) = (node.start(), node.val());
        for k in 0..block.ncols() {
            // Column k of the block: from its diagonal entry down to the end of the
            // diagonal block, then the rows that all columns of the supernode share.
            row_idx.extend(start + k..start + block.ncols());
            row_idx.extend_from_slice(node.pattern());
            values.extend(block.col(k).iter().skip(k));
            col_ptr.push(row_idx.len());
        }
    }
    (col_ptr, row_idx, values)
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
    use crate::testing::Lcg;
    use approx::{assert_abs_diff_eq, assert_relative_eq};
    use faer::Mat;
    use faer::linalg::solvers::{DenseSolveCore, Solve};

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
        let design = Design::from_full_codes(&[ev, st], &vec![0.0; n], &[], 0).unwrap();
        let mut values = vec![0.0; design.ztz.len()];
        design.fill_a(&[0.8, 1.3], &mut values);
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

    fn check_log_det_and_solve(kind: FactorKind) {
        for seed in 0..4 {
            let (design, values, dense) = random_matrix(seed, 120, 15, 11);
            let q = design.q();
            let mut factor = Factor::new(&design.col_ptr, &design.row_idx, 2, kind).unwrap();
            factor
                .factorize(&design.col_ptr, &design.row_idx, &values)
                .unwrap();
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

    #[test]
    fn inverse_diagonal_matches_dense_inverse() {
        for kind in [FactorKind::Simplicial, FactorKind::Supernodal] {
            for (n, events, sites) in SHAPES {
                for seed in 0..4 {
                    let (design, values, dense) = random_matrix(seed, n, events, sites);
                    let q = design.q();
                    let mut factor =
                        Factor::new(&design.col_ptr, &design.row_idx, 1, kind).unwrap();
                    assert_eq!(factor.is_simplicial(), kind == FactorKind::Simplicial);
                    factor
                        .factorize(&design.col_ptr, &design.row_idx, &values)
                        .unwrap();
                    let expected = dense.llt(Side::Lower).unwrap().inverse();
                    let diagonal = factor.inverse_diagonal();
                    for i in 0..q {
                        assert_abs_diff_eq!(diagonal[i], expected[(i, i)], epsilon = 1e-12);
                    }
                }
            }
        }
    }

    #[test]
    fn inverse_on_pattern_matches_dense_inverse() {
        // One value for each stored entry of A, in the order of its columns and rows.
        for kind in [FactorKind::Simplicial, FactorKind::Supernodal] {
            for (n, events, sites) in SHAPES {
                for seed in 0..4 {
                    let (design, values, dense) = random_matrix(seed, n, events, sites);
                    let mut factor =
                        Factor::new(&design.col_ptr, &design.row_idx, 1, kind).unwrap();
                    factor
                        .factorize(&design.col_ptr, &design.row_idx, &values)
                        .unwrap();
                    let expected = dense.llt(Side::Lower).unwrap().inverse();
                    let on_pattern = factor.inverse_on_pattern();
                    assert_eq!(on_pattern.len(), design.row_idx.len());
                    for c in 0..design.q() {
                        let entries = design.col_ptr[c]..design.col_ptr[c + 1];
                        let column = on_pattern[entries.clone()]
                            .iter()
                            .zip(&design.row_idx[entries]);
                        for (&value, &row) in column {
                            assert_abs_diff_eq!(value, expected[(row, c)], epsilon = 1e-12);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn the_sparse_shape_has_many_supernodes() {
        // What the two tests above rely on to reach every part of the supernodal code.
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
        let mut factor =
            Factor::new(&design.col_ptr, &design.row_idx, 1, FactorKind::Supernodal).unwrap();
        assert!(factor.is_simplicial());
        factor
            .factorize(&design.col_ptr, &design.row_idx, &values)
            .unwrap();
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
