//! Step 1 of a fit: reduce the data to the few sums that every later step needs.
//!
//! The optimiser evaluates the criterion at dozens of trial values of theta (see
//! `model.rs`). Nothing in this file depends on theta, so it is all computed once, when
//! the model is created:
//!
//! - `Z'Z`: how many observations each level has, and how many each pair of levels shares;
//! - `Z'[y X]` and `[y X]'[y X]`: sums of the response (and of the fixed-effect columns)
//!   within each level, and over all observations.
//!
//! (`'` means transpose, and `[y X]` is the matrix with `y` as its first column followed
//! by the columns of `X`.) After this the individual observations are never looked at
//! again. The cost of evaluating the criterion depends on the number of levels (events
//! plus sites) and on how many different event-site pairs have a record, not directly on
//! the number of records.
//!
//! # A worked example
//!
//! Five observations, three events and two sites. This is the `small()` design in the
//! tests at the bottom of this file:
//!
//! ```text
//! observation   event   site    y
//!      0          0       0     1
//!      1          0       1     2
//!      2          1       0     3
//!      3          1       1     4
//!      4          2       1     5
//! ```
//!
//! The levels of all factors are *stacked* into one list: the events first (stacked
//! indices 0, 1, 2), then the sites (3, 4). `Z` has one row per observation and one column
//! per stacked level, with a 1 where the observation belongs to the level:
//!
//! ```text
//!            events    sites
//!           0  1  2    0  1
//!     Z = [ 1  0  0    1  0 ]   observation 0: event 0, site 0
//!         [ 1  0  0    0  1 ]   observation 1: event 0, site 1
//!         [ 0  1  0    1  0 ]   observation 2: event 1, site 0
//!         [ 0  1  0    0  1 ]   observation 3: event 1, site 1
//!         [ 0  0  1    0  1 ]   observation 4: event 2, site 1
//! ```
//!
//! `Z` itself is never built. The fit only needs `Z'Z`, which is a table of counts:
//!
//! ```text
//!             [ 2  0  0  1  1 ]   event 0: 2 observations, 1 at site 0 and 1 at site 1
//!             [ 0  2  0  1  1 ]   event 1: the same
//!     Z'Z =   [ 0  0  1  0  1 ]   event 2: 1 observation, at site 1
//!             [ 1  1  0  2  0 ]   site 0: 2 observations
//!             [ 1  1  1  0  3 ]   site 1: 3 observations
//! ```
//!
//! - An entry on the diagonal is the number of observations of that level.
//! - An entry off the diagonal is the number of observations that two levels share. Two
//!   levels of the same factor never share one (an observation has only one event), so
//!   those entries are zero.
//!
//! It also needs `Z'y`, the sum of `y` over each level's observations, here
//! `[1+2, 3+4, 5, 1+3, 2+4+5] = [3, 7, 5, 4, 11]`, and `y'y = 55`, the sum of squares.
//!
//! # How `Z'Z` is stored
//!
//! In real data most entries of `Z'Z` are zero, because most events are not recorded at
//! most sites. So only the non-zero entries are stored (a *sparse* matrix), and since
//! `Z'Z` is symmetric, only those on or below the diagonal. The layout is "compressed
//! sparse column" (CSC), the layout of `scipy.sparse.csc_matrix`. It is three lists:
//!
//! ```text
//!     column:    0          1          2       3    4
//!     ztz     = [2, 1, 1,   2, 1, 1,   1, 1,   2,   3]        the values, column by column
//!     row_idx = [0, 3, 4,   1, 3, 4,   2, 4,   3,   4]        the row of each value
//!     col_ptr = [0,         3,         6,      8,   9,  10]   where each column starts
//! ```
//!
//! Column `c` owns the entries from `col_ptr[c]` up to, but not including,
//! `col_ptr[c + 1]`. Column 0 owns entries 0, 1 and 2: the values 2, 1, 1 in rows 0, 3, 4.
//!
//! The three lists are made by a sparse-matrix library (faer). `Design::new` only lists
//! what goes into the matrix: for each observation, a 1 for each pair of levels that it
//! has (event 0 with site 0, ...), and for each level, its number of observations. The
//! library sorts that list into column order and adds up the entries that fall in the
//! same place, so a pair of levels that shares three observations, listed three times
//! with the value 1, becomes one entry with the value 3. This is what
//! `scipy.sparse.coo_matrix((values, (rows, columns))).tocsc()` does in Python.
//!
//! # An observation without a level
//!
//! An observation need not have a level in every factor. If the sites are split into one
//! factor per site class (so that each class has its own standard deviation), an
//! observation at a site of one class has no level in the factors of the other classes.
//! Its row of `Z` is then all zeros in those factors' columns: it adds nothing to their
//! counts and sums, and gets no term from them. In `codes`, such an entry is `None`
//! where the others are `Some(level)`. (`Option`, with its two cases `Some(value)` and
//! `None`, is how Rust writes "a value, or nothing".)

use faer::sparse::{SparseColMat, Triplet};

use crate::LmmError;
use crate::error::invalid;

/// The design of a model with `K` crossed random-intercept factors and `p` fixed effects:
/// everything about the data that the fit needs.
///
/// The levels of all factors are stacked into one index `0..q`, factor by factor. `Z'Z` is
/// stored as its lower triangle in compressed sparse column form: each column holds its
/// diagonal entry first, then the remaining row indices in ascending order. The module
/// documentation at the top of this file works through an example.
#[derive(Debug, Clone)]
pub struct Design {
    /// Number of observations.
    pub n: usize,
    /// Number of fixed-effect columns: the bias, if one is fitted, and the covariates.
    pub p: usize,
    /// Number of levels of each factor, e.g. `[479 events, 212 sites]`.
    pub levels_per_factor: Vec<usize>,
    /// Factor index of every stacked level, e.g. `[0, 0, 0, 1, 1]` in the worked example.
    pub level_factor: Vec<usize>,
    /// Column pointers of the lower triangle of `Z'Z` (length `q + 1`).
    pub col_ptr: Vec<usize>,
    /// Row indices of the lower triangle of `Z'Z`.
    pub row_idx: Vec<usize>,
    /// Value of `Z'Z` at each stored entry.
    pub ztz: Vec<f64>,
    /// `Z'[y X]`, `q x (1 + p)`, column-major: the first `q` values are column 0 (`Z'y`),
    /// the next `q` are column 1, and so on.
    pub zt_yx: Vec<f64>,
    /// `[y X]'[y X]`, `(1 + p) x (1 + p)`, column-major.
    pub yx_yx: Vec<f64>,
}

impl Design {
    /// Builds the design from per-factor level codes, the number of levels of each
    /// factor, the response and an `n x p` column-major fixed-effects matrix (empty when
    /// `p == 0`).
    ///
    /// `codes[f][r]` is the level of observation `r` in factor `f`, as an integer from 0
    /// inside `Some(...)`, or `None` if the observation has no level in that factor. In
    /// the worked example, `codes` is `[[0, 0, 1, 1, 2], [0, 1, 0, 1, 1]]`, each number
    /// in a `Some`, and `n_levels` is `[3, 2]`.
    ///
    /// The return type `Result<Self, LmmError>` means "a `Design`, or an error": the
    /// checks below return `Err(...)` for unusable input, and the last line returns
    /// `Ok(...)` with the finished design.
    pub fn new(
        codes: &[Vec<Option<usize>>],
        n_levels: &[usize],
        y: &[f64],
        x: &[f64],
        p: usize,
    ) -> Result<Self, LmmError> {
        // ---- Check the input ----
        let n = y.len();
        let k = codes.len();
        if k == 0 {
            return Err(invalid("at least one grouping factor is required"));
        }
        if n_levels.len() != k {
            return Err(invalid(format!(
                "{} numbers of levels were given for {k} grouping factors",
                n_levels.len()
            )));
        }
        if n == 0 {
            return Err(invalid("the response is empty"));
        }
        if n <= p {
            return Err(invalid(format!(
                "need more observations ({n}) than fixed effects ({p})"
            )));
        }
        if x.len() != n * p {
            return Err(invalid(format!(
                "x has {} values, expected n * p = {}",
                x.len(),
                n * p
            )));
        }
        // If some factor does not have one code per observation (f is the first that
        // does not):
        if let Some(f) = codes.iter().position(|c| c.len() != n) {
            return Err(invalid(format!(
                "grouping factor {f} has {} codes but the response has {n} values",
                codes[f].len()
            )));
        }
        // Every code must be one of its factor's levels: the codes are used as indexes.
        // (`flatten` skips the `None` entries and unwraps the `Some` ones.)
        for (f, factor_codes) in codes.iter().enumerate() {
            if factor_codes
                .iter()
                .flatten()
                .any(|&code| code >= n_levels[f])
            {
                return Err(invalid(format!(
                    "grouping factor {f} has a code that is not one of its {} levels",
                    n_levels[f]
                )));
            }
        }
        if y.iter().chain(x).any(|v| !v.is_finite()) {
            return Err(invalid(
                "the response or fixed effects contain NaN or infinite values",
            ));
        }

        // ---- Stack the levels of all factors into one index, 0..q ----
        let levels_per_factor = n_levels.to_vec();
        // offset[f] is the stacked index of factor f's first level: [0, 3, 5] in the worked
        // example (events start at 0, sites at 3, and there are q = 5 levels in total).
        let mut offset = vec![0usize; k + 1];
        for f in 0..k {
            offset[f + 1] = offset[f] + levels_per_factor[f];
        }
        let q = offset[k];
        // The factor of every stacked level: a 0 for each level of factor 0, then a 1 for
        // each level of factor 1, and so on. [0, 0, 0, 1, 1] in the worked example.
        let level_factor: Vec<usize> = (0..k)
            .flat_map(|f| std::iter::repeat_n(f, levels_per_factor[f]))
            .collect();
        // A small local function (a "closure", like a Python lambda): the stacked index of
        // the level that observation r has in factor f, or None if it has none.
        let level = |f: usize, r: usize| codes[f][r].map(|code| offset[f] + code);

        // ---- Z'Z ----
        // The entries of the lower triangle of Z'Z, as (row, column, value). For each
        // observation, a 1 for each pair of levels that it has: the pair shares that
        // observation. A pair that shares several observations is listed several times.
        let mut diagonal = vec![0.0f64; q];
        let mut entries = Vec::with_capacity(q + n * k * (k - 1) / 2);
        for r in 0..n {
            for f in 0..k {
                if let Some(c) = level(f, r) {
                    diagonal[c] += 1.0;
                }
            }
            for a in 0..k {
                for b in a + 1..k {
                    // Later factors have larger stacked indices, so this is below the
                    // diagonal: the earlier factor's level is the column.
                    if let (Some(column), Some(row)) = (level(a, r), level(b, r)) {
                        entries.push(Triplet::new(row, column, 1.0));
                    }
                }
            }
        }
        // The diagonal: the number of observations of each level. Every level gets its
        // entry, even with no observations, because A has a 1 added there (`fill_a`).
        for (c, &count) in diagonal.iter().enumerate() {
            entries.push(Triplet::new(c, c, count));
        }
        // The library sorts the entries by column, then by row within a column, and adds
        // up those in the same place. The diagonal is the first row of each column of a
        // lower triangle, so each column starts with its diagonal entry.
        let ztz = SparseColMat::<usize, f64>::try_new_from_triplets(q, q, &entries)
            .map_err(|error| invalid(format!("could not assemble Z'Z: {error:?}")))?;
        let col_ptr = ztz.symbolic().col_ptr().to_vec();
        let row_idx = ztz.symbolic().row_idx().to_vec();
        let ztz = ztz.val().to_vec();

        // ---- Z'[y X] and [y X]'[y X] ----
        // nc is the number of columns of [y X]. yx(r, c) reads entry (r, c) of [y X]
        // without building it: column 0 is y, and column c >= 1 is column c - 1 of x.
        let nc = 1 + p;
        let yx = |r: usize, c: usize| if c == 0 { y[r] } else { x[r + n * (c - 1)] };
        let mut zt_yx = vec![0.0; q * nc];
        let mut yx_yx = vec![0.0; nc * nc];
        for r in 0..n {
            for c in 0..nc {
                let value = yx(r, c);
                // Z'[y X]: add this value to the sum of each level the observation has.
                for f in 0..k {
                    if let Some(stacked) = level(f, r) {
                        zt_yx[stacked + q * c] += value;
                    }
                }
                // [y X]'[y X]: sums of products of two columns over all observations.
                for c2 in 0..nc {
                    yx_yx[c + nc * c2] += value * yx(r, c2);
                }
            }
        }
        Ok(Self {
            n,
            p,
            levels_per_factor,
            level_factor,
            col_ptr,
            row_idx,
            ztz,
            zt_yx,
            yx_yx,
        })
    }

    /// For the tests: a design in which every observation has a level in every factor,
    /// and a factor has as many levels as its largest code plus one.
    #[cfg(test)]
    pub(crate) fn from_full_codes(
        codes: &[Vec<usize>],
        y: &[f64],
        x: &[f64],
        p: usize,
    ) -> Result<Self, LmmError> {
        let n_levels: Vec<usize> = codes
            .iter()
            .map(|c| c.iter().max().map_or(0, |m| m + 1))
            .collect();
        let codes: Vec<Vec<Option<usize>>> = codes
            .iter()
            .map(|c| c.iter().copied().map(Some).collect())
            .collect();
        Self::new(&codes, &n_levels, y, x, p)
    }

    /// Total number of levels over all factors (events + sites).
    pub fn q(&self) -> usize {
        self.level_factor.len()
    }

    /// Number of grouping factors.
    pub fn n_factors(&self) -> usize {
        self.levels_per_factor.len()
    }

    /// Writes `A = Lambda' Z'Z Lambda + I` at `theta` into `values`, one value per stored
    /// entry of `Z'Z`.
    ///
    /// `Lambda` is the diagonal matrix holding, for each stacked level, the theta of that
    /// level's factor, and `I` is the identity matrix. So entry `(row, c)` of `A` is
    /// `theta[factor of row] * theta[factor of c] * Z'Z[row, c]`, plus 1 on the diagonal.
    /// `A` has non-zero entries exactly where `Z'Z` does, which is why it can share the
    /// `col_ptr` and `row_idx` of `Z'Z` and only needs its own list of values.
    ///
    /// This is where theta enters `A`. (`CrossedModel::profile` in `model.rs` uses theta
    /// once more, to scale `Z'[y X]`.) In the worked example
    /// with `theta = [2, 3]`, the first column of `A` (event 0) is
    /// `[2*2*2 + 1, 2*3*1, 2*3*1] = [9, 6, 6]`.
    pub(crate) fn fill_a(&self, theta: &[f64], values: &mut [f64]) {
        for c in 0..self.q() {
            let theta_c = theta[self.level_factor[c]];
            let entries = self.col_ptr[c]..self.col_ptr[c + 1];
            // Walk down column c, reading its row indices and Z'Z values and writing to the
            // matching slots of `values`.
            let column = values[entries.clone()]
                .iter_mut()
                .zip(&self.row_idx[entries.clone()])
                .zip(&self.ztz[entries]);
            for ((value, &row), &ztz) in column {
                *value = theta_c * theta[self.level_factor[row]] * ztz;
                if row == c {
                    *value += 1.0;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    // The layout tests below are redundant with the Python tests, which check the
    // criterion and the fitted values against lme4's (tests/resources/lme4_reference):
    // an assembly bug would fail those too, but only as a wrong result. These tests are
    // kept because they pinpoint which part of the assembly broke.
    use super::*;

    // Five observations: events [0, 0, 1, 1, 2], sites [0, 1, 0, 1, 1].
    // Stacked levels: events 0..3, sites 3..5.
    // This is the worked example of the module documentation at the top of this file.
    fn small() -> Design {
        let events = vec![0, 0, 1, 1, 2];
        let sites = vec![0, 1, 0, 1, 1];
        Design::from_full_codes(&[events, sites], &[1.0, 2.0, 3.0, 4.0, 5.0], &[], 0).unwrap()
    }

    #[test]
    fn stacks_levels_factor_by_factor() {
        let d = small();
        assert_eq!(d.levels_per_factor, vec![3, 2]);
        assert_eq!(d.level_factor, vec![0, 0, 0, 1, 1]);
        assert_eq!((d.q(), d.n_factors(), d.n, d.p), (5, 2, 5, 0));
    }

    #[test]
    fn stores_lower_triangle_of_ztz_with_diagonal_first() {
        let d = small();
        assert_eq!(d.col_ptr, vec![0, 3, 6, 8, 9, 10]);
        assert_eq!(d.row_idx, vec![0, 3, 4, 1, 3, 4, 2, 4, 3, 4]);
        assert_eq!(
            d.ztz,
            vec![2.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 2.0, 3.0]
        );
    }

    #[test]
    fn cross_products_with_the_response() {
        let d = small();
        // events: 1+2, 3+4, 5; sites: 1+3, 2+4+5
        assert_eq!(d.zt_yx, vec![3.0, 7.0, 5.0, 4.0, 11.0]);
        assert_eq!(d.yx_yx, vec![55.0]);
    }

    #[test]
    fn fills_a_at_theta() {
        let d = small();
        let mut values = vec![0.0; d.ztz.len()];
        d.fill_a(&[2.0, 3.0], &mut values);
        // diagonal: theta^2 * count + 1; below it: theta_event * theta_site * count
        assert_eq!(
            values,
            vec![9.0, 6.0, 6.0, 9.0, 6.0, 6.0, 5.0, 6.0, 19.0, 28.0]
        );
    }

    #[test]
    fn repeated_event_site_pairs_are_counted() {
        let d = Design::from_full_codes(&[vec![0, 0, 0], vec![0, 0, 1]], &[1.0, 1.0, 1.0], &[], 0)
            .unwrap();
        // event 0 (level 0) meets site 0 (level 1) twice and site 1 (level 2) once
        assert_eq!(d.col_ptr, vec![0, 3, 4, 5]);
        assert_eq!(d.row_idx, vec![0, 1, 2, 1, 2]);
        assert_eq!(d.ztz, vec![3.0, 2.0, 1.0, 2.0, 1.0]);
    }

    #[test]
    fn fixed_effect_columns_enter_the_cross_products() {
        let y = [1.0, 2.0, 4.0];
        let d = Design::from_full_codes(&[vec![0, 1, 1]], &y, &[1.0, 1.0, 1.0], 1).unwrap();
        // Z'[y 1]: level 0 -> (1, 1), level 1 -> (6, 2); column-major
        assert_eq!(d.zt_yx, vec![1.0, 6.0, 1.0, 2.0]);
        // [y 1]'[y 1] = [[21, 7], [7, 3]]
        assert_eq!(d.yx_yx, vec![21.0, 7.0, 7.0, 3.0]);
    }

    // The worked example again, but observation 4 has no site: its site code is None.
    fn small_with_a_missing_site() -> Design {
        let events = [0, 0, 1, 1, 2].map(Some).to_vec();
        let sites = vec![Some(0), Some(1), Some(0), Some(1), None];
        Design::new(
            &[events, sites],
            &[3, 2],
            &[1.0, 2.0, 3.0, 4.0, 5.0],
            &[],
            0,
        )
        .unwrap()
    }

    #[test]
    fn an_observation_without_a_level_adds_nothing_to_that_factor() {
        let d = small_with_a_missing_site();
        // Site 1 now has 2 observations (1 and 3), and event 2 shares none with any site.
        assert_eq!(d.col_ptr, vec![0, 3, 6, 7, 8, 9]);
        assert_eq!(d.row_idx, vec![0, 3, 4, 1, 3, 4, 2, 3, 4]);
        assert_eq!(d.ztz, vec![2.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 2.0, 2.0]);
        // events: 1+2, 3+4, 5; sites: 1+3, 2+4. Observation 4 still counts in y'y.
        assert_eq!(d.zt_yx, vec![3.0, 7.0, 5.0, 4.0, 6.0]);
        assert_eq!(d.yx_yx, vec![55.0]);
        assert_eq!((d.q(), d.n), (5, 5));
    }

    #[test]
    fn a_level_without_observations_keeps_its_place() {
        // Three levels are declared, and level 1 has no observations.
        let codes = [0, 0, 2].map(Some).to_vec();
        let d = Design::new(&[codes], &[3], &[1.0, 2.0, 4.0], &[], 0).unwrap();
        assert_eq!(d.col_ptr, vec![0, 1, 2, 3]);
        assert_eq!(d.row_idx, vec![0, 1, 2]);
        assert_eq!(d.ztz, vec![2.0, 0.0, 1.0]);
        assert_eq!(d.zt_yx, vec![3.0, 0.0, 4.0]);
    }

    #[test]
    fn rejects_codes_that_are_not_levels() {
        let y = [1.0, 2.0, 3.0];
        let codes = [0, 1, 2].map(Some).to_vec();
        // code 2 is not one of 2 levels
        assert!(Design::new(std::slice::from_ref(&codes), &[2], &y, &[], 0).is_err());
        // one level count per factor is needed
        assert!(Design::new(std::slice::from_ref(&codes), &[3, 3], &y, &[], 0).is_err());
        assert!(Design::new(std::slice::from_ref(&codes), &[3], &y, &[], 0).is_ok());
    }

    #[test]
    fn rejects_invalid_input() {
        let ok = vec![0, 1, 1];
        let one_factor = std::slice::from_ref(&ok);
        assert!(Design::from_full_codes(&[], &[1.0, 2.0, 3.0], &[], 0).is_err());
        assert!(Design::from_full_codes(one_factor, &[], &[], 0).is_err());
        assert!(Design::from_full_codes(&[vec![0, 1]], &[1.0, 2.0, 3.0], &[], 0).is_err());
        assert!(Design::from_full_codes(one_factor, &[1.0, f64::NAN, 3.0], &[], 0).is_err());
        assert!(Design::from_full_codes(one_factor, &[1.0, 2.0, 3.0], &[1.0, 1.0], 1).is_err());
        assert!(Design::from_full_codes(one_factor, &[1.0, 2.0, 3.0], &[1.0; 9], 3).is_err());
    }
}
