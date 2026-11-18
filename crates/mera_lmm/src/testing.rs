//! Test helpers: a deterministic random-number generator, and random designs for the
//! property tests.
//!
//! Most tests need a random design and response, and should use the same one every time.
//! A small generator written here, rather than a random-number library, gives the same
//! numbers on every platform and compiler version, so such a test cannot pass on one
//! machine and fail on another.
//!
//! The *property tests* are the exception, on purpose. Each checks a rule that must hold
//! for every design (see `two_factor_codes` below), on designs that the `proptest`
//! library makes up afresh at every run, including awkward ones that nobody would think
//! of writing down. When a rule fails, proptest searches for a smaller design that still
//! breaks it, prints that design, and records it in a `proptest-regressions` folder,
//! which should be committed: the design is then tried first at every later run.

use std::ops::RangeInclusive;

use proptest::prelude::*;

/// Linear congruential generator with Knuth's MMIX constants; identical on every platform.
pub(crate) struct Lcg(u64);

impl Lcg {
    pub(crate) fn new(seed: u64) -> Self {
        // Mix the seed so that small seeds (0, 1, 2, ...) do not start with similar numbers.
        Self(seed ^ 0x9E37_79B9_7F4A_7C15)
    }

    /// Uniform number in `[0, 1)`.
    pub(crate) fn next_f64(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        // The top 53 bits, the precision of an f64, scaled to [0, 1).
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Uniform integer in `0..m`.
    pub(crate) fn below(&mut self, m: usize) -> usize {
        ((self.next_f64() * m as f64) as usize).min(m - 1)
    }

    /// Standard normal draw (Box–Muller).
    pub(crate) fn normal(&mut self) -> f64 {
        let u1 = self.next_f64().max(f64::MIN_POSITIVE);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

/// For the property tests: a random design with two grouping factors, as the event and
/// the site of each record.
///
/// What this returns is a *strategy*: a recipe that tells proptest how to make a random
/// input. This one first draws the number of events (1 to `max_events`), the number of
/// sites (1 to `max_sites`) and the number of records (within `n_records`), and then an
/// event and a site for each record. Nothing makes the events or sites equally common,
/// so some of them end up with a single record, or with none.
pub(crate) fn two_factor_codes(
    max_events: usize,
    max_sites: usize,
    n_records: RangeInclusive<usize>,
) -> impl Strategy<Value = (Vec<usize>, Vec<usize>)> {
    (1..=max_events, 1..=max_sites, n_records).prop_flat_map(|(events, sites, n)| {
        (
            prop::collection::vec(0..events, n),
            prop::collection::vec(0..sites, n),
        )
    })
}
