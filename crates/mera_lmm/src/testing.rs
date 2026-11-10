//! Test helpers: a deterministic RNG and a tolerance check.
//!
//! The tests need random designs and responses. A small generator written here, rather
//! than a random-number library, gives the same numbers on every platform and compiler
//! version, so a test cannot pass on one machine and fail on another.

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

/// Asserts `|a - b| <= tol * max(1, |b|)`.
pub(crate) fn assert_close(a: f64, b: f64, tol: f64, what: &str) {
    assert!(
        (a - b).abs() <= tol * b.abs().max(1.0),
        "{what}: {a} vs {b}"
    );
}
