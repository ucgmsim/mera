//! Benchmarks of the engine: how long its four calls take on three kinds of design.
//!
//! Run them with `cargo bench`. Criterion runs each call many times, prints the time per
//! call with its spread, compares it with the previous run, and writes an HTML report to
//! `target/criterion/report/index.html`. On Linux and macOS,
//! `cargo bench -- --profile-time 5` also writes a flamegraph for each benchmark, which
//! shows where the time goes.
//!
//! The calls are the ones that a fit makes (see `mera/lmm.py`): the model is set up once,
//! the search calls `deviance_and_gradient` about ten times, and `fit_at` is called once
//! at the end. `deviance` is the criterion without its slope, which only the fallback
//! search uses. The search itself is Python code, so it is not timed here.

use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use mera_lmm::{CrossedModel, Design};

/// A small random-number generator (the one in `src/testing.rs`), so that the designs
/// are the same on every run and on every machine.
struct Lcg(u64);

impl Lcg {
    /// Uniform number in `[0, 1)`.
    fn next_f64(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Uniform integer in `0..m`.
    fn below(&mut self, m: usize) -> usize {
        ((self.next_f64() * m as f64) as usize).min(m - 1)
    }
}

/// The data of one fit: for each grouping factor (events, then sites) the level of
/// every record and the number of levels, and the response.
struct Data {
    codes: Vec<Vec<Option<usize>>>,
    n_levels: Vec<usize>,
    y: Vec<f64>,
}

impl Data {
    /// Puts the records together, and gives each a response: a term for its event, a
    /// term for its site, and noise.
    fn new(events: Vec<usize>, sites: Vec<usize>, n_events: usize, n_sites: usize) -> Self {
        let mut rng = Lcg(1);
        let event_terms: Vec<f64> = (0..n_events).map(|_| rng.next_f64() - 0.5).collect();
        let site_terms: Vec<f64> = (0..n_sites).map(|_| rng.next_f64() - 0.5).collect();
        let y = events
            .iter()
            .zip(&sites)
            .map(|(&event, &site)| event_terms[event] + site_terms[site] + rng.next_f64() - 0.5)
            .collect();
        Self {
            codes: vec![
                events.into_iter().map(Some).collect(),
                sites.into_iter().map(Some).collect(),
            ],
            n_levels: vec![n_events, n_sites],
            y,
        }
    }

    /// Sites on a square grid, `side` by `side`, and each event recorded at the sites of
    /// a small square of the grid, `block` by `block`: an event is recorded by the
    /// sites near it, as in real networks.
    fn clustered(n_events: usize, side: usize, block: usize) -> Self {
        let mut rng = Lcg(2);
        let (mut events, mut sites) = (Vec::new(), Vec::new());
        for event in 0..n_events {
            // a corner of the event's square, anywhere on the grid
            let (row, column) = (rng.below(side), rng.below(side));
            for i in 0..block {
                for j in 0..block {
                    events.push(event);
                    // (the square wraps round at the edges of the grid)
                    sites.push((row + i) % side * side + (column + j) % side);
                }
            }
        }
        Self::new(events, sites, n_events, side * side)
    }

    /// Each event recorded at `per_event` sites chosen at random from all of them, which
    /// leaves the matrices with little sparsity to exploit.
    fn scattered(n_events: usize, n_sites: usize, per_event: usize) -> Self {
        let mut rng = Lcg(3);
        let (mut events, mut sites) = (Vec::new(), Vec::new());
        for event in 0..n_events {
            let mut chosen: Vec<usize> = Vec::with_capacity(per_event);
            while chosen.len() < per_event {
                let site = rng.below(n_sites);
                if !chosen.contains(&site) {
                    chosen.push(site);
                }
            }
            events.extend(std::iter::repeat_n(event, per_event));
            sites.extend(chosen);
        }
        Self::new(events, sites, n_events, n_sites)
    }
}

/// Times the four calls on one design. The benchmarks are named `<design>/<call>`.
fn bench_design(c: &mut Criterion, name: &str, data: &Data) {
    let n = data.y.len();
    // The fixed effects as `fit_lmm` hands them over: an intercept, as a column of
    // length 1.
    let x = vec![1.0 / (n as f64).sqrt(); n];
    let design = || Design::new(&data.codes, &data.n_levels, &data.y, &x, 1).unwrap();
    // a trial theta for events and sites, like those of ground-motion residuals
    let theta = [0.8, 0.9];

    let mut group = c.benchmark_group(name);
    // Reducing the data to sums, and analysing the pattern of the matrix: once per fit.
    group.bench_function("set-up", |b| {
        b.iter(|| CrossedModel::new(design(), true).unwrap())
    });
    // (`black_box` stops the compiler from working anything out ahead of time.)
    let mut model = CrossedModel::new(design(), true).unwrap();
    group.bench_function("deviance", |b| {
        b.iter(|| model.deviance(black_box(&theta)).unwrap())
    });
    group.bench_function("deviance_and_gradient", |b| {
        b.iter(|| model.deviance_and_gradient(black_box(&theta)).unwrap())
    });
    group.bench_function("fit_at", |b| {
        b.iter(|| model.fit_at(black_box(&theta)).unwrap())
    });
    group.finish();
}

fn engine(c: &mut Criterion) {
    // About the size of mera's example data: 500 events, 225 sites, 4,500 records.
    bench_design(c, "clustered_4500_records", &Data::clustered(500, 15, 3));
    // 2,000 events, 1,024 sites, 32,000 records.
    bench_design(c, "clustered_32000_records", &Data::clustered(2000, 32, 4));
    // 2,000 events, 1,000 sites, 30,000 records.
    bench_design(
        c,
        "scattered_30000_records",
        &Data::scattered(2000, 1000, 15),
    );
}

/// Criterion's settings: the defaults, with the profiler that writes flamegraphs where
/// it is available.
#[cfg(unix)]
fn settings() -> Criterion {
    use pprof::criterion::{Output, PProfProfiler};
    Criterion::default().with_profiler(PProfProfiler::new(100, Output::Flamegraph(None)))
}

#[cfg(not(unix))]
fn settings() -> Criterion {
    Criterion::default()
}

criterion_group! {
    name = benches;
    config = settings();
    targets = engine
}
criterion_main!(benches);
