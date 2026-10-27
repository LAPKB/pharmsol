//! Native bench matrix: macro-authored models, ODE + analytical, both workloads.
//! Default features only — this is the CI regression signal. DSL backends are in `dsl_matrix.rs`.
//!
//! Every cell except `native/cache-hit` measures solver work. The prediction
//! cache is keyed on (subject, parameters) and enabled by default, so a bench
//! that repeats the same call against a warm cache only times a hash lookup.
//!
//! IDs:
//! - `native/log-likelihood`    → `{workload}/{solver}` (cache disabled)
//! - `native/likelihood-matrix` → `{workload}/ode` (cache cleared before each iteration)
//! - `native/cache-hit`         → `{workload}/{solver}` (warm cache, lookup + clone only)

use std::hint::black_box;
use std::time::Duration;

use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use pharmsol::prelude::simulator::log_likelihood_matrix;
use pharmsol::prelude::*;
use pharmsol::{Cache, Equation, Parameters};

mod common;
use common::{
    assay_error_models, macro_analytical, macro_ode, matrix_data, named_params,
    subject_for_likelihood, support_points, SolverKind, Workload,
};

const MATRIX_N_SUBJECTS: usize = 32;
const MATRIX_N_SUPPORT: usize = 64;

fn id(workload: Workload, kind: SolverKind) -> BenchmarkId {
    BenchmarkId::from_parameter(format!("{}/{}", workload.label(), kind.label()))
}

fn bench_log_likelihood<E: Equation>(
    b: &mut criterion::Bencher,
    model: &E,
    subject: &Subject,
    theta: &Parameters,
    error_models: &AssayErrorModels,
) {
    b.iter(|| {
        black_box(
            model
                .log_likelihood(
                    black_box(subject),
                    black_box(theta),
                    black_box(error_models),
                )
                .unwrap(),
        )
    });
}

fn bench_matrix<E: Equation + Cache>(
    b: &mut criterion::Bencher,
    model: &E,
    data: &Data,
    theta: &ndarray::Array2<f64>,
    error_models: &AssayErrorModels,
) {
    // Clearing outside the timed region keeps the cache's miss + insert cost in
    // the measurement, which is what a fresh NPAG cycle pays.
    b.iter_batched(
        || model.clear_cache(),
        |()| {
            black_box(
                log_likelihood_matrix(
                    black_box(model),
                    black_box(data),
                    black_box(theta),
                    black_box(error_models),
                    false,
                )
                .unwrap(),
            )
        },
        BatchSize::PerIteration,
    );
}

fn log_likelihood_group(c: &mut Criterion) {
    let mut group = c.benchmark_group("native/log-likelihood");
    let error_models = assay_error_models();

    for workload in Workload::all() {
        let subject = subject_for_likelihood(workload);
        for kind in SolverKind::all() {
            group.bench_function(id(workload, kind), |b| match kind {
                SolverKind::Ode => {
                    let model = macro_ode(workload).disable_cache();
                    let theta = Parameters::with_model(&model, named_params(workload)).unwrap();
                    bench_log_likelihood(b, &model, &subject, &theta, &error_models);
                }
                SolverKind::Analytical => {
                    let model = macro_analytical(workload).disable_cache();
                    let theta = Parameters::with_model(&model, named_params(workload)).unwrap();
                    bench_log_likelihood(b, &model, &subject, &theta, &error_models);
                }
            });
        }
    }

    group.finish();
}

/// ODE only. An analytical matrix finishes in ~1 ms, so rayon scheduling
/// dominates and run-to-run drift reached ±10%; its solver cost is already
/// covered by `native/log-likelihood`.
fn likelihood_matrix_group(c: &mut Criterion) {
    let mut group = c.benchmark_group("native/likelihood-matrix");
    group.sample_size(20);
    group.measurement_time(Duration::from_secs(10));

    let error_models = assay_error_models();

    for workload in Workload::all() {
        let data = matrix_data(workload, MATRIX_N_SUBJECTS);
        let theta = support_points(workload, MATRIX_N_SUPPORT);
        group.bench_function(id(workload, SolverKind::Ode), |b| {
            bench_matrix(b, &macro_ode(workload), &data, &theta, &error_models)
        });
    }

    group.finish();
}

/// One cell guarding the cache-hit path (subject hash + parameter hash + lookup
/// + clone). ODE only: it is the solver whose cost the cache exists to hide.
fn cache_hit_group(c: &mut Criterion) {
    let mut group = c.benchmark_group("native/cache-hit");
    let error_models = assay_error_models();
    let workload = Workload::Repeat;
    let subject = subject_for_likelihood(workload);
    let model = macro_ode(workload);
    let theta = Parameters::with_model(&model, named_params(workload)).unwrap();
    model
        .log_likelihood(&subject, &theta, &error_models)
        .unwrap();

    group.bench_function(id(workload, SolverKind::Ode), |b| {
        bench_log_likelihood(b, &model, &subject, &theta, &error_models)
    });

    group.finish();
}

criterion_group!(
    native_matrix,
    log_likelihood_group,
    likelihood_matrix_group,
    cache_hit_group
);
criterion_main!(native_matrix);
