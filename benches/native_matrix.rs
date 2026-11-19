//! Native bench matrix: macro-authored models, ODE + analytical, both workloads.
//! Default features only — this is the CI regression signal. DSL backends are in `dsl_matrix.rs`.
//!
//! Every cell except `native/cache-hit` measures solver work. The prediction
//! cache is keyed on (subject, parameters) and enabled by default, so a bench
//! that repeats the same call against a warm cache only times a hash lookup.
//!
//! IDs:
//! - `native/predict`           → `{workload}/{solver}` (cache disabled)
//! - `native/prediction-matrix` → `{workload}/ode` (cache cleared before each iteration)
//! - `native/cache-hit`         → `{workload}/{solver}` (warm cache, lookup + `Arc` clone only)

use std::time::Duration;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use pharmsol::{Cache, Equation, Parameters};

mod common;
use common::{
    bench_matrix, bench_predict, macro_analytical, macro_ode, matrix_data, named_params,
    single_subject, support_points, SolverKind, Workload,
};

const MATRIX_N_SUBJECTS: usize = 32;
const MATRIX_N_SUPPORT: usize = 64;

fn id(workload: Workload, kind: SolverKind) -> BenchmarkId {
    BenchmarkId::from_parameter(format!("{}/{}", workload.label(), kind.label()))
}

fn predict_group(c: &mut Criterion) {
    let mut group = c.benchmark_group("native/predict");

    for workload in Workload::all() {
        let subject = single_subject(workload);
        for kind in SolverKind::all() {
            group.bench_function(id(workload, kind), |b| match kind {
                SolverKind::Ode => {
                    let model = macro_ode(workload).disable_cache();
                    let theta = Parameters::with_model(&model, named_params(workload)).unwrap();
                    bench_predict(b, &model, &subject, &theta);
                }
                SolverKind::Analytical => {
                    let model = macro_analytical(workload).disable_cache();
                    let theta = Parameters::with_model(&model, named_params(workload)).unwrap();
                    bench_predict(b, &model, &subject, &theta);
                }
            });
        }
    }

    group.finish();
}

/// ODE only. An analytical matrix finishes in ~1 ms, so rayon scheduling
/// dominates and run-to-run drift reached ±10%; its solver cost is already
/// covered by `native/predict`.
fn prediction_matrix_group(c: &mut Criterion) {
    let mut group = c.benchmark_group("native/prediction-matrix");
    group.sample_size(20);
    group.measurement_time(Duration::from_secs(10));

    for workload in Workload::all() {
        let data = matrix_data(workload, MATRIX_N_SUBJECTS);
        let theta = support_points(workload, MATRIX_N_SUPPORT);
        group.bench_function(id(workload, SolverKind::Ode), |b| {
            bench_matrix(b, &macro_ode(workload), &data, &theta)
        });
    }

    group.finish();
}

/// One cell guarding the cache-hit path: subject hash, parameter hash, lookup
/// and `Arc` clone. ODE only: it is the solver whose cost the cache exists to
/// hide.
fn cache_hit_group(c: &mut Criterion) {
    let mut group = c.benchmark_group("native/cache-hit");
    let workload = Workload::Repeat;
    let subject = single_subject(workload);
    let model = macro_ode(workload);
    let theta = Parameters::with_model(&model, named_params(workload)).unwrap();
    model.predict_shared(&subject, &theta).unwrap();

    group.bench_function(id(workload, SolverKind::Ode), |b| {
        bench_predict(b, &model, &subject, &theta)
    });

    group.finish();
}

criterion_group!(
    native_matrix,
    predict_group,
    prediction_matrix_group,
    cache_hit_group
);
criterion_main!(native_matrix);
