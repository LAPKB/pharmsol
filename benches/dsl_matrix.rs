//! DSL bench matrix (feature-gated): JIT across both workloads, ODE + analytical.
//! Mirrors `native_matrix.rs` but compiles models from DSL source, so each
//! `dsl/*` cell is directly comparable with its `native/*` twin.
//!
//! IDs:
//! - `dsl/compile`           → `{workload}/{kind}`
//! - `dsl/log-likelihood`    → `{workload}/{kind}` (cache disabled)
//! - `dsl/likelihood-matrix` → `{workload}/ode` (cache cleared before each iteration)

use std::hint::black_box;
use std::time::Duration;

use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};

use pharmsol::dsl::{
    compile_module_source_to_runtime, CompiledRuntimeModel, RuntimeAnalyticalModel, RuntimeOdeModel,
};
use pharmsol::prelude::simulator::log_likelihood_matrix;
use pharmsol::prelude::*;
use pharmsol::{Cache, Equation, Parameters};

mod common;
use common::{
    assay_error_models, dsl_model_name, dsl_source, matrix_data, named_params,
    subject_for_likelihood, support_points, SolverKind, Workload,
};

const MATRIX_N_SUBJECTS: usize = 32;
const MATRIX_N_SUPPORT: usize = 64;

fn id(workload: Workload, kind: SolverKind) -> BenchmarkId {
    BenchmarkId::from_parameter(format!("{}/{}", workload.label(), kind.label()))
}

/// Compile `(workload, kind)` and return the full `CompiledRuntimeModel`.
fn compile_runtime(workload: Workload, kind: SolverKind) -> CompiledRuntimeModel {
    let source = dsl_source(workload, kind);
    let name = dsl_model_name(workload, kind);
    compile_module_source_to_runtime(source, Some(name), |_, _| {})
        .unwrap_or_else(|e| panic!("compile {name} failed: {e:?}"))
}

fn compile_ode(workload: Workload) -> RuntimeOdeModel {
    match compile_runtime(workload, SolverKind::Ode) {
        CompiledRuntimeModel::Ode(model) => model,
        other => panic!(
            "expected Ode model for {}, got {:?}",
            workload.label(),
            other.kind()
        ),
    }
}

fn compile_analytical(workload: Workload) -> RuntimeAnalyticalModel {
    match compile_runtime(workload, SolverKind::Analytical) {
        CompiledRuntimeModel::Analytical(model) => model,
        other => panic!(
            "expected Analytical model for {}, got {:?}",
            workload.label(),
            other.kind()
        ),
    }
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
                .estimate_log_likelihood(
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

// ───────────────────────────── compile group ─────────────────────────

fn compile_group(c: &mut Criterion) {
    use std::time::Instant;

    let mut group = c.benchmark_group("dsl/compile");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(5));
    // Each compile leaks an executable mmap. Without a cap, a fast JIT compile
    // (~60 µs) lets Criterion request hundreds of thousands of iterations per
    // cell and exhausts the runner's executable memory pool /
    // `vm.max_map_count`. We hard-cap real iterations per Criterion batch to
    // `MAX_ITERS_PER_BATCH` and scale the reported elapsed time linearly so
    // per-iteration timings stay accurate.
    const MAX_ITERS_PER_BATCH: u64 = 25;

    for workload in Workload::all() {
        for kind in SolverKind::all() {
            group.bench_function(id(workload, kind), |b| {
                b.iter_custom(|iters| {
                    let actual = iters.min(MAX_ITERS_PER_BATCH).max(1);
                    let start = Instant::now();
                    for _ in 0..actual {
                        black_box(compile_runtime(black_box(workload), black_box(kind)));
                    }
                    let elapsed = start.elapsed();
                    elapsed.mul_f64(iters as f64 / actual as f64)
                });
            });
        }
    }

    group.finish();
}

// ───────────────────────────── log-likelihood group ──────────────────

fn log_likelihood_group(c: &mut Criterion) {
    let mut group = c.benchmark_group("dsl/log-likelihood");
    let error_models = assay_error_models();

    for workload in Workload::all() {
        let subject = subject_for_likelihood(workload);
        for kind in SolverKind::all() {
            match kind {
                SolverKind::Ode => {
                    let model = compile_ode(workload).disable_cache();
                    let theta = Parameters::with_model(&model, named_params(workload)).unwrap();
                    group.bench_function(id(workload, kind), |b| {
                        bench_log_likelihood(b, &model, &subject, &theta, &error_models)
                    });
                }
                SolverKind::Analytical => {
                    let model = compile_analytical(workload).disable_cache();
                    let theta = Parameters::with_model(&model, named_params(workload)).unwrap();
                    group.bench_function(id(workload, kind), |b| {
                        bench_log_likelihood(b, &model, &subject, &theta, &error_models)
                    });
                }
            }
        }
    }

    group.finish();
}

// ───────────────────────────── likelihood-matrix group ───────────────

/// ODE only, for the same reason as `native/likelihood-matrix`.
fn likelihood_matrix_group(c: &mut Criterion) {
    let mut group = c.benchmark_group("dsl/likelihood-matrix");
    group.sample_size(20);
    group.measurement_time(Duration::from_secs(10));

    let error_models = assay_error_models();

    for workload in Workload::all() {
        let data = matrix_data(workload, MATRIX_N_SUBJECTS);
        let theta = support_points(workload, MATRIX_N_SUPPORT);
        let model = compile_ode(workload);
        group.bench_function(id(workload, SolverKind::Ode), |b| {
            bench_matrix(b, &model, &data, &theta, &error_models)
        });
    }

    group.finish();
}

criterion_group!(
    dsl_matrix,
    compile_group,
    log_likelihood_group,
    likelihood_matrix_group
);
criterion_main!(dsl_matrix);
