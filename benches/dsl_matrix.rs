//! DSL bench matrix (feature-gated): JIT across both workloads, ODE + analytical.
//! Mirrors `native_matrix.rs` but compiles models from DSL source, so each
//! `dsl/*` cell is directly comparable with its `native/*` twin.
//!
//! IDs:
//! - `dsl/compile`           → `{workload}/{kind}`
//! - `dsl/predict`           → `{workload}/{kind}` (cache disabled)
//! - `dsl/prediction-matrix` → `{workload}/ode` (cache cleared before each iteration)

use std::hint::black_box;
use std::time::Duration;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};

use pharmsol::dsl::{
    compile_module_source_to_runtime, CompiledRuntimeModel, RuntimeAnalyticalModel, RuntimeOdeModel,
};
use pharmsol::{Cache, Parameters};

mod common;
use common::{
    bench_matrix, bench_predict, dsl_model_name, dsl_source, matrix_data, named_params,
    single_subject, support_points, SolverKind, Workload,
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
                    let actual = iters.clamp(1, MAX_ITERS_PER_BATCH);
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

// ───────────────────────────── predict group ────────────────────────

fn predict_group(c: &mut Criterion) {
    let mut group = c.benchmark_group("dsl/predict");

    for workload in Workload::all() {
        let subject = single_subject(workload);
        for kind in SolverKind::all() {
            match kind {
                SolverKind::Ode => {
                    let model = compile_ode(workload).disable_cache();
                    let theta = Parameters::with_model(&model, named_params(workload)).unwrap();
                    group.bench_function(id(workload, kind), |b| {
                        bench_predict(b, &model, &subject, &theta)
                    });
                }
                SolverKind::Analytical => {
                    let model = compile_analytical(workload).disable_cache();
                    let theta = Parameters::with_model(&model, named_params(workload)).unwrap();
                    group.bench_function(id(workload, kind), |b| {
                        bench_predict(b, &model, &subject, &theta)
                    });
                }
            }
        }
    }

    group.finish();
}

// ───────────────────────────── prediction-matrix group ───────────────

/// ODE only, for the same reason as `native/prediction-matrix`.
fn prediction_matrix_group(c: &mut Criterion) {
    let mut group = c.benchmark_group("dsl/prediction-matrix");
    group.sample_size(20);
    group.measurement_time(Duration::from_secs(10));

    for workload in Workload::all() {
        let data = matrix_data(workload, MATRIX_N_SUBJECTS);
        let theta = support_points(workload, MATRIX_N_SUPPORT);
        let model = compile_ode(workload);
        group.bench_function(id(workload, SolverKind::Ode), |b| {
            bench_matrix(b, &model, &data, &theta)
        });
    }

    group.finish();
}

criterion_group!(
    dsl_matrix,
    compile_group,
    predict_group,
    prediction_matrix_group
);
criterion_main!(dsl_matrix);
