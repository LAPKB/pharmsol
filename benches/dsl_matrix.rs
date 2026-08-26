//! DSL bench matrix (feature-gated): JIT across all workloads + solvers.
//! Mirrors `native_matrix.rs` but compiles models from DSL source.
//!
//! IDs:
//! - `dsl/compile`           → `{workload}/{kind}/{backend}`
//! - `dsl/predictions`       → `{workload}/{kind}/{backend}/{cache}`
//! - `dsl/log-likelihood`    → `{workload}/{kind}/{backend}/{cache}`
//! - `dsl/likelihood-matrix` → `{workload}/{kind}/{backend}`

use std::hint::black_box;
use std::time::Duration;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, SamplingMode};

use pharmsol::dsl::{
    compile_module_source_to_runtime, CompiledRuntimeModel, RuntimeAnalyticalModel,
    RuntimeOdeModel, RuntimeSdeModel,
};
use pharmsol::prelude::*;
use pharmsol::{Cache, Parameters};

mod common;
use common::{
    assay_error_models, dsl_model_name, dsl_source, matrix_data, named_params,
    subject_for_likelihood, subject_for_predictions, support_points, SolverKind, Workload,
};

const MATRIX_N_SUBJECTS: usize = 32;
const MATRIX_N_SUPPORT: usize = 64;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Backend {
    Jit,
}

impl Backend {
    fn label(self) -> &'static str {
        match self {
            Self::Jit => "dsl-jit",
        }
    }

    fn all() -> [Backend; 1] {
        [Backend::Jit]
    }
}

#[derive(Debug, Clone, Copy)]
enum CacheState {
    Hot,
    Cold,
}

impl CacheState {
    fn label(self) -> &'static str {
        match self {
            Self::Hot => "hot",
            Self::Cold => "cold",
        }
    }

    fn all() -> [CacheState; 2] {
        [CacheState::Hot, CacheState::Cold]
    }
}

/// Compile `(workload, kind)` with `backend` and return the full `CompiledRuntimeModel`.
fn compile_runtime(workload: Workload, kind: SolverKind, backend: Backend) -> CompiledRuntimeModel {
    let source = dsl_source(workload, kind);
    let name = dsl_model_name(workload, kind);
    compile_module_source_to_runtime(source, Some(name), |_, _| {})
        .unwrap_or_else(|e| panic!("compile {} via {} failed: {e:?}", name, backend.label()))
}

fn compile_ode(workload: Workload, backend: Backend) -> RuntimeOdeModel {
    match compile_runtime(workload, SolverKind::Ode, backend) {
        CompiledRuntimeModel::Ode(model) => model,
        other => panic!(
            "expected Ode model for {}, got {:?}",
            workload.label(),
            other.kind()
        ),
    }
}

fn compile_analytical(workload: Workload, backend: Backend) -> RuntimeAnalyticalModel {
    match compile_runtime(workload, SolverKind::Analytical, backend) {
        CompiledRuntimeModel::Analytical(model) => model,
        other => panic!(
            "expected Analytical model for {}, got {:?}",
            workload.label(),
            other.kind()
        ),
    }
}

fn compile_sde(workload: Workload, backend: Backend) -> RuntimeSdeModel {
    match compile_runtime(workload, SolverKind::Sde, backend) {
        CompiledRuntimeModel::Sde(model) => model,
        other => panic!(
            "expected Sde model for {}, got {:?}",
            workload.label(),
            other.kind()
        ),
    }
}

fn ode_parameters(model: &RuntimeOdeModel, workload: Workload) -> Parameters {
    Parameters::with_model(model, named_params(workload, SolverKind::Ode))
        .expect("DSL ODE bench parameters should validate")
}

fn analytical_parameters(model: &RuntimeAnalyticalModel, workload: Workload) -> Parameters {
    Parameters::with_model(model, named_params(workload, SolverKind::Analytical))
        .expect("DSL analytical bench parameters should validate")
}

fn sde_parameters(model: &RuntimeSdeModel, workload: Workload) -> Parameters {
    Parameters::with_model(model, named_params(workload, SolverKind::Sde))
        .expect("DSL SDE bench parameters should validate")
}

// ───────────────────────────── compile group ─────────────────────────

fn compile_group(c: &mut Criterion) {
    use std::time::Instant;

    let mut group = c.benchmark_group("dsl/compile");
    group.sampling_mode(SamplingMode::Flat);
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
            for backend in Backend::all() {
                let bench_id = BenchmarkId::from_parameter(format!(
                    "{}/{}/{}",
                    workload.label(),
                    kind.label(),
                    backend.label()
                ));
                group.bench_function(bench_id, |b| {
                    b.iter_custom(|iters| {
                        let actual = iters.min(MAX_ITERS_PER_BATCH).max(1);
                        let start = Instant::now();
                        for _ in 0..actual {
                            black_box(compile_runtime(
                                black_box(workload),
                                black_box(kind),
                                black_box(backend),
                            ));
                        }
                        let elapsed = start.elapsed();
                        elapsed.mul_f64(iters as f64 / actual as f64)
                    });
                });
            }
        }
    }

    group.finish();
}

// ───────────────────────────── predictions group ─────────────────────

fn predictions_group(c: &mut Criterion) {
    let mut group = c.benchmark_group("dsl/predictions");
    group.sampling_mode(SamplingMode::Flat);

    for workload in Workload::all() {
        let subject = subject_for_predictions(workload);
        for kind in SolverKind::all() {
            for backend in Backend::all() {
                for cache in CacheState::all() {
                    let bench_id = BenchmarkId::from_parameter(format!(
                        "{}/{}/{}/{}",
                        workload.label(),
                        kind.label(),
                        backend.label(),
                        cache.label()
                    ));
                    match kind {
                        SolverKind::Ode => {
                            let model = match cache {
                                CacheState::Hot => compile_ode(workload, backend),
                                CacheState::Cold => compile_ode(workload, backend).disable_cache(),
                            };
                            let theta = ode_parameters(&model, workload);
                            group.bench_function(bench_id, |b| {
                                b.iter(|| {
                                    black_box(
                                        model
                                            .estimate_predictions(
                                                black_box(&subject),
                                                black_box(&theta),
                                            )
                                            .unwrap(),
                                    )
                                });
                            });
                        }
                        SolverKind::Analytical => {
                            let model = match cache {
                                CacheState::Hot => compile_analytical(workload, backend),
                                CacheState::Cold => {
                                    compile_analytical(workload, backend).disable_cache()
                                }
                            };
                            let theta = analytical_parameters(&model, workload);
                            group.bench_function(bench_id, |b| {
                                b.iter(|| {
                                    black_box(
                                        model
                                            .estimate_predictions(
                                                black_box(&subject),
                                                black_box(&theta),
                                            )
                                            .unwrap(),
                                    )
                                });
                            });
                        }
                        SolverKind::Sde => {
                            let model = match cache {
                                CacheState::Hot => compile_sde(workload, backend),
                                CacheState::Cold => compile_sde(workload, backend).disable_cache(),
                            };
                            let theta = sde_parameters(&model, workload);
                            group.bench_function(bench_id, |b| {
                                b.iter(|| {
                                    black_box(
                                        model
                                            .estimate_predictions(
                                                black_box(&subject),
                                                black_box(&theta),
                                            )
                                            .unwrap(),
                                    )
                                });
                            });
                        }
                    }
                }
            }
        }
    }

    group.finish();
}

// ───────────────────────────── log-likelihood group ──────────────────

fn log_likelihood_group(c: &mut Criterion) {
    let mut group = c.benchmark_group("dsl/log-likelihood");
    group.sampling_mode(SamplingMode::Flat);

    let error_models = assay_error_models();

    for workload in Workload::all() {
        let subject = subject_for_likelihood(workload);
        for kind in SolverKind::all() {
            for backend in Backend::all() {
                for cache in CacheState::all() {
                    let bench_id = BenchmarkId::from_parameter(format!(
                        "{}/{}/{}/{}",
                        workload.label(),
                        kind.label(),
                        backend.label(),
                        cache.label()
                    ));
                    match kind {
                        SolverKind::Ode => {
                            let model = match cache {
                                CacheState::Hot => compile_ode(workload, backend),
                                CacheState::Cold => compile_ode(workload, backend).disable_cache(),
                            };
                            let theta = ode_parameters(&model, workload);
                            group.bench_function(bench_id, |b| {
                                b.iter(|| {
                                    black_box(
                                        model
                                            .estimate_log_likelihood(
                                                black_box(&subject),
                                                black_box(&theta),
                                                black_box(&error_models),
                                            )
                                            .unwrap(),
                                    )
                                });
                            });
                        }
                        SolverKind::Analytical => {
                            let model = match cache {
                                CacheState::Hot => compile_analytical(workload, backend),
                                CacheState::Cold => {
                                    compile_analytical(workload, backend).disable_cache()
                                }
                            };
                            let theta = analytical_parameters(&model, workload);
                            group.bench_function(bench_id, |b| {
                                b.iter(|| {
                                    black_box(
                                        model
                                            .estimate_log_likelihood(
                                                black_box(&subject),
                                                black_box(&theta),
                                                black_box(&error_models),
                                            )
                                            .unwrap(),
                                    )
                                });
                            });
                        }
                        SolverKind::Sde => {
                            let model = match cache {
                                CacheState::Hot => compile_sde(workload, backend),
                                CacheState::Cold => compile_sde(workload, backend).disable_cache(),
                            };
                            let theta = sde_parameters(&model, workload);
                            group.bench_function(bench_id, |b| {
                                b.iter(|| {
                                    black_box(
                                        model
                                            .estimate_log_likelihood(
                                                black_box(&subject),
                                                black_box(&theta),
                                                black_box(&error_models),
                                            )
                                            .unwrap(),
                                    )
                                });
                            });
                        }
                    }
                }
            }
        }
    }

    group.finish();
}

// ───────────────────────────── likelihood-matrix group ───────────────

fn likelihood_matrix_group(c: &mut Criterion) {
    use pharmsol::prelude::simulator::log_likelihood_matrix;

    let mut group = c.benchmark_group("dsl/likelihood-matrix");
    group.sampling_mode(SamplingMode::Flat);
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(20));

    let error_models = assay_error_models();

    for workload in Workload::all() {
        let data = matrix_data(workload, MATRIX_N_SUBJECTS);
        for kind in SolverKind::all() {
            let theta = support_points(workload, kind, MATRIX_N_SUPPORT);
            for backend in Backend::all() {
                let bench_id = BenchmarkId::from_parameter(format!(
                    "{}/{}/{}",
                    workload.label(),
                    kind.label(),
                    backend.label()
                ));
                match kind {
                    SolverKind::Ode => {
                        let model = compile_ode(workload, backend);
                        group.bench_function(bench_id, |b| {
                            b.iter(|| {
                                black_box(
                                    log_likelihood_matrix(
                                        black_box(&model),
                                        black_box(&data),
                                        black_box(&theta),
                                        black_box(&error_models),
                                        false,
                                    )
                                    .unwrap(),
                                )
                            });
                        });
                    }
                    SolverKind::Analytical => {
                        let model = compile_analytical(workload, backend);
                        group.bench_function(bench_id, |b| {
                            b.iter(|| {
                                black_box(
                                    log_likelihood_matrix(
                                        black_box(&model),
                                        black_box(&data),
                                        black_box(&theta),
                                        black_box(&error_models),
                                        false,
                                    )
                                    .unwrap(),
                                )
                            });
                        });
                    }
                    SolverKind::Sde => {
                        let model = compile_sde(workload, backend);
                        group.bench_function(bench_id, |b| {
                            b.iter(|| {
                                black_box(
                                    log_likelihood_matrix(
                                        black_box(&model),
                                        black_box(&data),
                                        black_box(&theta),
                                        black_box(&error_models),
                                        false,
                                    )
                                    .unwrap(),
                                )
                            });
                        });
                    }
                }
            }
        }
    }

    group.finish();
}

criterion_group!(
    dsl_matrix,
    compile_group,
    predictions_group,
    log_likelihood_group,
    likelihood_matrix_group
);
criterion_main!(dsl_matrix);
