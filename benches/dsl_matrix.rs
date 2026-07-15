//! DSL bench matrix (feature-gated): JIT, native AoT, WASM across all workloads + solvers.
//! Mirrors `native_matrix.rs` but compiles models from DSL source.
//!
//! IDs:
//! - `dsl/compile` → `{workload}/{kind}/{backend}`
//! - `dsl/predictions` → `{workload}/{kind}/{backend}/{cache}`

use std::hint::black_box;
use std::path::PathBuf;
use std::time::Duration;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, SamplingMode};
use tempfile::TempDir;

use pharmsol::dsl::{
    compile_module_source_to_runtime, CompiledRuntimeModel, NativeAnalyticalModel,
    NativeAotCompileOptions, NativeOdeModel, RuntimeCompilationTarget,
};
use pharmsol::{Cache, Parameters};

mod common;
use common::{
    dsl_model_name, dsl_source, named_params, subject_for_predictions, SolverKind, Workload,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)] // Retained for opt-in backend profiling outside the default matrix.
enum Backend {
    Jit,
    Aot,
    Wasm,
}

impl Backend {
    fn label(self) -> &'static str {
        match self {
            Self::Jit => "dsl-jit",
            Self::Aot => "dsl-aot",
            Self::Wasm => "dsl-wasm",
        }
    }

    // The default matrix bounds runtime to JIT; AoT and WASM remain opt-in.
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

/// One `TempDir` shared across the bench binary; each compile gets a fresh subdir.
struct AotWorkspace {
    root: TempDir,
    counter: std::cell::Cell<usize>,
}

impl AotWorkspace {
    fn new() -> Self {
        Self {
            root: tempfile::Builder::new()
                .prefix("pharmsol-bench-dsl-aot-")
                .tempdir()
                .expect("create AoT workspace tempdir"),
            counter: std::cell::Cell::new(0),
        }
    }

    fn fresh(&self, stem: &str) -> PathBuf {
        let n = self.counter.get();
        self.counter.set(n + 1);
        self.root.path().join(format!("{stem}-{n:04}"))
    }
}

/// Compile `(workload, kind)` with `backend` and return the full `CompiledRuntimeModel`.
fn compile_runtime(
    workload: Workload,
    kind: SolverKind,
    backend: Backend,
    aot: &AotWorkspace,
) -> CompiledRuntimeModel {
    let source = dsl_source(workload, kind);
    let name = dsl_model_name(workload, kind);
    let target = match backend {
        Backend::Jit => RuntimeCompilationTarget::Jit,
        Backend::Aot => {
            let dir = aot.fresh(&format!("{}-{}", workload.label(), kind.label()));
            RuntimeCompilationTarget::NativeAot(NativeAotCompileOptions::new(dir))
        }
        Backend::Wasm => RuntimeCompilationTarget::Wasm,
    };
    compile_module_source_to_runtime(source, Some(name), target, |_, _| {})
        .unwrap_or_else(|e| panic!("compile {} via {} failed: {e:?}", name, backend.label()))
}

fn compile_ode(workload: Workload, backend: Backend, aot: &AotWorkspace) -> NativeOdeModel {
    match compile_runtime(workload, SolverKind::Ode, backend, aot) {
        CompiledRuntimeModel::Ode(model) => model,
        other => panic!(
            "expected Ode model for {}, got {:?}",
            workload.label(),
            other.backend()
        ),
    }
}

fn compile_analytical(
    workload: Workload,
    backend: Backend,
    aot: &AotWorkspace,
) -> NativeAnalyticalModel {
    match compile_runtime(workload, SolverKind::Analytical, backend, aot) {
        CompiledRuntimeModel::Analytical(model) => model,
        other => panic!(
            "expected Analytical model for {}, got {:?}",
            workload.label(),
            other.backend()
        ),
    }
}

fn ode_parameters(model: &NativeOdeModel, workload: Workload) -> Parameters {
    Parameters::with_model(model, named_params(workload, SolverKind::Ode))
        .expect("DSL ODE bench parameters should validate")
}

fn analytical_parameters(model: &NativeAnalyticalModel, workload: Workload) -> Parameters {
    Parameters::with_model(model, named_params(workload, SolverKind::Analytical))
        .expect("DSL analytical bench parameters should validate")
}

// ───────────────────────────── compile group ─────────────────────────

fn compile_group(c: &mut Criterion) {
    use std::time::Instant;

    let mut group = c.benchmark_group("dsl/compile");
    group.sampling_mode(SamplingMode::Flat);
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(5));
    // Each compile leaks an executable mmap (JIT) or runs rustc (AoT). Without
    // a cap, a fast JIT compile (~60 µs) lets Criterion request hundreds of
    // thousands of iterations per cell and exhausts the runner's executable
    // memory pool / `vm.max_map_count`. We hard-cap real iterations per
    // Criterion batch to `MAX_ITERS_PER_BATCH` and scale the reported elapsed
    // time linearly so per-iteration timings stay accurate.
    const MAX_ITERS_PER_BATCH: u64 = 25;

    let aot = AotWorkspace::new();

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
                        let actual = iters.clamp(1, MAX_ITERS_PER_BATCH);
                        let start = Instant::now();
                        for _ in 0..actual {
                            black_box(compile_runtime(
                                black_box(workload),
                                black_box(kind),
                                black_box(backend),
                                &aot,
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

    let aot = AotWorkspace::new();

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
                                CacheState::Hot => compile_ode(workload, backend, &aot),
                                CacheState::Cold => {
                                    compile_ode(workload, backend, &aot).disable_cache()
                                }
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
                                CacheState::Hot => compile_analytical(workload, backend, &aot),
                                CacheState::Cold => {
                                    compile_analytical(workload, backend, &aot).disable_cache()
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
                        SolverKind::Sde => unreachable!("SDE has a dedicated benchmark"),
                    }
                }
            }
        }
    }

    group.finish();
}

criterion_group!(dsl_matrix, compile_group, predictions_group);
criterion_main!(dsl_matrix);
