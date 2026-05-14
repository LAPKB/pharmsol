//! Native bench matrix: handwritten + macro models, all solvers, both workloads.
//! Default features only — this is the CI regression signal. DSL backends are in `dsl_matrix.rs`.
//!
//! IDs:
//! - `native/predictions`       → `{workload}/{solver}/{authoring}/{cache}`
//! - `native/log-likelihood`    → `{workload}/{solver}/{authoring}/{cache}`
//! - `native/likelihood-matrix` → `{workload}/{solver}/{authoring}`

use std::hint::black_box;
use std::time::Duration;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, SamplingMode};
use pharmsol::prelude::*;
use pharmsol::Cache;

mod common;
use common::{
    assay_error_models, handwritten_analytical, handwritten_ode, handwritten_sde, macro_analytical,
    macro_ode, macro_sde, matrix_data, params, subject_for_likelihood, subject_for_predictions,
    support_points, SolverKind, Workload,
};

const MATRIX_N_SUBJECTS: usize = 32;
const MATRIX_N_SUPPORT: usize = 64;

#[derive(Debug, Clone, Copy)]
enum Authoring {
    Handwritten,
    Macro,
}

impl Authoring {
    fn label(self) -> &'static str {
        match self {
            Self::Handwritten => "handwritten",
            Self::Macro => "macro",
        }
    }

    fn all() -> [Authoring; 2] {
        [Authoring::Handwritten, Authoring::Macro]
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

fn id(workload: Workload, kind: SolverKind, authoring: Authoring, cache: CacheState) -> String {
    format!(
        "{}/{}/{}/{}",
        workload.label(),
        kind.label(),
        authoring.label(),
        cache.label()
    )
}

fn predictions_group(c: &mut Criterion) {
    let mut group = c.benchmark_group("native/predictions");
    group.sampling_mode(SamplingMode::Flat);

    for workload in Workload::all() {
        let subject = subject_for_predictions(workload);
        for kind in SolverKind::all() {
            let theta = params(workload, kind);
            for authoring in Authoring::all() {
                for cache in CacheState::all() {
                    let bench_id =
                        BenchmarkId::from_parameter(id(workload, kind, authoring, cache));
                    group.bench_function(bench_id, |b| match (kind, authoring, cache) {
                        (SolverKind::Ode, Authoring::Handwritten, CacheState::Hot) => {
                            let model = handwritten_ode(workload);
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
                        }
                        (SolverKind::Ode, Authoring::Handwritten, CacheState::Cold) => {
                            let model = handwritten_ode(workload).disable_cache();
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
                        }
                        (SolverKind::Ode, Authoring::Macro, CacheState::Hot) => {
                            let model = macro_ode(workload);
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
                        }
                        (SolverKind::Ode, Authoring::Macro, CacheState::Cold) => {
                            let model = macro_ode(workload).disable_cache();
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
                        }
                        (SolverKind::Analytical, Authoring::Handwritten, CacheState::Hot) => {
                            let model = handwritten_analytical(workload);
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
                        }
                        (SolverKind::Analytical, Authoring::Handwritten, CacheState::Cold) => {
                            let model = handwritten_analytical(workload).disable_cache();
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
                        }
                        (SolverKind::Analytical, Authoring::Macro, CacheState::Hot) => {
                            let model = macro_analytical(workload);
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
                        }
                        (SolverKind::Analytical, Authoring::Macro, CacheState::Cold) => {
                            let model = macro_analytical(workload).disable_cache();
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
                        }
                        (SolverKind::Sde, Authoring::Handwritten, CacheState::Hot) => {
                            let model = handwritten_sde(workload);
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
                        }
                        (SolverKind::Sde, Authoring::Handwritten, CacheState::Cold) => {
                            let model = handwritten_sde(workload).disable_cache();
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
                        }
                        (SolverKind::Sde, Authoring::Macro, CacheState::Hot) => {
                            let model = macro_sde(workload);
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
                        }
                        (SolverKind::Sde, Authoring::Macro, CacheState::Cold) => {
                            let model = macro_sde(workload).disable_cache();
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
                        }
                    });
                }
            }
        }
    }

    group.finish();
}

fn log_likelihood_group(c: &mut Criterion) {
    let mut group = c.benchmark_group("native/log-likelihood");
    group.sampling_mode(SamplingMode::Flat);
    let error_models = assay_error_models();

    for workload in Workload::all() {
        let subject = subject_for_likelihood(workload);
        for kind in SolverKind::all() {
            let theta = params(workload, kind);
            for authoring in Authoring::all() {
                for cache in CacheState::all() {
                    let bench_id =
                        BenchmarkId::from_parameter(id(workload, kind, authoring, cache));
                    group.bench_function(bench_id, |b| match (kind, authoring, cache) {
                        (SolverKind::Ode, Authoring::Handwritten, CacheState::Hot) => {
                            let model = handwritten_ode(workload);
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
                        }
                        (SolverKind::Ode, Authoring::Handwritten, CacheState::Cold) => {
                            let model = handwritten_ode(workload).disable_cache();
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
                        }
                        (SolverKind::Ode, Authoring::Macro, CacheState::Hot) => {
                            let model = macro_ode(workload);
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
                        }
                        (SolverKind::Ode, Authoring::Macro, CacheState::Cold) => {
                            let model = macro_ode(workload).disable_cache();
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
                        }
                        (SolverKind::Analytical, Authoring::Handwritten, CacheState::Hot) => {
                            let model = handwritten_analytical(workload);
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
                        }
                        (SolverKind::Analytical, Authoring::Handwritten, CacheState::Cold) => {
                            let model = handwritten_analytical(workload).disable_cache();
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
                        }
                        (SolverKind::Analytical, Authoring::Macro, CacheState::Hot) => {
                            let model = macro_analytical(workload);
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
                        }
                        (SolverKind::Analytical, Authoring::Macro, CacheState::Cold) => {
                            let model = macro_analytical(workload).disable_cache();
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
                        }
                        (SolverKind::Sde, Authoring::Handwritten, CacheState::Hot) => {
                            let model = handwritten_sde(workload);
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
                        }
                        (SolverKind::Sde, Authoring::Handwritten, CacheState::Cold) => {
                            let model = handwritten_sde(workload).disable_cache();
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
                        }
                        (SolverKind::Sde, Authoring::Macro, CacheState::Hot) => {
                            let model = macro_sde(workload);
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
                        }
                        (SolverKind::Sde, Authoring::Macro, CacheState::Cold) => {
                            let model = macro_sde(workload).disable_cache();
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
                        }
                    });
                }
            }
        }
    }

    group.finish();
}

fn likelihood_matrix_group(c: &mut Criterion) {
    use pharmsol::prelude::simulator::log_likelihood_matrix;

    let mut group = c.benchmark_group("native/likelihood-matrix");
    group.sampling_mode(SamplingMode::Flat);
    // One matrix iteration is multi-second; small sample budget keeps wall-clock sane.
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(20));

    let error_models = assay_error_models();

    for workload in Workload::all() {
        let data = matrix_data(workload, MATRIX_N_SUBJECTS);
        for kind in SolverKind::all() {
            let theta = support_points(workload, kind, MATRIX_N_SUPPORT);
            for authoring in Authoring::all() {
                let bench_id = BenchmarkId::from_parameter(format!(
                    "{}/{}/{}",
                    workload.label(),
                    kind.label(),
                    authoring.label()
                ));
                group.bench_function(bench_id, |b| match (kind, authoring) {
                    (SolverKind::Ode, Authoring::Handwritten) => {
                        let model = handwritten_ode(workload);
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
                    }
                    (SolverKind::Ode, Authoring::Macro) => {
                        let model = macro_ode(workload);
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
                    }
                    (SolverKind::Analytical, Authoring::Handwritten) => {
                        let model = handwritten_analytical(workload);
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
                    }
                    (SolverKind::Analytical, Authoring::Macro) => {
                        let model = macro_analytical(workload);
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
                    }
                    (SolverKind::Sde, Authoring::Handwritten) => {
                        let model = handwritten_sde(workload);
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
                    }
                    (SolverKind::Sde, Authoring::Macro) => {
                        let model = macro_sde(workload);
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
                    }
                });
            }
        }
    }

    group.finish();
}

criterion_group!(
    native_matrix,
    predictions_group,
    log_likelihood_group,
    likelihood_matrix_group
);
criterion_main!(native_matrix);
