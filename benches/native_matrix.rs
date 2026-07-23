//! Native bench matrix: handwritten + macro models, all solvers, both workloads.
//! Default features only — this is the CI regression signal. DSL backends are in `dsl_matrix.rs`.
//!
//! IDs:
//! - `native/predictions` → `{workload}/{solver}/{authoring}/{cache}`

use std::hint::black_box;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, SamplingMode};
use pharmsol::equation::{Analytical, ODE};
use pharmsol::prelude::*;
use pharmsol::{Cache, Parameters};

mod common;
use common::{
    handwritten_analytical, handwritten_ode, macro_analytical, macro_ode, named_params,
    subject_for_predictions, SolverKind, Workload,
};

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

fn ode_parameters(model: &ODE, workload: Workload) -> Parameters {
    Parameters::with_model(model, named_params(workload, SolverKind::Ode))
        .expect("native ODE bench parameters should validate")
}

fn analytical_parameters(model: &Analytical, workload: Workload) -> Parameters {
    Parameters::with_model(model, named_params(workload, SolverKind::Analytical))
        .expect("native analytical bench parameters should validate")
}

fn predictions_group(c: &mut Criterion) {
    let mut group = c.benchmark_group("native/predictions");
    group.sampling_mode(SamplingMode::Flat);

    for workload in Workload::all() {
        let subject = subject_for_predictions(workload);
        for kind in SolverKind::all() {
            for authoring in Authoring::all() {
                for cache in CacheState::all() {
                    let bench_id =
                        BenchmarkId::from_parameter(id(workload, kind, authoring, cache));
                    group.bench_function(bench_id, |b| match (kind, authoring, cache) {
                        (SolverKind::Ode, Authoring::Handwritten, CacheState::Hot) => {
                            let model = handwritten_ode(workload);
                            let theta = ode_parameters(&model, workload);
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
                            let theta = ode_parameters(&model, workload);
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
                            let theta = ode_parameters(&model, workload);
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
                            let theta = ode_parameters(&model, workload);
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
                            let theta = analytical_parameters(&model, workload);
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
                            let theta = analytical_parameters(&model, workload);
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
                            let theta = analytical_parameters(&model, workload);
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
                            let theta = analytical_parameters(&model, workload);
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
                        (SolverKind::Sde, _, _) => unreachable!("SDE has a dedicated benchmark"),
                    });
                }
            }
        }
    }

    group.finish();
}

criterion_group!(native_matrix, predictions_group);
criterion_main!(native_matrix);
