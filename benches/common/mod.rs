//! Shared bench fixtures for the native + DSL benchmark suites.
//!
//! This module centralizes the model definitions, subjects, parameter vectors,
//! and error-model fixtures that every backend variant in `benches/native_matrix.rs`
//! and `benches/dsl_matrix.rs` reuses. The goal is to keep all backends
//! (handwritten, macro, dsl-jit, dsl-aot, dsl-wasm) measuring **the same**
//! model, subject, and parameters so the only thing that differs between
//! cells is the engine itself.
//!
//! Two workloads are exposed:
//! - [`Workload::Short`]: 1-compartment, single 100 mg PO bolus at t=0,
//!   9 observations spread across 12 hours.
//! - [`Workload::Repeat`]: 2-compartment, 100 mg IV bolus q12h × 10
//!   (doses at t=0,12,…,108), 14 observations spread across 120 hours.
//!
//! For each workload three solver kinds (ODE, Analytical, SDE) and three
//! authoring/backend flavors (handwritten, macro, DSL-source) are provided.
//!
//! Some helpers (`SDE_PARTICLES`, `matrix_data`, `support_points`) are
//! deliberately public — both bench binaries consume them, but cargo compiles
//! each bench separately so unused helpers would trigger dead-code lints in
//! one binary or the other. We silence those with `#![allow(dead_code)]`.

#![allow(dead_code)]

use ndarray::Array2;
use pharmsol::prelude::*;
use pharmsol::simulator::equation::analytical::{
    one_compartment_with_absorption, two_compartments,
};
use pharmsol::{
    equation::{self, Route},
    Analytical, ResidualErrorModel, ResidualErrorModels, ODE, SDE,
};

/// Build the `ModelMetadata` for `(workload, kind)`. Handwritten factories
/// attach this so they resolve string route labels (`"po"`/`"iv"`) and output
/// labels (`"plasma"`) the same way the `ode!` / `analytical!` / `sde!`
/// macros and the DSL do.
fn model_metadata(workload: Workload, kind: SolverKind) -> equation::ModelMetadata {
    let name = match (workload, kind) {
        (Workload::Short, SolverKind::Ode) => "bench_one_cpt_po_ode",
        (Workload::Short, SolverKind::Analytical) => "bench_one_cpt_po_analytical",
        (Workload::Short, SolverKind::Sde) => "bench_one_cpt_po_sde",
        (Workload::Repeat, SolverKind::Ode) => "bench_two_cpt_iv_ode",
        (Workload::Repeat, SolverKind::Analytical) => "bench_two_cpt_iv_analytical",
        (Workload::Repeat, SolverKind::Sde) => "bench_two_cpt_iv_sde",
    };
    match workload {
        Workload::Short => {
            let params: &[&str] = match kind {
                SolverKind::Sde => &["ka", "ke", "v", "sigma_ke"],
                _ => &["ka", "ke", "v"],
            };
            equation::metadata::new(name)
                .parameters(params.iter().copied())
                .states(["depot", "central"])
                .outputs(["plasma"])
                .route(
                    Route::bolus("po")
                        .to_state("depot")
                        .inject_input_to_destination(),
                )
        }
        Workload::Repeat => {
            let params: &[&str] = match kind {
                SolverKind::Sde => &["ke", "kcp", "kpc", "v", "sigma_ke"],
                _ => &["ke", "kcp", "kpc", "v"],
            };
            equation::metadata::new(name)
                .parameters(params.iter().copied())
                .states(["central", "peripheral"])
                .outputs(["plasma"])
                .route(
                    Route::bolus("iv")
                        .to_state("central")
                        .inject_input_to_destination(),
                )
        }
    }
}

/// SDE particle count used across every SDE bench cell. Kept low so SDE
/// wall-clock stays comparable to ODE/Analytical at the same workload.
pub const SDE_PARTICLES: usize = 16;

/// Two scenarios that every backend variant must implement identically.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Workload {
    /// 1-compartment, single 100 mg PO bolus, 12 h horizon.
    Short,
    /// 2-compartment, 100 mg IV bolus q12h × 10, 120 h horizon.
    Repeat,
}

impl Workload {
    pub fn label(self) -> &'static str {
        match self {
            Self::Short => "1cpt-12h-po",
            Self::Repeat => "2cpt-120h-q12h",
        }
    }

    pub fn all() -> [Workload; 2] {
        [Workload::Short, Workload::Repeat]
    }
}

/// Solver kinds covered by the matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SolverKind {
    Ode,
    Analytical,
    Sde,
}

impl SolverKind {
    pub fn label(self) -> &'static str {
        match self {
            Self::Ode => "ode",
            Self::Analytical => "analytical",
            Self::Sde => "sde",
        }
    }

    pub fn all() -> [SolverKind; 3] {
        [SolverKind::Ode, SolverKind::Analytical, SolverKind::Sde]
    }
}

// ───────────────────────────── Subjects ──────────────────────────────

/// Time grid for the short workload (1-cpt, 12 h). 9 sampling points.
const SHORT_TIMES: &[f64] = &[0.25, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0];

/// Synthetic concentration values for the short workload. These do not need to
/// match any specific model exactly — they only feed into the likelihood
/// calculation so that we exercise the same code path across backends.
const SHORT_OBS: &[f64] = &[0.50, 0.90, 1.60, 2.40, 2.10, 1.50, 1.05, 0.72, 0.48];

/// Time grid for the repeat workload (2-cpt, 120 h). 14 sampling points.
const REPEAT_TIMES: &[f64] = &[
    0.5, 2.0, 6.0, 10.0, 14.0, 24.0, 36.0, 48.0, 60.0, 72.0, 84.0, 96.0, 108.0, 120.0,
];

const REPEAT_OBS: &[f64] = &[
    1.80, 1.45, 1.10, 0.90, 1.30, 1.60, 1.55, 1.50, 1.48, 1.45, 1.43, 1.42, 1.41, 0.95,
];

/// Build a subject for the given workload, using `missing_observation` so it
/// can drive `estimate_predictions` benchmarks without paying for observation
/// allocation.
pub fn subject_for_predictions(workload: Workload) -> Subject {
    let id = format!("{}-pred", workload.label());
    let mut builder = Subject::builder(id);
    match workload {
        Workload::Short => {
            builder = builder.bolus(0.0, 100.0, "po");
            for &t in SHORT_TIMES {
                builder = builder.missing_observation(t, "plasma");
            }
        }
        Workload::Repeat => {
            for dose in 0..10 {
                let t = dose as f64 * 12.0;
                builder = builder.bolus(t, 100.0, "iv");
            }
            for &t in REPEAT_TIMES {
                builder = builder.missing_observation(t, "plasma");
            }
        }
    }
    builder.build()
}

/// Build a subject for the given workload with real observations, suitable
/// for `estimate_log_likelihood` and `log_likelihood_matrix` benchmarks.
pub fn subject_for_likelihood(workload: Workload) -> Subject {
    let id = format!("{}-lik", workload.label());
    let mut builder = Subject::builder(id);
    match workload {
        Workload::Short => {
            builder = builder.bolus(0.0, 100.0, "po");
            for (&t, &y) in SHORT_TIMES.iter().zip(SHORT_OBS.iter()) {
                builder = builder.observation(t, y, "plasma");
            }
        }
        Workload::Repeat => {
            for dose in 0..10 {
                let t = dose as f64 * 12.0;
                builder = builder.bolus(t, 100.0, "iv");
            }
            for (&t, &y) in REPEAT_TIMES.iter().zip(REPEAT_OBS.iter()) {
                builder = builder.observation(t, y, "plasma");
            }
        }
    }
    builder.build()
}

/// Population dataset for `log_likelihood_matrix`. `n` subjects share the
/// same template with small per-subject perturbations on observation values
/// (mirrors the pattern used in the original `benches/likelihood.rs`).
pub fn matrix_data(workload: Workload, n: usize) -> Data {
    let subjects = (0..n)
        .map(|i| {
            let offset = i as f64 * 0.01;
            let id = format!("{}-{i:03}", workload.label());
            let mut builder = Subject::builder(id);
            match workload {
                Workload::Short => {
                    builder = builder.bolus(0.0, 100.0, "po");
                    for (&t, &y) in SHORT_TIMES.iter().zip(SHORT_OBS.iter()) {
                        builder = builder.observation(t, y + offset, "plasma");
                    }
                }
                Workload::Repeat => {
                    for dose in 0..10 {
                        let t = dose as f64 * 12.0;
                        builder = builder.bolus(t, 100.0, "iv");
                    }
                    for (&t, &y) in REPEAT_TIMES.iter().zip(REPEAT_OBS.iter()) {
                        builder = builder.observation(t, y + offset, "plasma");
                    }
                }
            }
            builder.build()
        })
        .collect();
    Data::new(subjects)
}

// ───────────────────────────── Parameters ────────────────────────────

/// Reference parameter vector for the given (workload, solver) cell.
///
/// **Short / ODE**: `[ka, ke, v]` = `[1.0, 0.2, 50.0]`
/// **Short / Analytical**: `[ka, ke, v]` = `[1.0, 0.2, 50.0]`
/// **Short / SDE**: `[ka, ke, v, sigma_ke]` = `[1.0, 0.2, 50.0, 0.05]`
/// **Repeat / ODE**: `[ke, kcp, kpc, v]` = `[0.10, 0.05, 0.04, 50.0]`
/// **Repeat / Analytical**: `[ke, kcp, kpc, v]` = `[0.10, 0.05, 0.04, 50.0]`
/// **Repeat / SDE**: `[ke, kcp, kpc, v, sigma_ke]` = `[0.10, 0.05, 0.04, 50.0, 0.01]`
pub fn params(workload: Workload, kind: SolverKind) -> Vec<f64> {
    match (workload, kind) {
        (Workload::Short, SolverKind::Ode | SolverKind::Analytical) => vec![1.0, 0.2, 50.0],
        (Workload::Short, SolverKind::Sde) => vec![1.0, 0.2, 50.0, 0.05],
        (Workload::Repeat, SolverKind::Ode | SolverKind::Analytical) => {
            vec![0.10, 0.05, 0.04, 50.0]
        }
        (Workload::Repeat, SolverKind::Sde) => vec![0.10, 0.05, 0.04, 50.0, 0.01],
    }
}

/// Support-point grid for `log_likelihood_matrix`. Shape: `(n, nparams)`.
/// Rows are small perturbations of [`params`] so every grid cell stays inside
/// a numerically well-behaved region.
pub fn support_points(workload: Workload, kind: SolverKind, n: usize) -> Array2<f64> {
    let base = params(workload, kind);
    let nparams = base.len();
    Array2::from_shape_fn((n, nparams), |(row, col)| {
        let p = base[col];
        let perturbation = (row as f64) * 0.001 * p.abs().max(1e-3);
        p + perturbation
    })
}

// ───────────────────────────── Error models ──────────────────────────

/// Assay error model for `estimate_log_likelihood` / `log_likelihood_matrix`.
pub fn assay_error_models() -> AssayErrorModels {
    AssayErrorModels::new()
        .add(
            "plasma",
            AssayErrorModel::additive(ErrorPoly::new(0.1, 0.1, 0.0, 0.0), 0.0),
        )
        .expect("plasma assay error model")
}

/// Residual error model (kept for symmetry with the existing benches even
/// though the new suite does not currently exercise `log_likelihood_batch`).
pub fn residual_error_models() -> ResidualErrorModels {
    // `ResidualErrorModels` indexes outputs by their dense `usize` index
    // (not by label), so this assumes a single output equation at index 0.
    ResidualErrorModels::new().add(0, ResidualErrorModel::constant(0.2))
}

// ───────────────────────────── Handwritten factories ─────────────────

pub fn handwritten_ode(workload: Workload) -> ODE {
    match workload {
        Workload::Short => {
            // 1-cpt, oral absorption. Params = [ka, ke, v].
            ODE::new(
                |x, p, _t, dx, b, _rateiv, _cov| {
                    let ka = p[0];
                    let ke = p[1];
                    dx[0] = -ka * x[0] + b[0];
                    dx[1] = ka * x[0] - ke * x[1];
                },
                |_p, _t, _cov| lag! {},
                |_p, _t, _cov| fa! {},
                |_p, _t, _cov, _x| {},
                |x, p, _t, _cov, y| {
                    let v = p[2];
                    y[0] = x[1] / v;
                },
            )
            .with_nstates(2)
            .with_ndrugs(1)
            .with_nout(1)
            .with_metadata(model_metadata(workload, SolverKind::Ode))
            .expect("short ODE metadata validates")
        }
        Workload::Repeat => {
            // 2-cpt IV bolus. Params = [ke, kcp, kpc, v].
            ODE::new(
                |x, p, _t, dx, b, _rateiv, _cov| {
                    let ke = p[0];
                    let kcp = p[1];
                    let kpc = p[2];
                    dx[0] = -ke * x[0] - kcp * x[0] + kpc * x[1] + b[0];
                    dx[1] = kcp * x[0] - kpc * x[1];
                },
                |_p, _t, _cov| lag! {},
                |_p, _t, _cov| fa! {},
                |_p, _t, _cov, _x| {},
                |x, p, _t, _cov, y| {
                    let v = p[3];
                    y[0] = x[0] / v;
                },
            )
            .with_nstates(2)
            .with_ndrugs(1)
            .with_nout(1)
            .with_metadata(model_metadata(workload, SolverKind::Ode))
            .expect("repeat ODE metadata validates")
        }
    }
}

pub fn handwritten_analytical(workload: Workload) -> Analytical {
    match workload {
        Workload::Short => Analytical::new(
            one_compartment_with_absorption,
            |_x, _t, _cov| {},
            |_p, _t, _cov| lag! {},
            |_p, _t, _cov| fa! {},
            |_p, _t, _cov, _x| {},
            |x, p, _t, _cov, y| {
                let v = p[2];
                y[0] = x[1] / v;
            },
        )
        .with_nstates(2)
        .with_ndrugs(1)
        .with_nout(1)
        .with_metadata(model_metadata(workload, SolverKind::Analytical))
        .expect("short Analytical metadata validates"),
        Workload::Repeat => Analytical::new(
            two_compartments,
            |_x, _t, _cov| {},
            |_p, _t, _cov| lag! {},
            |_p, _t, _cov| fa! {},
            |_p, _t, _cov, _x| {},
            |x, p, _t, _cov, y| {
                let v = p[3];
                y[0] = x[0] / v;
            },
        )
        .with_nstates(2)
        .with_ndrugs(1)
        .with_nout(1)
        .with_metadata(model_metadata(workload, SolverKind::Analytical))
        .expect("repeat Analytical metadata validates"),
    }
}

pub fn handwritten_sde(workload: Workload) -> SDE {
    match workload {
        Workload::Short => {
            // Drift mirrors handwritten_ode short. Diffusion = sigma_ke on central.
            // Params = [ka, ke, v, sigma_ke].
            SDE::new(
                |x, p, _t, dx, b, _cov| {
                    let ka = p[0];
                    let ke = p[1];
                    dx[0] = -ka * x[0] + b[0];
                    dx[1] = ka * x[0] - ke * x[1];
                },
                |p, sigma| {
                    let sigma_ke = p[3];
                    sigma[0] = 0.0;
                    sigma[1] = sigma_ke;
                },
                |_p, _t, _cov| lag! {},
                |_p, _t, _cov| fa! {},
                |_p, _t, _cov, _x| {},
                |x, p, _t, _cov, y| {
                    let v = p[2];
                    y[0] = x[1] / v;
                },
                SDE_PARTICLES,
            )
            .with_nstates(2)
            .with_ndrugs(1)
            .with_nout(1)
            .with_metadata(model_metadata(workload, SolverKind::Sde))
            .expect("short SDE metadata validates")
        }
        Workload::Repeat => {
            // 2-cpt IV bolus SDE. Params = [ke, kcp, kpc, v, sigma_ke].
            SDE::new(
                |x, p, _t, dx, b, _cov| {
                    let ke = p[0];
                    let kcp = p[1];
                    let kpc = p[2];
                    dx[0] = -ke * x[0] - kcp * x[0] + kpc * x[1] + b[0];
                    dx[1] = kcp * x[0] - kpc * x[1];
                },
                |p, sigma| {
                    let sigma_ke = p[4];
                    sigma[0] = sigma_ke;
                    sigma[1] = 0.0;
                },
                |_p, _t, _cov| lag! {},
                |_p, _t, _cov| fa! {},
                |_p, _t, _cov, _x| {},
                |x, p, _t, _cov, y| {
                    let v = p[3];
                    y[0] = x[0] / v;
                },
                SDE_PARTICLES,
            )
            .with_nstates(2)
            .with_ndrugs(1)
            .with_nout(1)
            .with_metadata(model_metadata(workload, SolverKind::Sde))
            .expect("repeat SDE metadata validates")
        }
    }
}

// ───────────────────────────── Macro factories ───────────────────────

pub fn macro_ode(workload: Workload) -> ODE {
    match workload {
        Workload::Short => ode! {
            name: "bench_one_cpt_po_ode",
            params: [ka, ke, v],
            states: [depot, central],
            outputs: [plasma],
            routes: [
                bolus(po) -> depot,
            ],
            diffeq: |x, _p, _t, dx, _cov| {
                dx[depot] = -ka * x[depot];
                dx[central] = ka * x[depot] - ke * x[central];
            },
            out: |x, _p, _t, _cov, y| {
                y[plasma] = x[central] / v;
            },
        },
        Workload::Repeat => ode! {
            name: "bench_two_cpt_iv_ode",
            params: [ke, kcp, kpc, v],
            states: [central, peripheral],
            outputs: [plasma],
            routes: [
                bolus(iv) -> central,
            ],
            diffeq: |x, _p, _t, dx, _cov| {
                dx[central] = -ke * x[central] - kcp * x[central] + kpc * x[peripheral];
                dx[peripheral] = kcp * x[central] - kpc * x[peripheral];
            },
            out: |x, _p, _t, _cov, y| {
                y[plasma] = x[central] / v;
            },
        },
    }
}

pub fn macro_analytical(workload: Workload) -> Analytical {
    match workload {
        Workload::Short => analytical! {
            name: "bench_one_cpt_po_analytical",
            params: [ka, ke, v],
            states: [depot, central],
            outputs: [plasma],
            routes: [
                bolus(po) -> depot,
            ],
            structure: one_compartment_with_absorption,
            out: |x, _p, _t, _cov, y| {
                y[plasma] = x[central] / v;
            },
        },
        Workload::Repeat => analytical! {
            name: "bench_two_cpt_iv_analytical",
            params: [ke, kcp, kpc, v],
            states: [central, peripheral],
            outputs: [plasma],
            routes: [
                bolus(iv) -> central,
            ],
            structure: two_compartments,
            out: |x, _p, _t, _cov, y| {
                y[plasma] = x[central] / v;
            },
        },
    }
}

pub fn macro_sde(workload: Workload) -> SDE {
    match workload {
        Workload::Short => sde! {
            name: "bench_one_cpt_po_sde",
            params: [ka, ke, v, sigma_ke],
            states: [depot, central],
            outputs: [plasma],
            particles: 16,
            routes: [
                bolus(po) -> depot,
            ],
            drift: |x, _p, _t, dx, _cov| {
                dx[depot] = -ka * x[depot];
                dx[central] = ka * x[depot] - ke * x[central];
            },
            diffusion: |_p, sigma| {
                sigma[central] = sigma_ke;
            },
            out: |x, _p, _t, _cov, y| {
                y[plasma] = x[central] / v;
            },
        },
        Workload::Repeat => sde! {
            name: "bench_two_cpt_iv_sde",
            params: [ke, kcp, kpc, v, sigma_ke],
            states: [central, peripheral],
            outputs: [plasma],
            particles: 16,
            routes: [
                bolus(iv) -> central,
            ],
            drift: |x, _p, _t, dx, _cov| {
                dx[central] = -ke * x[central] - kcp * x[central] + kpc * x[peripheral];
                dx[peripheral] = kcp * x[central] - kpc * x[peripheral];
            },
            diffusion: |_p, sigma| {
                sigma[central] = sigma_ke;
            },
            out: |x, _p, _t, _cov, y| {
                y[plasma] = x[central] / v;
            },
        },
    }
}

// ───────────────────────────── DSL source strings ────────────────────

/// DSL source for `(workload, kind)`. Consumed by `benches/dsl_matrix.rs`
/// via `compile_module_source_to_runtime` (JIT/AOT) and
/// `compile_module_source_to_runtime_wasm` (WASM).
pub fn dsl_source(workload: Workload, kind: SolverKind) -> &'static str {
    match (workload, kind) {
        (Workload::Short, SolverKind::Ode) => SHORT_ODE_DSL,
        (Workload::Short, SolverKind::Analytical) => SHORT_ANALYTICAL_DSL,
        (Workload::Short, SolverKind::Sde) => SHORT_SDE_DSL,
        (Workload::Repeat, SolverKind::Ode) => REPEAT_ODE_DSL,
        (Workload::Repeat, SolverKind::Analytical) => REPEAT_ANALYTICAL_DSL,
        (Workload::Repeat, SolverKind::Sde) => REPEAT_SDE_DSL,
    }
}

/// DSL `name = ...` field for `(workload, kind)`. Used by `Some(model_name)`
/// arguments on the DSL compile entrypoints.
pub fn dsl_model_name(workload: Workload, kind: SolverKind) -> &'static str {
    match (workload, kind) {
        (Workload::Short, SolverKind::Ode) => "bench_one_cpt_po_ode",
        (Workload::Short, SolverKind::Analytical) => "bench_one_cpt_po_analytical",
        (Workload::Short, SolverKind::Sde) => "bench_one_cpt_po_sde",
        (Workload::Repeat, SolverKind::Ode) => "bench_two_cpt_iv_ode",
        (Workload::Repeat, SolverKind::Analytical) => "bench_two_cpt_iv_analytical",
        (Workload::Repeat, SolverKind::Sde) => "bench_two_cpt_iv_sde",
    }
}

const SHORT_ODE_DSL: &str = r#"
name = bench_one_cpt_po_ode
kind = ode

params = ka, ke, v
states = depot, central
outputs = plasma

bolus(po) -> depot

dx(depot) = -ka * depot
dx(central) = ka * depot - ke * central

out(plasma) = central / v ~ continuous()
"#;

const SHORT_ANALYTICAL_DSL: &str = r#"
name = bench_one_cpt_po_analytical
kind = analytical

params = ka, ke, v
states = depot, central
outputs = plasma

bolus(po) -> depot

structure = one_compartment_with_absorption

out(plasma) = central / v ~ continuous()
"#;

const SHORT_SDE_DSL: &str = r#"
name = bench_one_cpt_po_sde
kind = sde

params = ka, ke, v, sigma_ke
states = depot, central
particles = 16
outputs = plasma

bolus(po) -> depot

dx(depot) = -ka * depot
dx(central) = ka * depot - ke * central

noise(central) = sigma_ke

out(plasma) = central / v ~ continuous()
"#;

const REPEAT_ODE_DSL: &str = r#"
name = bench_two_cpt_iv_ode
kind = ode

params = ke, kcp, kpc, v
states = central, peripheral
outputs = plasma

bolus(iv) -> central

dx(central) = -ke * central - kcp * central + kpc * peripheral
dx(peripheral) = kcp * central - kpc * peripheral

out(plasma) = central / v ~ continuous()
"#;

const REPEAT_ANALYTICAL_DSL: &str = r#"
name = bench_two_cpt_iv_analytical
kind = analytical

params = ke, kcp, kpc, v
states = central, peripheral
outputs = plasma

bolus(iv) -> central

structure = two_compartments

out(plasma) = central / v ~ continuous()
"#;

const REPEAT_SDE_DSL: &str = r#"
name = bench_two_cpt_iv_sde
kind = sde

params = ke, kcp, kpc, v, sigma_ke
states = central, peripheral
particles = 16
outputs = plasma

bolus(iv) -> central

dx(central) = -ke * central - kcp * central + kpc * peripheral
dx(peripheral) = kcp * central - kpc * peripheral

noise(central) = sigma_ke

out(plasma) = central / v ~ continuous()
"#;
