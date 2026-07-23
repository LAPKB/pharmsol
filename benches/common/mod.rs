//! Shared bench fixtures (workloads, subjects, parameters, and model factories)
//! used by both `native_matrix.rs` and `dsl_matrix.rs`. Every backend measures the
//! same model + subject + params so only the engine varies between cells.
//!
//! Workloads:
//! - [`Workload::Short`]: 1-cpt, 100 mg PO at t=0, 9 obs over 12 h.
//! - [`Workload::Repeat`]: 2-cpt, 100 mg IV q12h × 10, 14 obs over 120 h.
//!
//! `#![allow(dead_code)]` since each bench binary compiles separately and may
//! not use every helper.

#![allow(dead_code)]

use pharmsol::equation::{self, Analytical, Route, ODE, SDE};
use pharmsol::prelude::*;
use pharmsol::simulator::equation::analytical::{
    one_compartment_with_absorption, two_compartments,
};

/// `ModelMetadata` for handwritten factories so route/output labels resolve like the macro/DSL paths.
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

/// SDE particle count. Kept low so SDE wall-clock stays in the same ballpark as ODE/Analytical.
pub const SDE_PARTICLES: usize = 16;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Workload {
    /// 1-cpt, 100 mg PO, 12 h.
    Short,
    /// 2-cpt, 100 mg IV q12h × 10, 120 h.
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

/// Solver kinds.
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

    // Stochastic workloads use the dedicated SDE benchmark target.
    pub fn all() -> [SolverKind; 2] {
        [SolverKind::Ode, SolverKind::Analytical]
    }
}

// ───────────────────────────── Subjects ──────────────────────────────

/// 9 sampling points for the short workload.
const SHORT_TIMES: &[f64] = &[0.25, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0];

/// 14 sampling points for the repeat workload.
const REPEAT_TIMES: &[f64] = &[
    0.5, 2.0, 6.0, 10.0, 14.0, 24.0, 36.0, 48.0, 60.0, 72.0, 84.0, 96.0, 108.0, 120.0,
];

/// Subject with `missing_observation` slots — used for prediction benches.
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

// ───────────────────────────── Parameters ────────────────────────────

/// Reference named parameters per `(workload, kind)` in the source order shared by bench fixtures.
pub fn named_params(workload: Workload, kind: SolverKind) -> Vec<(&'static str, f64)> {
    match (workload, kind) {
        (Workload::Short, SolverKind::Ode | SolverKind::Analytical) => {
            vec![("ka", 1.0), ("ke", 0.2), ("v", 50.0)]
        }
        (Workload::Short, SolverKind::Sde) => {
            vec![("ka", 1.0), ("ke", 0.2), ("v", 50.0), ("sigma_ke", 0.05)]
        }
        (Workload::Repeat, SolverKind::Ode | SolverKind::Analytical) => {
            vec![("ke", 0.10), ("kcp", 0.05), ("kpc", 0.04), ("v", 50.0)]
        }
        (Workload::Repeat, SolverKind::Sde) => vec![
            ("ke", 0.10),
            ("kcp", 0.05),
            ("kpc", 0.04),
            ("v", 50.0),
            ("sigma_ke", 0.01),
        ],
    }
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

/// DSL source for `(workload, kind)`. Compiled via `compile_module_source_to_runtime`.
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

/// DSL `name = ...` field for `(workload, kind)`.
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
