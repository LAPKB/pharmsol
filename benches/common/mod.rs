//! Shared bench fixtures (workloads, subjects, params, error models, model factories)
//! used by both `native_matrix.rs` and `dsl_matrix.rs`. Every backend measures the
//! same model + subject + params so only the engine varies between cells.
//!
//! Workloads:
//! - [`Workload::Short`]: 1-cpt, 100 mg PO at t=0, 9 obs over 12 h.
//! - [`Workload::Repeat`]: 2-cpt, 100 mg IV q12h × 10, 14 obs over 120 h.
//!
//! Native models come from the `ode!` / `analytical!` macros. Handwritten
//! `ODE::new` / `Analytical::new` factories are not benched separately: the
//! macros expand to the same constructors with plain `fn` pointers, so the two
//! measured the same solver path and only added noise.
//!
//! `#![allow(dead_code)]` since each bench binary compiles separately and may
//! not use every helper.

#![allow(dead_code)]

use ndarray::Array2;
use pharmsol::prelude::*;
use pharmsol::{Analytical, ODE};

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

/// Solver kinds. SDE is not benched: at any particle count that exercises the
/// particle filter it dwarfs the rest of the suite.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SolverKind {
    Ode,
    Analytical,
}

impl SolverKind {
    pub fn label(self) -> &'static str {
        match self {
            Self::Ode => "ode",
            Self::Analytical => "analytical",
        }
    }

    pub fn all() -> [SolverKind; 2] {
        [SolverKind::Ode, SolverKind::Analytical]
    }
}

// ───────────────────────────── Subjects ──────────────────────────────

/// 9 sampling points for the short workload.
const SHORT_TIMES: &[f64] = &[0.25, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0];

/// Synthetic plasma concentrations — only need to exercise the likelihood path, not match any model.
const SHORT_OBS: &[f64] = &[0.50, 0.90, 1.60, 2.40, 2.10, 1.50, 1.05, 0.72, 0.48];

/// 14 sampling points for the repeat workload.
const REPEAT_TIMES: &[f64] = &[
    0.5, 2.0, 6.0, 10.0, 14.0, 24.0, 36.0, 48.0, 60.0, 72.0, 84.0, 96.0, 108.0, 120.0,
];

const REPEAT_OBS: &[f64] = &[
    1.80, 1.45, 1.10, 0.90, 1.30, 1.60, 1.55, 1.50, 1.48, 1.45, 1.43, 1.42, 1.41, 0.95,
];

/// Subject `id` with the workload's dosing and observations shifted by `offset`.
fn build_subject(workload: Workload, id: String, offset: f64) -> Subject {
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
}

/// Single subject with observations — used for log-likelihood benches.
pub fn subject_for_likelihood(workload: Workload) -> Subject {
    build_subject(workload, format!("{}-lik", workload.label()), 0.0)
}

/// Population dataset for `log_likelihood_matrix`: `n` subjects, same template with small obs perturbations.
pub fn matrix_data(workload: Workload, n: usize) -> Data {
    let subjects = (0..n)
        .map(|i| {
            build_subject(
                workload,
                format!("{}-{i:03}", workload.label()),
                i as f64 * 0.01,
            )
        })
        .collect();
    Data::new(subjects)
}

// ───────────────────────────── Parameters ────────────────────────────

/// Reference named parameters per workload in the source order shared by bench fixtures.
pub fn named_params(workload: Workload) -> Vec<(&'static str, f64)> {
    match workload {
        Workload::Short => vec![("ka", 1.0), ("ke", 0.2), ("v", 50.0)],
        Workload::Repeat => vec![("ke", 0.10), ("kcp", 0.05), ("kpc", 0.04), ("v", 50.0)],
    }
}

/// Support-point grid for `log_likelihood_matrix`, shape `(n, nparams)`.
/// Rows are small perturbations of [`named_params`], so every row is a distinct cache key.
pub fn support_points(workload: Workload, n: usize) -> Array2<f64> {
    let base: Vec<f64> = named_params(workload).into_iter().map(|(_, v)| v).collect();
    let nparams = base.len();
    Array2::from_shape_fn((n, nparams), |(row, col)| {
        let p = base[col];
        let perturbation = (row as f64) * 0.001 * p.abs().max(1e-3);
        p + perturbation
    })
}

// ───────────────────────────── Error models ──────────────────────────

/// Assay error model for `log_likelihood` / `log_likelihood_matrix`.
pub fn assay_error_models() -> AssayErrorModels {
    AssayErrorModels::new()
        .add(
            "plasma",
            AssayErrorModel::additive(ErrorPoly::new(0.1, 0.1, 0.0, 0.0), 0.0),
        )
        .expect("plasma assay error model")
}

// ───────────────────────────── Macro factories ───────────────────────

pub fn macro_ode(workload: Workload) -> ODE {
    match workload {
        Workload::Short => ode! {
            name: "bench_one_cpt_po_ode",
            params: [ka, ke, v],
            states: [depot, central],
            outputs: [plasma],

            diffeq: |x, _p, _t, dx, _cov| {
                dx[depot] = bolus[po] + -ka * x[depot];
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

            diffeq: |x, _p, _t, dx, _cov| {
                dx[central] = bolus[iv] + -ke * x[central] - kcp * x[central] + kpc * x[peripheral];
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

// ───────────────────────────── DSL source strings ────────────────────

/// DSL source for `(workload, kind)`. Compiled via `compile_module_source_to_runtime`.
pub fn dsl_source(workload: Workload, kind: SolverKind) -> &'static str {
    match (workload, kind) {
        (Workload::Short, SolverKind::Ode) => SHORT_ODE_DSL,
        (Workload::Short, SolverKind::Analytical) => SHORT_ANALYTICAL_DSL,
        (Workload::Repeat, SolverKind::Ode) => REPEAT_ODE_DSL,
        (Workload::Repeat, SolverKind::Analytical) => REPEAT_ANALYTICAL_DSL,
    }
}

/// DSL `name = ...` field for `(workload, kind)`.
pub fn dsl_model_name(workload: Workload, kind: SolverKind) -> &'static str {
    match (workload, kind) {
        (Workload::Short, SolverKind::Ode) => "bench_one_cpt_po_ode",
        (Workload::Short, SolverKind::Analytical) => "bench_one_cpt_po_analytical",
        (Workload::Repeat, SolverKind::Ode) => "bench_two_cpt_iv_ode",
        (Workload::Repeat, SolverKind::Analytical) => "bench_two_cpt_iv_analytical",
    }
}

const SHORT_ODE_DSL: &str = r#"
name = bench_one_cpt_po_ode
kind = ode

params = ka, ke, v
states = depot, central
outputs = plasma


dx(depot) = bolus(po) + -ka * depot
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

const REPEAT_ODE_DSL: &str = r#"
name = bench_two_cpt_iv_ode
kind = ode

params = ke, kcp, kpc, v
states = central, peripheral
outputs = plasma


dx(central) = bolus(iv) + -ke * central - kcp * central + kpc * peripheral
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
