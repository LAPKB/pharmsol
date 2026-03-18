//! Shows how to select different ODE solvers for the same model.
//!
//! pharmsol wraps diffsol's solver families:
//!
//! - `Bdf`               — implicit multistep, best for stiff systems (default)
//! - `Sdirk(TrBdf2)`     — implicit single-step, good all-rounder
//! - `Sdirk(Esdirk34)`   — implicit single-step, higher accuracy
//! - `ExplicitRk(Tsit45)` — explicit Runge-Kutta, fastest for non-stiff systems
//!
//!     cargo run --release --example compare_solvers

use std::time::Instant;

use pharmsol::prelude::*;

// ── Model ──────────────────────────────────────────────────────────
// Two-compartment IV model.  The solver is the only thing that changes
// between runs — the ODE, output equation and dimensions stay the same.

fn two_cpt(solver: OdeSolver) -> equation::ODE {
    ode! {
        diffeq: |x, p, _t, dx, b, rateiv, _cov| {
            fetch_params!(p, ke, kcp, kpc, _v);
            dx[0] = rateiv[0] + b[0] - ke * x[0] - kcp * x[0] + kpc * x[1];
            dx[1] = kcp * x[0] - kpc * x[1];
        },
        out: |x, p, _t, _cov, y| {
            fetch_params!(p, _ke, _kcp, _kpc, v);
            y[0] = x[0] / v;
        },
    }
    .with_solver(solver)
}

// ── Main ───────────────────────────────────────────────────────────

fn main() {
    let subject = Subject::builder("id1")
        .bolus(0.0, 100.0, 0)
        .infusion(12.0, 200.0, 0, 2.0)
        .missing_observation(0.5, 0)
        .missing_observation(1.0, 0)
        .missing_observation(2.0, 0)
        .missing_observation(4.0, 0)
        .missing_observation(8.0, 0)
        .missing_observation(12.0, 0)
        .missing_observation(12.5, 0)
        .missing_observation(13.0, 0)
        .missing_observation(14.0, 0)
        .missing_observation(16.0, 0)
        .missing_observation(24.0, 0)
        .build();

    let spp = vec![0.1, 0.05, 0.03, 50.0]; // ke, kcp, kpc, V

    // Run each solver and collect predictions
    let bdf = two_cpt(OdeSolver::Bdf);
    let tsit45 = two_cpt(OdeSolver::ExplicitRk(ExplicitRkTableau::Tsit45));
    let trbdf2 = two_cpt(OdeSolver::Sdirk(SdirkTableau::TrBdf2));
    let esdirk34 = two_cpt(OdeSolver::Sdirk(SdirkTableau::Esdirk34));

    let results: Vec<(&str, equation::ODE)> = vec![
        ("Bdf", bdf),
        ("Sdirk(TrBdf2)", trbdf2),
        ("Sdirk(Esdirk34)", esdirk34),
        ("ExplicitRk(Tsit45)", tsit45),
    ];

    // ── Print comparison table ─────────────────────────────────────
    println!("{:<12} {:>8}   predictions", "solver", "µs");
    println!("{}", "-".repeat(80));

    let mut ref_preds: Option<Vec<f64>> = None;

    for (name, ode) in &results {
        let (preds, us) = timed(|| ode.estimate_predictions(&subject, &spp).unwrap());
        let preds: Vec<f64> = preds.flat_predictions().to_vec();

        let fmt: Vec<String> = preds.iter().map(|v| format!("{v:.4}")).collect();
        println!("{:<12} {:>8}   [{}]", name, us, fmt.join(", "));

        if let Some(ref rpreds) = ref_preds {
            let max_diff = max_abs_diff(&preds, rpreds);
            if max_diff > 1e-3 {
                println!("             max diff from Bdf: {max_diff:.6}");
            }
        } else {
            ref_preds = Some(preds);
        }
    }
}

fn timed<T>(f: impl FnOnce() -> T) -> (T, u128) {
    let t0 = Instant::now();
    let result = f();
    (result, t0.elapsed().as_micros())
}

fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f64, f64::max)
}
