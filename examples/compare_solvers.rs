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

    let parameters = vec![0.1, 0.05, 0.03, 50.0]; // ke, kcp, kpc, V

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

    // ── Run all solvers and collect results ───────────────────────
    let mut rows: Vec<(&str, u128, Vec<f64>)> = Vec::new();
    for (name, ode) in &results {
        let (preds, us) = timed(|| ode.estimate_predictions(&subject, &parameters).unwrap());
        let preds: Vec<f64> = preds.flat_predictions().to_vec();
        rows.push((name, us, preds));
    }

    let obs_times = [0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 12.5, 13.0, 14.0, 16.0, 24.0];
    let ref_preds = &rows[0].2;

    // ── Summary ────────────────────────────────────────────────────
    println!();
    println!("  Solver Performance");
    println!("  {}", "─".repeat(48));
    for (i, (name, us, preds)) in rows.iter().enumerate() {
        let diff = if i == 0 {
            "(reference)".to_string()
        } else {
            let d = max_abs_diff(preds, ref_preds);
            format!("max \u{0394} {d:.2e}")
        };
        println!("  {:<22} {:>6} \u{00B5}s   {}", name, us, diff);
    }
    println!();

    // ── Predictions (long format) ──────────────────────────────────
    // Short labels for the column headers
    let labels: Vec<&str> = rows.iter().map(|(n, _, _)| *n).collect();
    let col_w = labels.iter().map(|l| l.len().max(8)).collect::<Vec<_>>();

    print!("  {:>6}", "t");
    for (j, label) in labels.iter().enumerate() {
        print!("  {:>w$}", label, w = col_w[j]);
    }
    println!();

    print!("  {:>6}", "──────");
    for j in 0..labels.len() {
        print!("  {:─>w$}", "", w = col_w[j]);
    }
    println!();

    for (k, t) in obs_times.iter().enumerate() {
        print!("  {:>6.1}", t);
        for (j, (_, _, preds)) in rows.iter().enumerate() {
            print!("  {:>w$.4}", preds[k], w = col_w[j]);
        }
        println!();
    }
    println!();
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
