//! Shows how to select different ODE solvers for the same declaration-first model.
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
// Two-compartment IV model. The solver is the only thing that changes
// between runs; the declaration-first `ode!` surface and the generated
// metadata stay the same.

fn two_cpt(solver: OdeSolver) -> equation::ODE {
    ode! {
        name: "two_cpt",
        params: [ke, kcp, kpc, v],
        states: [central, peripheral],
        outputs: [cp],
        routes: [
            bolus(load) -> central,
            infusion(iv) -> central,
        ],
        diffeq: |x, _p, _t, dx, _cov| {
            dx[central] = -ke * x[central] - kcp * x[central] + kpc * x[peripheral];
            dx[peripheral] = kcp * x[central] - kpc * x[peripheral];
        },
        out: |x, _p, _t, _cov, y| {
            y[cp] = x[central] / v;
        },
    }
    .with_solver(solver)
}

// ── Main ───────────────────────────────────────────────────────────

fn main() {
    // Run each solver and collect predictions
    let bdf = two_cpt(OdeSolver::Bdf);
    let tsit45 = two_cpt(OdeSolver::ExplicitRk(ExplicitRkTableau::Tsit45));
    let trbdf2 = two_cpt(OdeSolver::Sdirk(SdirkTableau::TrBdf2));
    let esdirk34 = two_cpt(OdeSolver::Sdirk(SdirkTableau::Esdirk34));

    // Both declarations resolve to the same shared input, so subject
    // authoring still uses one numeric index for the loading bolus and the
    // maintenance infusion.

    let subject = Subject::builder("id1")
        .bolus(0.0, 100.0, "load")
        .infusion(12.0, 200.0, "iv", 2.0)
        .missing_observation(0.5, "cp")
        .missing_observation(1.0, "cp")
        .missing_observation(2.0, "cp")
        .missing_observation(4.0, "cp")
        .missing_observation(8.0, "cp")
        .missing_observation(12.0, "cp")
        .missing_observation(12.5, "cp")
        .missing_observation(13.0, "cp")
        .missing_observation(14.0, "cp")
        .missing_observation(16.0, "cp")
        .missing_observation(24.0, "cp")
        .build();

    let parameters = vec![0.1, 0.05, 0.03, 50.0]; // ke, kcp, kpc, V

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
    for (label, width) in labels.iter().zip(&col_w) {
        print!("  {:>w$}", label, w = *width);
    }
    println!();

    print!("  {:>6}", "──────");
    for width in &col_w {
        print!("  {:─>w$}", "", w = *width);
    }
    println!();

    for (k, t) in obs_times.iter().enumerate() {
        print!("  {:>6.1}", t);
        for ((_, _, preds), width) in rows.iter().zip(&col_w) {
            print!("  {:>w$.4}", preds[k], w = *width);
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
