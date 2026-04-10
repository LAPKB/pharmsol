//! Compares analytical and ODE solutions for standard PK models.
//!
//! For each model configuration (one-compartment IV, one-compartment oral,
//! two-compartment IV, two-compartment oral), this example runs both the
//! closed-form analytical solution and the equivalent ODE, then prints
//! the predictions side by side so you can verify they match.
//!
//!     cargo run --release --example analytical_vs_ode

use pharmsol::prelude::*;

// ── Subjects ───────────────────────────────────────────────────────

fn subject_iv() -> Subject {
    Subject::builder("1")
        .infusion(0.0, 500.0, 0, 0.5)
        .observation(0.5, 0.0, 0)
        .observation(1.0, 0.0, 0)
        .observation(2.0, 0.0, 0)
        .observation(4.0, 0.0, 0)
        .observation(8.0, 0.0, 0)
        .observation(12.0, 0.0, 0)
        .observation(24.0, 0.0, 0)
        .build()
}

fn subject_oral() -> Subject {
    Subject::builder("1")
        .bolus(0.0, 500.0, 0)
        .observation(0.5, 0.0, 0)
        .observation(1.0, 0.0, 0)
        .observation(2.0, 0.0, 0)
        .observation(4.0, 0.0, 0)
        .observation(8.0, 0.0, 0)
        .observation(12.0, 0.0, 0)
        .observation(24.0, 0.0, 0)
        .build()
}

// ── Helpers ────────────────────────────────────────────────────────

fn print_comparison(label: &str, analytical: &SubjectPredictions, ode: &SubjectPredictions) {
    println!("\n=== {label} ===");
    println!(
        "{:<8} {:>14} {:>14} {:>14}",
        "Time", "Analytical", "ODE", "Diff"
    );
    println!("{:-<56}", "");
    for (a, o) in analytical
        .predictions()
        .iter()
        .zip(ode.predictions().iter())
    {
        let diff = (a.prediction() - o.prediction()).abs();
        println!(
            "{:<8.1} {:>14.6} {:>14.6} {:>14.2e}",
            a.time(),
            a.prediction(),
            o.prediction(),
            diff
        );
    }
}

// ── One-compartment IV ─────────────────────────────────────────────

fn one_cmt_iv(subject: &Subject, params: &[f64]) {
    let analytical = equation::Analytical::new(
        one_compartment,
        |_p, _t, _cov| {},
        |_p, _t, _cov| lag! {},
        |_p, _t, _cov| fa! {},
        |_p, _t, _cov, _x| {},
        |x, p, _t, _cov, y| {
            fetch_params!(p, _ke, v);
            y[0] = x[0] / v;
        },
    )
    .with_nstates(1)
    .with_nout(1);

    let ode = equation::ODE::new(
        |x, p, _t, dx, _b, rateiv, _cov| {
            fetch_params!(p, ke, _v);
            dx[0] = -ke * x[0] + rateiv[0];
        },
        |_p, _t, _cov| lag! {},
        |_p, _t, _cov| fa! {},
        |_p, _t, _cov, _x| {},
        |x, p, _t, _cov, y| {
            fetch_params!(p, _ke, v);
            y[0] = x[0] / v;
        },
    )
    .with_nstates(1)
    .with_nout(1);

    let pred_a = analytical.estimate_predictions(subject, params).unwrap();
    let pred_o = ode.estimate_predictions(subject, params).unwrap();
    print_comparison("One-compartment IV", &pred_a, &pred_o);
}

// ── One-compartment oral ───────────────────────────────────────────

fn one_cmt_oral(subject: &Subject, params: &[f64]) {
    let analytical = equation::Analytical::new(
        one_compartment_with_absorption,
        |_p, _t, _cov| {},
        |_p, _t, _cov| lag! {},
        |_p, _t, _cov| fa! {},
        |_p, _t, _cov, _x| {},
        |x, p, _t, _cov, y| {
            fetch_params!(p, _ka, _ke, v);
            y[0] = x[1] / v;
        },
    )
    .with_nstates(2)
    .with_nout(1);

    let ode = equation::ODE::new(
        |x, p, _t, dx, _b, _rateiv, _cov| {
            fetch_params!(p, ka, ke, _v);
            dx[0] = -ka * x[0];
            dx[1] = ka * x[0] - ke * x[1];
        },
        |_p, _t, _cov| lag! {},
        |_p, _t, _cov| fa! {},
        |_p, _t, _cov, _x| {},
        |x, p, _t, _cov, y| {
            fetch_params!(p, _ka, _ke, v);
            y[0] = x[1] / v;
        },
    )
    .with_nstates(2)
    .with_nout(1);

    let pred_a = analytical.estimate_predictions(subject, params).unwrap();
    let pred_o = ode.estimate_predictions(subject, params).unwrap();
    print_comparison("One-compartment oral", &pred_a, &pred_o);
}

// ── Two-compartment IV ─────────────────────────────────────────────

fn two_cmt_iv(subject: &Subject, params: &[f64]) {
    let analytical = equation::Analytical::new(
        two_compartments,
        |_p, _t, _cov| {},
        |_p, _t, _cov| lag! {},
        |_p, _t, _cov| fa! {},
        |_p, _t, _cov, _x| {},
        |x, p, _t, _cov, y| {
            fetch_params!(p, _ke, _k12, _k21, v);
            y[0] = x[0] / v;
        },
    )
    .with_nstates(2)
    .with_nout(1);

    let ode = equation::ODE::new(
        |x, p, _t, dx, _b, rateiv, _cov| {
            fetch_params!(p, ke, k12, k21, _v);
            dx[0] = -ke * x[0] - k12 * x[0] + k21 * x[1] + rateiv[0];
            dx[1] = k12 * x[0] - k21 * x[1];
        },
        |_p, _t, _cov| lag! {},
        |_p, _t, _cov| fa! {},
        |_p, _t, _cov, _x| {},
        |x, p, _t, _cov, y| {
            fetch_params!(p, _ke, _k12, _k21, v);
            y[0] = x[0] / v;
        },
    )
    .with_nstates(2)
    .with_nout(1);

    let pred_a = analytical.estimate_predictions(subject, params).unwrap();
    let pred_o = ode.estimate_predictions(subject, params).unwrap();
    print_comparison("Two-compartment IV", &pred_a, &pred_o);
}

// ── Two-compartment oral ───────────────────────────────────────────

fn two_cmt_oral(subject: &Subject, params: &[f64]) {
    let analytical = equation::Analytical::new(
        two_compartments_with_absorption,
        |_p, _t, _cov| {},
        |_p, _t, _cov| lag! {},
        |_p, _t, _cov| fa! {},
        |_p, _t, _cov, _x| {},
        |x, p, _t, _cov, y| {
            fetch_params!(p, _ka, _ke, _k12, _k21, v);
            y[0] = x[1] / v;
        },
    )
    .with_nstates(3)
    .with_nout(1);

    let ode = equation::ODE::new(
        |x, p, _t, dx, _b, _rateiv, _cov| {
            fetch_params!(p, ka, ke, k12, k21, _v);
            dx[0] = -ka * x[0];
            dx[1] = ka * x[0] - ke * x[1] - k12 * x[1] + k21 * x[2];
            dx[2] = k12 * x[1] - k21 * x[2];
        },
        |_p, _t, _cov| lag! {},
        |_p, _t, _cov| fa! {},
        |_p, _t, _cov, _x| {},
        |x, p, _t, _cov, y| {
            fetch_params!(p, _ka, _ke, _k12, _k21, v);
            y[0] = x[1] / v;
        },
    )
    .with_nstates(3)
    .with_nout(1);

    let pred_a = analytical.estimate_predictions(subject, params).unwrap();
    let pred_o = ode.estimate_predictions(subject, params).unwrap();
    print_comparison("Two-compartment oral", &pred_a, &pred_o);
}

// ── Main ───────────────────────────────────────────────────────────

fn main() {
    let iv = subject_iv();
    let oral = subject_oral();

    one_cmt_iv(&iv, &[0.1, 50.0]); // ke, v
    one_cmt_oral(&oral, &[1.0, 0.1, 50.0]); // ka, ke, v
    two_cmt_iv(&iv, &[0.1, 0.3, 0.2, 50.0]); // ke, k12, k21, v
    two_cmt_oral(&oral, &[1.0, 0.1, 0.3, 0.2, 50.0]); // ka, ke, k12, k21, v
}
