//! Compares analytical and ODE solutions for standard PK models.
//!
//! For each model configuration (one-compartment IV, one-compartment oral,
//! two-compartment IV, two-compartment oral), this example runs both the
//! closed-form analytical solution and the equivalent ODE, then prints
//! the predictions side by side so you can verify they match.
//! Both authoring paths use the declaration-first macro surface so the
//! example stays on the preferred public authoring story.
//! Built-in analytical structures are positional: the `params: [...]`
//! declaration becomes metadata, but the runtime kernel still expects values
//! in the structure's native positional order.
//!
//!     cargo run --release --example analytical_vs_ode

use pharmsol::prelude::*;

// ── Subjects ───────────────────────────────────────────────────────

fn subject_iv(input: impl ToString, output: impl ToString) -> Subject {
    Subject::builder("1")
        .infusion(0.0, 500.0, input, 0.5)
        .observation(0.5, 0.0, output.to_string())
        .observation(1.0, 0.0, output.to_string())
        .observation(2.0, 0.0, output.to_string())
        .observation(4.0, 0.0, output.to_string())
        .observation(8.0, 0.0, output.to_string())
        .observation(12.0, 0.0, output.to_string())
        .observation(24.0, 0.0, output)
        .build()
}

fn subject_oral(input: impl ToString, output: impl ToString) -> Subject {
    Subject::builder("1")
        .bolus(0.0, 500.0, input)
        .observation(0.5, 0.0, output.to_string())
        .observation(1.0, 0.0, output.to_string())
        .observation(2.0, 0.0, output.to_string())
        .observation(4.0, 0.0, output.to_string())
        .observation(8.0, 0.0, output.to_string())
        .observation(12.0, 0.0, output.to_string())
        .observation(24.0, 0.0, output)
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

fn one_cmt_iv(params: &[f64]) {
    let analytical = analytical! {
        name: "one_cmt_iv",
        params: [ke, v],
        states: [central],
        outputs: [cp],
        routes: [
            infusion(iv) -> central,
        ],
        structure: one_compartment,
        out: |x, _p, _t, _cov, y| {
            y[cp] = x[central] / v;
        },
    };

    let ode = ode! {
        name: "one_cmt_iv",
        params: [ke, v],
        states: [central],
        outputs: [cp],
        routes: [
            infusion(iv) -> central,
        ],
        diffeq: |x, _p, _t, dx, _cov| {
            dx[central] = -ke * x[central];
        },
        out: |x, _p, _t, _cov, y| {
            y[cp] = x[central] / v;
        },
    };

    let subject = subject_iv("iv", "cp");

    let pred_a = analytical.estimate_predictions(&subject, params).unwrap();
    let pred_o = ode.estimate_predictions(&subject, params).unwrap();
    print_comparison("One-compartment IV", &pred_a, &pred_o);
}

// ── One-compartment oral ───────────────────────────────────────────

fn one_cmt_oral(params: &[f64]) {
    let analytical = analytical! {
        name: "one_cmt_oral",
        params: [ka, ke, v],
        states: [gut, central],
        outputs: [cp],
        routes: [
            bolus(oral) -> gut,
        ],
        structure: one_compartment_with_absorption,
        out: |x, _p, _t, _cov, y| {
            y[cp] = x[central] / v;
        },
    };

    let ode = ode! {
        name: "one_cmt_oral",
        params: [ka, ke, v],
        states: [gut, central],
        outputs: [cp],
        routes: [
            bolus(oral) -> gut,
        ],
        diffeq: |x, _p, _t, dx, _cov| {
            dx[gut] = -ka * x[gut];
            dx[central] = ka * x[gut] - ke * x[central];
        },
        out: |x, _p, _t, _cov, y| {
            y[cp] = x[central] / v;
        },
    };

    let subject = subject_oral("oral", "cp");

    let pred_a = analytical.estimate_predictions(&subject, params).unwrap();
    let pred_o = ode.estimate_predictions(&subject, params).unwrap();
    print_comparison("One-compartment oral", &pred_a, &pred_o);
}

// ── Two-compartment IV ─────────────────────────────────────────────

fn two_cmt_iv(params: &[f64]) {
    let analytical = analytical! {
        name: "two_cmt_iv",
        params: [ke, k12, k21, v],
        states: [central, peripheral],
        outputs: [cp],
        routes: [
            infusion(iv) -> central,
        ],
        structure: two_compartments,
        out: |x, _p, _t, _cov, y| {
            y[cp] = x[central] / v;
        },
    };

    let ode = ode! {
        name: "two_cmt_iv",
        params: [ke, k12, k21, v],
        states: [central, peripheral],
        outputs: [cp],
        routes: [
            infusion(iv) -> central,
        ],
        diffeq: |x, _p, _t, dx, _cov| {
            dx[central] = -ke * x[central] - k12 * x[central] + k21 * x[peripheral];
            dx[peripheral] = k12 * x[central] - k21 * x[peripheral];
        },
        out: |x, _p, _t, _cov, y| {
            y[cp] = x[central] / v;
        },
    };

    let subject = subject_iv("iv", "cp");

    let pred_a = analytical.estimate_predictions(&subject, params).unwrap();
    let pred_o = ode.estimate_predictions(&subject, params).unwrap();
    print_comparison("Two-compartment IV", &pred_a, &pred_o);
}

// ── Two-compartment oral ───────────────────────────────────────────

fn two_cmt_oral(params: &[f64]) {
    let analytical = analytical! {
        name: "two_cmt_oral",
        params: [ke, ka, k12, k21, v],
        states: [gut, central, peripheral],
        outputs: [cp],
        routes: [
            bolus(oral) -> gut,
        ],
        structure: two_compartments_with_absorption,
        out: |x, _p, _t, _cov, y| {
            y[cp] = x[central] / v;
        },
    };

    let ode = ode! {
        name: "two_cmt_oral",
        params: [ka, ke, k12, k21, v],
        states: [gut, central, peripheral],
        outputs: [cp],
        routes: [
            bolus(oral) -> gut,
        ],
        diffeq: |x, _p, _t, dx, _cov| {
            dx[gut] = -ka * x[gut];
            dx[central] = ka * x[gut] - ke * x[central] - k12 * x[central] + k21 * x[peripheral];
            dx[peripheral] = k12 * x[central] - k21 * x[peripheral];
        },
        out: |x, _p, _t, _cov, y| {
            y[cp] = x[central] / v;
        },
    };

    let subject = subject_oral("oral", "cp");

    // `two_compartments_with_absorption` is positional and expects
    // `ke, ka, k12, k21, v`, while the ODE closure below is authored as
    // `ka, ke, k12, k21, v`.
    let analytical_params = [params[1], params[0], params[2], params[3], params[4]];

    let pred_a = analytical
        .estimate_predictions(&subject, &analytical_params)
        .unwrap();
    let pred_o = ode.estimate_predictions(&subject, params).unwrap();
    print_comparison("Two-compartment oral", &pred_a, &pred_o);
}

// ── Main ───────────────────────────────────────────────────────────

fn main() {
    one_cmt_iv(&[0.1, 50.0]); // ke, v
    one_cmt_oral(&[1.0, 0.1, 50.0]); // ka, ke, v
    two_cmt_iv(&[0.1, 0.3, 0.2, 50.0]); // ke, k12, k21, v
    two_cmt_oral(&[1.0, 0.1, 0.3, 0.2, 50.0]); // ka, ke, k12, k21, v
}
