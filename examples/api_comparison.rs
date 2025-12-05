//! Comparison example demonstrating both the old (tuple-based) and new (builder) APIs.
//!
//! This example shows that both APIs produce identical results, ensuring backward compatibility
//! while providing a more ergonomic builder pattern for new code.
//!
//! The new builder API uses the type-state pattern to enforce required fields at compile time,
//! while making optional fields (lag, fa, init) truly optional with sensible defaults.

use pharmsol::prelude::models::one_compartment;
use pharmsol::*;

fn main() {
    println!("=== API Comparison: Old (Tuple) vs New (Builder) ===\n");

    // Create a simple subject for testing
    let subject = Subject::builder("comparison_test")
        .infusion(0.0, 500.0, 0, 0.5)
        .observation(0.5, 0.0, 0)
        .observation(1.0, 0.0, 0)
        .observation(2.0, 0.0, 0)
        .observation(4.0, 0.0, 0)
        .observation(8.0, 0.0, 0)
        .build();

    let params = vec![1.02282724609375, 194.51904296875]; // ke, v

    println!("--- ODE Models ---");

    let ode_old = equation::ODE::new(
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
        (1, 1),
    );

    let ode_minimal = equation::ODE::builder()
        .diffeq(|x, p, _t, dx, _b, rateiv, _cov| {
            fetch_params!(p, ke, _v);
            dx[0] = -ke * x[0] + rateiv[0];
        })
        .out(|x, p, _t, _cov, y| {
            fetch_params!(p, _ke, v);
            y[0] = x[0] / v;
        })
        .nstates(1)
        .nouteqs(1)
        .build();

    // Also show full specification with optional fields
    let ode_full = equation::ODE::builder()
        .diffeq(|x, p, _t, dx, _b, rateiv, _cov| {
            fetch_params!(p, ke, _v);
            dx[0] = -ke * x[0] + rateiv[0];
        })
        .out(|x, p, _t, _cov, y| {
            fetch_params!(p, _ke, v);
            y[0] = x[0] / v;
        })
        .neqs(Neqs::new(1, 1)) // Can also use Neqs struct
        .build();

    // Compare predictions
    let pred_old = ode_old.estimate_predictions(&subject, &params).unwrap();
    let pred_minimal = ode_minimal.estimate_predictions(&subject, &params).unwrap();
    let pred_full = ode_full.estimate_predictions(&subject, &params).unwrap();

    println!("ODE Predictions (Old API with all 6 args):");
    for p in pred_old.flat_predictions() {
        print!("{:.9} ", p);
    }
    println!("\n");

    println!("ODE Predictions (New Builder - minimal, 4 required fields only):");
    for p in pred_minimal.flat_predictions() {
        print!("{:.9} ", p);
    }
    println!("\n");

    println!("ODE Predictions (New Builder - all fields explicit):");
    for p in pred_full.flat_predictions() {
        print!("{:.9} ", p);
    }
    println!("\n");

    // Verify they match
    let old_preds = pred_old.flat_predictions();
    let minimal_preds = pred_minimal.flat_predictions();
    let full_preds = pred_full.flat_predictions();

    let all_match = old_preds
        .iter()
        .zip(minimal_preds.iter())
        .zip(full_preds.iter())
        .all(|((a, b), c)| (a - b).abs() < 1e-12 && (a - c).abs() < 1e-12);

    println!("All ODE APIs produce identical results: {} ✓", all_match);
    println!();

    // =========================================================================
    // Analytical: Old API (tuple-based)
    // =========================================================================
    println!("--- Analytical Models ---");

    let analytical_old = equation::Analytical::new(
        one_compartment,
        |_p, _t, _cov| {},
        |_p, _t, _cov| lag! {},
        |_p, _t, _cov| fa! {},
        |_p, _t, _cov, _x| {},
        |x, p, _t, _cov, y| {
            fetch_params!(p, _ke, v);
            y[0] = x[0] / v;
        },
        (1, 1), // Old tuple-based Neqs
    );

    // =========================================================================
    // Analytical: New API (builder pattern) - MINIMAL version
    // Required fields: eq, seq_eq, out, nstates, nouteqs
    // =========================================================================
    let analytical_minimal = equation::Analytical::builder()
        .eq(one_compartment)
        .seq_eq(|_p, _t, _cov| {})
        .out(|x, p, _t, _cov, y| {
            fetch_params!(p, _ke, v);
            y[0] = x[0] / v;
        })
        .nstates(1)
        .nouteqs(1)
        .build();

    // Compare predictions
    let an_pred_old = analytical_old
        .estimate_predictions(&subject, &params)
        .unwrap();
    let an_pred_minimal = analytical_minimal
        .estimate_predictions(&subject, &params)
        .unwrap();

    println!("Analytical Predictions (Old API with all 7 args):");
    for p in an_pred_old.flat_predictions() {
        print!("{:.9} ", p);
    }
    println!("\n");

    println!("Analytical Predictions (New Builder - minimal, 5 required fields only):");
    for p in an_pred_minimal.flat_predictions() {
        print!("{:.9} ", p);
    }
    println!("\n");

    // Verify they match
    let an_old_preds = an_pred_old.flat_predictions();
    let an_minimal_preds = an_pred_minimal.flat_predictions();

    let analytical_match = an_old_preds
        .iter()
        .zip(an_minimal_preds.iter())
        .all(|(a, b)| (a - b).abs() < 1e-12);

    println!(
        "Analytical APIs produce identical results: {} ✓",
        analytical_match
    );
    println!();
}
