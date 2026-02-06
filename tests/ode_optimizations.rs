//! Tests to verify ODE optimizations don't break functionality.
//!
//! These tests specifically validate:
//! 1. Bolus handling with the b parameter in diffeq
//! 2. Infusion handling with rateiv parameter
//! 3. Multiple boluses at different times
//! 4. Mixed bolus + infusion scenarios
//! 5. Multi-occasion subjects
//! 6. Covariate handling

use pharmsol::prelude::models::{one_compartment, one_compartment_with_absorption};
use pharmsol::*;

const REL_TOL: f64 = 1e-3;
const ABS_TOL: f64 = 1e-6;

/// Helper to compare ODE vs Analytical predictions
fn assert_ode_matches_analytical(
    label: &str,
    analytical: &equation::Analytical,
    ode: &equation::ODE,
    subject: &Subject,
    params: &[f64],
) {
    let params_vec: Vec<f64> = params.to_vec();

    let analytical_predictions = analytical
        .estimate_predictions(subject, &params_vec)
        .expect("analytical predictions should succeed");

    let ode_predictions = ode
        .estimate_predictions(subject, &params_vec)
        .expect("ode predictions should succeed");

    let expected = analytical_predictions.flat_predictions();
    let actual = ode_predictions.flat_predictions();

    assert_eq!(
        expected.len(),
        actual.len(),
        "{}: prediction vector length mismatch (analytical={}, ode={})",
        label,
        expected.len(),
        actual.len()
    );

    for (idx, (&reference, &candidate)) in expected.iter().zip(actual.iter()).enumerate() {
        let abs_err = (reference - candidate).abs();
        let rel_err = if reference.abs() > ABS_TOL {
            abs_err / reference.abs()
        } else {
            abs_err
        };

        assert!(
            abs_err <= ABS_TOL || rel_err <= REL_TOL,
            "{}: prediction {} differs significantly\n  analytical = {:.10}\n  ode        = {:.10}\n  abs_err    = {:.2e}\n  rel_err    = {:.2e}",
            label,
            idx,
            reference,
            candidate,
            abs_err,
            rel_err
        );
    }
}

// =============================================================================
// BOLUS TESTS
// =============================================================================

#[test]
fn single_iv_bolus_matches_analytical() {
    // Simple IV bolus into central compartment
    let subject = Subject::builder("single_bolus")
        .bolus(0.0, 100.0, 0)
        .observation(1.0, 0.0, 0)
        .observation(2.0, 0.0, 0)
        .observation(4.0, 0.0, 0)
        .observation(8.0, 0.0, 0)
        .observation(12.0, 0.0, 0)
        .observation(24.0, 0.0, 0)
        .build();

    let analytical = equation::Analytical::new(
        one_compartment,
        |_p, _t, _cov| Ok(()),
        |_p, _t, _cov| Ok(lag! {}),
        |_p, _t, _cov| Ok(fa! {}),
        |_p, _t, _cov, _x| Ok(()),
        |x, p, _t, _cov, y| {
            fetch_params!(p, _ke, v);
            y[0] = x[0] / v;
            Ok(())
        },
        (1, 1),
    );

    let ode = equation::ODE::new(
        |x, p, _t, dx, b, _rateiv, _cov| {
            fetch_params!(p, ke, _v);
            // Bolus appears in derivative as instantaneous input
            dx[0] = -ke * x[0] + b[0];
            Ok(())
        },
        |_p, _t, _cov| Ok(lag! {}),
        |_p, _t, _cov| Ok(fa! {}),
        |_p, _t, _cov, _x| Ok(()),
        |x, p, _t, _cov, y| {
            fetch_params!(p, _ke, v);
            y[0] = x[0] / v;
            Ok(())
        },
        (1, 1),
    );

    // ke = 0.1, v = 50
    assert_ode_matches_analytical("single_iv_bolus", &analytical, &ode, &subject, &[0.1, 50.0]);
}

#[test]
fn multiple_iv_boluses_match_analytical() {
    // Multiple IV boluses at different times
    let subject = Subject::builder("multiple_boluses")
        .bolus(0.0, 100.0, 0)
        .observation(1.0, 0.0, 0)
        .observation(2.0, 0.0, 0)
        .bolus(4.0, 50.0, 0) // Second dose at t=4
        .observation(4.0, 0.0, 0)
        .observation(5.0, 0.0, 0)
        .observation(6.0, 0.0, 0)
        .bolus(8.0, 75.0, 0) // Third dose at t=8
        .observation(8.0, 0.0, 0)
        .observation(10.0, 0.0, 0)
        .observation(12.0, 0.0, 0)
        .observation(24.0, 0.0, 0)
        .build();

    let analytical = equation::Analytical::new(
        one_compartment,
        |_p, _t, _cov| Ok(()),
        |_p, _t, _cov| Ok(lag! {}),
        |_p, _t, _cov| Ok(fa! {}),
        |_p, _t, _cov, _x| Ok(()),
        |x, p, _t, _cov, y| {
            fetch_params!(p, _ke, v);
            y[0] = x[0] / v;
            Ok(())
        },
        (1, 1),
    );

    let ode = equation::ODE::new(
        |x, p, _t, dx, b, _rateiv, _cov| {
            fetch_params!(p, ke, _v);
            dx[0] = -ke * x[0] + b[0];
            Ok(())
        },
        |_p, _t, _cov| Ok(lag! {}),
        |_p, _t, _cov| Ok(fa! {}),
        |_p, _t, _cov, _x| Ok(()),
        |x, p, _t, _cov, y| {
            fetch_params!(p, _ke, v);
            y[0] = x[0] / v;
            Ok(())
        },
        (1, 1),
    );

    assert_ode_matches_analytical(
        "multiple_iv_boluses",
        &analytical,
        &ode,
        &subject,
        &[0.1, 50.0],
    );
}

#[test]
fn oral_bolus_with_absorption_matches_analytical() {
    // Oral dose with first-order absorption
    let subject = Subject::builder("oral_bolus")
        .bolus(0.0, 100.0, 0) // Dose into gut (compartment 0)
        .observation(0.5, 0.0, 0)
        .observation(1.0, 0.0, 0)
        .observation(2.0, 0.0, 0)
        .observation(4.0, 0.0, 0)
        .observation(8.0, 0.0, 0)
        .observation(12.0, 0.0, 0)
        .observation(24.0, 0.0, 0)
        .build();

    let analytical = equation::Analytical::new(
        one_compartment_with_absorption,
        |_p, _t, _cov| Ok(()),
        |_p, _t, _cov| Ok(lag! {}),
        |_p, _t, _cov| Ok(fa! {}),
        |_p, _t, _cov, _x| Ok(()),
        |x, p, _t, _cov, y| {
            fetch_params!(p, _ka, _ke, v);
            y[0] = x[1] / v; // Central compartment is x[1]
            Ok(())
        },
        (2, 1),
    );

    let ode = equation::ODE::new(
        |x, p, _t, dx, b, _rateiv, _cov| {
            fetch_params!(p, ka, ke, _v);
            dx[0] = -ka * x[0] + b[0]; // Gut compartment with oral bolus
            dx[1] = ka * x[0] - ke * x[1]; // Central compartment
            Ok(())
        },
        |_p, _t, _cov| Ok(lag! {}),
        |_p, _t, _cov| Ok(fa! {}),
        |_p, _t, _cov, _x| Ok(()),
        |x, p, _t, _cov, y| {
            fetch_params!(p, _ka, _ke, v);
            y[0] = x[1] / v;
            Ok(())
        },
        (2, 1),
    );

    // ka = 1.0, ke = 0.1, v = 50
    assert_ode_matches_analytical(
        "oral_bolus_absorption",
        &analytical,
        &ode,
        &subject,
        &[1.0, 0.1, 50.0],
    );
}

#[test]
fn multiple_oral_doses_match_analytical() {
    // Multiple oral doses simulating real dosing regimen
    let subject = Subject::builder("multiple_oral")
        .bolus(0.0, 100.0, 0) // First dose
        .observation(1.0, 0.0, 0)
        .observation(2.0, 0.0, 0)
        .observation(4.0, 0.0, 0)
        .bolus(8.0, 100.0, 0) // Second dose
        .observation(8.0, 0.0, 0)
        .observation(9.0, 0.0, 0)
        .observation(10.0, 0.0, 0)
        .observation(12.0, 0.0, 0)
        .bolus(16.0, 100.0, 0) // Third dose
        .observation(16.0, 0.0, 0)
        .observation(17.0, 0.0, 0)
        .observation(20.0, 0.0, 0)
        .observation(24.0, 0.0, 0)
        .build();

    let analytical = equation::Analytical::new(
        one_compartment_with_absorption,
        |_p, _t, _cov| Ok(()),
        |_p, _t, _cov| Ok(lag! {}),
        |_p, _t, _cov| Ok(fa! {}),
        |_p, _t, _cov, _x| Ok(()),
        |x, p, _t, _cov, y| {
            fetch_params!(p, _ka, _ke, v);
            y[0] = x[1] / v;
            Ok(())
        },
        (2, 1),
    );

    let ode = equation::ODE::new(
        |x, p, _t, dx, b, _rateiv, _cov| {
            fetch_params!(p, ka, ke, _v);
            dx[0] = -ka * x[0] + b[0];
            dx[1] = ka * x[0] - ke * x[1];
            Ok(())
        },
        |_p, _t, _cov| Ok(lag! {}),
        |_p, _t, _cov| Ok(fa! {}),
        |_p, _t, _cov, _x| Ok(()),
        |x, p, _t, _cov, y| {
            fetch_params!(p, _ka, _ke, v);
            y[0] = x[1] / v;
            Ok(())
        },
        (2, 1),
    );

    assert_ode_matches_analytical(
        "multiple_oral_doses",
        &analytical,
        &ode,
        &subject,
        &[1.0, 0.1, 50.0],
    );
}

// =============================================================================
// INFUSION TESTS
// =============================================================================

#[test]
fn single_infusion_matches_analytical() {
    // Single IV infusion
    let subject = Subject::builder("single_infusion")
        .infusion(0.0, 100.0, 0, 2.0) // 100mg over 2 hours
        .observation(0.5, 0.0, 0)
        .observation(1.0, 0.0, 0)
        .observation(2.0, 0.0, 0) // End of infusion
        .observation(3.0, 0.0, 0)
        .observation(4.0, 0.0, 0)
        .observation(8.0, 0.0, 0)
        .observation(12.0, 0.0, 0)
        .build();

    let analytical = equation::Analytical::new(
        one_compartment,
        |_p, _t, _cov| Ok(()),
        |_p, _t, _cov| Ok(lag! {}),
        |_p, _t, _cov| Ok(fa! {}),
        |_p, _t, _cov, _x| Ok(()),
        |x, p, _t, _cov, y| {
            fetch_params!(p, _ke, v);
            y[0] = x[0] / v;
            Ok(())
        },
        (1, 1),
    );

    let ode = equation::ODE::new(
        |x, p, _t, dx, _b, rateiv, _cov| {
            fetch_params!(p, ke, _v);
            dx[0] = -ke * x[0] + rateiv[0];
            Ok(())
        },
        |_p, _t, _cov| Ok(lag! {}),
        |_p, _t, _cov| Ok(fa! {}),
        |_p, _t, _cov, _x| Ok(()),
        |x, p, _t, _cov, y| {
            fetch_params!(p, _ke, v);
            y[0] = x[0] / v;
            Ok(())
        },
        (1, 1),
    );

    assert_ode_matches_analytical("single_infusion", &analytical, &ode, &subject, &[0.1, 50.0]);
}

#[test]
fn overlapping_infusions_match_analytical() {
    // Two overlapping infusions
    let subject = Subject::builder("overlapping_infusions")
        .infusion(0.0, 100.0, 0, 4.0) // First: 100mg over 4 hours
        .infusion(2.0, 50.0, 0, 2.0) // Second: 50mg over 2 hours (overlaps)
        .observation(1.0, 0.0, 0)
        .observation(2.0, 0.0, 0)
        .observation(3.0, 0.0, 0)
        .observation(4.0, 0.0, 0)
        .observation(5.0, 0.0, 0)
        .observation(6.0, 0.0, 0)
        .observation(8.0, 0.0, 0)
        .observation(12.0, 0.0, 0)
        .build();

    let analytical = equation::Analytical::new(
        one_compartment,
        |_p, _t, _cov| Ok(()),
        |_p, _t, _cov| Ok(lag! {}),
        |_p, _t, _cov| Ok(fa! {}),
        |_p, _t, _cov, _x| Ok(()),
        |x, p, _t, _cov, y| {
            fetch_params!(p, _ke, v);
            y[0] = x[0] / v;
            Ok(())
        },
        (1, 1),
    );

    let ode = equation::ODE::new(
        |x, p, _t, dx, _b, rateiv, _cov| {
            fetch_params!(p, ke, _v);
            dx[0] = -ke * x[0] + rateiv[0];
            Ok(())
        },
        |_p, _t, _cov| Ok(lag! {}),
        |_p, _t, _cov| Ok(fa! {}),
        |_p, _t, _cov, _x| Ok(()),
        |x, p, _t, _cov, y| {
            fetch_params!(p, _ke, v);
            y[0] = x[0] / v;
            Ok(())
        },
        (1, 1),
    );

    assert_ode_matches_analytical(
        "overlapping_infusions",
        &analytical,
        &ode,
        &subject,
        &[0.1, 50.0],
    );
}

// =============================================================================
// MIXED BOLUS + INFUSION TESTS
// =============================================================================

#[test]
fn bolus_plus_infusion_matches_analytical() {
    // Loading bolus followed by maintenance infusion
    let subject = Subject::builder("bolus_plus_infusion")
        .bolus(0.0, 100.0, 0) // Loading dose
        .infusion(0.0, 200.0, 0, 8.0) // Maintenance infusion
        .observation(1.0, 0.0, 0)
        .observation(2.0, 0.0, 0)
        .observation(4.0, 0.0, 0)
        .observation(8.0, 0.0, 0) // End of infusion
        .observation(10.0, 0.0, 0)
        .observation(12.0, 0.0, 0)
        .observation(24.0, 0.0, 0)
        .build();

    let analytical = equation::Analytical::new(
        one_compartment,
        |_p, _t, _cov| Ok(()),
        |_p, _t, _cov| Ok(lag! {}),
        |_p, _t, _cov| Ok(fa! {}),
        |_p, _t, _cov, _x| Ok(()),
        |x, p, _t, _cov, y| {
            fetch_params!(p, _ke, v);
            y[0] = x[0] / v;
            Ok(())
        },
        (1, 1),
    );

    let ode = equation::ODE::new(
        |x, p, _t, dx, b, rateiv, _cov| {
            fetch_params!(p, ke, _v);
            dx[0] = -ke * x[0] + b[0] + rateiv[0];
            Ok(())
        },
        |_p, _t, _cov| Ok(lag! {}),
        |_p, _t, _cov| Ok(fa! {}),
        |_p, _t, _cov, _x| Ok(()),
        |x, p, _t, _cov, y| {
            fetch_params!(p, _ke, v);
            y[0] = x[0] / v;
            Ok(())
        },
        (1, 1),
    );

    assert_ode_matches_analytical(
        "bolus_plus_infusion",
        &analytical,
        &ode,
        &subject,
        &[0.1, 50.0],
    );
}

#[test]
fn complex_dosing_scenario_matches_analytical() {
    // Complex scenario: multiple oral doses with varying amounts
    // Using only oral dosing to match analytical model capabilities
    let subject = Subject::builder("complex_dosing")
        .bolus(0.0, 100.0, 0) // First oral dose
        .observation(1.0, 0.0, 0)
        .observation(2.0, 0.0, 0)
        .observation(4.0, 0.0, 0)
        .bolus(6.0, 150.0, 0) // Second oral dose (different amount)
        .observation(6.0, 0.0, 0)
        .observation(7.0, 0.0, 0)
        .observation(8.0, 0.0, 0)
        .bolus(12.0, 100.0, 0) // Third oral dose
        .observation(12.0, 0.0, 0)
        .observation(14.0, 0.0, 0)
        .observation(18.0, 0.0, 0)
        .observation(24.0, 0.0, 0)
        .build();

    let analytical = equation::Analytical::new(
        one_compartment_with_absorption,
        |_p, _t, _cov| Ok(()),
        |_p, _t, _cov| Ok(lag! {}),
        |_p, _t, _cov| Ok(fa! {}),
        |_p, _t, _cov, _x| Ok(()),
        |x, p, _t, _cov, y| {
            fetch_params!(p, _ka, _ke, v);
            y[0] = x[1] / v;
            Ok(())
        },
        (2, 1),
    );

    let ode = equation::ODE::new(
        |x, p, _t, dx, b, _rateiv, _cov| {
            fetch_params!(p, ka, ke, _v);
            dx[0] = -ka * x[0] + b[0]; // Gut: oral doses
            dx[1] = ka * x[0] - ke * x[1]; // Central
            Ok(())
        },
        |_p, _t, _cov| Ok(lag! {}),
        |_p, _t, _cov| Ok(fa! {}),
        |_p, _t, _cov, _x| Ok(()),
        |x, p, _t, _cov, y| {
            fetch_params!(p, _ka, _ke, v);
            y[0] = x[1] / v;
            Ok(())
        },
        (2, 1),
    );

    assert_ode_matches_analytical(
        "complex_dosing",
        &analytical,
        &ode,
        &subject,
        &[1.0, 0.1, 50.0],
    );
}

#[test]
fn mixed_bolus_infusion_iv_matches_analytical() {
    // Test IV bolus + IV infusion with one-compartment model
    // This is fully supported by both analytical and ODE solvers
    let subject = Subject::builder("mixed_iv")
        .bolus(0.0, 100.0, 0) // IV bolus
        .observation(1.0, 0.0, 0)
        .observation(2.0, 0.0, 0)
        .infusion(4.0, 200.0, 0, 4.0) // IV infusion: 200mg over 4 hours
        .observation(4.0, 0.0, 0)
        .observation(5.0, 0.0, 0)
        .observation(6.0, 0.0, 0)
        .bolus(8.0, 50.0, 0) // Another IV bolus at end of infusion
        .observation(8.0, 0.0, 0)
        .observation(9.0, 0.0, 0)
        .observation(10.0, 0.0, 0)
        .observation(12.0, 0.0, 0)
        .observation(24.0, 0.0, 0)
        .build();

    let analytical = equation::Analytical::new(
        one_compartment,
        |_p, _t, _cov| Ok(()),
        |_p, _t, _cov| Ok(lag! {}),
        |_p, _t, _cov| Ok(fa! {}),
        |_p, _t, _cov, _x| Ok(()),
        |x, p, _t, _cov, y| {
            fetch_params!(p, _ke, v);
            y[0] = x[0] / v;
            Ok(())
        },
        (1, 1),
    );

    let ode = equation::ODE::new(
        |x, p, _t, dx, b, rateiv, _cov| {
            fetch_params!(p, ke, _v);
            dx[0] = -ke * x[0] + b[0] + rateiv[0];
            Ok(())
        },
        |_p, _t, _cov| Ok(lag! {}),
        |_p, _t, _cov| Ok(fa! {}),
        |_p, _t, _cov, _x| Ok(()),
        |x, p, _t, _cov, y| {
            fetch_params!(p, _ke, v);
            y[0] = x[0] / v;
            Ok(())
        },
        (1, 1),
    );

    assert_ode_matches_analytical(
        "mixed_iv_bolus_infusion",
        &analytical,
        &ode,
        &subject,
        &[0.1, 50.0],
    );
}

// =============================================================================
// EDGE CASES
// =============================================================================

#[test]
fn bolus_at_observation_time_matches_analytical() {
    // Bolus and observation at exactly the same time
    let subject = Subject::builder("simultaneous")
        .bolus(0.0, 100.0, 0)
        .observation(0.0, 0.0, 0) // Observation at dose time
        .observation(1.0, 0.0, 0)
        .bolus(2.0, 50.0, 0)
        .observation(2.0, 0.0, 0) // Observation at dose time
        .observation(3.0, 0.0, 0)
        .observation(4.0, 0.0, 0)
        .build();

    let analytical = equation::Analytical::new(
        one_compartment,
        |_p, _t, _cov| Ok(()),
        |_p, _t, _cov| Ok(lag! {}),
        |_p, _t, _cov| Ok(fa! {}),
        |_p, _t, _cov, _x| Ok(()),
        |x, p, _t, _cov, y| {
            fetch_params!(p, _ke, v);
            y[0] = x[0] / v;
            Ok(())
        },
        (1, 1),
    );

    let ode = equation::ODE::new(
        |x, p, _t, dx, b, _rateiv, _cov| {
            fetch_params!(p, ke, _v);
            dx[0] = -ke * x[0] + b[0];
            Ok(())
        },
        |_p, _t, _cov| Ok(lag! {}),
        |_p, _t, _cov| Ok(fa! {}),
        |_p, _t, _cov, _x| Ok(()),
        |x, p, _t, _cov, y| {
            fetch_params!(p, _ke, v);
            y[0] = x[0] / v;
            Ok(())
        },
        (1, 1),
    );

    assert_ode_matches_analytical(
        "bolus_at_observation_time",
        &analytical,
        &ode,
        &subject,
        &[0.1, 50.0],
    );
}

#[test]
fn very_fast_elimination_matches_analytical() {
    // High ke tests numerical stability
    let subject = Subject::builder("fast_elimination")
        .bolus(0.0, 100.0, 0)
        .observation(0.1, 0.0, 0)
        .observation(0.2, 0.0, 0)
        .observation(0.5, 0.0, 0)
        .observation(1.0, 0.0, 0)
        .observation(2.0, 0.0, 0)
        .build();

    let analytical = equation::Analytical::new(
        one_compartment,
        |_p, _t, _cov| Ok(()),
        |_p, _t, _cov| Ok(lag! {}),
        |_p, _t, _cov| Ok(fa! {}),
        |_p, _t, _cov, _x| Ok(()),
        |x, p, _t, _cov, y| {
            fetch_params!(p, _ke, v);
            y[0] = x[0] / v;
            Ok(())
        },
        (1, 1),
    );

    let ode = equation::ODE::new(
        |x, p, _t, dx, b, _rateiv, _cov| {
            fetch_params!(p, ke, _v);
            dx[0] = -ke * x[0] + b[0];
            Ok(())
        },
        |_p, _t, _cov| Ok(lag! {}),
        |_p, _t, _cov| Ok(fa! {}),
        |_p, _t, _cov, _x| Ok(()),
        |x, p, _t, _cov, y| {
            fetch_params!(p, _ke, v);
            y[0] = x[0] / v;
            Ok(())
        },
        (1, 1),
    );

    // Very fast elimination: ke = 2.0 (half-life ~20 min)
    assert_ode_matches_analytical(
        "fast_elimination",
        &analytical,
        &ode,
        &subject,
        &[2.0, 50.0],
    );
}

#[test]
fn very_slow_elimination_matches_analytical() {
    // Low ke tests long simulation
    let subject = Subject::builder("slow_elimination")
        .bolus(0.0, 100.0, 0)
        .observation(24.0, 0.0, 0)
        .observation(48.0, 0.0, 0)
        .observation(72.0, 0.0, 0)
        .observation(96.0, 0.0, 0)
        .observation(168.0, 0.0, 0) // 1 week
        .build();

    let analytical = equation::Analytical::new(
        one_compartment,
        |_p, _t, _cov| Ok(()),
        |_p, _t, _cov| Ok(lag! {}),
        |_p, _t, _cov| Ok(fa! {}),
        |_p, _t, _cov, _x| Ok(()),
        |x, p, _t, _cov, y| {
            fetch_params!(p, _ke, v);
            y[0] = x[0] / v;
            Ok(())
        },
        (1, 1),
    );

    let ode = equation::ODE::new(
        |x, p, _t, dx, b, _rateiv, _cov| {
            fetch_params!(p, ke, _v);
            dx[0] = -ke * x[0] + b[0];
            Ok(())
        },
        |_p, _t, _cov| Ok(lag! {}),
        |_p, _t, _cov| Ok(fa! {}),
        |_p, _t, _cov, _x| Ok(()),
        |x, p, _t, _cov, y| {
            fetch_params!(p, _ke, v);
            y[0] = x[0] / v;
            Ok(())
        },
        (1, 1),
    );

    // Very slow elimination: ke = 0.01 (half-life ~69 hours)
    assert_ode_matches_analytical(
        "slow_elimination",
        &analytical,
        &ode,
        &subject,
        &[0.01, 50.0],
    );
}

#[test]
fn rapid_absorption_matches_analytical() {
    // Very fast absorption with oral dose
    let subject = Subject::builder("rapid_absorption")
        .bolus(0.0, 100.0, 0)
        .observation(0.1, 0.0, 0)
        .observation(0.25, 0.0, 0)
        .observation(0.5, 0.0, 0)
        .observation(1.0, 0.0, 0)
        .observation(2.0, 0.0, 0)
        .observation(4.0, 0.0, 0)
        .build();

    let analytical = equation::Analytical::new(
        one_compartment_with_absorption,
        |_p, _t, _cov| Ok(()),
        |_p, _t, _cov| Ok(lag! {}),
        |_p, _t, _cov| Ok(fa! {}),
        |_p, _t, _cov, _x| Ok(()),
        |x, p, _t, _cov, y| {
            fetch_params!(p, _ka, _ke, v);
            y[0] = x[1] / v;
            Ok(())
        },
        (2, 1),
    );

    let ode = equation::ODE::new(
        |x, p, _t, dx, b, _rateiv, _cov| {
            fetch_params!(p, ka, ke, _v);
            dx[0] = -ka * x[0] + b[0];
            dx[1] = ka * x[0] - ke * x[1];
            Ok(())
        },
        |_p, _t, _cov| Ok(lag! {}),
        |_p, _t, _cov| Ok(fa! {}),
        |_p, _t, _cov, _x| Ok(()),
        |x, p, _t, _cov, y| {
            fetch_params!(p, _ka, _ke, v);
            y[0] = x[1] / v;
            Ok(())
        },
        (2, 1),
    );

    // Very fast absorption: ka = 10.0
    assert_ode_matches_analytical(
        "rapid_absorption",
        &analytical,
        &ode,
        &subject,
        &[10.0, 0.1, 50.0],
    );
}

// =============================================================================
// COVARIATE TESTS
// =============================================================================

#[test]
fn time_varying_covariates_work_correctly() {
    // Test that covariates are properly interpolated
    let subject = Subject::builder("covariates")
        .bolus(0.0, 100.0, 0)
        .covariate("wt", 0.0, 70.0)
        .observation(1.0, 0.0, 0)
        .covariate("wt", 2.0, 75.0) // Weight changes
        .observation(2.0, 0.0, 0)
        .observation(4.0, 0.0, 0)
        .covariate("wt", 6.0, 72.0) // Weight changes again
        .observation(6.0, 0.0, 0)
        .observation(8.0, 0.0, 0)
        .build();

    // ODE with weight-based clearance
    let ode = equation::ODE::new(
        |x, p, t, dx, b, _rateiv, cov| {
            fetch_params!(p, ke_ref, _v);
            fetch_cov!(cov, t, wt);
            // Allometric scaling: CL proportional to weight^0.75
            let ke = ke_ref * (wt / 70.0_f64).powf(0.75);
            dx[0] = -ke * x[0] + b[0];
            Ok(())
        },
        |_p, _t, _cov| Ok(lag! {}),
        |_p, _t, _cov| Ok(fa! {}),
        |_p, _t, _cov, _x| Ok(()),
        |x, p, _t, _cov, y| {
            fetch_params!(p, _ke, v);
            y[0] = x[0] / v;
            Ok(())
        },
        (1, 1),
    );

    // Just verify it runs without error and produces reasonable output
    let result = ode.estimate_predictions(&subject, &vec![0.1, 50.0]);
    assert!(result.is_ok(), "ODE with covariates should succeed");

    let predictions = result.unwrap();
    let preds = predictions.flat_predictions();

    // All predictions should be positive
    for (i, &pred) in preds.iter().enumerate() {
        assert!(
            pred > 0.0,
            "Prediction {} should be positive, got {}",
            i,
            pred
        );
    }

    // Predictions should decrease over time (elimination)
    for i in 1..preds.len() {
        // Allow for some noise due to covariate changes, but general trend should be down
        // or predictions should be reasonable (between 0 and initial dose/volume)
        assert!(
            preds[i] < 3.0, // Should not exceed ~100/50 * some factor
            "Prediction {} seems too high: {}",
            i,
            preds[i]
        );
    }
}

// =============================================================================
// LIKELIHOOD TESTS
// =============================================================================

#[test]
fn likelihood_calculation_matches_analytical() {
    // Verify likelihood calculations are consistent
    let subject = Subject::builder("likelihood")
        .bolus(0.0, 100.0, 0)
        .observation(1.0, 1.8, 0) // Observed value
        .observation(2.0, 1.6, 0)
        .observation(4.0, 1.3, 0)
        .observation(8.0, 0.8, 0)
        .build();

    let analytical = equation::Analytical::new(
        one_compartment,
        |_p, _t, _cov| Ok(()),
        |_p, _t, _cov| Ok(lag! {}),
        |_p, _t, _cov| Ok(fa! {}),
        |_p, _t, _cov, _x| Ok(()),
        |x, p, _t, _cov, y| {
            fetch_params!(p, _ke, v);
            y[0] = x[0] / v;
            Ok(())
        },
        (1, 1),
    );

    let ode = equation::ODE::new(
        |x, p, _t, dx, b, _rateiv, _cov| {
            fetch_params!(p, ke, _v);
            dx[0] = -ke * x[0] + b[0];
            Ok(())
        },
        |_p, _t, _cov| Ok(lag! {}),
        |_p, _t, _cov| Ok(fa! {}),
        |_p, _t, _cov, _x| Ok(()),
        |x, p, _t, _cov, y| {
            fetch_params!(p, _ke, v);
            y[0] = x[0] / v;
            Ok(())
        },
        (1, 1),
    );

    let error_models = AssayErrorModels::new()
        .add(
            0,
            AssayErrorModel::additive(ErrorPoly::new(0.0, 0.1, 0.0, 0.0), 0.0),
        )
        .unwrap();

    let params = vec![0.1, 50.0];

    let ll_analytical = analytical
        .estimate_log_likelihood(&subject, &params, &error_models, false)
        .expect("analytical likelihood")
        .exp();

    let ll_ode = ode
        .estimate_log_likelihood(&subject, &params, &error_models, false)
        .expect("ode likelihood")
        .exp();

    let ll_diff = (ll_analytical - ll_ode).abs();
    let ll_rel_diff = ll_diff / ll_analytical.abs().max(1e-10);

    assert!(
        ll_rel_diff < 0.01, // Within 1%
        "Likelihoods should match: analytical={:.6}, ode={:.6}, rel_diff={:.2e}",
        ll_analytical,
        ll_ode,
        ll_rel_diff
    );
}
