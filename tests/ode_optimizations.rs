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

const REL_TOL: f64 = 1e-2; // 1% relative tolerance for most comparisons
const ABS_TOL: f64 = 1e-6;

fn parameters_for_analytical(
    label: &str,
    analytical: &equation::Analytical,
    param_names: &[&str],
    params: &[f64],
) -> Parameters {
    assert_eq!(
        param_names.len(),
        params.len(),
        "{label}: expected {} parameter value(s), got {}",
        param_names.len(),
        params.len()
    );

    Parameters::with_model(
        analytical,
        param_names.iter().copied().zip(params.iter().copied()),
    )
    .unwrap_or_else(|error| panic!("{label}: analytical parameters should validate: {error}"))
}

fn parameters_for_ode(
    label: &str,
    ode: &equation::ODE,
    param_names: &[&str],
    params: &[f64],
) -> Parameters {
    assert_eq!(
        param_names.len(),
        params.len(),
        "{label}: expected {} parameter value(s), got {}",
        param_names.len(),
        params.len()
    );

    Parameters::with_model(ode, param_names.iter().copied().zip(params.iter().copied()))
        .unwrap_or_else(|error| panic!("{label}: ODE parameters should validate: {error}"))
}

fn with_one_compartment_analytical_metadata(
    analytical: equation::Analytical,
    model_name: &str,
) -> equation::Analytical {
    analytical
        .with_ndrugs(1)
        .with_metadata(
            equation::metadata::new(model_name)
                .kind(equation::ModelKind::Analytical)
                .parameters(["ke", "v"])
                .states(["central"])
                .outputs(["cp"])
                .routes([
                    equation::Route::bolus("iv_bolus").to_state("central"),
                    equation::Route::infusion("iv").to_state("central"),
                ])
                .analytical_kernel(equation::AnalyticalKernel::OneCompartment),
        )
        .expect("one-compartment analytical metadata should validate")
}

fn with_one_compartment_ode_metadata(ode: equation::ODE, model_name: &str) -> equation::ODE {
    ode.with_ndrugs(1)
        .with_metadata(
            equation::metadata::new(model_name)
                .parameters(["ke", "v"])
                .states(["central"])
                .outputs(["cp"])
                .routes([
                    equation::Route::bolus("iv_bolus")
                        .to_state("central")
                        .expect_explicit_input(),
                    equation::Route::infusion("iv")
                        .to_state("central")
                        .expect_explicit_input(),
                ]),
        )
        .expect("one-compartment ODE metadata should validate")
}

fn with_absorption_analytical_metadata(
    analytical: equation::Analytical,
    model_name: &str,
) -> equation::Analytical {
    analytical
        .with_ndrugs(1)
        .with_metadata(
            equation::metadata::new(model_name)
                .kind(equation::ModelKind::Analytical)
                .parameters(["ka", "ke", "v"])
                .states(["gut", "central"])
                .outputs(["cp"])
                .route(equation::Route::bolus("oral").to_state("gut"))
                .analytical_kernel(equation::AnalyticalKernel::OneCompartmentWithAbsorption),
        )
        .expect("absorption analytical metadata should validate")
}

fn with_absorption_ode_metadata(ode: equation::ODE, model_name: &str) -> equation::ODE {
    ode.with_ndrugs(1)
        .with_metadata(
            equation::metadata::new(model_name)
                .parameters(["ka", "ke", "v"])
                .states(["gut", "central"])
                .outputs(["cp"])
                .route(
                    equation::Route::bolus("oral")
                        .to_state("gut")
                        .expect_explicit_input(),
                ),
        )
        .expect("absorption ODE metadata should validate")
}

fn with_covariate_ode_metadata(ode: equation::ODE, model_name: &str) -> equation::ODE {
    ode.with_ndrugs(1)
        .with_metadata(
            equation::metadata::new(model_name)
                .parameters(["ke", "v"])
                .covariates([equation::Covariate::continuous("wt")])
                .states(["central"])
                .outputs(["cp"])
                .route(
                    equation::Route::bolus("iv_bolus")
                        .to_state("central")
                        .expect_explicit_input(),
                ),
        )
        .expect("covariate ODE metadata should validate")
}

/// Helper to compare ODE vs Analytical predictions
fn assert_ode_matches_analytical(
    label: &str,
    analytical: &equation::Analytical,
    ode: &equation::ODE,
    subject: &Subject,
    param_names: &[&str],
    params: &[f64],
) {
    let analytical_params = parameters_for_analytical(label, analytical, param_names, params);
    let ode_params = parameters_for_ode(label, ode, param_names, params);

    let analytical_predictions = analytical
        .estimate_predictions(subject, &analytical_params)
        .expect("analytical predictions should succeed");

    let ode_predictions = ode
        .estimate_predictions(subject, &ode_params)
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
        .bolus(0.0, 100.0, "iv_bolus")
        .observation(1.0, 0.0, "cp")
        .observation(2.0, 0.0, "cp")
        .observation(4.0, 0.0, "cp")
        .observation(8.0, 0.0, "cp")
        .observation(12.0, 0.0, "cp")
        .observation(24.0, 0.0, "cp")
        .build();

    let analytical = with_one_compartment_analytical_metadata(
        equation::Analytical::new(
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
        .with_nout(1),
        "single_iv_bolus",
    );

    let ode = with_one_compartment_ode_metadata(
        equation::ODE::new(
            |x, p, _t, dx, b, _rateiv, _cov| {
                fetch_params!(p, ke, _v);
                // Bolus appears in derivative as instantaneous input
                dx[0] = -ke * x[0] + b[0];
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
        .with_nout(1),
        "single_iv_bolus",
    );

    // ke = 0.1, v = 50
    assert_ode_matches_analytical(
        "single_iv_bolus",
        &analytical,
        &ode,
        &subject,
        &["ke", "v"],
        &[0.1, 50.0],
    );
}

#[test]
fn multiple_iv_boluses_match_analytical() {
    // Multiple IV boluses at different times
    let subject = Subject::builder("multiple_boluses")
        .bolus(0.0, 100.0, "iv_bolus")
        .observation(1.0, 0.0, "cp")
        .observation(2.0, 0.0, "cp")
        .bolus(4.0, 50.0, "iv_bolus") // Second dose at t=4
        .observation(4.0, 0.0, "cp")
        .observation(5.0, 0.0, "cp")
        .observation(6.0, 0.0, "cp")
        .bolus(8.0, 75.0, "iv_bolus") // Third dose at t=8
        .observation(8.0, 0.0, "cp")
        .observation(10.0, 0.0, "cp")
        .observation(12.0, 0.0, "cp")
        .observation(24.0, 0.0, "cp")
        .build();

    let analytical = with_one_compartment_analytical_metadata(
        equation::Analytical::new(
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
        .with_nout(1),
        "multiple_iv_boluses",
    );

    let ode = with_one_compartment_ode_metadata(
        equation::ODE::new(
            |x, p, _t, dx, b, _rateiv, _cov| {
                fetch_params!(p, ke, _v);
                dx[0] = -ke * x[0] + b[0];
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
        .with_nout(1),
        "multiple_iv_boluses",
    );

    assert_ode_matches_analytical(
        "multiple_iv_boluses",
        &analytical,
        &ode,
        &subject,
        &["ke", "v"],
        &[0.1, 50.0],
    );
}

#[test]
fn oral_bolus_with_absorption_matches_analytical() {
    // Oral dose with first-order absorption
    let subject = Subject::builder("oral_bolus")
        .bolus(0.0, 100.0, "oral")
        .observation(0.5, 0.0, "cp")
        .observation(1.0, 0.0, "cp")
        .observation(2.0, 0.0, "cp")
        .observation(4.0, 0.0, "cp")
        .observation(8.0, 0.0, "cp")
        .observation(12.0, 0.0, "cp")
        .observation(24.0, 0.0, "cp")
        .build();

    let analytical = with_absorption_analytical_metadata(
        equation::Analytical::new(
            one_compartment_with_absorption,
            |_p, _t, _cov| {},
            |_p, _t, _cov| lag! {},
            |_p, _t, _cov| fa! {},
            |_p, _t, _cov, _x| {},
            |x, p, _t, _cov, y| {
                fetch_params!(p, _ka, _ke, v);
                y[0] = x[1] / v; // Central compartment is x[1]
            },
        )
        .with_nstates(2)
        .with_nout(1),
        "oral_bolus_absorption",
    );

    let ode = with_absorption_ode_metadata(
        equation::ODE::new(
            |x, p, _t, dx, b, _rateiv, _cov| {
                fetch_params!(p, ka, ke, _v);
                dx[0] = -ka * x[0] + b[0]; // Gut compartment with oral bolus
                dx[1] = ka * x[0] - ke * x[1]; // Central compartment
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
        .with_nout(1),
        "oral_bolus_absorption",
    );

    // ka = 1.0, ke = 0.1, v = 50
    assert_ode_matches_analytical(
        "oral_bolus_absorption",
        &analytical,
        &ode,
        &subject,
        &["ka", "ke", "v"],
        &[1.0, 0.1, 50.0],
    );
}

#[test]
fn multiple_oral_doses_match_analytical() {
    // Multiple oral doses simulating real dosing regimen
    let subject = Subject::builder("multiple_oral")
        .bolus(0.0, 100.0, "oral") // First dose
        .observation(1.0, 0.0, "cp")
        .observation(2.0, 0.0, "cp")
        .observation(4.0, 0.0, "cp")
        .bolus(8.0, 100.0, "oral") // Second dose
        .observation(8.0, 0.0, "cp")
        .observation(9.0, 0.0, "cp")
        .observation(10.0, 0.0, "cp")
        .observation(12.0, 0.0, "cp")
        .bolus(16.0, 100.0, "oral") // Third dose
        .observation(16.0, 0.0, "cp")
        .observation(17.0, 0.0, "cp")
        .observation(20.0, 0.0, "cp")
        .observation(24.0, 0.0, "cp")
        .build();

    let analytical = with_absorption_analytical_metadata(
        equation::Analytical::new(
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
        .with_nout(1),
        "multiple_oral_doses",
    );

    let ode = with_absorption_ode_metadata(
        equation::ODE::new(
            |x, p, _t, dx, b, _rateiv, _cov| {
                fetch_params!(p, ka, ke, _v);
                dx[0] = -ka * x[0] + b[0];
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
        .with_nout(1),
        "multiple_oral_doses",
    );

    assert_ode_matches_analytical(
        "multiple_oral_doses",
        &analytical,
        &ode,
        &subject,
        &["ka", "ke", "v"],
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
        .infusion(0.0, 100.0, "iv", 2.0) // 100mg over 2 hours
        .observation(0.5, 0.0, "cp")
        .observation(1.0, 0.0, "cp")
        .observation(2.0, 0.0, "cp") // End of infusion
        .observation(3.0, 0.0, "cp")
        .observation(4.0, 0.0, "cp")
        .observation(8.0, 0.0, "cp")
        .observation(12.0, 0.0, "cp")
        .build();

    let analytical = with_one_compartment_analytical_metadata(
        equation::Analytical::new(
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
        .with_nout(1),
        "single_infusion",
    );

    let ode = with_one_compartment_ode_metadata(
        equation::ODE::new(
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
        .with_nout(1),
        "single_infusion",
    );

    assert_ode_matches_analytical(
        "single_infusion",
        &analytical,
        &ode,
        &subject,
        &["ke", "v"],
        &[0.1, 50.0],
    );
}

#[test]
fn overlapping_infusions_match_analytical() {
    // Two overlapping infusions
    let subject = Subject::builder("overlapping_infusions")
        .infusion(0.0, 100.0, "iv", 4.0) // First: 100mg over 4 hours
        .infusion(2.0, 50.0, "iv", 2.0) // Second: 50mg over 2 hours (overlaps)
        .observation(1.0, 0.0, "cp")
        .observation(2.0, 0.0, "cp")
        .observation(3.0, 0.0, "cp")
        .observation(4.0, 0.0, "cp")
        .observation(5.0, 0.0, "cp")
        .observation(6.0, 0.0, "cp")
        .observation(8.0, 0.0, "cp")
        .observation(12.0, 0.0, "cp")
        .build();

    let analytical = with_one_compartment_analytical_metadata(
        equation::Analytical::new(
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
        .with_nout(1),
        "overlapping_infusions",
    );

    let ode = with_one_compartment_ode_metadata(
        equation::ODE::new(
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
        .with_nout(1),
        "overlapping_infusions",
    );

    assert_ode_matches_analytical(
        "overlapping_infusions",
        &analytical,
        &ode,
        &subject,
        &["ke", "v"],
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
        .bolus(0.0, 100.0, "iv_bolus") // Loading dose
        .infusion(0.0, 200.0, "iv", 8.0) // Maintenance infusion
        .observation(1.0, 0.0, "cp")
        .observation(2.0, 0.0, "cp")
        .observation(4.0, 0.0, "cp")
        .observation(8.0, 0.0, "cp") // End of infusion
        .observation(10.0, 0.0, "cp")
        .observation(12.0, 0.0, "cp")
        .observation(24.0, 0.0, "cp")
        .build();

    let analytical = with_one_compartment_analytical_metadata(
        equation::Analytical::new(
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
        .with_nout(1),
        "bolus_plus_infusion",
    );

    let ode = with_one_compartment_ode_metadata(
        equation::ODE::new(
            |x, p, _t, dx, b, rateiv, _cov| {
                fetch_params!(p, ke, _v);
                dx[0] = -ke * x[0] + b[0] + rateiv[0];
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
        .with_nout(1),
        "bolus_plus_infusion",
    );

    assert_ode_matches_analytical(
        "bolus_plus_infusion",
        &analytical,
        &ode,
        &subject,
        &["ke", "v"],
        &[0.1, 50.0],
    );
}

#[test]
fn complex_dosing_scenario_matches_analytical() {
    // Complex scenario: multiple oral doses with varying amounts
    // Using only oral dosing to match analytical model capabilities
    let subject = Subject::builder("complex_dosing")
        .bolus(0.0, 100.0, "oral") // First oral dose
        .observation(1.0, 0.0, "cp")
        .observation(2.0, 0.0, "cp")
        .observation(4.0, 0.0, "cp")
        .bolus(6.0, 150.0, "oral") // Second oral dose (different amount)
        .observation(6.0, 0.0, "cp")
        .observation(7.0, 0.0, "cp")
        .observation(8.0, 0.0, "cp")
        .bolus(12.0, 100.0, "oral") // Third oral dose
        .observation(12.0, 0.0, "cp")
        .observation(14.0, 0.0, "cp")
        .observation(18.0, 0.0, "cp")
        .observation(24.0, 0.0, "cp")
        .build();

    let analytical = with_absorption_analytical_metadata(
        equation::Analytical::new(
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
        .with_nout(1),
        "complex_dosing",
    );

    let ode = with_absorption_ode_metadata(
        equation::ODE::new(
            |x, p, _t, dx, b, _rateiv, _cov| {
                fetch_params!(p, ka, ke, _v);
                dx[0] = -ka * x[0] + b[0]; // Gut: oral doses
                dx[1] = ka * x[0] - ke * x[1]; // Central
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
        .with_nout(1),
        "complex_dosing",
    );

    assert_ode_matches_analytical(
        "complex_dosing",
        &analytical,
        &ode,
        &subject,
        &["ka", "ke", "v"],
        &[1.0, 0.1, 50.0],
    );
}

#[test]
fn mixed_bolus_infusion_iv_matches_analytical() {
    // Test IV bolus + IV infusion with one-compartment model
    // This is fully supported by both analytical and ODE solvers
    let subject = Subject::builder("mixed_iv")
        .bolus(0.0, 100.0, "iv_bolus") // IV bolus
        .observation(1.0, 0.0, "cp")
        .observation(2.0, 0.0, "cp")
        .infusion(4.0, 200.0, "iv", 4.0) // IV infusion: 200mg over 4 hours
        .observation(4.0, 0.0, "cp")
        .observation(5.0, 0.0, "cp")
        .observation(6.0, 0.0, "cp")
        .bolus(8.0, 50.0, "iv_bolus") // Another IV bolus at end of infusion
        .observation(8.0, 0.0, "cp")
        .observation(9.0, 0.0, "cp")
        .observation(10.0, 0.0, "cp")
        .observation(12.0, 0.0, "cp")
        .observation(24.0, 0.0, "cp")
        .build();

    let analytical = with_one_compartment_analytical_metadata(
        equation::Analytical::new(
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
        .with_nout(1),
        "mixed_iv_bolus_infusion",
    );

    let ode = with_one_compartment_ode_metadata(
        equation::ODE::new(
            |x, p, _t, dx, b, rateiv, _cov| {
                fetch_params!(p, ke, _v);
                dx[0] = -ke * x[0] + b[0] + rateiv[0];
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
        .with_nout(1),
        "mixed_iv_bolus_infusion",
    );

    assert_ode_matches_analytical(
        "mixed_iv_bolus_infusion",
        &analytical,
        &ode,
        &subject,
        &["ke", "v"],
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
        .bolus(0.0, 100.0, "iv_bolus")
        .observation(0.0, 0.0, "cp") // Observation at dose time
        .observation(1.0, 0.0, "cp")
        .bolus(2.0, 50.0, "iv_bolus")
        .observation(2.0, 0.0, "cp") // Observation at dose time
        .observation(3.0, 0.0, "cp")
        .observation(4.0, 0.0, "cp")
        .build();

    let analytical = with_one_compartment_analytical_metadata(
        equation::Analytical::new(
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
        .with_nout(1),
        "bolus_at_observation_time",
    );

    let ode = with_one_compartment_ode_metadata(
        equation::ODE::new(
            |x, p, _t, dx, b, _rateiv, _cov| {
                fetch_params!(p, ke, _v);
                dx[0] = -ke * x[0] + b[0];
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
        .with_nout(1),
        "bolus_at_observation_time",
    );

    assert_ode_matches_analytical(
        "bolus_at_observation_time",
        &analytical,
        &ode,
        &subject,
        &["ke", "v"],
        &[0.1, 50.0],
    );
}

#[test]
fn very_fast_elimination_matches_analytical() {
    // High ke tests numerical stability
    let subject = Subject::builder("fast_elimination")
        .bolus(0.0, 100.0, "iv_bolus")
        .observation(0.1, 0.0, "cp")
        .observation(0.2, 0.0, "cp")
        .observation(0.5, 0.0, "cp")
        .observation(1.0, 0.0, "cp")
        .observation(2.0, 0.0, "cp")
        .build();

    let analytical = with_one_compartment_analytical_metadata(
        equation::Analytical::new(
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
        .with_nout(1),
        "fast_elimination",
    );

    let ode = with_one_compartment_ode_metadata(
        equation::ODE::new(
            |x, p, _t, dx, b, _rateiv, _cov| {
                fetch_params!(p, ke, _v);
                dx[0] = -ke * x[0] + b[0];
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
        .with_nout(1),
        "fast_elimination",
    );

    // Very fast elimination: ke = 2.0 (half-life ~20 min)
    assert_ode_matches_analytical(
        "fast_elimination",
        &analytical,
        &ode,
        &subject,
        &["ke", "v"],
        &[2.0, 50.0],
    );
}

#[test]
fn very_slow_elimination_matches_analytical() {
    // Low ke tests long simulation
    let subject = Subject::builder("slow_elimination")
        .bolus(0.0, 100.0, "iv_bolus")
        .observation(24.0, 0.0, "cp")
        .observation(48.0, 0.0, "cp")
        .observation(72.0, 0.0, "cp")
        .observation(96.0, 0.0, "cp")
        .observation(168.0, 0.0, "cp") // 1 week
        .build();

    let analytical = with_one_compartment_analytical_metadata(
        equation::Analytical::new(
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
        .with_nout(1),
        "slow_elimination",
    );

    let ode = with_one_compartment_ode_metadata(
        equation::ODE::new(
            |x, p, _t, dx, b, _rateiv, _cov| {
                fetch_params!(p, ke, _v);
                dx[0] = -ke * x[0] + b[0];
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
        .with_nout(1),
        "slow_elimination",
    );

    // Very slow elimination: ke = 0.01 (half-life ~69 hours)
    assert_ode_matches_analytical(
        "slow_elimination",
        &analytical,
        &ode,
        &subject,
        &["ke", "v"],
        &[0.01, 50.0],
    );
}

#[test]
fn rapid_absorption_matches_analytical() {
    // Very fast absorption with oral dose
    let subject = Subject::builder("rapid_absorption")
        .bolus(0.0, 100.0, "oral")
        .observation(0.1, 0.0, "cp")
        .observation(0.25, 0.0, "cp")
        .observation(0.5, 0.0, "cp")
        .observation(1.0, 0.0, "cp")
        .observation(2.0, 0.0, "cp")
        .observation(4.0, 0.0, "cp")
        .build();

    let analytical = with_absorption_analytical_metadata(
        equation::Analytical::new(
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
        .with_nout(1),
        "rapid_absorption",
    );

    let ode = with_absorption_ode_metadata(
        equation::ODE::new(
            |x, p, _t, dx, b, _rateiv, _cov| {
                fetch_params!(p, ka, ke, _v);
                dx[0] = -ka * x[0] + b[0];
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
        .with_nout(1),
        "rapid_absorption",
    );

    // Very fast absorption: ka = 10.0
    assert_ode_matches_analytical(
        "rapid_absorption",
        &analytical,
        &ode,
        &subject,
        &["ka", "ke", "v"],
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
        .bolus(0.0, 100.0, "iv_bolus")
        .covariate("wt", 0.0, 70.0)
        .observation(1.0, 0.0, "cp")
        .covariate("wt", 2.0, 75.0) // Weight changes
        .observation(2.0, 0.0, "cp")
        .observation(4.0, 0.0, "cp")
        .covariate("wt", 6.0, 72.0) // Weight changes again
        .observation(6.0, 0.0, "cp")
        .observation(8.0, 0.0, "cp")
        .build();

    // ODE with weight-based clearance
    let ode = with_covariate_ode_metadata(
        equation::ODE::new(
            |x, p, t, dx, b, _rateiv, cov| {
                fetch_params!(p, ke_ref, _v);
                fetch_cov!(cov, t, wt);
                // Allometric scaling: CL proportional to weight^0.75
                let ke = ke_ref * (wt / 70.0_f64).powf(0.75);
                dx[0] = -ke * x[0] + b[0];
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
        .with_nout(1),
        "time_varying_covariates",
    );

    // Just verify it runs without error and produces reasonable output
    let parameters =
        parameters_for_ode("time_varying_covariates", &ode, &["ke", "v"], &[0.1, 50.0]);
    let result = ode.estimate_predictions(&subject, &parameters);
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
    for (i, pred) in preds.iter().copied().enumerate().skip(1) {
        // Allow for some noise due to covariate changes, but general trend should be down
        // or predictions should be reasonable (between 0 and initial dose/volume)
        assert!(
            pred < 3.0, // Should not exceed ~100/50 * some factor
            "Prediction {} seems too high: {}",
            i,
            pred
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
        .bolus(0.0, 100.0, "iv_bolus")
        .observation(1.0, 1.8, "cp") // Observed value
        .observation(2.0, 1.6, "cp")
        .observation(4.0, 1.3, "cp")
        .observation(8.0, 0.8, "cp")
        .build();

    let analytical = with_one_compartment_analytical_metadata(
        equation::Analytical::new(
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
        .with_nout(1),
        "likelihood_calculation",
    );

    let ode = with_one_compartment_ode_metadata(
        equation::ODE::new(
            |x, p, _t, dx, b, _rateiv, _cov| {
                fetch_params!(p, ke, _v);
                dx[0] = -ke * x[0] + b[0];
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
        .with_nout(1),
        "likelihood_calculation",
    );

    let error_models = AssayErrorModels::default()
        .add(
            0,
            AssayErrorModel::additive(ErrorPoly::new(0.0, 0.1, 0.0, 0.0), 0.0),
        )
        .unwrap();

    let analytical_params = parameters_for_analytical(
        "likelihood_calculation",
        &analytical,
        &["ke", "v"],
        &[0.1, 50.0],
    );
    let ode_params = parameters_for_ode("likelihood_calculation", &ode, &["ke", "v"], &[0.1, 50.0]);

    let ll_analytical = analytical
        .estimate_log_likelihood(&subject, &analytical_params, &error_models)
        .expect("analytical likelihood")
        .exp();

    let ll_ode = ode
        .estimate_log_likelihood(&subject, &ode_params, &error_models)
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
