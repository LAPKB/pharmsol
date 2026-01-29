//! Tests for NCA parameter calculations
//!
//! Tests all derived parameters via the public API:
//! - Clearance
//! - Volume of distribution
//! - Half-life
//! - Mean residence time
//! - Steady-state parameters
//!
//! Note: These tests use the public NCA API via Subject::builder().nca()

use approx::assert_relative_eq;
use pharmsol::data::Subject;
use pharmsol::nca::{LambdaZOptions, NCAOptions};
use pharmsol::SubjectBuilderExt;

/// Helper to create a subject from time/concentration arrays with a specific dose
fn build_subject_with_dose(times: &[f64], concs: &[f64], dose: f64) -> Subject {
    let mut builder = Subject::builder("test").bolus(0.0, dose, 0);
    for (&t, &c) in times.iter().zip(concs.iter()) {
        builder = builder.observation(t, c, 0);
    }
    builder.build()
}

#[test]
fn test_clearance_calculation() {
    // IV-like profile with known parameters
    let times = vec![0.0, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0];
    let lambda: f64 = 0.1;
    let concs: Vec<f64> = times.iter().map(|&t| 100.0 * (-lambda * t).exp()).collect();
    let dose = 1000.0;

    let subject = build_subject_with_dose(&times, &concs, dose);
    let options = NCAOptions::default();

    let results = subject.nca(&options, 0);
    let result = results
        .first()
        .unwrap()
        .as_ref()
        .expect("NCA should succeed");

    // If we have clearance, verify it's reasonable
    // CL = Dose / AUCinf, for this profile AUCinf should be around 1000
    if let Some(ref clearance) = result.clearance {
        // CL = 1000 / 1000 = 1.0 L/h (approximately)
        assert!(clearance.cl_f > 0.5 && clearance.cl_f < 2.0);
    }
}

#[test]
fn test_volume_distribution() {
    let times = vec![0.0, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0];
    let lambda: f64 = 0.1;
    let concs: Vec<f64> = times.iter().map(|&t| 100.0 * (-lambda * t).exp()).collect();
    let dose = 1000.0;

    let subject = build_subject_with_dose(&times, &concs, dose);
    let options = NCAOptions::default();

    let results = subject.nca(&options, 0);
    let result = results
        .first()
        .unwrap()
        .as_ref()
        .expect("NCA should succeed");

    // Vz = CL / lambda_z
    // If CL ~ 1.0 and lambda ~ 0.1, then Vz ~ 10 L
    if let Some(ref clearance) = result.clearance {
        assert!(clearance.vz_f > 5.0 && clearance.vz_f < 20.0);
    }
}

#[test]
fn test_half_life() {
    let times = vec![0.0, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0];
    let lambda: f64 = 0.0693; // ln(2)/10 = half-life of 10h
    let concs: Vec<f64> = times.iter().map(|&t| 100.0 * (-lambda * t).exp()).collect();

    let subject = build_subject_with_dose(&times, &concs, 100.0);
    let options = NCAOptions::default().with_lambda_z(LambdaZOptions {
        min_r_squared: 0.90,
        min_span_ratio: 1.0,
        ..Default::default()
    });

    let results = subject.nca(&options, 0);
    let result = results
        .first()
        .unwrap()
        .as_ref()
        .expect("NCA should succeed");

    if let Some(ref terminal) = result.terminal {
        // Half-life should be close to 10 hours
        assert_relative_eq!(terminal.half_life, 10.0, epsilon = 1.0);
    }
}

#[test]
fn test_cmax_tmax() {
    // Typical oral PK profile
    let times = vec![0.0, 0.5, 1.0, 2.0, 4.0, 8.0];
    let concs = vec![0.0, 50.0, 80.0, 90.0, 60.0, 30.0];

    let subject = build_subject_with_dose(&times, &concs, 100.0);
    let options = NCAOptions::default();

    let results = subject.nca(&options, 0);
    let result = results
        .first()
        .unwrap()
        .as_ref()
        .expect("NCA should succeed");

    assert_relative_eq!(result.exposure.cmax, 90.0, epsilon = 0.001);
    assert_relative_eq!(result.exposure.tmax, 2.0, epsilon = 0.001);
}

#[test]
fn test_iv_bolus_cmax_at_first_point() {
    // IV bolus - Cmax at t=0
    let times = vec![0.0, 1.0, 2.0, 4.0];
    let concs = vec![100.0, 80.0, 60.0, 40.0];

    let subject = build_subject_with_dose(&times, &concs, 100.0);
    let options = NCAOptions::default();

    let results = subject.nca(&options, 0);
    let result = results
        .first()
        .unwrap()
        .as_ref()
        .expect("NCA should succeed");

    assert_relative_eq!(result.exposure.cmax, 100.0, epsilon = 0.001);
    assert_relative_eq!(result.exposure.tmax, 0.0, epsilon = 0.001);
}

#[test]
fn test_clast_tlast() {
    let times = vec![0.0, 1.0, 2.0, 4.0, 8.0];
    let concs = vec![100.0, 80.0, 60.0, 30.0, 10.0];

    let subject = build_subject_with_dose(&times, &concs, 100.0);
    let options = NCAOptions::default();

    let results = subject.nca(&options, 0);
    let result = results
        .first()
        .unwrap()
        .as_ref()
        .expect("NCA should succeed");

    // Last positive concentration
    assert_relative_eq!(result.exposure.clast, 10.0, epsilon = 0.001);
    assert_relative_eq!(result.exposure.tlast, 8.0, epsilon = 0.001);
}

#[test]
fn test_steady_state_parameters() {
    // Steady-state profile with dosing interval
    let times = vec![0.0, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0];
    let concs = vec![50.0, 80.0, 70.0, 55.0, 48.0, 45.0, 50.0];
    let tau = 12.0;

    let subject = build_subject_with_dose(&times, &concs, 100.0);
    let options = NCAOptions::default().with_tau(tau);

    let results = subject.nca(&options, 0);
    let result = results
        .first()
        .unwrap()
        .as_ref()
        .expect("NCA should succeed");

    if let Some(ref ss) = result.steady_state {
        // Cmin should be around 45-50
        assert!(ss.cmin > 40.0 && ss.cmin < 55.0);
        // Cavg = AUC_tau / tau
        assert!(ss.cavg > 50.0 && ss.cavg < 70.0);
        // Fluctuation should be moderate
        assert!(ss.fluctuation > 0.0);
    }
}

#[test]
fn test_extrapolation_percent() {
    let times = vec![0.0, 1.0, 2.0, 4.0, 8.0, 12.0];
    let concs = vec![100.0, 80.0, 65.0, 45.0, 25.0, 15.0];

    let subject = build_subject_with_dose(&times, &concs, 100.0);
    let options = NCAOptions::default();

    let results = subject.nca(&options, 0);
    let result = results
        .first()
        .unwrap()
        .as_ref()
        .expect("NCA should succeed");

    // Extrapolation percent should be reasonable for good data
    if let Some(extrap_pct) = result.exposure.auc_pct_extrap {
        // For well-sampled data, extrapolation should be under 30%
        assert!(extrap_pct < 50.0, "Extrapolation too high: {}", extrap_pct);
    }
}

#[test]
fn test_complete_parameter_workflow() {
    // Complete workflow: all parameters from raw data
    let times = vec![0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0];
    let concs = vec![100.0, 91.0, 83.0, 70.0, 49.0, 24.0, 12.0, 1.5];
    let dose = 1000.0;

    let subject = build_subject_with_dose(&times, &concs, dose);
    let options = NCAOptions::default();

    let results = subject.nca(&options, 0);
    let result = results
        .first()
        .unwrap()
        .as_ref()
        .expect("NCA should succeed");

    // Verify basic parameters exist
    assert_eq!(result.exposure.cmax, 100.0);
    assert_eq!(result.exposure.tmax, 0.0);
    assert!(result.exposure.auc_last > 400.0 && result.exposure.auc_last < 600.0);

    // If terminal phase estimated
    if let Some(ref terminal) = result.terminal {
        assert!(terminal.lambda_z > 0.05 && terminal.lambda_z < 0.20);
        assert!(terminal.half_life > 3.0 && terminal.half_life < 15.0);
    }

    // If clearance calculated
    if let Some(ref clearance) = result.clearance {
        assert!(clearance.cl_f > 0.0);
        assert!(clearance.vz_f > 0.0);
    }

    println!("Complete parameter set:");
    println!("  Cmax: {:.2}", result.exposure.cmax);
    println!("  Tmax: {:.2}", result.exposure.tmax);
    println!("  AUClast: {:.2}", result.exposure.auc_last);
    if let Some(auc_inf) = result.exposure.auc_inf {
        println!("  AUCinf: {:.2}", auc_inf);
    }
    if let Some(ref terminal) = result.terminal {
        println!("  Lambda_z: {:.4}", terminal.lambda_z);
        println!("  Half-life: {:.2}", terminal.half_life);
    }
}
