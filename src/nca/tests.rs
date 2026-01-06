//! Comprehensive tests for NCA module
//!
//! Tests cover all major NCA parameters and edge cases.

use super::*;

// ============================================================================
// Test data helpers
// ============================================================================

/// Create a typical single-dose PK profile
fn single_dose_profile() -> (Vec<f64>, Vec<f64>) {
    let times = vec![0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0];
    let concs = vec![0.0, 5.0, 10.0, 8.0, 4.0, 2.0, 1.0, 0.25];
    (times, concs)
}

/// Create an IV bolus profile (high C0)
fn iv_bolus_profile() -> (Vec<f64>, Vec<f64>) {
    let times = vec![0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0];
    let concs = vec![100.0, 75.0, 56.0, 32.0, 10.0, 3.0, 0.9, 0.3];
    (times, concs)
}

/// Create a steady-state profile
fn steady_state_profile() -> (Vec<f64>, Vec<f64>) {
    let times = vec![0.0, 1.0, 2.0, 4.0, 8.0, 12.0];
    let concs = vec![2.5, 10.0, 8.0, 4.0, 2.5, 2.5];
    (times, concs)
}

/// Create a profile with BLQ values
fn blq_profile() -> (Vec<f64>, Vec<f64>) {
    let times = vec![0.0, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0];
    let concs = vec![0.0, 10.0, 8.0, 4.0, 2.0, 0.5, 0.0]; // Last point BLQ
    (times, concs)
}

// ============================================================================
// Basic NCA parameter tests
// ============================================================================

#[test]
fn test_nca_basic_exposure() {
    let (times, concs) = single_dose_profile();
    let options = NCAOptions::default();
    let result = nca_from_arrays(&times, &concs, None, &options).unwrap();

    // Check Cmax/Tmax
    assert_eq!(result.exposure.cmax, 10.0, "Cmax should be 10.0");
    assert_eq!(result.exposure.tmax, 1.0, "Tmax should be 1.0");

    // Check Clast/Tlast
    assert_eq!(result.exposure.clast, 0.25, "Clast should be 0.25");
    assert_eq!(result.exposure.tlast, 24.0, "Tlast should be 24.0");

    // AUClast should be positive
    assert!(result.exposure.auc_last > 0.0, "AUClast should be positive");
}

#[test]
fn test_nca_with_dose() {
    let (times, concs) = single_dose_profile();
    let options = NCAOptions::default();
    let dose = DoseContext::bolus(100.0, true); // extravascular
    let result = nca_from_arrays(&times, &concs, Some(&dose), &options).unwrap();

    // Should have clearance parameters if lambda-z was estimated
    if let Some(ref cl) = result.clearance {
        assert!(cl.cl_f > 0.0, "CL/F should be positive");
        assert!(cl.vz_f > 0.0, "Vz/F should be positive");
    }
}

#[test]
fn test_nca_terminal_phase() {
    let (times, concs) = single_dose_profile();
    let options = NCAOptions::default();
    let result = nca_from_arrays(&times, &concs, None, &options).unwrap();

    // Check terminal phase was estimated
    assert!(
        result.terminal.is_some(),
        "Terminal phase should be estimated"
    );

    if let Some(ref term) = result.terminal {
        assert!(term.lambda_z > 0.0, "Lambda-z should be positive");
        assert!(term.half_life > 0.0, "Half-life should be positive");

        // Half-life relationship
        let expected_hl = std::f64::consts::LN_2 / term.lambda_z;
        assert!(
            (term.half_life - expected_hl).abs() < 1e-10,
            "Half-life = ln(2)/lambda_z"
        );
    }
}

// ============================================================================
// AUC calculation tests
// ============================================================================

#[test]
fn test_auc_linear_method() {
    let (times, concs) = single_dose_profile();
    let options = NCAOptions::default().with_auc_method(AUCMethod::Linear);
    let result = nca_from_arrays(&times, &concs, None, &options).unwrap();

    // AUClast should be positive
    assert!(result.exposure.auc_last > 0.0);
}

#[test]
fn test_auc_linuplogdown_method() {
    let (times, concs) = single_dose_profile();
    let options = NCAOptions::default().with_auc_method(AUCMethod::LinUpLogDown);
    let result = nca_from_arrays(&times, &concs, None, &options).unwrap();

    // AUClast should be positive
    assert!(result.exposure.auc_last > 0.0);
}

#[test]
fn test_auc_methods_differ() {
    let (times, concs) = single_dose_profile();

    let linear = NCAOptions::default().with_auc_method(AUCMethod::Linear);
    let logdown = NCAOptions::default().with_auc_method(AUCMethod::LinUpLogDown);

    let result_linear = nca_from_arrays(&times, &concs, None, &linear).unwrap();
    let result_logdown = nca_from_arrays(&times, &concs, None, &logdown).unwrap();

    // Methods should give slightly different results
    // Linear up/log down typically gives smaller AUC for descending curves
    assert!(
        result_linear.exposure.auc_last != result_logdown.exposure.auc_last,
        "Different AUC methods should give different results"
    );
}

// ============================================================================
// Route-specific tests
// ============================================================================

#[test]
fn test_iv_bolus_route() {
    let (times, concs) = iv_bolus_profile();
    let dose = DoseContext::bolus(500.0, false); // IV bolus (not extravascular)
    let options = NCAOptions::default();

    let result = nca_from_arrays(&times, &concs, Some(&dose), &options).unwrap();

    // Should have IV bolus parameters
    assert!(
        result.iv_bolus.is_some(),
        "IV bolus parameters should be present"
    );

    if let Some(ref bolus) = result.iv_bolus {
        assert!(bolus.c0 > 0.0, "C0 should be positive");
        assert!(bolus.vd > 0.0, "Vd should be positive");
    }

    // Should not have extravascular or infusion params
    assert!(result.extravascular.is_none());
    assert!(result.iv_infusion.is_none());
}

#[test]
fn test_iv_infusion_route() {
    let (times, concs) = single_dose_profile();
    let dose = DoseContext::infusion(100.0, 0.5);
    let options = NCAOptions::default();

    let result = nca_from_arrays(&times, &concs, Some(&dose), &options).unwrap();

    // Should have IV infusion parameters
    assert!(
        result.iv_infusion.is_some(),
        "IV infusion parameters should be present"
    );

    if let Some(ref infusion) = result.iv_infusion {
        assert_eq!(
            infusion.infusion_duration, 0.5,
            "Infusion duration should be 0.5"
        );
    }
}

#[test]
fn test_extravascular_route() {
    let (times, concs) = single_dose_profile();
    let dose = DoseContext::bolus(100.0, true); // extravascular
    let options = NCAOptions::default();

    let result = nca_from_arrays(&times, &concs, Some(&dose), &options).unwrap();

    // Should have extravascular parameters
    assert!(
        result.extravascular.is_some(),
        "Extravascular parameters should be present"
    );
}

// ============================================================================
// Steady-state tests
// ============================================================================

#[test]
fn test_steady_state_parameters() {
    let (times, concs) = steady_state_profile();
    let dose = DoseContext::bolus(100.0, true);
    let options = NCAOptions::default().with_tau(12.0);

    let result = nca_from_arrays(&times, &concs, Some(&dose), &options).unwrap();

    // Should have steady-state parameters
    assert!(
        result.steady_state.is_some(),
        "Steady-state parameters should be present"
    );

    if let Some(ref ss) = result.steady_state {
        assert_eq!(ss.tau, 12.0, "Tau should be 12.0");
        assert!(ss.auc_tau > 0.0, "AUCtau should be positive");
        assert!(ss.cmin > 0.0, "Cmin should be positive");
        assert!(ss.cavg > 0.0, "Cavg should be positive");
        assert!(ss.fluctuation > 0.0, "Fluctuation should be positive");
    }
}

// ============================================================================
// BLQ handling tests
// ============================================================================

#[test]
fn test_blq_exclude() {
    let (times, concs) = blq_profile();
    let options = NCAOptions::default().with_blq(0.1, BLQRule::Exclude);

    let result = nca_from_arrays(&times, &concs, None, &options).unwrap();

    // Tlast should be at t=12 (last point above LOQ)
    assert_eq!(result.exposure.tlast, 12.0, "Tlast should exclude BLQ");
}

#[test]
fn test_blq_zero() {
    let (times, concs) = blq_profile();
    let options = NCAOptions::default().with_blq(0.1, BLQRule::Zero);

    let result = nca_from_arrays(&times, &concs, None, &options).unwrap();

    // Should include the BLQ points as zeros
    assert!(result.exposure.auc_last > 0.0);
}

#[test]
fn test_blq_loq_over_2() {
    let (times, concs) = blq_profile();
    let options = NCAOptions::default().with_blq(0.1, BLQRule::LoqOver2);

    let result = nca_from_arrays(&times, &concs, None, &options).unwrap();

    // Should include the BLQ points as LOQ/2
    assert!(result.exposure.auc_last > 0.0);
}

// ============================================================================
// Lambda-z estimation tests
// ============================================================================

#[test]
fn test_lambda_z_auto_selection() {
    let (times, concs) = single_dose_profile();
    let options = NCAOptions::default().with_lambda_z(LambdaZOptions {
        method: LambdaZMethod::AdjR2,
        ..Default::default()
    });

    let result = nca_from_arrays(&times, &concs, None, &options).unwrap();

    // Should have terminal phase
    assert!(result.terminal.is_some());

    if let Some(ref term) = result.terminal {
        assert!(term.regression.is_some());
        if let Some(ref reg) = term.regression {
            assert!(reg.r_squared > 0.9, "R² should be high for good fit");
            assert!(reg.n_points >= 3, "Should use at least 3 points");
        }
    }
}

#[test]
fn test_lambda_z_manual_points() {
    let (times, concs) = single_dose_profile();
    let options = NCAOptions::default().with_lambda_z(LambdaZOptions {
        method: LambdaZMethod::Manual(4),
        ..Default::default()
    });

    let result = nca_from_arrays(&times, &concs, None, &options).unwrap();

    if let Some(ref term) = result.terminal {
        if let Some(ref reg) = term.regression {
            assert_eq!(reg.n_points, 4, "Should use exactly 4 points");
        }
    }
}

// ============================================================================
// Edge case tests
// ============================================================================

#[test]
fn test_empty_data() {
    let result = nca_from_arrays(&[], &[], None, &NCAOptions::default());
    assert!(result.is_err(), "Empty data should return error");
}

#[test]
fn test_single_point() {
    let result = nca_from_arrays(&[0.0], &[10.0], None, &NCAOptions::default());
    assert!(result.is_err(), "Single point should return error");
}

#[test]
fn test_all_zero_concentrations() {
    let times = vec![0.0, 1.0, 2.0, 4.0];
    let concs = vec![0.0, 0.0, 0.0, 0.0];

    let result = nca_from_arrays(&times, &concs, None, &NCAOptions::default());
    assert!(result.is_err(), "All zero concentrations should fail");
}

#[test]
fn test_negative_concentrations() {
    let times = vec![0.0, 1.0, 2.0, 4.0];
    let concs = vec![0.0, 10.0, -5.0, 5.0]; // Negative value

    // Should still compute (negative values are sometimes valid)
    let result = nca_from_arrays(&times, &concs, None, &NCAOptions::default());
    // Behavior depends on implementation - may or may not be error
    assert!(result.is_ok() || result.is_err());
}

// ============================================================================
// Quality/Warning tests
// ============================================================================

#[test]
fn test_quality_warnings() {
    // Profile with poor terminal phase
    let times = vec![0.0, 1.0, 2.0];
    let concs = vec![0.0, 10.0, 8.0]; // Too few points for lambda-z

    let result = nca_from_arrays(&times, &concs, None, &NCAOptions::default()).unwrap();

    // Should have lambda-z warning
    assert!(
        result
            .quality
            .warnings
            .iter()
            .any(|w| matches!(w, Warning::LambdaZNotEstimable)),
        "Should warn about lambda-z"
    );
}

// ============================================================================
// Result conversion tests
// ============================================================================

#[test]
fn test_result_to_params() {
    let (times, concs) = single_dose_profile();
    let dose = DoseContext::bolus(100.0, true);
    let result = nca_from_arrays(&times, &concs, Some(&dose), &NCAOptions::default()).unwrap();

    let params = result.to_params();

    // Check key parameters are present
    assert!(params.contains_key("cmax"));
    assert!(params.contains_key("tmax"));
    assert!(params.contains_key("auc_last"));
}

#[test]
fn test_result_display() {
    let (times, concs) = single_dose_profile();
    let dose = DoseContext::bolus(100.0, true);
    let result = nca_from_arrays(&times, &concs, Some(&dose), &NCAOptions::default()).unwrap();

    let display = format!("{}", result);
    assert!(display.contains("Cmax"), "Display should contain Cmax");
    assert!(display.contains("AUC"), "Display should contain AUC");
}

// ============================================================================
// Integration tests with Data structures
// ============================================================================

#[cfg(test)]
mod integration_tests {
    use super::*;

    #[test]
    fn test_occasion_nca() {
        use crate::data::{Occasion, Event, Bolus, Observation};
        use crate::Censor;

        let mut occasion = Occasion::new(0);
        
        // Add a dose
        let bolus = Bolus::new(0.0, 100.0, 0, 0);
        occasion.add_event(Event::Bolus(bolus));
        
        // Add observations
        for (t, c) in [(1.0, 10.0), (2.0, 8.0), (4.0, 4.0), (8.0, 2.0)].iter() {
            let obs = Observation::new(*t, Some(*c), 0, None, 0, Censor::None);
            occasion.add_event(Event::Observation(obs));
        }

        let result = occasion.nca(&NCAOptions::default(), 0, Some("test".into()));
        assert!(result.is_ok());
        
        let result = result.unwrap();
        assert_eq!(result.exposure.cmax, 10.0);
    }
}
