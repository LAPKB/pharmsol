//! Comprehensive tests for NCA module
//!
//! Tests cover all major NCA parameters and edge cases.
//! All tests use Subject::builder() as the single entry point.

use crate::data::Subject;
use crate::nca::*;
use crate::SubjectBuilderExt;

// ============================================================================
// Test subject builders
// ============================================================================

/// Create a typical single-dose oral PK subject
fn single_dose_oral() -> Subject {
    Subject::builder("test")
        .bolus(0.0, 100.0, 0) // 100 mg to depot (extravascular)
        .observation(0.0, 0.0, 0)
        .observation(0.5, 5.0, 0)
        .observation(1.0, 10.0, 0)
        .observation(2.0, 8.0, 0)
        .observation(4.0, 4.0, 0)
        .observation(8.0, 2.0, 0)
        .observation(12.0, 1.0, 0)
        .observation(24.0, 0.25, 0)
        .build()
}

/// Create an IV bolus subject (high C0, dose to central)
fn iv_bolus_subject() -> Subject {
    Subject::builder("test")
        .bolus(0.0, 500.0, 1) // 500 mg to central (IV)
        .observation(0.0, 100.0, 0)
        .observation(0.25, 75.0, 0)
        .observation(0.5, 56.0, 0)
        .observation(1.0, 32.0, 0)
        .observation(2.0, 10.0, 0)
        .observation(4.0, 3.0, 0)
        .observation(8.0, 0.9, 0)
        .observation(12.0, 0.3, 0)
        .build()
}

/// Create an IV infusion subject
fn iv_infusion_subject() -> Subject {
    Subject::builder("test")
        .infusion(0.0, 100.0, 1, 0.5) // 100 mg over 0.5h to central
        .observation(0.0, 0.0, 0)
        .observation(0.5, 5.0, 0)
        .observation(1.0, 10.0, 0)
        .observation(2.0, 8.0, 0)
        .observation(4.0, 4.0, 0)
        .observation(8.0, 2.0, 0)
        .observation(12.0, 1.0, 0)
        .observation(24.0, 0.25, 0)
        .build()
}

/// Create a steady-state profile subject
fn steady_state_subject() -> Subject {
    Subject::builder("test")
        .bolus(0.0, 100.0, 0) // 100 mg oral
        .observation(0.0, 5.0, 0)
        .observation(1.0, 15.0, 0)
        .observation(2.0, 12.0, 0)
        .observation(4.0, 8.0, 0)
        .observation(6.0, 6.0, 0)
        .observation(8.0, 5.5, 0)
        .observation(12.0, 5.0, 0)
        .build()
}

/// Create a subject with BLQ values
fn blq_subject() -> Subject {
    use crate::Censor;

    Subject::builder("test")
        .bolus(0.0, 100.0, 0)
        .observation(0.0, 0.0, 0)
        .observation(1.0, 10.0, 0)
        .observation(2.0, 8.0, 0)
        .observation(4.0, 4.0, 0)
        .observation(8.0, 2.0, 0)
        .observation(12.0, 0.5, 0)
        .censored_observation(24.0, 0.1, 0, Censor::BLOQ) // BLQ with LOQ=0.1
        .build()
}

/// Create a minimal subject (no dose)
fn no_dose_subject() -> Subject {
    Subject::builder("test")
        .observation(0.0, 0.0, 0)
        .observation(1.0, 10.0, 0)
        .observation(2.0, 8.0, 0)
        .observation(4.0, 4.0, 0)
        .build()
}

// ============================================================================
// Basic NCA parameter tests
// ============================================================================

#[test]
fn test_nca_basic_exposure() {
    let subject = single_dose_oral();
    let options = NCAOptions::default();
    let results = subject.nca(&options, 0);
    let result = results[0].as_ref().unwrap();

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
    let subject = single_dose_oral();
    let options = NCAOptions::default();
    let results = subject.nca(&options, 0);
    let result = results[0].as_ref().unwrap();

    // Should have clearance parameters if lambda-z was estimated
    if let Some(ref cl) = result.clearance {
        assert!(cl.cl_f > 0.0, "CL/F should be positive");
        assert!(cl.vz_f > 0.0, "Vz/F should be positive");
    }
}

#[test]
fn test_nca_without_dose() {
    let subject = no_dose_subject();
    let options = NCAOptions::default();
    let results = subject.nca(&options, 0);
    let result = results[0].as_ref().unwrap();

    // Exposure should still be computed
    assert!(result.exposure.cmax > 0.0);
    // But clearance should be None (no dose)
    assert!(result.clearance.is_none());
}

#[test]
fn test_nca_terminal_phase() {
    let subject = single_dose_oral();
    let options = NCAOptions::default();
    let results = subject.nca(&options, 0);
    let result = results[0].as_ref().unwrap();

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
    let subject = single_dose_oral();
    let options = NCAOptions::default().with_auc_method(AUCMethod::Linear);
    let results = subject.nca(&options, 0);
    let result = results[0].as_ref().unwrap();

    assert!(result.exposure.auc_last > 0.0);
}

#[test]
fn test_auc_linuplogdown_method() {
    let subject = single_dose_oral();
    let options = NCAOptions::default().with_auc_method(AUCMethod::LinUpLogDown);
    let results = subject.nca(&options, 0);
    let result = results[0].as_ref().unwrap();

    assert!(result.exposure.auc_last > 0.0);
}

#[test]
fn test_auc_methods_differ() {
    let subject = single_dose_oral();

    let linear = NCAOptions::default().with_auc_method(AUCMethod::Linear);
    let logdown = NCAOptions::default().with_auc_method(AUCMethod::LinUpLogDown);

    let result_linear = subject.nca(&linear, 0)[0]
        .as_ref()
        .unwrap()
        .exposure
        .auc_last;
    let result_logdown = subject.nca(&logdown, 0)[0]
        .as_ref()
        .unwrap()
        .exposure
        .auc_last;

    // Methods should give slightly different results
    assert!(
        result_linear != result_logdown,
        "Different AUC methods should give different results"
    );
}

// ============================================================================
// Route-specific tests
// ============================================================================

#[test]
fn test_iv_bolus_route() {
    let subject = iv_bolus_subject();
    let options = NCAOptions::default();
    let results = subject.nca(&options, 0);
    let result = results[0].as_ref().unwrap();

    // Should have IV bolus parameters
    assert!(
        result.iv_bolus.is_some(),
        "IV bolus parameters should be present"
    );

    if let Some(ref bolus) = result.iv_bolus {
        assert!(bolus.c0 > 0.0, "C0 should be positive");
        assert!(bolus.vd > 0.0, "Vd should be positive");
    }

    // Should not have infusion params
    assert!(result.iv_infusion.is_none());
}

#[test]
fn test_iv_infusion_route() {
    let subject = iv_infusion_subject();
    let options = NCAOptions::default();
    let results = subject.nca(&options, 0);
    let result = results[0].as_ref().unwrap();

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
    let subject = single_dose_oral();
    let options = NCAOptions::default();
    let results = subject.nca(&options, 0);
    let result = results[0].as_ref().unwrap();

    // Tlag should be in exposure params (may be None if no lag detected)
    // For extravascular, should not have IV-specific params
    assert!(result.iv_bolus.is_none());
    assert!(result.iv_infusion.is_none());
}

// ============================================================================
// Steady-state tests
// ============================================================================

#[test]
fn test_steady_state_parameters() {
    let subject = steady_state_subject();
    let options = NCAOptions::default().with_tau(12.0);
    let results = subject.nca(&options, 0);
    let result = results[0].as_ref().unwrap();

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
    let subject = blq_subject();
    let options = NCAOptions::default().with_blq_rule(BLQRule::Exclude);
    let results = subject.nca(&options, 0);
    let result = results[0].as_ref().unwrap();

    // Tlast should be at t=12 (last non-BLQ point)
    assert_eq!(result.exposure.tlast, 12.0, "Tlast should exclude BLQ");
}

#[test]
fn test_blq_zero() {
    let subject = blq_subject();
    let options = NCAOptions::default().with_blq_rule(BLQRule::Zero);
    let results = subject.nca(&options, 0);
    let result = results[0].as_ref().unwrap();

    // Should include the BLQ points as zeros
    assert!(result.exposure.auc_last > 0.0);
}

#[test]
fn test_blq_loq_over_2() {
    let subject = blq_subject();
    let options = NCAOptions::default().with_blq_rule(BLQRule::LoqOver2);
    let results = subject.nca(&options, 0);
    let result = results[0].as_ref().unwrap();

    // Should include the BLQ points as LOQ/2 (0.1 / 2 = 0.05)
    assert!(result.exposure.auc_last > 0.0);
}

// ============================================================================
// Lambda-z estimation tests
// ============================================================================

#[test]
fn test_lambda_z_auto_selection() {
    let subject = single_dose_oral();
    let options = NCAOptions::default().with_lambda_z(LambdaZOptions {
        method: LambdaZMethod::AdjR2,
        ..Default::default()
    });
    let results = subject.nca(&options, 0);
    let result = results[0].as_ref().unwrap();

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
    let subject = single_dose_oral();
    let options = NCAOptions::default().with_lambda_z(LambdaZOptions {
        method: LambdaZMethod::Manual(4),
        ..Default::default()
    });
    let results = subject.nca(&options, 0);
    let result = results[0].as_ref().unwrap();

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
fn test_insufficient_observations() {
    let subject = Subject::builder("test")
        .bolus(0.0, 100.0, 0)
        .observation(1.0, 10.0, 0)
        .build();

    let results = subject.nca(&NCAOptions::default(), 0);
    // Should fail with insufficient data
    assert!(
        results[0].is_err(),
        "Single observation should return error"
    );
}

#[test]
fn test_all_zero_concentrations() {
    let subject = Subject::builder("test")
        .bolus(0.0, 100.0, 0)
        .observation(0.0, 0.0, 0)
        .observation(1.0, 0.0, 0)
        .observation(2.0, 0.0, 0)
        .observation(4.0, 0.0, 0)
        .build();

    let results = subject.nca(&NCAOptions::default(), 0);
    assert!(results[0].is_err(), "All zero concentrations should fail");
}

// ============================================================================
// Quality/Warning tests
// ============================================================================

#[test]
fn test_quality_warnings_lambda_z() {
    // Profile with too few points for lambda-z
    let subject = Subject::builder("test")
        .bolus(0.0, 100.0, 0)
        .observation(0.0, 0.0, 0)
        .observation(1.0, 10.0, 0)
        .observation(2.0, 8.0, 0)
        .build();

    let results = subject.nca(&NCAOptions::default(), 0);
    let result = results[0].as_ref().unwrap();

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
    let subject = single_dose_oral();
    let results = subject.nca(&NCAOptions::default(), 0);
    let result = results[0].as_ref().unwrap();

    let params = result.to_params();

    // Check key parameters are present
    assert!(params.contains_key("cmax"));
    assert!(params.contains_key("tmax"));
    assert!(params.contains_key("auc_last"));
}

#[test]
fn test_result_display() {
    let subject = single_dose_oral();
    let results = subject.nca(&NCAOptions::default(), 0);
    let result = results[0].as_ref().unwrap();

    let display = format!("{}", result);
    assert!(display.contains("Cmax"), "Display should contain Cmax");
    assert!(display.contains("AUC"), "Display should contain AUC");
}

// ============================================================================
// Subject/Occasion identification tests
// ============================================================================

#[test]
fn test_result_subject_id() {
    let subject = Subject::builder("patient_001")
        .bolus(0.0, 100.0, 0)
        .observation(1.0, 10.0, 0)
        .observation(2.0, 8.0, 0)
        .observation(4.0, 4.0, 0)
        .observation(8.0, 2.0, 0)
        .build();

    let results = subject.nca(&NCAOptions::default(), 0);
    let result = results[0].as_ref().unwrap();

    assert_eq!(result.subject_id.as_deref(), Some("patient_001"));
    assert_eq!(result.occasion, Some(0));
}

// ============================================================================
// Presets tests
// ============================================================================

#[test]
fn test_bioequivalence_preset() {
    let options = NCAOptions::bioequivalence();
    assert_eq!(options.lambda_z.min_r_squared, 0.90);
    assert_eq!(options.max_auc_extrap_pct, 20.0);
}

#[test]
fn test_sparse_preset() {
    let options = NCAOptions::sparse();
    assert_eq!(options.lambda_z.min_r_squared, 0.80);
    assert_eq!(options.max_auc_extrap_pct, 30.0);
}

// ============================================================================
// Partial AUC tests
// ============================================================================

#[test]
fn test_partial_auc_interval() {
    let subject = single_dose_oral();
    let options = NCAOptions::default().with_auc_interval(0.0, 4.0);
    let results = subject.nca(&options, 0);
    let result = results[0].as_ref().unwrap();

    // Partial AUC should be calculated
    assert!(
        result.exposure.auc_partial.is_some(),
        "Partial AUC should be computed when interval specified"
    );

    let auc_partial = result.exposure.auc_partial.unwrap();
    assert!(auc_partial > 0.0, "Partial AUC should be positive");

    // Partial AUC (0-4h) should be less than AUClast (0-24h)
    assert!(
        auc_partial < result.exposure.auc_last,
        "Partial AUC should be less than AUClast"
    );
}

#[test]
fn test_positional_blq_rule() {
    use crate::Censor;

    // Create subject with BLQ at start, middle, and end
    let subject = Subject::builder("test")
        .bolus(0.0, 100.0, 0)
        .censored_observation(0.0, 0.1, 0, Censor::BLOQ) // First - keep as 0
        .observation(1.0, 10.0, 0)
        .censored_observation(2.0, 0.1, 0, Censor::BLOQ) // Middle - drop
        .observation(4.0, 4.0, 0)
        .observation(8.0, 2.0, 0)
        .censored_observation(12.0, 0.1, 0, Censor::BLOQ) // Last - keep as 0
        .build();

    // With positional BLQ handling
    let options = NCAOptions::default().with_blq_rule(BLQRule::Positional);
    let results = subject.nca(&options, 0);
    let result = results[0].as_ref().unwrap();

    // Middle BLQ at t=2 should be dropped, but first and last kept as 0 (PKNCA behavior)
    // With last BLQ kept as 0 (not LOQ), tlast remains at 8.0 (last positive conc)
    assert_eq!(result.exposure.cmax, 10.0, "Cmax should be 10.0");
    // tlast is the last time with positive concentration (8.0), the BLQ at 12 is 0
    assert_eq!(result.exposure.tlast, 8.0, "Tlast should be 8.0 (last positive concentration)");
    assert_eq!(result.exposure.clast, 2.0, "Clast should be 2.0 (last positive value)");
}

