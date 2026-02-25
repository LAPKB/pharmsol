//! Comprehensive tests for NCA module
//!
//! Tests cover all major NCA parameters and edge cases.
//! All tests use Subject::builder() as the single entry point.

use crate::data::Subject;
use crate::nca::*;
use crate::Data;
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
    let results = subject.nca_all(&options);
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
    let results = subject.nca_all(&options);
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
    let results = subject.nca_all(&options);
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
    let results = subject.nca_all(&options);
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
    let results = subject.nca_all(&options);
    let result = results[0].as_ref().unwrap();

    assert!(result.exposure.auc_last > 0.0);
}

#[test]
fn test_auc_linuplogdown_method() {
    let subject = single_dose_oral();
    let options = NCAOptions::default().with_auc_method(AUCMethod::LinUpLogDown);
    let results = subject.nca_all(&options);
    let result = results[0].as_ref().unwrap();

    assert!(result.exposure.auc_last > 0.0);
}

#[test]
fn test_auc_methods_differ() {
    let subject = single_dose_oral();

    let linear = NCAOptions::default().with_auc_method(AUCMethod::Linear);
    let logdown = NCAOptions::default().with_auc_method(AUCMethod::LinUpLogDown);

    let result_linear = subject.nca_all(&linear)[0]
        .as_ref()
        .unwrap()
        .exposure
        .auc_last;
    let result_logdown = subject.nca_all(&logdown)[0]
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
    let results = subject.nca_all(&options);
    let result = results[0].as_ref().unwrap();

    // Should have IV bolus parameters
    assert!(
        matches!(result.route_params, Some(RouteParams::IVBolus(_))),
        "IV bolus parameters should be present"
    );

    if let Some(RouteParams::IVBolus(ref bolus)) = result.route_params {
        assert!(bolus.c0 > 0.0, "C0 should be positive");
        assert!(bolus.vd > 0.0, "Vd should be positive");
    }
}

#[test]
fn test_iv_infusion_route() {
    let subject = iv_infusion_subject();
    let options = NCAOptions::default();
    let results = subject.nca_all(&options);
    let result = results[0].as_ref().unwrap();

    // Should have IV infusion parameters
    assert!(
        matches!(result.route_params, Some(RouteParams::IVInfusion(_))),
        "IV infusion parameters should be present"
    );

    if let Some(RouteParams::IVInfusion(ref infusion)) = result.route_params {
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
    let results = subject.nca_all(&options);
    let result = results[0].as_ref().unwrap();

    // Tlag should be in exposure params (may be None if no lag detected)
    // For extravascular, should have Extravascular route params
    assert!(
        matches!(result.route_params, Some(RouteParams::Extravascular)),
        "Extravascular route should not have IV-specific params"
    );
}

// ============================================================================
// Steady-state tests
// ============================================================================

#[test]
fn test_steady_state_parameters() {
    let subject = steady_state_subject();
    let options = NCAOptions::default().with_tau(12.0);
    let results = subject.nca_all(&options);
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
    let results = subject.nca_all(&options);
    let result = results[0].as_ref().unwrap();

    // Tlast should be at t=12 (last non-BLQ point)
    assert_eq!(result.exposure.tlast, 12.0, "Tlast should exclude BLQ");
}

#[test]
fn test_blq_zero() {
    let subject = blq_subject();
    let options = NCAOptions::default().with_blq_rule(BLQRule::Zero);
    let results = subject.nca_all(&options);
    let result = results[0].as_ref().unwrap();

    // Should include the BLQ points as zeros
    assert!(result.exposure.auc_last > 0.0);
}

#[test]
fn test_blq_loq_over_2() {
    let subject = blq_subject();
    let options = NCAOptions::default().with_blq_rule(BLQRule::LoqOver2);
    let results = subject.nca_all(&options);
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
    let results = subject.nca_all(&options);
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
    let results = subject.nca_all(&options);
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

    let results = subject.nca_all(&NCAOptions::default());
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

    let results = subject.nca_all(&NCAOptions::default());
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

    let results = subject.nca_all(&NCAOptions::default());
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
    let results = subject.nca_all(&NCAOptions::default());
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
    let results = subject.nca_all(&NCAOptions::default());
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

    let results = subject.nca_all(&NCAOptions::default());
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
    let results = subject.nca_all(&options);
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
    let results = subject.nca_all(&options);
    let result = results[0].as_ref().unwrap();

    // Middle BLQ at t=2 should be dropped, but first and last kept as 0 (PKNCA behavior)
    // With last BLQ kept as 0 (not LOQ), tlast remains at 8.0 (last positive conc)
    assert_eq!(result.exposure.cmax, 10.0, "Cmax should be 10.0");
    // tlast is the last time with positive concentration (8.0), the BLQ at 12 is 0
    assert_eq!(
        result.exposure.tlast, 8.0,
        "Tlast should be 8.0 (last positive concentration)"
    );
    assert_eq!(
        result.exposure.clast, 2.0,
        "Clast should be 2.0 (last positive value)"
    );
}

// ============================================================================
// Lambda-z Candidates API tests
// ============================================================================

#[test]
fn test_lambda_z_candidates_returns_multiple() {
    let subject = single_dose_oral();
    let options = NCAOptions::default();
    let results = subject.nca_all(&options);
    let result = results[0].as_ref().unwrap();
    let auc_last = result.exposure.auc_last;

    // Get ObservationProfile for the first occasion
    let occasion = &subject.occasions()[0];
    let profile = ObservationProfile::from_occasion(occasion, 0, &options.blq_rule).unwrap();

    let candidates = lambda_z_candidates(&profile, &options.lambda_z, auc_last);
    assert!(
        candidates.len() >= 2,
        "Should produce multiple candidates, got {}",
        candidates.len()
    );

    // Exactly one should be selected
    let selected_count = candidates.iter().filter(|c| c.is_selected).count();
    assert_eq!(
        selected_count, 1,
        "Exactly one candidate should be selected"
    );
}

#[test]
fn test_lambda_z_candidates_selected_matches_nca_result() {
    let subject = single_dose_oral();
    let options = NCAOptions::default();
    let results = subject.nca_all(&options);
    let result = results[0].as_ref().unwrap();
    let auc_last = result.exposure.auc_last;

    let occasion = &subject.occasions()[0];
    let profile = ObservationProfile::from_occasion(occasion, 0, &options.blq_rule).unwrap();

    let candidates = lambda_z_candidates(&profile, &options.lambda_z, auc_last);
    let selected = candidates.iter().find(|c| c.is_selected).unwrap();

    // Selected candidate's lambda_z should match what NCA computed
    let terminal = result.terminal.as_ref().unwrap();
    let rel_diff = (selected.lambda_z - terminal.lambda_z).abs() / terminal.lambda_z;
    assert!(
        rel_diff < 1e-10,
        "Selected λz ({}) should match NCA result ({})",
        selected.lambda_z,
        terminal.lambda_z
    );

    // Half-life should also match
    let hl_diff = (selected.half_life - terminal.half_life).abs() / terminal.half_life;
    assert!(
        hl_diff < 1e-10,
        "Selected t½ ({}) should match NCA result ({})",
        selected.half_life,
        terminal.half_life
    );
}

#[test]
fn test_lambda_z_candidates_all_have_positive_lambda_z() {
    let subject = single_dose_oral();
    let options = NCAOptions::default();
    let results = subject.nca_all(&options);
    let auc_last = results[0].as_ref().unwrap().exposure.auc_last;

    let occasion = &subject.occasions()[0];
    let profile = ObservationProfile::from_occasion(occasion, 0, &options.blq_rule).unwrap();

    let candidates = lambda_z_candidates(&profile, &options.lambda_z, auc_last);
    for c in &candidates {
        assert!(c.lambda_z > 0.0, "λz must be positive, got {}", c.lambda_z);
        assert!(
            c.half_life > 0.0,
            "t½ must be positive, got {}",
            c.half_life
        );
        assert!(c.n_points >= 3, "Must have at least 3 points");
        assert!(c.r_squared >= 0.0 && c.r_squared <= 1.0, "R² out of range");
    }
}

#[test]
fn test_lambda_z_candidates_empty_for_insufficient_points() {
    // Subject with too few observations for terminal regression
    let subject = Subject::builder("short")
        .bolus(0.0, 100.0, 0)
        .observation(0.0, 0.0, 0)
        .observation(1.0, 10.0, 0)
        .observation(2.0, 5.0, 0)
        .build();

    let options = NCAOptions::default();
    let occasion = &subject.occasions()[0];

    if let Ok(profile) = ObservationProfile::from_occasion(occasion, 0, &options.blq_rule) {
        let candidates = lambda_z_candidates(&profile, &options.lambda_z, 10.0);
        // Either empty or no selected candidate (not enough points after Cmax)
        let selected = candidates.iter().filter(|c| c.is_selected).count();
        assert!(
            candidates.is_empty() || selected == 0,
            "Should have no selected candidate with insufficient terminal points"
        );
    }
}

#[test]
fn test_lambda_z_candidates_span_ratio_and_extrap() {
    let subject = single_dose_oral();
    let options = NCAOptions::default();
    let results = subject.nca_all(&options);
    let auc_last = results[0].as_ref().unwrap().exposure.auc_last;

    let occasion = &subject.occasions()[0];
    let profile = ObservationProfile::from_occasion(occasion, 0, &options.blq_rule).unwrap();

    let candidates = lambda_z_candidates(&profile, &options.lambda_z, auc_last);
    for c in &candidates {
        // span_ratio = (end_time - start_time) / half_life
        let expected_span = (c.end_time - c.start_time) / c.half_life;
        let diff = (c.span_ratio - expected_span).abs();
        assert!(
            diff < 1e-10,
            "Span ratio mismatch: {} vs expected {}",
            c.span_ratio,
            expected_span
        );

        // auc_inf should be >= auc_last
        assert!(
            c.auc_inf >= auc_last,
            "AUC∞ ({}) should be >= AUClast ({})",
            c.auc_inf,
            auc_last
        );

        // extrap pct should be 0..100
        assert!(
            c.auc_pct_extrap >= 0.0 && c.auc_pct_extrap <= 100.0,
            "Extrap % ({}) out of range",
            c.auc_pct_extrap
        );
    }
}

// ============================================================================
// Phase 8: nca() / nca_all() and to_row() tests
// ============================================================================

#[test]
fn test_nca_returns_single_result() {
    let subject = single_dose_oral();
    let options = NCAOptions::default();
    let result = subject.nca(&options);
    assert!(result.is_ok(), "nca() should succeed for a valid subject");
    let r = result.unwrap();
    assert!(r.exposure.cmax > 0.0);
    assert_eq!(r.subject_id.as_deref(), Some("test"));
}

#[test]
fn test_nca_matches_nca_all_vec() {
    let subject = single_dose_oral();
    let options = NCAOptions::default();

    let first = subject.nca(&options).unwrap();
    let vec_result = subject.nca_all(&options);
    let vec_first = vec_result[0].as_ref().unwrap();

    assert!((first.exposure.cmax - vec_first.exposure.cmax).abs() < 1e-10);
    assert!((first.exposure.auc_last - vec_first.exposure.auc_last).abs() < 1e-10);
}

#[test]
fn test_nca_error_on_empty_outeq() {
    // A subject with no observations for outeq=99
    let subject = Subject::builder("empty")
        .bolus(0.0, 100.0, 0)
        .observation(1.0, 10.0, 0)
        .build();
    let options = NCAOptions::default().with_outeq(99);
    let result = subject.nca(&options);
    assert!(result.is_err(), "nca() should fail for missing outeq");
}

#[test]
fn test_to_row_contains_expected_keys() {
    let subject = single_dose_oral();
    let options = NCAOptions::default();
    let result = subject.nca(&options).unwrap();
    let row = result.to_row();

    let keys: Vec<&str> = row.iter().map(|(k, _)| *k).collect();
    assert!(keys.contains(&"cmax"), "to_row should contain cmax");
    assert!(keys.contains(&"tmax"), "to_row should contain tmax");
    assert!(keys.contains(&"auc_last"), "to_row should contain auc_last");
    assert!(keys.contains(&"clast"), "to_row should contain clast");
    assert!(keys.contains(&"tlast"), "to_row should contain tlast");
}

#[test]
fn test_to_row_values_match_result() {
    let subject = single_dose_oral();
    let options = NCAOptions::default();
    let result = subject.nca(&options).unwrap();
    let row = result.to_row();

    let find =
        |key: &str| -> Option<f64> { row.iter().find(|(k, _)| *k == key).and_then(|(_, v)| *v) };

    assert!((find("cmax").unwrap() - result.exposure.cmax).abs() < 1e-10);
    assert!((find("tmax").unwrap() - result.exposure.tmax).abs() < 1e-10);
    assert!((find("auc_last").unwrap() - result.exposure.auc_last).abs() < 1e-10);
}

#[test]
fn test_to_row_terminal_params_present_when_lambda_z_succeeds() {
    let subject = single_dose_oral();
    let options = NCAOptions::default();
    let result = subject.nca(&options).unwrap();

    // Verify terminal phase succeeded
    assert!(
        result.terminal.is_some(),
        "Expected terminal phase to succeed"
    );

    let row = result.to_row();
    let find =
        |key: &str| -> Option<f64> { row.iter().find(|(k, _)| *k == key).and_then(|(_, v)| *v) };

    assert!(
        find("lambda_z").is_some(),
        "to_row should have lambda_z when terminal succeeds"
    );
    assert!(
        find("half_life").is_some(),
        "to_row should have half_life when terminal succeeds"
    );
}

// ============================================================================
// Phase 9: ObservationProfile NCA tests
// ============================================================================

#[test]
fn test_nca_with_dose_matches_subject() {
    use crate::data::observation::ObservationProfile;
    use crate::data::Route;

    let subject = single_dose_oral();
    let options = NCAOptions::default();
    let subject_result = subject.nca(&options).unwrap();

    // Build a profile from the same raw data as single_dose_oral()
    let times = vec![0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0];
    let concs = vec![0.0, 5.0, 10.0, 8.0, 4.0, 2.0, 1.0, 0.25];
    let profile = ObservationProfile::from_raw(&times, &concs).unwrap();
    let profile_result = profile
        .nca_with_dose(Some(100.0), Route::Extravascular, None, &options)
        .unwrap();

    // Cmax and tmax should match exactly (same data, same filtering)
    assert!(
        (subject_result.exposure.cmax - profile_result.exposure.cmax).abs() < 1e-10,
        "Cmax should match"
    );
    assert!(
        (subject_result.exposure.tmax - profile_result.exposure.tmax).abs() < 1e-10,
        "Tmax should match"
    );
    // AUClast should be very close (tlag may differ slightly)
    assert!(
        (subject_result.exposure.auc_last - profile_result.exposure.auc_last).abs()
            / subject_result.exposure.auc_last
            < 0.01,
        "AUClast should be within 1%"
    );
}

#[test]
fn test_nca_with_dose_no_dose() {
    use crate::data::observation::ObservationProfile;
    use crate::data::Route;

    let profile =
        ObservationProfile::from_raw(&[0.0, 1.0, 4.0, 8.0], &[0.0, 10.0, 5.0, 1.0]).unwrap();
    let options = NCAOptions::default();
    let result = profile
        .nca_with_dose(None, Route::Extravascular, None, &options)
        .unwrap();

    // Should work but dose-normalized params should be None
    assert!(result.exposure.cmax > 0.0);
    assert!(result.exposure.cmax_dn.is_none());
}

// ============================================================================
// Phase 10: Population error isolation (Task 4.5)
// ============================================================================

#[test]
fn test_population_error_isolation() {
    // Create a population: one good subject, one with no observations (will fail)
    let good = Subject::builder("good")
        .bolus(0.0, 100.0, 0)
        .observation(1.0, 10.0, 0)
        .observation(2.0, 8.0, 0)
        .observation(4.0, 4.0, 0)
        .observation(8.0, 2.0, 0)
        .build();

    let bad = Subject::builder("bad")
        .bolus(0.0, 100.0, 0)
        // No observations → will fail
        .build();

    let data = Data::new(vec![good, bad]);
    let opts = NCAOptions::default();
    let grouped = data.nca_grouped(&opts);

    assert_eq!(grouped.len(), 2);

    // Good subject
    let good_result = grouped.iter().find(|r| r.subject_id == "good").unwrap();
    assert_eq!(good_result.successes().len(), 1);
    assert_eq!(good_result.errors().len(), 0);

    // Bad subject
    let bad_result = grouped.iter().find(|r| r.subject_id == "bad").unwrap();
    assert_eq!(bad_result.successes().len(), 0);
    assert_eq!(bad_result.errors().len(), 1);

    // nca_all() should have both success and failure
    let all = data.nca_all(&opts);
    let ok_count = all.iter().filter(|r| r.is_ok()).count();
    let err_count = all.iter().filter(|r| r.is_err()).count();
    assert_eq!(ok_count, 1);
    assert_eq!(err_count, 1);
}
