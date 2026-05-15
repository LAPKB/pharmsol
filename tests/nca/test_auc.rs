//! Comprehensive tests for AUC calculation algorithms
//!
//! Tests cover:
//! - Linear trapezoidal rule
//! - Linear up / log down
//! - Edge cases (zeros, single points, etc.)
//! - Partial AUC intervals
//!
//! Note: These tests use the public NCA API via Subject::builder().nca()

use approx::assert_relative_eq;
use pharmsol::data::Subject;
use pharmsol::nca::{AUCMethod, NCAOptions, NCA};
use pharmsol::SubjectBuilderExt;

/// Helper to create a subject from time/concentration arrays
fn build_subject(times: &[f64], concs: &[f64]) -> Subject {
    let mut builder = Subject::builder("test").bolus(0.0, 100.0, 0);
    for (&t, &c) in times.iter().zip(concs.iter()) {
        builder = builder.observation(t, c, 0);
    }
    builder.build()
}

#[test]
fn test_linear_trapezoidal_simple_decreasing() {
    let times = vec![0.0, 1.0, 2.0, 4.0, 8.0];
    let concs = vec![10.0, 8.0, 6.0, 4.0, 2.0];

    let subject = build_subject(&times, &concs);
    let options = NCAOptions::default().with_auc_method(AUCMethod::Linear);

    let result = subject.nca(&options).expect("NCA should succeed");

    // Manual calculation: (10+8)/2*1 + (8+6)/2*1 + (6+4)/2*2 + (4+2)/2*4 = 38.0
    assert_relative_eq!(result.exposure.auc_last, 38.0, epsilon = 1e-6);
}

#[test]
fn test_linear_trapezoidal_exponential_decay() {
    let times = vec![0.0, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0];
    let concs = vec![100.0, 90.48, 81.87, 67.03, 44.93, 30.12, 9.07];

    let subject = build_subject(&times, &concs);
    let options = NCAOptions::default().with_auc_method(AUCMethod::Linear);

    let result = subject.nca(&options).expect("NCA should succeed");

    // For exponential decay with lambda = 0.1, true AUC to 24h is around 909
    assert!(
        result.exposure.auc_last > 900.0 && result.exposure.auc_last < 950.0,
        "AUClast = {} not in expected range",
        result.exposure.auc_last
    );
}

#[test]
fn test_linear_up_log_down() {
    let times = vec![0.0, 0.5, 1.0, 2.0, 4.0, 8.0];
    let concs = vec![0.0, 5.0, 8.0, 6.0, 3.0, 1.0];

    let subject = build_subject(&times, &concs);
    let options = NCAOptions::default().with_auc_method(AUCMethod::LinUpLogDown);

    let result = subject.nca(&options).expect("NCA should succeed");

    assert!(result.exposure.auc_last > 0.0);
    assert!(result.exposure.auc_last < 50.0);
}

#[test]
fn test_auc_with_zero_concentration() {
    let times = vec![0.0, 1.0, 2.0, 3.0, 4.0];
    let concs = vec![10.0, 5.0, 0.0, 0.0, 0.0];

    let subject = build_subject(&times, &concs);
    let options = NCAOptions::default().with_auc_method(AUCMethod::Linear);

    let result = subject.nca(&options).expect("NCA should succeed");

    // NCA calculates AUC to Tlast (last positive concentration)
    // Tlast = 1.0 (concentration 5.0), so AUC is only segment 1: (10+5)/2*1 = 7.5
    assert_relative_eq!(result.exposure.auc_last, 7.5, epsilon = 1e-6);
    assert!(result.exposure.auc_last.is_finite());
}

#[test]
fn test_auc_two_points() {
    let times = vec![0.0, 4.0];
    let concs = vec![10.0, 6.0];

    let subject = build_subject(&times, &concs);
    let options = NCAOptions::default().with_auc_method(AUCMethod::Linear);

    let result = subject.nca(&options).expect("NCA should succeed");

    // (10+6)/2 * 4 = 32.0
    assert_relative_eq!(result.exposure.auc_last, 32.0, epsilon = 1e-6);
}

#[test]
fn test_auc_plateau() {
    let times = vec![0.0, 1.0, 2.0, 3.0, 4.0];
    let concs = vec![5.0, 5.0, 5.0, 5.0, 5.0];

    let subject = build_subject(&times, &concs);
    let options = NCAOptions::default().with_auc_method(AUCMethod::Linear);

    let result = subject.nca(&options).expect("NCA should succeed");

    // 5.0 * 4.0 = 20.0
    assert_relative_eq!(result.exposure.auc_last, 20.0, epsilon = 1e-6);
}

#[test]
fn test_auc_unequal_spacing() {
    let times = vec![0.0, 0.25, 1.0, 2.5, 8.0];
    let concs = vec![100.0, 95.0, 80.0, 55.0, 20.0];

    let subject = build_subject(&times, &concs);
    let options = NCAOptions::default().with_auc_method(AUCMethod::Linear);

    let result = subject.nca(&options).expect("NCA should succeed");

    // Total: 397.5
    assert_relative_eq!(result.exposure.auc_last, 397.5, epsilon = 1e-6);
}

#[test]
fn test_auc_methods_comparison() {
    let times = vec![0.0, 1.0, 2.0, 4.0, 8.0, 12.0];
    let concs = vec![100.0, 86.07, 74.08, 54.88, 30.12, 16.53];

    let subject = build_subject(&times, &concs);

    let options_linear = NCAOptions::default().with_auc_method(AUCMethod::Linear);
    let options_linlog = NCAOptions::default().with_auc_method(AUCMethod::LinUpLogDown);

    let result_linear = subject.nca(&options_linear).unwrap();
    let result_linlog = subject.nca(&options_linlog).unwrap();

    let auc_linear = result_linear.exposure.auc_last;
    let auc_linlog = result_linlog.exposure.auc_last;

    // Both should be reasonably close (within 5%)
    let true_auc = 555.6;
    assert!((auc_linear - true_auc).abs() / true_auc < 0.05);
    assert!((auc_linlog - true_auc).abs() / true_auc < 0.05);
}

#[test]
fn test_partial_auc() {
    let times = vec![0.0, 1.0, 2.0, 4.0, 8.0, 12.0];
    let concs = vec![100.0, 90.0, 80.0, 60.0, 35.0, 20.0];

    let subject = build_subject(&times, &concs);
    let options = NCAOptions::default()
        .with_auc_method(AUCMethod::Linear)
        .with_auc_interval(2.0, 8.0);

    let result = subject.nca(&options).expect("NCA should succeed");

    if let Some(auc_partial) = result.exposure.auc_partial {
        // (80+60)/2*2 + (60+35)/2*4 = 330
        assert_relative_eq!(auc_partial, 330.0, epsilon = 1.0);
    }
}

#[test]
fn test_auc_inf_calculation() {
    let times = vec![0.0, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0];
    let lambda: f64 = 0.1;
    let concs: Vec<f64> = times.iter().map(|&t| 100.0 * (-lambda * t).exp()).collect();

    let subject = build_subject(&times, &concs);
    let options = NCAOptions::default();

    let result = subject.nca(&options).expect("NCA should succeed");

    if let Some(auc_inf) = result.exposure.auc_inf_obs {
        assert!(auc_inf > result.exposure.auc_last);
        // True AUCinf = C0/lambda = 100/0.1 = 1000
        assert_relative_eq!(auc_inf, 1000.0, epsilon = 50.0);
    }
}
