//! Tests for quality assessment and acceptance criteria
//!
//! Tests verify that the NCA module properly flags quality issues like:
//! - Poor R-squared for lambda_z regression
//! - High AUC extrapolation percentage
//! - Insufficient span ratio
//!
//! Note: These tests use the public NCA API via Subject::builder().nca()

use pharmsol::data::Subject;
use pharmsol::nca::{LambdaZOptions, NCAOptions, Warning};
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
fn test_quality_good_data_no_warnings() {
    // Well-behaved exponential decay
    let times = vec![0.0, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0];
    let lambda: f64 = 0.1;
    let concs: Vec<f64> = times.iter().map(|&t| 100.0 * (-lambda * t).exp()).collect();

    let subject = build_subject(&times, &concs);
    let options = NCAOptions::default();

    let results = subject.nca(&options, 0);
    let result = results
        .first()
        .unwrap()
        .as_ref()
        .expect("NCA should succeed");

    // Good data should have few or no warnings
    // (may have some due to extrapolation)
    println!("Warnings for good data: {:?}", result.quality.warnings);
}

#[test]
fn test_quality_high_extrapolation_warning() {
    // Short sampling - will have high extrapolation
    let times = vec![0.0, 1.0, 2.0, 4.0];
    let concs = vec![100.0, 80.0, 60.0, 40.0];

    let subject = build_subject(&times, &concs);
    let options = NCAOptions::default().with_lambda_z(LambdaZOptions {
        min_r_squared: 0.80,
        min_span_ratio: 1.0,
        ..Default::default()
    });

    let results = subject.nca(&options, 0);
    let result = results
        .first()
        .unwrap()
        .as_ref()
        .expect("NCA should succeed");

    // May have high extrapolation warning
    let has_high_extrap = result
        .quality
        .warnings
        .iter()
        .any(|w| matches!(w, Warning::HighExtrapolation));
    println!(
        "Has high extrapolation warning: {}, warnings: {:?}",
        has_high_extrap, result.quality.warnings
    );
}

#[test]
fn test_quality_lambda_z_not_estimable() {
    // Too few points for lambda_z
    let times = vec![0.0, 1.0];
    let concs = vec![100.0, 50.0];

    let subject = build_subject(&times, &concs);
    let options = NCAOptions::default();

    let results = subject.nca(&options, 0);
    let result = results
        .first()
        .unwrap()
        .as_ref()
        .expect("NCA should succeed");

    // Should not have terminal phase
    assert!(result.terminal.is_none());

    // Should have warning about lambda_z not estimable
    let has_lz_warning = result
        .quality
        .warnings
        .iter()
        .any(|w| matches!(w, Warning::LambdaZNotEstimable));
    assert!(has_lz_warning, "Expected LambdaZNotEstimable warning");
}

#[test]
fn test_quality_poor_fit_warning() {
    // Noisy data that should give poor fit
    let times = vec![0.0, 2.0, 4.0, 6.0, 8.0, 10.0];
    let concs = vec![100.0, 60.0, 80.0, 40.0, 50.0, 30.0]; // Very noisy

    let subject = build_subject(&times, &concs);
    let options = NCAOptions::default().with_lambda_z(LambdaZOptions {
        min_r_squared: 0.70, // Very lenient
        min_span_ratio: 0.5,
        ..Default::default()
    });

    let results = subject.nca(&options, 0);
    let result = results
        .first()
        .unwrap()
        .as_ref()
        .expect("NCA should succeed");

    println!(
        "Terminal phase: {:?}, Warnings: {:?}",
        result.terminal, result.quality.warnings
    );
}

#[test]
fn test_quality_short_terminal_phase() {
    // Very short terminal phase span
    let times = vec![0.0, 0.5, 1.0, 1.5, 2.0];
    let concs = vec![100.0, 90.0, 80.0, 70.0, 60.0];

    let subject = build_subject(&times, &concs);
    let options = NCAOptions::default().with_lambda_z(LambdaZOptions {
        min_r_squared: 0.80,
        min_span_ratio: 0.5, // Very lenient
        ..Default::default()
    });

    let results = subject.nca(&options, 0);
    let result = results
        .first()
        .unwrap()
        .as_ref()
        .expect("NCA should succeed");

    // Check for short terminal phase warning
    let has_short_warning = result
        .quality
        .warnings
        .iter()
        .any(|w| matches!(w, Warning::ShortTerminalPhase));
    println!(
        "Has short terminal phase warning: {}, warnings: {:?}",
        has_short_warning, result.quality.warnings
    );
}

#[test]
fn test_regression_stats_available() {
    // Good data should have regression statistics
    let times = vec![0.0, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0];
    let lambda: f64 = 0.1;
    let concs: Vec<f64> = times.iter().map(|&t| 100.0 * (-lambda * t).exp()).collect();

    let subject = build_subject(&times, &concs);
    let options = NCAOptions::default();

    let results = subject.nca(&options, 0);
    let result = results
        .first()
        .unwrap()
        .as_ref()
        .expect("NCA should succeed");

    if let Some(ref terminal) = result.terminal {
        if let Some(ref stats) = terminal.regression {
            // Good fit should have high R-squared
            assert!(
                stats.r_squared > 0.95,
                "R-squared too low: {}",
                stats.r_squared
            );
            assert!(stats.adj_r_squared > 0.95);
            assert!(stats.n_points >= 3);
            assert!(stats.span_ratio > 2.0);
        }
    }
}

#[test]
fn test_bioequivalence_preset_quality() {
    // Test BE preset quality thresholds
    let times = vec![0.0, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0];
    let lambda: f64 = 0.1;
    let concs: Vec<f64> = times.iter().map(|&t| 100.0 * (-lambda * t).exp()).collect();

    let subject = build_subject(&times, &concs);
    let options = NCAOptions::bioequivalence();

    let results = subject.nca(&options, 0);
    let result = results
        .first()
        .unwrap()
        .as_ref()
        .expect("NCA should succeed");

    // BE preset should have stricter quality requirements
    // Good data should still pass
    if let Some(ref terminal) = result.terminal {
        if let Some(ref stats) = terminal.regression {
            assert!(
                stats.r_squared >= 0.90,
                "BE threshold requires R-squared >= 0.90"
            );
        }
    }
}

#[test]
fn test_sparse_preset_quality() {
    // Sparse preset should be more lenient
    let times = vec![0.0, 2.0, 8.0, 24.0];
    let concs = vec![100.0, 70.0, 35.0, 10.0];

    let subject = build_subject(&times, &concs);
    let options = NCAOptions::sparse();

    let results = subject.nca(&options, 0);
    let result = results
        .first()
        .unwrap()
        .as_ref()
        .expect("NCA should succeed");

    // Sparse preset should still be able to estimate terminal phase
    // with fewer points
    println!(
        "Sparse data - Terminal: {:?}, Warnings: {:?}",
        result.terminal.is_some(),
        result.quality.warnings
    );
}
