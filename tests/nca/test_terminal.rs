//! Tests for terminal phase (lambda_z) calculations
//!
//! Tests various methods using the public NCA API:
//! - Adjusted R²
//! - R²
//! - Manual point selection
//!
//! Note: Tests use Subject::builder() with .nca() as the entry point,
//! which internally computes lambda_z via regression on the terminal phase.

use approx::assert_relative_eq;
use pharmsol::data::Subject;
use pharmsol::nca::{LambdaZMethod, LambdaZOptions, NCAOptions, NCA};
use pharmsol::SubjectBuilderExt;

/// Helper to create a subject from time/concentration arrays
fn build_subject(times: &[f64], concs: &[f64]) -> Subject {
    let mut builder = Subject::builder("test").bolus(0.0, 100.0, 0); // Dose at depot
    for (&t, &c) in times.iter().zip(concs.iter()) {
        builder = builder.observation(t, c, 0);
    }
    builder.build()
}

#[test]
fn test_lambda_z_simple_exponential() {
    // Perfect exponential decay: C = 100 * e^(-0.1*t)
    // lambda_z should be exactly 0.1
    let times = vec![0.0, 4.0, 8.0, 12.0, 16.0, 24.0];
    let concs = vec![
        100.0, 67.03, // 100 * e^(-0.1*4)
        44.93, // 100 * e^(-0.1*8)
        30.12, // 100 * e^(-0.1*12)
        20.19, // 100 * e^(-0.1*16)
        9.07,  // 100 * e^(-0.1*24)
    ];

    let subject = build_subject(&times, &concs);
    let options = NCAOptions::default().with_lambda_z(LambdaZOptions {
        min_r_squared: 0.90,
        ..Default::default()
    });

    let results = subject.nca(&options, 0);
    let result = results
        .first()
        .unwrap()
        .as_ref()
        .expect("NCA should succeed");

    // Terminal params should exist
    let terminal = result
        .terminal
        .as_ref()
        .expect("Terminal phase should be estimated");

    // Lambda_z should be very close to 0.1
    assert_relative_eq!(terminal.lambda_z, 0.1, epsilon = 0.01);

    // R² should be high (check regression stats in terminal params)
    if let Some(ref stats) = terminal.regression {
        assert!(stats.r_squared > 0.99);
        assert!(stats.adj_r_squared > 0.99);
    }
}

#[test]
fn test_lambda_z_with_noise() {
    // Exponential decay with some realistic noise
    let times = vec![0.0, 4.0, 6.0, 8.0, 12.0, 24.0];
    let concs = vec![100.0, 65.0, 52.0, 43.0, 29.5, 9.5];

    let subject = build_subject(&times, &concs);
    let options = NCAOptions::default().with_lambda_z(LambdaZOptions {
        min_r_squared: 0.90,
        ..Default::default()
    });

    let results = subject.nca(&options, 0);
    let result = results
        .first()
        .unwrap()
        .as_ref()
        .expect("NCA should succeed");

    let terminal = result
        .terminal
        .as_ref()
        .expect("Terminal phase should be estimated");

    // Lambda should be around 0.09-0.11
    assert!(
        terminal.lambda_z > 0.08 && terminal.lambda_z < 0.12,
        "lambda_z = {} not in expected range",
        terminal.lambda_z
    );

    // R² should still be reasonable
    if let Some(ref stats) = terminal.regression {
        assert!(stats.r_squared > 0.95);
    }
}

#[test]
fn test_lambda_z_manual_points() {
    // Test using manual N points method
    let times = vec![0.0, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0];
    let concs = vec![0.0, 80.0, 100.0, 80.0, 50.0, 30.0, 10.0];

    let subject = build_subject(&times, &concs);

    // Use manual 3 points
    let options = NCAOptions::default().with_lambda_z(LambdaZOptions {
        method: LambdaZMethod::Manual(3),
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

    if let Some(ref terminal) = result.terminal {
        if let Some(ref stats) = terminal.regression {
            // Should use exactly 3 points
            assert_eq!(stats.n_points, 3);
            // Should use terminal points
            assert_eq!(stats.time_last, 24.0);
        }
    }
}

#[test]
fn test_lambda_z_insufficient_points() {
    // Only 2 points - insufficient for terminal phase
    let times = vec![0.0, 2.0];
    let concs = vec![100.0, 50.0];

    let subject = build_subject(&times, &concs);
    let options = NCAOptions::default();

    let results = subject.nca(&options, 0);
    let result = results
        .first()
        .unwrap()
        .as_ref()
        .expect("NCA should succeed");

    // Terminal params should be None due to insufficient data
    assert!(
        result.terminal.is_none(),
        "Terminal phase should not be estimated with only 2 points"
    );
}

#[test]
fn test_adjusted_r2_vs_r2_method() {
    let times = vec![0.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0];
    let concs = vec![100.0, 70.0, 55.0, 45.0, 30.0, 22.0, 10.0];

    let subject = build_subject(&times, &concs);

    // Test with AdjR2 method (default)
    let options_adj = NCAOptions::default().with_lambda_z(LambdaZOptions {
        method: LambdaZMethod::AdjR2,
        min_r_squared: 0.90,
        ..Default::default()
    });

    let results_adj = subject.nca(&options_adj, 0);
    let result_adj = results_adj
        .first()
        .unwrap()
        .as_ref()
        .expect("NCA should succeed");

    if let Some(ref terminal) = result_adj.terminal {
        if let Some(ref stats) = terminal.regression {
            // Adjusted R² should be ≤ R²
            assert!(stats.adj_r_squared <= stats.r_squared);
            // For good fit, they should be close
            assert!((stats.r_squared - stats.adj_r_squared) < 0.05);
        }
    }
}

#[test]
fn test_half_life_from_lambda_z() {
    // Build a subject with known lambda_z ≈ 0.0693 (half-life = 10h)
    let lambda: f64 = 0.0693;
    let times = vec![0.0, 5.0, 10.0, 15.0, 20.0];
    let concs: Vec<f64> = times.iter().map(|&t| 100.0 * (-lambda * t).exp()).collect();

    let subject = build_subject(&times, &concs);
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

    let terminal = result
        .terminal
        .as_ref()
        .expect("Terminal phase should be estimated");

    // Half-life should be close to 10.0 hours
    assert_relative_eq!(terminal.half_life, 10.0, epsilon = 0.5);
}

#[test]
fn test_lambda_z_quality_metrics() {
    let times = vec![0.0, 4.0, 8.0, 12.0, 16.0, 24.0];
    let concs = vec![100.0, 80.0, 60.0, 45.0, 30.0, 12.0];

    let subject = build_subject(&times, &concs);
    let options = NCAOptions::default();

    let results = subject.nca(&options, 0);
    let result = results
        .first()
        .unwrap()
        .as_ref()
        .expect("NCA should succeed");

    // Check quality metrics in terminal.regression
    if let Some(ref terminal) = result.terminal {
        if let Some(ref stats) = terminal.regression {
            assert!(stats.r_squared > 0.95, "R² too low: {}", stats.r_squared);
            assert!(
                stats.adj_r_squared > 0.95,
                "Adjusted R² too low: {}",
                stats.adj_r_squared
            );
            assert!(
                stats.span_ratio > 2.0,
                "Span ratio too small: {}",
                stats.span_ratio
            );
            assert!(stats.n_points >= 3, "Too few points: {}", stats.n_points);
        }
    }
}

#[test]
fn test_auc_inf_extrapolation() {
    // Test that AUCinf is properly calculated
    let times = vec![0.0, 1.0, 2.0, 4.0, 8.0, 12.0];
    let concs = vec![100.0, 90.0, 80.0, 65.0, 40.0, 25.0];

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

    // AUClast should exist
    assert!(result.exposure.auc_last > 0.0);

    // If terminal phase estimated, AUCinf should be > AUClast
    if result.terminal.is_some() {
        if let Some(auc_inf) = result.exposure.auc_inf_obs {
            assert!(
                auc_inf > result.exposure.auc_last,
                "AUCinf should be > AUClast"
            );
        }
    }
}

#[test]
fn test_terminal_phase_with_absorption() {
    // Typical oral PK profile: absorption then elimination
    let times = vec![0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0];
    let concs = vec![0.0, 5.0, 10.0, 8.0, 4.0, 2.0, 1.0, 0.25];

    let subject = build_subject(&times, &concs);
    let options = NCAOptions::default();

    let results = subject.nca(&options, 0);
    let result = results
        .first()
        .unwrap()
        .as_ref()
        .expect("NCA should succeed");

    // Cmax should be at 1.0h
    assert_eq!(result.exposure.cmax, 10.0);
    assert_eq!(result.exposure.tmax, 1.0);

    // Terminal phase should be estimated from post-Tmax points
    if let Some(ref terminal) = result.terminal {
        if let Some(ref stats) = terminal.regression {
            // Should not include Tmax by default
            assert!(stats.time_first > 1.0);
        }
    }
}
