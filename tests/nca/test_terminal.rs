//! Tests for terminal phase (lambda_z) calculations
//!
//! Tests various methods:
//! - Adjusted R²
//! - R²
//! - Interval method
//! - Points method

use approx::assert_relative_eq;
use pharmsol::nca::terminal::*;

#[test]
fn test_lambda_z_simple_exponential() {
    // Perfect exponential decay: C = 100 * e^(-0.1*t)
    // lambda_z should be exactly 0.1
    let times = vec![4.0, 8.0, 12.0, 16.0, 24.0];
    let concs = vec![
        67.03, // 100 * e^(-0.1*4)
        44.93, // 100 * e^(-0.1*8)
        30.12, // 100 * e^(-0.1*12)
        20.19, // 100 * e^(-0.1*16)
        9.07,  // 100 * e^(-0.1*24)
    ];

    let result = calculate_lambda_z_adjusted_r2(&times, &concs, None);

    assert!(result.is_ok());
    let lambda_z = result.unwrap();

    // Should be very close to 0.1
    assert_relative_eq!(lambda_z.lambda, 0.1, epsilon = 0.001);

    // R² should be very close to 1.0
    assert!(lambda_z.r_squared > 0.999);
    assert!(lambda_z.adjusted_r_squared > 0.999);
}

#[test]
fn test_lambda_z_with_noise() {
    // Exponential decay with some realistic noise
    let times = vec![4.0, 6.0, 8.0, 12.0, 24.0];
    let concs = vec![65.0, 52.0, 43.0, 29.5, 9.5];

    let result = calculate_lambda_z_adjusted_r2(&times, &concs, None);

    assert!(result.is_ok());
    let lambda_z = result.unwrap();

    // Lambda should be around 0.09-0.11
    assert!(lambda_z.lambda > 0.08 && lambda_z.lambda < 0.12);

    // R² should still be high
    assert!(lambda_z.r_squared > 0.95);
}

#[test]
fn test_lambda_z_manual_range() {
    let times = vec![0.0, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0];
    let concs = vec![0.0, 80.0, 100.0, 80.0, 50.0, 30.0, 10.0];

    // Manually specify to use only points from 8h onwards
    let range = Some((8.0, 24.0));
    let result = calculate_lambda_z_adjusted_r2(&times, &concs, range);

    assert!(result.is_ok());
    let lambda_z = result.unwrap();

    // Should only use last 3 points
    assert_eq!(lambda_z.n_points, 3);
    assert_eq!(lambda_z.time_first, 8.0);
    assert_eq!(lambda_z.time_last, 24.0);
}

#[test]
fn test_lambda_z_insufficient_points() {
    let times = vec![0.0, 2.0];
    let concs = vec![100.0, 50.0];

    let result = calculate_lambda_z_adjusted_r2(&times, &concs, None);

    // Should fail - need at least 3 points
    assert!(result.is_err());
}

#[test]
fn test_lambda_z_all_same_concentration() {
    let times = vec![4.0, 8.0, 12.0, 16.0];
    let concs = vec![10.0, 10.0, 10.0, 10.0];

    let result = calculate_lambda_z_adjusted_r2(&times, &concs, None);

    // Should fail or return lambda ≈ 0
    // (no elimination)
    if let Ok(lambda_z) = result {
        assert!(lambda_z.lambda < 0.001);
    }
}

#[test]
fn test_lambda_z_increasing_concentrations() {
    let times = vec![4.0, 8.0, 12.0];
    let concs = vec![10.0, 20.0, 30.0];

    let result = calculate_lambda_z_adjusted_r2(&times, &concs, None);

    // Should detect this is not a terminal phase
    // (concentrations increasing)
    assert!(result.is_err() || result.unwrap().lambda < 0.0);
}

#[test]
fn test_adjusted_r2_vs_r2() {
    let times = vec![4.0, 6.0, 8.0, 12.0, 16.0, 24.0];
    let concs = vec![70.0, 55.0, 45.0, 30.0, 22.0, 10.0];

    let result = calculate_lambda_z_adjusted_r2(&times, &concs, None);
    assert!(result.is_ok());
    let lambda_z = result.unwrap();

    // Adjusted R² should be ≤ R²
    assert!(lambda_z.adjusted_r_squared <= lambda_z.r_squared);

    // For good fit, they should be close
    assert!((lambda_z.r_squared - lambda_z.adjusted_r_squared) < 0.05);
}

#[test]
fn test_lambda_z_span_calculation() {
    let times = vec![4.0, 8.0, 12.0, 16.0, 24.0];
    let concs = vec![100.0, 60.0, 40.0, 25.0, 10.0];

    let result = calculate_lambda_z_adjusted_r2(&times, &concs, None);
    assert!(result.is_ok());
    let lambda_z = result.unwrap();

    // Span = (time_last - time_first) * lambda_z
    let expected_span = (24.0 - 4.0) * lambda_z.lambda;
    assert_relative_eq!(lambda_z.span, expected_span, epsilon = 0.001);

    // For a good terminal phase, span should be > 2
    assert!(lambda_z.span > 2.0);
}

#[test]
fn test_lambda_z_extrapolation_percent() {
    let times = vec![0.0, 1.0, 2.0, 4.0, 8.0, 12.0];
    let concs = vec![100.0, 90.0, 80.0, 65.0, 40.0, 25.0];

    // Calculate total AUC
    let auc_last = auc_linear_trapezoidal(&times, &concs);

    // Calculate lambda_z
    let lambda_z_result = calculate_lambda_z_adjusted_r2(&times, &concs, Some((4.0, 12.0)));
    assert!(lambda_z_result.is_ok());
    let lambda_z = lambda_z_result.unwrap().lambda;

    // Extrapolated AUC
    let c_last = concs.last().unwrap();
    let auc_extrap = c_last / lambda_z;

    let auc_total = auc_last + auc_extrap;
    let extrap_percent = (auc_extrap / auc_total) * 100.0;

    // Should be reasonable (< 20% for good data)
    assert!(extrap_percent < 50.0);
}

#[test]
fn test_interval_method() {
    // Multiple possible intervals, algorithm should choose best
    let times = vec![0.0, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 24.0];
    let concs = vec![0.0, 80.0, 100.0, 90.0, 75.0, 60.0, 40.0, 15.0];

    // Try to find best interval automatically
    let result = find_best_lambda_z_interval(&times, &concs);

    assert!(result.is_ok());
    let best = result.unwrap();

    // Should select points from terminal phase (likely 6h onwards)
    assert!(best.time_first >= 4.0);
    assert!(best.r_squared > 0.95);
}

#[test]
fn test_points_method() {
    // Test selecting best N consecutive points
    let times = vec![0.0, 1.0, 2.0, 4.0, 8.0, 12.0, 16.0, 24.0];
    let concs = vec![0.0, 85.0, 100.0, 90.0, 65.0, 45.0, 30.0, 12.0];

    // Try 3, 4, and 5 points
    let result_3 = find_best_lambda_z_n_points(&times, &concs, 3);
    let result_4 = find_best_lambda_z_n_points(&times, &concs, 4);
    let result_5 = find_best_lambda_z_n_points(&times, &concs, 5);

    assert!(result_3.is_ok());
    assert!(result_4.is_ok());
    assert!(result_5.is_ok());

    // All should have good R²
    assert!(result_3.unwrap().r_squared > 0.95);
    assert!(result_4.unwrap().r_squared > 0.95);
}

#[test]
fn test_half_life_calculation() {
    let lambda_z = 0.0693; // ln(2)/10
    let half_life = calculate_half_life(lambda_z);

    // Should be exactly 10.0 hours
    assert_relative_eq!(half_life, 10.0, epsilon = 0.001);
}

#[test]
fn test_lambda_z_quality_metrics() {
    let times = vec![4.0, 8.0, 12.0, 16.0, 24.0];
    let concs = vec![80.0, 60.0, 45.0, 30.0, 12.0];

    let result = calculate_lambda_z_adjusted_r2(&times, &concs, None);
    assert!(result.is_ok());
    let lambda_z = result.unwrap();

    // Check quality metrics
    assert!(lambda_z.r_squared > 0.95, "R² too low");
    assert!(lambda_z.adjusted_r_squared > 0.95, "Adjusted R² too low");
    assert!(lambda_z.span > 2.0, "Span too small");
    assert!(lambda_z.n_points >= 3, "Too few points");
}
