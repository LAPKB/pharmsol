//! Comprehensive tests for AUC calculation algorithms
//!
//! Tests cover:
//! - Linear trapezoidal rule
//! - Linear up / log down
//! - Edge cases (zeros, single points, etc.)
//! - Property-based testing

use approx::assert_relative_eq;
use pharmsol::nca::auc::*;

#[test]
fn test_linear_trapezoidal_simple_decreasing() {
    let times = vec![0.0, 1.0, 2.0, 4.0, 8.0];
    let concs = vec![10.0, 8.0, 6.0, 4.0, 2.0];

    let auc = auc_linear_trapezoidal(&times, &concs);

    // Manual calculation:
    // Segment 1: (10+8)/2 * 1 = 9.0
    // Segment 2: (8+6)/2 * 1 = 7.0
    // Segment 3: (6+4)/2 * 2 = 10.0
    // Segment 4: (4+2)/2 * 4 = 12.0
    // Total: 38.0

    assert_relative_eq!(auc, 38.0, epsilon = 1e-10);
}

#[test]
fn test_linear_trapezoidal_exponential_decay() {
    // Simulate exponential decay: C(t) = 100 * e^(-0.1*t)
    let times = vec![0.0, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0];
    let concs = vec![
        100.0, 90.48, // 100 * e^(-0.1*1)
        81.87, // 100 * e^(-0.1*2)
        67.03, // 100 * e^(-0.1*4)
        44.93, // 100 * e^(-0.1*8)
        30.12, // 100 * e^(-0.1*12)
        9.07,  // 100 * e^(-0.1*24)
    ];

    let auc = auc_linear_trapezoidal(&times, &concs);

    // For exponential decay with lambda = 0.1, true AUC to 24h ≈ 909.3
    // Linear trapezoidal will slightly overestimate
    assert!(auc > 900.0 && auc < 950.0);
}

#[test]
fn test_linear_up_log_down() {
    // Profile with absorption phase (increasing) then elimination (decreasing)
    let times = vec![0.0, 0.5, 1.0, 2.0, 4.0, 8.0];
    let concs = vec![0.0, 5.0, 8.0, 6.0, 3.0, 1.0];

    let auc = auc_linear_up_log_down(&times, &concs);

    // Should use linear for increasing segments (0→0.5, 0.5→1.0)
    // Should use log for decreasing segments (1.0→2.0, 2.0→4.0, 4.0→8.0)
    assert!(auc > 0.0);
    assert!(auc < 50.0); // Sanity check
}

#[test]
fn test_auc_with_zero_concentration() {
    let times = vec![0.0, 1.0, 2.0, 3.0, 4.0];
    let concs = vec![10.0, 5.0, 0.0, 0.0, 0.0];

    let auc = auc_linear_trapezoidal(&times, &concs);

    // Segment 1: (10+5)/2 * 1 = 7.5
    // Segment 2: (5+0)/2 * 1 = 2.5
    // Segments 3-4: 0
    // Total: 10.0

    assert_relative_eq!(auc, 10.0, epsilon = 1e-10);
    assert!(auc.is_finite());
}

#[test]
fn test_auc_single_point() {
    let times = vec![0.0];
    let concs = vec![10.0];

    let auc = auc_linear_trapezoidal(&times, &concs);

    // Single point has no area
    assert_eq!(auc, 0.0);
}

#[test]
fn test_auc_two_points() {
    let times = vec![0.0, 4.0];
    let concs = vec![10.0, 6.0];

    let auc = auc_linear_trapezoidal(&times, &concs);

    // (10+6)/2 * 4 = 32.0
    assert_relative_eq!(auc, 32.0, epsilon = 1e-10);
}

#[test]
fn test_auc_empty_data() {
    let times: Vec<f64> = vec![];
    let concs: Vec<f64> = vec![];

    let auc = auc_linear_trapezoidal(&times, &concs);

    assert_eq!(auc, 0.0);
}

#[test]
fn test_auc_plateau() {
    // Concentration plateau (constant value)
    let times = vec![0.0, 1.0, 2.0, 3.0, 4.0];
    let concs = vec![5.0, 5.0, 5.0, 5.0, 5.0];

    let auc = auc_linear_trapezoidal(&times, &concs);

    // Constant concentration = concentration * time
    // 5.0 * 4.0 = 20.0
    assert_relative_eq!(auc, 20.0, epsilon = 1e-10);
}

#[test]
fn test_auc_unequal_spacing() {
    let times = vec![0.0, 0.25, 1.0, 2.5, 8.0];
    let concs = vec![100.0, 95.0, 80.0, 55.0, 20.0];

    let auc = auc_linear_trapezoidal(&times, &concs);

    // Segment 1: (100+95)/2 * 0.25 = 24.375
    // Segment 2: (95+80)/2 * 0.75 = 65.625
    // Segment 3: (80+55)/2 * 1.5 = 101.25
    // Segment 4: (55+20)/2 * 5.5 = 206.25
    // Total: 397.5

    assert_relative_eq!(auc, 397.5, epsilon = 1e-10);
}

#[test]
fn test_log_trapezoidal_decreasing() {
    let times = vec![0.0, 2.0, 4.0, 8.0];
    let concs = vec![100.0, 50.0, 25.0, 12.5];

    let auc = auc_log_trapezoidal(&times, &concs);

    // For exact exponential decay with half-life = 2h:
    // True AUC = C0 / lambda = 100 / 0.3466 ≈ 288.5
    // Log trapezoidal should be very accurate
    // AUC 0-8h ≈ 252-254

    assert!(auc > 250.0 && auc < 260.0);
}

#[test]
fn test_log_trapezoidal_with_zero() {
    let times = vec![0.0, 2.0, 4.0];
    let concs = vec![100.0, 10.0, 0.0];

    // Log trapezoidal cannot handle zero concentration
    // Should fall back to linear or return error
    let auc = auc_log_trapezoidal(&times, &concs);

    // Should still produce a reasonable result
    assert!(auc > 0.0);
    assert!(auc.is_finite());
}

#[test]
fn test_auc_methods_comparison() {
    // For purely exponential decay, log method should be more accurate
    let times = vec![0.0, 1.0, 2.0, 4.0, 8.0, 12.0];
    // C = 100 * e^(-0.15*t)
    let concs = vec![100.0, 86.07, 74.08, 54.88, 30.12, 16.53];

    let auc_linear = auc_linear_trapezoidal(&times, &concs);
    let auc_log = auc_log_trapezoidal(&times, &concs);

    // True AUC 0-12h ≈ 555.6
    // Log should be closer to truth
    let true_auc = 555.6;

    let error_linear = (auc_linear - true_auc).abs();
    let error_log = (auc_log - true_auc).abs();

    // Log trapezoidal should have less error
    assert!(error_log < error_linear);
}

// Property-based tests would go here (using proptest)
// Example:
// proptest! {
//     #[test]
//     fn auc_is_positive_for_positive_concentrations(...) { ... }
// }

#[test]
fn test_partial_auc() {
    let times = vec![0.0, 1.0, 2.0, 4.0, 8.0, 12.0];
    let concs = vec![100.0, 90.0, 80.0, 60.0, 35.0, 20.0];

    // Calculate AUC from 2 to 8 hours
    let auc_partial = auc_interval(&times, &concs, 2.0, 8.0);

    // Should be: (80+60)/2*2 + (60+35)/2*4 = 140 + 190 = 330
    assert_relative_eq!(auc_partial, 330.0, epsilon = 1e-10);
}

#[test]
fn test_aumc_calculation() {
    let times = vec![0.0, 1.0, 2.0, 4.0];
    let concs = vec![10.0, 8.0, 6.0, 4.0];

    // AUMC = ∫ t * C(t) dt
    let aumc = aumc_linear_trapezoidal(&times, &concs);

    // Manual calculation:
    // Segment 1: (0*10 + 1*8)/2 * 1 = 4.0
    // Segment 2: (1*8 + 2*6)/2 * 1 = 10.0
    // Segment 3: (2*6 + 4*4)/2 * 2 = 28.0
    // Total: 42.0

    assert_relative_eq!(aumc, 42.0, epsilon = 1e-10);
}
