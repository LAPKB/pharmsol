//! Tests for NCA parameter calculations
//!
//! Tests all derived parameters:
//! - Clearance
//! - Volume of distribution
//! - Mean residence time
//! - etc.

use approx::assert_relative_eq;
use pharmsol::nca::params::*;

#[test]
fn test_calculate_auc_inf_obs() {
    let auc_last = 450.0; // ng*h/mL
    let c_last = 15.0; // ng/mL
    let lambda_z = 0.1; // 1/h

    let auc_inf = calculate_auc_inf_obs(auc_last, c_last, lambda_z);

    // AUC_inf = AUC_last + C_last / lambda_z
    // = 450 + 15 / 0.1 = 450 + 150 = 600
    assert_relative_eq!(auc_inf, 600.0, epsilon = 0.001);
}

#[test]
fn test_calculate_auc_inf_pred() {
    let auc_last = 450.0;
    let c_last_pred = 16.0; // Predicted from regression
    let lambda_z = 0.1;

    let auc_inf = calculate_auc_inf_pred(auc_last, c_last_pred, lambda_z);

    // AUC_inf = AUC_last + C_last_pred / lambda_z
    // = 450 + 16 / 0.1 = 450 + 160 = 610
    assert_relative_eq!(auc_inf, 610.0, epsilon = 0.001);
}

#[test]
fn test_extrapolation_percent() {
    let auc_last = 450.0;
    let auc_inf = 500.0;

    let extrap_pct = calculate_extrapolation_percent(auc_last, auc_inf);

    // (500 - 450) / 500 * 100 = 10%
    assert_relative_eq!(extrap_pct, 10.0, epsilon = 0.001);
}

#[test]
fn test_calculate_clearance() {
    let dose = 1000.0; // mg
    let auc = 500.0; // mg*h/L

    let cl = calculate_clearance(dose, auc);

    // CL = Dose / AUC = 1000 / 500 = 2.0 L/h
    assert_relative_eq!(cl, 2.0, epsilon = 0.001);
}

#[test]
fn test_calculate_volume_distribution() {
    let cl = 2.0; // L/h
    let lambda_z = 0.1; // 1/h

    let vd = calculate_volume_distribution(cl, lambda_z);

    // Vd = CL / lambda_z = 2.0 / 0.1 = 20.0 L
    assert_relative_eq!(vd, 20.0, epsilon = 0.001);
}

#[test]
fn test_calculate_half_life() {
    let lambda_z = 0.0693; // 1/h

    let t_half = calculate_half_life(lambda_z);

    // T1/2 = ln(2) / lambda_z = 0.693 / 0.0693 ≈ 10.0 h
    assert_relative_eq!(t_half, 10.0, epsilon = 0.01);
}

#[test]
fn test_calculate_mrt() {
    let aumc = 5000.0; // ng*h²/mL
    let auc = 500.0; // ng*h/mL

    let mrt = calculate_mrt(aumc, auc);

    // MRT = AUMC / AUC = 5000 / 500 = 10.0 h
    assert_relative_eq!(mrt, 10.0, epsilon = 0.001);
}

#[test]
fn test_calculate_vss() {
    let cl = 2.0; // L/h
    let mrt = 10.0; // h

    let vss = calculate_vss(cl, mrt);

    // Vss = CL * MRT = 2.0 * 10.0 = 20.0 L
    assert_relative_eq!(vss, 20.0, epsilon = 0.001);
}

#[test]
fn test_find_cmax_tmax() {
    let times = vec![0.0, 0.5, 1.0, 2.0, 4.0, 8.0];
    let concs = vec![0.0, 50.0, 80.0, 90.0, 60.0, 30.0];

    let (cmax, tmax) = find_cmax_tmax(&times, &concs);

    assert_relative_eq!(cmax, 90.0, epsilon = 0.001);
    assert_relative_eq!(tmax, 2.0, epsilon = 0.001);
}

#[test]
fn test_find_cmax_at_first_point() {
    // IV bolus - Cmax at t=0
    let times = vec![0.0, 1.0, 2.0, 4.0];
    let concs = vec![100.0, 80.0, 60.0, 40.0];

    let (cmax, tmax) = find_cmax_tmax(&times, &concs);

    assert_relative_eq!(cmax, 100.0, epsilon = 0.001);
    assert_relative_eq!(tmax, 0.0, epsilon = 0.001);
}

#[test]
fn test_calculate_c0_extrapolation() {
    // For IV bolus, extrapolate back to t=0
    let times = vec![0.25, 0.5, 1.0, 2.0];
    let concs = vec![95.0, 90.0, 81.0, 66.0];

    let c0 = calculate_c0_extrapolation(&times, &concs);

    // Should be around 100 (depends on extrapolation method)
    assert!(c0 > 98.0 && c0 < 102.0);
}

#[test]
fn test_steady_state_auc_tau() {
    let times = vec![0.0, 1.0, 2.0, 4.0, 6.0, 8.0];
    let concs = vec![50.0, 60.0, 70.0, 65.0, 55.0, 50.0];
    let tau = 8.0; // Dosing interval

    let auc_tau = calculate_auc_tau(&times, &concs, tau);

    // Should integrate over the dosing interval
    assert!(auc_tau > 0.0);
}

#[test]
fn test_accumulation_ratio() {
    let auc_tau_ss = 500.0; // AUC at steady-state
    let auc_tau_sd = 400.0; // AUC after single dose

    let rac = calculate_accumulation_ratio(auc_tau_ss, auc_tau_sd);

    // Rac = AUC_tau_ss / AUC_tau_sd = 500 / 400 = 1.25
    assert_relative_eq!(rac, 1.25, epsilon = 0.001);
}

#[test]
fn test_fluctuation() {
    let cmax_ss = 80.0;
    let cmin_ss = 40.0;

    let fluct = calculate_fluctuation(cmax_ss, cmin_ss);

    // Fluctuation = (Cmax - Cmin) / Cmin * 100
    // = (80 - 40) / 40 * 100 = 100%
    assert_relative_eq!(fluct, 100.0, epsilon = 0.001);
}

#[test]
fn test_swing() {
    let cmax_ss = 80.0;
    let cmin_ss = 40.0;

    let swing = calculate_swing(cmax_ss, cmin_ss);

    // Swing = (Cmax - Cmin) / Cmin
    // = (80 - 40) / 40 = 1.0
    assert_relative_eq!(swing, 1.0, epsilon = 0.001);
}

#[test]
fn test_cave_steady_state() {
    let auc_tau = 480.0; // ng*h/mL
    let tau = 8.0; // h

    let cave = calculate_cave(auc_tau, tau);

    // Cave = AUC_tau / tau = 480 / 8 = 60.0 ng/mL
    assert_relative_eq!(cave, 60.0, epsilon = 0.001);
}

#[test]
fn test_all_parameters_integration() {
    // Complete workflow: calculate all parameters from raw data
    let times = vec![0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0];
    let concs = vec![100.0, 91.0, 83.0, 70.0, 49.0, 24.0, 12.0, 1.5];
    let dose = 1000.0;

    // Step 1: Find Cmax/Tmax
    let (cmax, tmax) = find_cmax_tmax(&times, &concs);
    assert_relative_eq!(cmax, 100.0, epsilon = 0.1);
    assert_relative_eq!(tmax, 0.0, epsilon = 0.1);

    // Step 2: Calculate AUC_last
    let auc_last = auc_linear_trapezoidal(&times, &concs);
    assert!(auc_last > 400.0 && auc_last < 600.0);

    // Step 3: Calculate lambda_z
    let lambda_z_result = calculate_lambda_z_adjusted_r2(&times, &concs, None).unwrap();
    let lambda_z = lambda_z_result.lambda;
    assert!(lambda_z > 0.05 && lambda_z < 0.15);

    // Step 4: Calculate AUC_inf
    let c_last = *concs.last().unwrap();
    let auc_inf = calculate_auc_inf_obs(auc_last, c_last, lambda_z);
    assert!(auc_inf > auc_last);

    // Step 5: Calculate clearance
    let cl = calculate_clearance(dose, auc_inf);
    assert!(cl > 0.0);

    // Step 6: Calculate Vd
    let vd = calculate_volume_distribution(cl, lambda_z);
    assert!(vd > 0.0);

    // Step 7: Calculate T1/2
    let t_half = calculate_half_life(lambda_z);
    assert!(t_half > 0.0);

    println!("Complete parameter set:");
    println!("  Cmax: {:.2} ng/mL", cmax);
    println!("  Tmax: {:.2} h", tmax);
    println!("  AUC_last: {:.2} ng*h/mL", auc_last);
    println!("  AUC_inf: {:.2} ng*h/mL", auc_inf);
    println!("  Lambda_z: {:.4} 1/h", lambda_z);
    println!("  T1/2: {:.2} h", t_half);
    println!("  CL: {:.2} L/h", cl);
    println!("  Vd: {:.2} L", vd);
}
