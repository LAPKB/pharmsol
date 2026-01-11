//! Pure calculation functions for NCA parameters
//!
//! This module contains stateless functions that compute individual NCA parameters.
//! All functions take validated inputs and return calculated values.

use super::profile::Profile;
use super::types::{AUCMethod, LambdaZMethod, LambdaZOptions, RegressionStats};

// ============================================================================
// AUC Calculations
// ============================================================================

/// Check if log-linear method should be used for this segment
#[inline]
fn use_log_linear(c1: f64, c2: f64) -> bool {
    c2 < c1 && c1 > 0.0 && c2 > 0.0 && ((c1 / c2) - 1.0).abs() >= 1e-10
}

/// Linear trapezoidal AUC for a segment
#[inline]
fn auc_linear(c1: f64, c2: f64, dt: f64) -> f64 {
    (c1 + c2) / 2.0 * dt
}

/// Log-linear AUC for a segment (assumes c1 > c2 > 0)
#[inline]
fn auc_log(c1: f64, c2: f64, dt: f64) -> f64 {
    (c1 - c2) * dt / (c1 / c2).ln()
}

/// Linear trapezoidal AUMC for a segment
#[inline]
fn aumc_linear(t1: f64, c1: f64, t2: f64, c2: f64, dt: f64) -> f64 {
    (t1 * c1 + t2 * c2) / 2.0 * dt
}

/// Log-linear AUMC for a segment (PKNCA formula)
#[inline]
fn aumc_log(t1: f64, c1: f64, t2: f64, c2: f64, dt: f64) -> f64 {
    let k = (c1 / c2).ln() / dt;
    (t1 * c1 - t2 * c2) / k + (c1 - c2) / (k * k)
}

/// Calculate AUC for a single segment between two time points
///
/// For [`AUCMethod::LinLog`], this uses linear trapezoidal since segment-level
/// calculation cannot know Tmax context. Use [`auc_last`] for proper LinLog handling.
#[inline]
pub fn auc_segment(t1: f64, c1: f64, t2: f64, c2: f64, method: AUCMethod) -> f64 {
    let dt = t2 - t1;
    if dt <= 0.0 {
        return 0.0;
    }

    match method {
        AUCMethod::Linear | AUCMethod::LinLog => auc_linear(c1, c2, dt),
        AUCMethod::LinUpLogDown => {
            if use_log_linear(c1, c2) {
                auc_log(c1, c2, dt)
            } else {
                auc_linear(c1, c2, dt)
            }
        }
    }
}

/// Calculate AUC for a segment with Tmax context (for LinLog method)
#[inline]
fn auc_segment_with_tmax(t1: f64, c1: f64, t2: f64, c2: f64, tmax: f64, method: AUCMethod) -> f64 {
    let dt = t2 - t1;
    if dt <= 0.0 {
        return 0.0;
    }

    match method {
        AUCMethod::Linear => auc_linear(c1, c2, dt),
        AUCMethod::LinUpLogDown => {
            if use_log_linear(c1, c2) {
                auc_log(c1, c2, dt)
            } else {
                auc_linear(c1, c2, dt)
            }
        }
        AUCMethod::LinLog => {
            // Linear before/at Tmax, log-linear after Tmax (for descending)
            if t2 <= tmax || !use_log_linear(c1, c2) {
                auc_linear(c1, c2, dt)
            } else {
                auc_log(c1, c2, dt)
            }
        }
    }
}

/// Calculate AUC from time 0 to Tlast
pub fn auc_last(profile: &Profile, method: AUCMethod) -> f64 {
    let mut auc = 0.0;
    let tmax = profile.tmax(); // Get Tmax for LinLog method

    for i in 1..=profile.tlast_idx {
        auc += auc_segment_with_tmax(
            profile.times[i - 1],
            profile.concentrations[i - 1],
            profile.times[i],
            profile.concentrations[i],
            tmax,
            method,
        );
    }

    auc
}

/// Calculate AUMC for a segment with Tmax context (for LinLog method)
#[inline]
fn aumc_segment_with_tmax(t1: f64, c1: f64, t2: f64, c2: f64, tmax: f64, method: AUCMethod) -> f64 {
    let dt = t2 - t1;
    if dt <= 0.0 {
        return 0.0;
    }

    match method {
        AUCMethod::Linear => aumc_linear(t1, c1, t2, c2, dt),
        AUCMethod::LinUpLogDown => {
            if use_log_linear(c1, c2) {
                aumc_log(t1, c1, t2, c2, dt)
            } else {
                aumc_linear(t1, c1, t2, c2, dt)
            }
        }
        AUCMethod::LinLog => {
            // Linear before/at Tmax, log-linear after Tmax (for descending)
            if t2 <= tmax || !use_log_linear(c1, c2) {
                aumc_linear(t1, c1, t2, c2, dt)
            } else {
                aumc_log(t1, c1, t2, c2, dt)
            }
        }
    }
}

/// Calculate AUMC from time 0 to Tlast
pub fn aumc_last(profile: &Profile, method: AUCMethod) -> f64 {
    let mut aumc = 0.0;
    let tmax_val = profile.tmax();

    for i in 1..=profile.tlast_idx {
        aumc += aumc_segment_with_tmax(
            profile.times[i - 1],
            profile.concentrations[i - 1],
            profile.times[i],
            profile.concentrations[i],
            tmax_val,
            method,
        );
    }

    aumc
}

/// Calculate AUC over a specific interval (for steady-state AUCτ)
pub fn auc_interval(profile: &Profile, start: f64, end: f64, method: AUCMethod) -> f64 {
    if end <= start {
        return 0.0;
    }

    let mut auc = 0.0;

    for i in 1..profile.times.len() {
        let t1 = profile.times[i - 1];
        let t2 = profile.times[i];

        // Skip segments entirely outside the interval
        if t2 <= start || t1 >= end {
            continue;
        }

        // Clamp to interval boundaries
        let seg_start = t1.max(start);
        let seg_end = t2.min(end);

        // Interpolate concentrations at boundaries if needed
        let c1 = if t1 < start {
            interpolate_concentration(profile, start)
        } else {
            profile.concentrations[i - 1]
        };

        let c2 = if t2 > end {
            interpolate_concentration(profile, end)
        } else {
            profile.concentrations[i]
        };

        auc += auc_segment(seg_start, c1, seg_end, c2, method);
    }

    auc
}

/// Linear interpolation of concentration at a given time
fn interpolate_concentration(profile: &Profile, time: f64) -> f64 {
    if time <= profile.times[0] {
        return profile.concentrations[0];
    }
    if time >= profile.times[profile.times.len() - 1] {
        return profile.concentrations[profile.times.len() - 1];
    }

    // Find bracketing indices
    let upper_idx = profile
        .times
        .iter()
        .position(|&t| t >= time)
        .unwrap_or(profile.times.len() - 1);
    let lower_idx = upper_idx.saturating_sub(1);

    let t1 = profile.times[lower_idx];
    let t2 = profile.times[upper_idx];
    let c1 = profile.concentrations[lower_idx];
    let c2 = profile.concentrations[upper_idx];

    if (t2 - t1).abs() < 1e-10 {
        c1
    } else {
        c1 + (c2 - c1) * (time - t1) / (t2 - t1)
    }
}

// ============================================================================
// Lambda-z Calculations
// ============================================================================

/// Result of lambda-z estimation
#[derive(Debug, Clone)]
pub struct LambdaZResult {
    pub lambda_z: f64,
    pub intercept: f64,
    pub r_squared: f64,
    pub adj_r_squared: f64,
    pub n_points: usize,
    pub time_first: f64,
    pub time_last: f64,
    pub clast_pred: f64,
}

impl From<LambdaZResult> for RegressionStats {
    fn from(lz: LambdaZResult) -> Self {
        let half_life = std::f64::consts::LN_2 / lz.lambda_z;
        let span = lz.time_last - lz.time_first;
        RegressionStats {
            r_squared: lz.r_squared,
            adj_r_squared: lz.adj_r_squared,
            n_points: lz.n_points,
            time_first: lz.time_first,
            time_last: lz.time_last,
            span_ratio: span / half_life,
        }
    }
}

/// Estimate lambda-z using log-linear regression
pub fn lambda_z(profile: &Profile, options: &LambdaZOptions) -> Option<LambdaZResult> {
    // Determine start index (exclude or include Tmax)
    let start_idx = if options.include_tmax {
        0
    } else {
        profile.cmax_idx + 1
    };

    // Need at least min_points between start and tlast
    if profile.tlast_idx < start_idx + options.min_points - 1 {
        return None;
    }

    match options.method {
        LambdaZMethod::Manual(n) => lambda_z_with_n_points(profile, start_idx, n, options),
        LambdaZMethod::R2 | LambdaZMethod::AdjR2 => lambda_z_best_fit(profile, start_idx, options),
    }
}

/// Lambda-z with specified number of terminal points
fn lambda_z_with_n_points(
    profile: &Profile,
    start_idx: usize,
    n_points: usize,
    options: &LambdaZOptions,
) -> Option<LambdaZResult> {
    if n_points < options.min_points {
        return None;
    }

    let first_idx = profile.tlast_idx.saturating_sub(n_points - 1);
    if first_idx < start_idx {
        return None;
    }

    fit_lambda_z(profile, first_idx, profile.tlast_idx, options)
}

/// Lambda-z with best fit selection
fn lambda_z_best_fit(
    profile: &Profile,
    start_idx: usize,
    options: &LambdaZOptions,
) -> Option<LambdaZResult> {
    let mut best_result: Option<LambdaZResult> = None;

    // Determine max points to try
    let max_n = if let Some(max) = options.max_points {
        (profile.tlast_idx - start_idx + 1).min(max)
    } else {
        profile.tlast_idx - start_idx + 1
    };

    // Try all valid point counts
    for n_points in options.min_points..=max_n {
        let first_idx = profile.tlast_idx - n_points + 1;

        if first_idx < start_idx {
            continue;
        }

        if let Some(result) = fit_lambda_z(profile, first_idx, profile.tlast_idx, options) {
            // Check quality criteria
            if result.r_squared < options.min_r_squared {
                continue;
            }

            let half_life = std::f64::consts::LN_2 / result.lambda_z;
            let span = result.time_last - result.time_first;
            let span_ratio = span / half_life;

            if span_ratio < options.min_span_ratio {
                continue;
            }

            // Select best based on method, using adj_r_squared_factor to prefer more points
            let is_better = match &best_result {
                None => true,
                Some(best) => {
                    // PKNCA formula: adj_r_squared + factor * n_points
                    // This allows preferring regressions with more points when R² is similar
                    let factor = options.adj_r_squared_factor;
                    let current_score = match options.method {
                        LambdaZMethod::AdjR2 => {
                            result.adj_r_squared + factor * result.n_points as f64
                        }
                        _ => result.r_squared,
                    };
                    let best_score = match options.method {
                        LambdaZMethod::AdjR2 => best.adj_r_squared + factor * best.n_points as f64,
                        _ => best.r_squared,
                    };

                    current_score > best_score
                }
            };

            if is_better {
                best_result = Some(result);
            }
        }
    }

    best_result
}

/// Fit log-linear regression for lambda-z
fn fit_lambda_z(
    profile: &Profile,
    first_idx: usize,
    last_idx: usize,
    _options: &LambdaZOptions,
) -> Option<LambdaZResult> {
    // Extract points with positive concentrations
    let mut times = Vec::new();
    let mut log_concs = Vec::new();

    for i in first_idx..=last_idx {
        if profile.concentrations[i] > 0.0 {
            times.push(profile.times[i]);
            log_concs.push(profile.concentrations[i].ln());
        }
    }

    if times.len() < 2 {
        return None;
    }

    // Simple linear regression: ln(C) = intercept + slope * t
    let (slope, intercept, r_squared) = linear_regression(&times, &log_concs)?;

    let lambda_z = -slope;

    // Lambda-z must be positive
    if lambda_z <= 0.0 {
        return None;
    }

    let n = times.len() as f64;
    let adj_r_squared = 1.0 - (1.0 - r_squared) * (n - 1.0) / (n - 2.0);

    // Predicted concentration at Tlast
    let clast_pred = (intercept + slope * profile.times[last_idx]).exp();

    Some(LambdaZResult {
        lambda_z,
        intercept,
        r_squared,
        adj_r_squared,
        n_points: times.len(),
        time_first: times[0],
        time_last: times[times.len() - 1],
        clast_pred,
    })
}

/// Simple linear regression: y = a + b*x
/// Returns (slope, intercept, r_squared)
fn linear_regression(x: &[f64], y: &[f64]) -> Option<(f64, f64, f64)> {
    let n = x.len() as f64;
    if n < 2.0 {
        return None;
    }

    let sum_x: f64 = x.iter().sum();
    let sum_y: f64 = y.iter().sum();
    let sum_xy: f64 = x.iter().zip(y.iter()).map(|(xi, yi)| xi * yi).sum();
    let sum_x2: f64 = x.iter().map(|xi| xi * xi).sum();
    let sum_y2: f64 = y.iter().map(|yi| yi * yi).sum();

    let denom = n * sum_x2 - sum_x * sum_x;
    if denom.abs() < 1e-15 {
        return None;
    }

    let slope = (n * sum_xy - sum_x * sum_y) / denom;
    let intercept = (sum_y - slope * sum_x) / n;

    // Calculate R²
    let ss_tot = sum_y2 - sum_y * sum_y / n;
    let ss_res: f64 = x
        .iter()
        .zip(y.iter())
        .map(|(xi, yi)| {
            let pred = intercept + slope * xi;
            (yi - pred).powi(2)
        })
        .sum();

    let r_squared = if ss_tot.abs() < 1e-15 {
        1.0
    } else {
        1.0 - ss_res / ss_tot
    };

    Some((slope, intercept, r_squared))
}

// ============================================================================
// Derived Parameters
// ============================================================================

/// Calculate terminal half-life
#[inline]
pub fn half_life(lambda_z: f64) -> f64 {
    std::f64::consts::LN_2 / lambda_z
}

/// Calculate AUC extrapolated to infinity
#[inline]
pub fn auc_inf(auc_last: f64, clast: f64, lambda_z: f64) -> f64 {
    if lambda_z <= 0.0 {
        return f64::NAN;
    }
    auc_last + clast / lambda_z
}

/// Calculate percentage of AUC extrapolated
#[inline]
pub fn auc_extrap_pct(auc_last: f64, auc_inf: f64) -> f64 {
    if auc_inf <= 0.0 || !auc_inf.is_finite() {
        return f64::NAN;
    }
    (auc_inf - auc_last) / auc_inf * 100.0
}

/// Calculate AUMC extrapolated to infinity
pub fn aumc_inf(aumc_last: f64, clast: f64, tlast: f64, lambda_z: f64) -> f64 {
    if lambda_z <= 0.0 {
        return f64::NAN;
    }
    aumc_last + clast * tlast / lambda_z + clast / (lambda_z * lambda_z)
}

/// Calculate mean residence time
#[inline]
pub fn mrt(aumc_inf: f64, auc_inf: f64) -> f64 {
    if auc_inf <= 0.0 || !auc_inf.is_finite() {
        return f64::NAN;
    }
    aumc_inf / auc_inf
}

/// Calculate clearance
#[inline]
pub fn clearance(dose: f64, auc_inf: f64) -> f64 {
    if auc_inf <= 0.0 || !auc_inf.is_finite() {
        return f64::NAN;
    }
    dose / auc_inf
}

/// Calculate volume of distribution
#[inline]
pub fn vz(dose: f64, lambda_z: f64, auc_inf: f64) -> f64 {
    if lambda_z <= 0.0 || auc_inf <= 0.0 || !auc_inf.is_finite() {
        return f64::NAN;
    }
    dose / (lambda_z * auc_inf)
}

// ============================================================================
// Route-Specific Parameters
// ============================================================================

use super::types::C0Method;

/// Estimate C0 using a cascade of methods (first success wins)
///
/// Methods are tried in order. Default cascade: `[Observed, LogSlope, FirstConc]`
pub fn c0(profile: &Profile, methods: &[C0Method], lambda_z: f64) -> f64 {
    methods
        .iter()
        .filter_map(|m| try_c0_method(profile, *m, lambda_z))
        .next()
        .unwrap_or(f64::NAN)
}

/// Try a single C0 estimation method
fn try_c0_method(profile: &Profile, method: C0Method, _lambda_z: f64) -> Option<f64> {
    match method {
        C0Method::Observed => {
            // Use concentration at t=0 if present and positive
            if !profile.times.is_empty() && profile.times[0].abs() < 1e-10 {
                let c = profile.concentrations[0];
                if c > 0.0 {
                    return Some(c);
                }
            }
            None
        }
        C0Method::LogSlope => c0_logslope(profile),
        C0Method::FirstConc => {
            // Use first positive concentration
            profile.concentrations.iter().find(|&&c| c > 0.0).copied()
        }
        C0Method::Cmin => {
            // Use minimum positive concentration
            profile
                .concentrations
                .iter()
                .filter(|&&c| c > 0.0)
                .min_by(|a, b| a.partial_cmp(b).unwrap())
                .copied()
        }
        C0Method::Zero => Some(0.0),
    }
}

/// Semilog back-extrapolation from first two positive points (PKNCA logslope method)
fn c0_logslope(profile: &Profile) -> Option<f64> {
    if profile.concentrations.is_empty() {
        return None;
    }

    // Find first two positive concentrations
    let positive_points: Vec<(f64, f64)> = profile
        .times
        .iter()
        .zip(profile.concentrations.iter())
        .filter(|(_, &c)| c > 0.0)
        .map(|(&t, &c)| (t, c))
        .take(2)
        .collect();

    if positive_points.len() < 2 {
        return None;
    }

    let (t1, c1) = positive_points[0];
    let (t2, c2) = positive_points[1];

    // PKNCA requires c2 < c1 (declining) for logslope
    if c2 >= c1 || (t2 - t1).abs() < 1e-10 {
        return None;
    }

    // Semilog extrapolation: C0 = exp(ln(c1) - slope * t1)
    let slope = (c2.ln() - c1.ln()) / (t2 - t1);
    Some((c1.ln() - slope * t1).exp())
}

/// Legacy C0 back-extrapolation (kept for compatibility)
#[deprecated(note = "Use c0() with C0Method cascade instead")]
#[allow(dead_code)]
pub fn c0_backextrap(profile: &Profile, _lambda_z: f64) -> f64 {
    c0(
        profile,
        &[C0Method::Observed, C0Method::LogSlope, C0Method::FirstConc],
        _lambda_z,
    )
}

/// Calculate Vd for IV bolus
#[inline]
pub fn vd_bolus(dose: f64, c0: f64) -> f64 {
    if c0 <= 0.0 || !c0.is_finite() {
        return f64::NAN;
    }
    dose / c0
}

/// Calculate Vss for IV administration
pub fn vss(dose: f64, aumc_inf: f64, auc_inf: f64) -> f64 {
    if auc_inf <= 0.0 || !auc_inf.is_finite() {
        return f64::NAN;
    }
    dose * aumc_inf / (auc_inf * auc_inf)
}

/// Calculate MRT corrected for infusion duration
#[inline]
pub fn mrt_infusion(mrt: f64, duration: f64) -> f64 {
    mrt - duration / 2.0
}

/// Detect lag time for extravascular administration from raw concentration data
///
/// This matches PKNCA's approach: tlag is calculated on raw data with BLQ treated as 0,
/// BEFORE any BLQ filtering is applied to the profile.
///
/// Returns the time at which concentration first increases (PKNCA method).
/// For profiles starting at t=0 with C=0 (or BLQ), this returns 0 if there's
/// an increase to the next point.
pub fn tlag_from_raw(
    times: &[f64],
    concentrations: &[f64],
    censoring: &[crate::Censor],
) -> Option<f64> {
    if times.len() < 2 || concentrations.len() < 2 {
        return None;
    }

    // Convert BLQ to 0, keep other values as-is (matching PKNCA)
    let concs: Vec<f64> = concentrations
        .iter()
        .zip(censoring.iter())
        .map(|(&c, censor)| {
            if matches!(censor, crate::Censor::BLOQ) {
                0.0
            } else {
                c
            }
        })
        .collect();

    // Find first time when concentration increases (PKNCA method)
    for i in 0..concs.len().saturating_sub(1) {
        if concs[i + 1] > concs[i] {
            return Some(times[i]);
        }
    }
    // No increase found - either flat or all decreasing
    None
}

/// Detect lag time for extravascular administration from processed profile
///
/// Returns the time at which concentration first increases (PKNCA method).
/// This is more appropriate than finding "time before first positive" because
/// it captures when absorption actually begins, not just when drug is detectable.
///
/// For profiles starting at t=0 with C=0, this returns the time point where
/// C[i+1] > C[i] for the first time.
#[deprecated(note = "Use tlag_from_raw for PKNCA-compatible tlag calculation")]
pub fn tlag(profile: &Profile) -> Option<f64> {
    // Find first time when concentration increases
    for i in 0..profile.concentrations.len().saturating_sub(1) {
        if profile.concentrations[i + 1] > profile.concentrations[i] {
            return Some(profile.times[i]);
        }
    }
    // No increase found - either flat or all decreasing
    None
}

// ============================================================================
// Steady-State Parameters
// ============================================================================

/// Calculate Cmin from profile
pub fn cmin(profile: &Profile) -> f64 {
    profile
        .concentrations
        .iter()
        .copied()
        .filter(|&c| c > 0.0)
        .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or(0.0)
}

/// Calculate average concentration
#[inline]
pub fn cavg(auc_tau: f64, tau: f64) -> f64 {
    if tau <= 0.0 {
        return f64::NAN;
    }
    auc_tau / tau
}

/// Calculate fluctuation percentage
pub fn fluctuation(cmax: f64, cmin: f64, cavg: f64) -> f64 {
    if cavg <= 0.0 {
        return f64::NAN;
    }
    (cmax - cmin) / cavg * 100.0
}

/// Calculate swing
pub fn swing(cmax: f64, cmin: f64) -> f64 {
    if cmin <= 0.0 {
        return f64::NAN;
    }
    (cmax - cmin) / cmin
}

/// Calculate accumulation ratio
#[inline]
#[allow(dead_code)] // Reserved for future steady-state analysis
pub fn accumulation(auc_tau: f64, auc_inf_single: f64) -> f64 {
    if auc_inf_single <= 0.0 || !auc_inf_single.is_finite() {
        return f64::NAN;
    }
    auc_tau / auc_inf_single
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Censor;

    fn make_test_profile() -> Profile {
        let censoring = vec![Censor::None; 6];
        Profile::from_arrays(
            &[0.0, 1.0, 2.0, 4.0, 8.0, 12.0],
            &[0.0, 10.0, 8.0, 4.0, 2.0, 1.0],
            &censoring,
            super::super::types::BLQRule::Exclude,
        )
        .unwrap()
    }

    #[test]
    fn test_auc_segment_linear() {
        let auc = auc_segment(0.0, 10.0, 1.0, 8.0, AUCMethod::Linear);
        assert!((auc - 9.0).abs() < 1e-10); // (10 + 8) / 2 * 1
    }

    #[test]
    fn test_auc_segment_log_down() {
        // Descending - should use log-linear
        let auc = auc_segment(0.0, 10.0, 1.0, 5.0, AUCMethod::LinUpLogDown);
        let expected = 5.0 / (10.0_f64 / 5.0).ln(); // (C1-C2) * dt / ln(C1/C2)
        assert!((auc - expected).abs() < 1e-10);
    }

    #[test]
    fn test_auc_last() {
        let profile = make_test_profile();
        let auc = auc_last(&profile, AUCMethod::Linear);

        // Manual calculation:
        // 0-1: (0 + 10) / 2 * 1 = 5
        // 1-2: (10 + 8) / 2 * 1 = 9
        // 2-4: (8 + 4) / 2 * 2 = 12
        // 4-8: (4 + 2) / 2 * 4 = 12
        // 8-12: (2 + 1) / 2 * 4 = 6
        // Total = 44
        assert!((auc - 44.0).abs() < 1e-10);
    }

    #[test]
    fn test_half_life() {
        let hl = half_life(0.1);
        assert!((hl - 6.931).abs() < 0.01); // ln(2) / 0.1 ≈ 6.931
    }

    #[test]
    fn test_clearance() {
        let cl = clearance(100.0, 50.0);
        assert!((cl - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_vz() {
        let v = vz(100.0, 0.1, 50.0);
        assert!((v - 20.0).abs() < 1e-10); // 100 / (0.1 * 50) = 20
    }

    #[test]
    fn test_linear_regression() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0]; // Perfect line: y = 2x

        let (slope, intercept, r_squared) = linear_regression(&x, &y).unwrap();
        assert!((slope - 2.0).abs() < 1e-10);
        assert!(intercept.abs() < 1e-10);
        assert!((r_squared - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_fluctuation() {
        let fluct = fluctuation(10.0, 2.0, 5.0);
        assert!((fluct - 160.0).abs() < 1e-10); // (10-2)/5 * 100 = 160%
    }

    #[test]
    fn test_swing() {
        let s = swing(10.0, 2.0);
        assert!((s - 4.0).abs() < 1e-10); // (10-2)/2 = 4
    }
}
