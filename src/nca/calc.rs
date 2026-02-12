//! Pure calculation functions for NCA parameters
//!
//! This module contains stateless functions that compute individual NCA parameters.
//! All functions take validated inputs and return calculated values.
//!
//! AUC segment calculations are delegated to [`crate::data::auc`].

use crate::observation::Profile;

use super::types::*;
use serde::{Deserialize, Serialize};

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
        // corrxy is -sqrt(R²) since the terminal slope is negative
        let corrxy = if lz.r_squared >= 0.0 {
            -(lz.r_squared.sqrt())
        } else {
            f64::NAN
        };
        RegressionStats {
            r_squared: lz.r_squared,
            adj_r_squared: lz.adj_r_squared,
            corrxy,
            n_points: lz.n_points,
            time_first: lz.time_first,
            time_last: lz.time_last,
            span_ratio: span / half_life,
        }
    }
}

/// A single candidate regression for λz estimation
///
/// Each candidate represents a different set of terminal points used for
/// log-linear regression. Use [`lambda_z_candidates`] to enumerate all
/// valid candidates, or call `.nca()` which auto-selects the best.
///
/// # Example
///
/// ```rust,ignore
/// use pharmsol::nca::{lambda_z_candidates, LambdaZOptions, ObservationProfile};
///
/// let candidates = lambda_z_candidates(&profile, &LambdaZOptions::default(), auc_last);
/// for c in &candidates {
///     println!("{} pts: λz={:.4} t½={:.2} R²={:.4} {}",
///         c.n_points, c.lambda_z, c.half_life, c.r_squared,
///         if c.is_selected { "← selected" } else { "" });
/// }
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LambdaZCandidate {
    /// Number of points used in regression
    pub n_points: usize,
    /// Index of first point in the profile
    pub start_idx: usize,
    /// Index of last point in the profile
    pub end_idx: usize,
    /// Time of first point
    pub start_time: f64,
    /// Time of last point
    pub end_time: f64,
    /// Terminal elimination rate constant
    pub lambda_z: f64,
    /// Terminal half-life (ln(2) / λz)
    pub half_life: f64,
    /// Regression intercept (in log-concentration space)
    pub intercept: f64,
    /// Coefficient of determination
    pub r_squared: f64,
    /// Adjusted R²
    pub adj_r_squared: f64,
    /// Span ratio (time span / half-life)
    pub span_ratio: f64,
    /// AUC∞ computed from this candidate's λz
    pub auc_inf: f64,
    /// Percentage of AUC extrapolated
    pub auc_pct_extrap: f64,
    /// Whether this candidate was auto-selected as best
    pub is_selected: bool,
}

/// Enumerate all valid λz regression candidates for a profile
///
/// Returns every valid regression from `min_points` to `max_points` terminal
/// points, each with its computed λz, half-life, R², and derived AUC∞. The
/// auto-selected best candidate has `is_selected = true`.
///
/// This is useful for interactive exploration: a GUI can display all candidates
/// and let the user override the automatic selection.
///
/// # Arguments
/// * `profile` - Validated observation profile
/// * `options` - Lambda-z estimation options (controls point range, R² thresholds)
/// * `auc_last` - AUC from time 0 to Tlast (needed to compute AUC∞ for each candidate)
pub fn lambda_z_candidates(
    profile: &Profile,
    options: &LambdaZOptions,
    auc_last: f64,
) -> Vec<LambdaZCandidate> {
    let start_idx = if options.include_tmax {
        0
    } else {
        profile.cmax_idx + 1
    };

    if profile.tlast_idx < start_idx + options.min_points - 1 {
        return Vec::new();
    }

    let max_n = if let Some(max) = options.max_points {
        (profile.tlast_idx - start_idx + 1).min(max)
    } else {
        profile.tlast_idx - start_idx + 1
    };

    let clast_obs = profile.concentrations[profile.tlast_idx];

    let mut candidates = Vec::new();
    let mut best_idx: Option<usize> = None;
    let mut best_score = f64::NEG_INFINITY;

    for n_points in options.min_points..=max_n {
        let first_idx = profile.tlast_idx - n_points + 1;
        if first_idx < start_idx {
            continue;
        }

        if let Some(result) = fit_lambda_z(profile, first_idx, profile.tlast_idx, options) {
            let hl = std::f64::consts::LN_2 / result.lambda_z;
            let span = result.time_last - result.time_first;
            let span_ratio = span / hl;
            let auc_inf_val = auc_inf(auc_last, clast_obs, result.lambda_z);
            let extrap_pct = auc_extrap_pct(auc_last, auc_inf_val);

            let candidate = LambdaZCandidate {
                n_points: result.n_points,
                start_idx: first_idx,
                end_idx: profile.tlast_idx,
                start_time: result.time_first,
                end_time: result.time_last,
                lambda_z: result.lambda_z,
                half_life: hl,
                intercept: result.intercept,
                r_squared: result.r_squared,
                adj_r_squared: result.adj_r_squared,
                span_ratio,
                auc_inf: auc_inf_val,
                auc_pct_extrap: extrap_pct,
                is_selected: false,
            };

            // Check if this candidate qualifies for "best" selection
            let qualifies =
                result.r_squared >= options.min_r_squared && span_ratio >= options.min_span_ratio;

            if qualifies {
                let factor = options.adj_r_squared_factor;
                let score = match options.method {
                    LambdaZMethod::AdjR2 => result.adj_r_squared + factor * result.n_points as f64,
                    _ => result.r_squared,
                };
                if score > best_score {
                    best_score = score;
                    best_idx = Some(candidates.len());
                }
            }

            candidates.push(candidate);
        }
    }

    // Mark the selected candidate
    if let Some(idx) = best_idx {
        candidates[idx].is_selected = true;
    }

    candidates
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
///
/// Delegates to [`lambda_z_candidates`] and returns the selected candidate's
/// underlying [`LambdaZResult`]. We use `auc_last = 0.0` here because the
/// caller only needs the regression result, not AUC∞ (which is computed later).
fn lambda_z_best_fit(
    profile: &Profile,
    _start_idx: usize,
    options: &LambdaZOptions,
) -> Option<LambdaZResult> {
    let candidates = lambda_z_candidates(profile, options, 0.0);
    let selected = candidates.iter().find(|c| c.is_selected)?;

    // Reconstruct LambdaZResult from the selected candidate
    let clast_pred =
        (selected.intercept - selected.lambda_z * profile.times[selected.end_idx]).exp();

    Some(LambdaZResult {
        lambda_z: selected.lambda_z,
        intercept: selected.intercept,
        r_squared: selected.r_squared,
        adj_r_squared: selected.adj_r_squared,
        n_points: selected.n_points,
        time_first: selected.start_time,
        time_last: selected.end_time,
        clast_pred,
    })
}

/// Fit log-linear regression for lambda-z
fn fit_lambda_z(
    profile: &Profile,
    first_idx: usize,
    last_idx: usize,
    options: &LambdaZOptions,
) -> Option<LambdaZResult> {
    // Extract points with positive concentrations, respecting exclusion list
    let mut times = Vec::new();
    let mut log_concs = Vec::new();

    for i in first_idx..=last_idx {
        // Skip excluded indices
        if options.exclude_indices.contains(&i) {
            continue;
        }
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

/// Numerically stable linear regression using Kahan (compensated) summation.
///
/// Uses compensated summation for all accumulations to avoid catastrophic
/// cancellation with large time values (e.g., time in minutes > 10,000).
///
/// Returns (slope, intercept, r_squared)
fn linear_regression(x: &[f64], y: &[f64]) -> Option<(f64, f64, f64)> {
    let n = x.len() as f64;
    if n < 2.0 {
        return None;
    }

    // Kahan compensated summation for all sums
    let sum_x = kahan_sum(x.iter().copied());
    let sum_y = kahan_sum(y.iter().copied());
    let sum_xy = kahan_sum(x.iter().zip(y.iter()).map(|(xi, yi)| xi * yi));
    let sum_x2 = kahan_sum(x.iter().map(|xi| xi * xi));

    let denom = n * sum_x2 - sum_x * sum_x;
    if denom.abs() < 1e-15 {
        return None;
    }

    let slope = (n * sum_xy - sum_x * sum_y) / denom;
    let intercept = (sum_y - slope * sum_x) / n;

    // Calculate R² using residuals (more stable than sum_y2 formula)
    let mean_y = sum_y / n;
    let ss_tot = kahan_sum(y.iter().map(|yi| (yi - mean_y).powi(2)));
    let ss_res = kahan_sum(x.iter().zip(y.iter()).map(|(xi, yi)| {
        let pred = intercept + slope * xi;
        (yi - pred).powi(2)
    }));

    let r_squared = if ss_tot.abs() < 1e-15 {
        1.0
    } else {
        1.0 - ss_res / ss_tot
    };

    Some((slope, intercept, r_squared))
}

/// Kahan (compensated) summation for improved numerical precision.
///
/// Reduces floating-point accumulation error from O(n·ε) to O(ε) where
/// ε is machine epsilon, making it safe for large values and long sums.
#[inline]
fn kahan_sum(iter: impl Iterator<Item = f64>) -> f64 {
    let mut sum = 0.0_f64;
    let mut comp = 0.0_f64; // compensation for lost low-order bits
    for val in iter {
        let y = val - comp;
        let t = sum + y;
        comp = (t - sum) - y;
        sum = t;
    }
    sum
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
///
/// Returns `(c0_value, method_used)` or `(NaN, None)` if all methods fail.
pub fn c0(profile: &Profile, methods: &[C0Method], lambda_z: f64) -> (f64, Option<C0Method>) {
    for m in methods {
        if let Some(val) = try_c0_method(profile, *m, lambda_z) {
            return (val, Some(*m));
        }
    }
    (f64::NAN, None)
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

    // Use iterator-based approach to avoid allocating a Vec for BLQ-converted concentrations.
    // Convert BLQ to 0 on-the-fly, keep other values as-is (matching PKNCA).
    let conc_iter = concentrations
        .iter()
        .zip(censoring.iter())
        .map(|(&c, censor)| {
            if matches!(censor, crate::Censor::BLOQ) {
                0.0
            } else {
                c
            }
        });

    // Find first time when concentration increases (PKNCA method)
    // We need to compare adjacent elements, so we use a sliding window via zip
    let mut prev = None;
    for (i, c) in conc_iter.enumerate() {
        if let Some(prev_c) = prev {
            if c > prev_c {
                return Some(times[i - 1]);
            }
        }
        prev = Some(c);
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
// Derived Parameters — Phase 2 additions
// ============================================================================

/// Calculate effective half-life: t½,eff = ln(2) × MRT
///
/// Useful for drugs with nonlinear pharmacokinetics where terminal half-life
/// may not reflect the effective duration of drug persistence.
#[inline]
pub fn effective_half_life(mrt: f64) -> f64 {
    if !mrt.is_finite() || mrt <= 0.0 {
        return f64::NAN;
    }
    std::f64::consts::LN_2 * mrt
}

/// Calculate elimination rate constant: Kel = 1 / MRT
///
/// Alternative representation of overall elimination.
#[inline]
pub fn kel(mrt: f64) -> f64 {
    if !mrt.is_finite() || mrt <= 0.0 {
        return f64::NAN;
    }
    1.0 / mrt
}

/// Calculate peak-to-trough ratio: PTR = Cmax / Cmin
///
/// Used in steady-state analysis to assess PK variability within a dosing interval.
#[inline]
pub fn peak_trough_ratio(cmax: f64, cmin: f64) -> f64 {
    if cmin <= 0.0 || !cmin.is_finite() {
        return f64::NAN;
    }
    cmax / cmin
}

/// Calculate time above a target concentration
///
/// Uses linear interpolation to find exact crossing times.
/// Returns the total time spent above the threshold within the profile.
///
/// This is PD-relevant for concentration-dependent drugs (e.g., antibiotics)
/// where efficacy correlates with the time the drug concentration exceeds
/// a minimum inhibitory concentration (MIC).
pub fn time_above_concentration(times: &[f64], concentrations: &[f64], threshold: f64) -> f64 {
    if times.len() < 2 || concentrations.len() < 2 {
        return 0.0;
    }

    let mut total_time = 0.0;

    for i in 0..times.len() - 1 {
        let (t1, c1) = (times[i], concentrations[i]);
        let (t2, c2) = (times[i + 1], concentrations[i + 1]);
        let dt = t2 - t1;

        if c1 >= threshold && c2 >= threshold {
            // Both above: entire interval counts
            total_time += dt;
        } else if c1 >= threshold && c2 < threshold {
            // Crosses below: interpolate the crossing time
            let t_cross = t1 + dt * (c1 - threshold) / (c1 - c2);
            total_time += t_cross - t1;
        } else if c1 < threshold && c2 >= threshold {
            // Crosses above: interpolate the crossing time
            let t_cross = t1 + dt * (threshold - c1) / (c2 - c1);
            total_time += t2 - t_cross;
        }
        // Both below: nothing added
    }

    total_time
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::auc::auc_segment;
    use crate::data::builder::SubjectBuilderExt;
    use crate::Subject;

    fn make_test_profile() -> Profile {
        let subject = Subject::builder("test")
            .bolus(0.0, 100.0, 0)
            .observation(0.0, 0.0, 0)
            .observation(1.0, 10.0, 0)
            .observation(2.0, 8.0, 0)
            .observation(4.0, 4.0, 0)
            .observation(8.0, 2.0, 0)
            .observation(12.0, 1.0, 0)
            .build();
        let occ = &subject.occasions()[0];
        Profile::from_occasion(occ, 0, &BLQRule::Exclude).unwrap()
    }

    #[test]
    fn test_auc_segment_linear() {
        let auc = auc_segment(0.0, 10.0, 1.0, 8.0, &AUCMethod::Linear);
        assert!((auc - 9.0).abs() < 1e-10); // (10 + 8) / 2 * 1
    }

    #[test]
    fn test_auc_segment_log_down() {
        // Descending - should use log-linear
        let auc = auc_segment(0.0, 10.0, 1.0, 5.0, &AUCMethod::LinUpLogDown);
        let expected = 5.0 / (10.0_f64 / 5.0).ln(); // (C1-C2) * dt / ln(C1/C2)
        assert!((auc - expected).abs() < 1e-10);
    }

    #[test]
    fn test_auc_last() {
        let profile = make_test_profile();
        let auc = profile.auc_last(&AUCMethod::Linear);

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
