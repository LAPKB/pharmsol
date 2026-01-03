//! Area Under the Curve (AUC) calculation functions
//!
//! This module provides functions for calculating AUC using different integration methods.
//! The functions are designed to be usable both:
//! - Directly with raw time/concentration arrays (for PMcore's bestdose)
//! - Through higher-level NCA functions on Subject data
//!
//! # Methods
//!
//! ## Linear Trapezoidal
//!
//! The simplest method, uses the arithmetic mean of adjacent concentrations:
//!
//! ```text
//! AUC_segment = (C₁ + C₂) / 2 × (t₂ - t₁)
//! ```
//!
//! ## Linear Up / Log Down (Recommended)
//!
//! The industry standard method that provides more accurate estimates:
//! - **Linear** for ascending concentrations (absorption phase)
//! - **Log-linear** for descending concentrations (elimination phase)
//!
//! ```text
//! Ascending (C₂ ≥ C₁):   AUC = (C₁ + C₂) / 2 × Δt
//! Descending (C₂ < C₁):  AUC = (C₁ - C₂) × Δt / ln(C₁ / C₂)
//! ```
//!
//! The log-linear formula is mathematically equivalent to integrating an exponential
//! decay between the two points, which better reflects elimination kinetics.

use serde::{Deserialize, Serialize};

/// Method for calculating AUC segments
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum AUCMethod {
    /// Linear trapezoidal rule: (C₁ + C₂) / 2 × Δt
    Linear,
    /// Linear up/log down: linear for ascending, log-linear for descending
    /// This is the industry standard method and is more accurate for PK data
    #[default]
    LinUpLogDown,
}

/// Calculate AUC for a single segment between two time points
///
/// This is the fundamental building block for all AUC calculations.
///
/// # Arguments
///
/// * `t1` - Time at first point
/// * `c1` - Concentration at first point
/// * `t2` - Time at second point
/// * `c2` - Concentration at second point
/// * `method` - Integration method to use
///
/// # Returns
///
/// The AUC for this segment. Returns 0.0 if times are invalid (t2 <= t1).
///
/// # Examples
///
/// ```rust
/// use pharmsol::nca::auc_segment;
/// use pharmsol::nca::AUCMethod;
///
/// // Linear method
/// let auc_lin = auc_segment(0.0, 10.0, 1.0, 8.0, AUCMethod::Linear);
/// assert!((auc_lin - 9.0).abs() < 1e-10); // (10 + 8) / 2 * 1 = 9
///
/// // Log-down method (descending)
/// let auc_log = auc_segment(0.0, 10.0, 1.0, 8.0, AUCMethod::LinUpLogDown);
/// // Log-linear: (10 - 8) * 1 / ln(10/8) ≈ 8.96
/// ```
#[inline]
pub fn auc_segment(t1: f64, c1: f64, t2: f64, c2: f64, method: AUCMethod) -> f64 {
    let dt = t2 - t1;

    // Guard against invalid time intervals
    if dt <= 0.0 {
        return 0.0;
    }

    match method {
        AUCMethod::Linear => (c1 + c2) / 2.0 * dt,
        AUCMethod::LinUpLogDown => {
            // Use linear for:
            // - Ascending concentrations (absorption)
            // - Non-positive concentrations (can't take log)
            // - Very small concentrations (numerical stability)
            if c2 >= c1 || c1 <= 0.0 || c2 <= 0.0 {
                (c1 + c2) / 2.0 * dt
            } else {
                // Log-linear for descending positive concentrations
                // Formula: (C1 - C2) * dt / ln(C1/C2)
                let ratio = c1 / c2;

                // Guard against numerical issues when concentrations are very close
                if (ratio - 1.0).abs() < 1e-10 {
                    (c1 + c2) / 2.0 * dt
                } else {
                    (c1 - c2) * dt / ratio.ln()
                }
            }
        }
    }
}

/// Calculate AUC from time 0 to the last measurable concentration (AUClast)
///
/// This function calculates the area from the first time point to the last
/// concentration that is greater than zero (or the limit of quantification).
///
/// # Arguments
///
/// * `times` - Time points (must be sorted in ascending order)
/// * `concentrations` - Concentration values at each time point
/// * `method` - Integration method
///
/// # Returns
///
/// The cumulative AUC from the first time point to Tlast (last positive concentration).
///
/// # Panics
///
/// Panics if `times` and `concentrations` have different lengths.
///
/// # Examples
///
/// ```rust
/// use pharmsol::nca::{auc_last, AUCMethod};
///
/// let times = vec![0.0, 1.0, 2.0, 4.0, 8.0];
/// let concs = vec![0.0, 10.0, 8.0, 4.0, 1.0];
///
/// let auc = auc_last(&times, &concs, AUCMethod::LinUpLogDown);
/// println!("AUClast = {:.2}", auc);
/// ```
pub fn auc_last(times: &[f64], concentrations: &[f64], method: AUCMethod) -> f64 {
    assert_eq!(
        times.len(),
        concentrations.len(),
        "times and concentrations must have the same length"
    );

    if times.len() < 2 {
        return 0.0;
    }

    // Find tlast (last index where concentration > 0)
    let tlast_idx = concentrations
        .iter()
        .rposition(|&c| c > 0.0)
        .unwrap_or(times.len() - 1);

    let mut auc = 0.0;

    for i in 1..=tlast_idx {
        auc += auc_segment(
            times[i - 1],
            concentrations[i - 1],
            times[i],
            concentrations[i],
            method,
        );
    }

    auc
}

/// Calculate AUC including all data points (AUCall)
///
/// Unlike [`auc_last`], this includes integration to BLQ (below limit of quantification)
/// values after Tlast, using linear interpolation to zero.
///
/// # Arguments
///
/// * `times` - Time points (must be sorted in ascending order)
/// * `concentrations` - Concentration values at each time point
/// * `method` - Integration method
///
/// # Returns
///
/// The cumulative AUC including the segment from Tlast to the first BLQ after Tlast.
pub fn auc_all(times: &[f64], concentrations: &[f64], method: AUCMethod) -> f64 {
    assert_eq!(
        times.len(),
        concentrations.len(),
        "times and concentrations must have the same length"
    );

    if times.len() < 2 {
        return 0.0;
    }

    // Find tlast (last index where concentration > 0)
    let tlast_idx = concentrations
        .iter()
        .rposition(|&c| c > 0.0)
        .unwrap_or(times.len() - 1);

    let mut auc = 0.0;

    // Calculate AUC to tlast
    for i in 1..=tlast_idx {
        auc += auc_segment(
            times[i - 1],
            concentrations[i - 1],
            times[i],
            concentrations[i],
            method,
        );
    }

    // Add segment from tlast to first BLQ after tlast (linear to 0)
    if tlast_idx < times.len() - 1 {
        let first_blq_idx = tlast_idx + 1;
        // Linear interpolation from Clast to 0
        auc += auc_segment(
            times[tlast_idx],
            concentrations[tlast_idx],
            times[first_blq_idx],
            0.0,
            AUCMethod::Linear,
        );
    }

    auc
}

/// Calculate AUC over a specific interval
///
/// Calculates AUC from `start_time` to `end_time`. If the exact start or end times
/// are not in the data, linear interpolation is used.
///
/// This is particularly useful for:
/// - Partial AUC calculations (AUC0-4h, AUC4-12h, etc.)
/// - Dosing interval AUC (AUCτ) for steady-state calculations
/// - PMcore's bestdose interval AUC calculations
///
/// # Arguments
///
/// * `times` - Time points (must be sorted in ascending order)
/// * `concentrations` - Concentration values at each time point
/// * `start_time` - Start of the interval
/// * `end_time` - End of the interval
/// * `method` - Integration method
///
/// # Returns
///
/// The AUC over the specified interval. Returns 0.0 if the interval is invalid.
///
/// # Examples
///
/// ```rust
/// use pharmsol::nca::{auc_interval, AUCMethod};
///
/// let times = vec![0.0, 1.0, 2.0, 4.0, 8.0, 12.0];
/// let concs = vec![0.0, 10.0, 8.0, 4.0, 2.0, 1.0];
///
/// // AUC from 0 to 4 hours
/// let auc_0_4 = auc_interval(&times, &concs, 0.0, 4.0, AUCMethod::LinUpLogDown);
///
/// // AUC from 4 to 12 hours (dosing interval)
/// let auc_4_12 = auc_interval(&times, &concs, 4.0, 12.0, AUCMethod::LinUpLogDown);
/// ```
pub fn auc_interval(
    times: &[f64],
    concentrations: &[f64],
    start_time: f64,
    end_time: f64,
    method: AUCMethod,
) -> f64 {
    assert_eq!(
        times.len(),
        concentrations.len(),
        "times and concentrations must have the same length"
    );

    if times.len() < 2 || end_time <= start_time {
        return 0.0;
    }

    // Find the range of indices that overlap with [start_time, end_time]
    let first_idx = times.iter().position(|&t| t >= start_time).unwrap_or(0);
    let last_idx = times
        .iter()
        .rposition(|&t| t <= end_time)
        .unwrap_or(times.len() - 1);

    if first_idx > last_idx || first_idx >= times.len() {
        return 0.0;
    }

    let mut auc = 0.0;

    // Handle start interpolation if start_time is before the first point in range
    if first_idx > 0 && times[first_idx] > start_time {
        let c_start = interpolate_concentration(times, concentrations, start_time);
        auc += auc_segment(
            start_time,
            c_start,
            times[first_idx],
            concentrations[first_idx],
            method,
        );
    } else if first_idx == 0 && times[0] > start_time {
        // If start_time is before all data, start from first point
        // (can't extrapolate backwards)
    }

    // Calculate AUC for complete segments within the interval
    let effective_start = first_idx.max(1);
    for i in effective_start..=last_idx {
        let t1 = times[i - 1].max(start_time);
        let t2 = times[i].min(end_time);

        if t2 > t1 {
            let c1 = if times[i - 1] < start_time {
                interpolate_concentration(times, concentrations, start_time)
            } else {
                concentrations[i - 1]
            };

            let c2 = if times[i] > end_time {
                interpolate_concentration(times, concentrations, end_time)
            } else {
                concentrations[i]
            };

            auc += auc_segment(t1, c1, t2, c2, method);
        }
    }

    // Handle end interpolation if end_time is after the last point in range
    if last_idx < times.len() - 1 && times[last_idx] < end_time {
        let c_end = interpolate_concentration(times, concentrations, end_time);
        auc += auc_segment(
            times[last_idx],
            concentrations[last_idx],
            end_time,
            c_end,
            method,
        );
    }

    auc
}

/// Calculate cumulative AUC at each time point
///
/// Returns a vector of cumulative AUC values, one for each time point.
/// This is useful for PMcore's bestdose when predictions at specific times are needed.
///
/// # Arguments
///
/// * `times` - Time points (must be sorted in ascending order)
/// * `concentrations` - Concentration values at each time point
/// * `method` - Integration method
///
/// # Returns
///
/// Vector of cumulative AUC values at each time point. The first value is always 0.0.
///
/// # Examples
///
/// ```rust
/// use pharmsol::nca::{auc_cumulative, AUCMethod};
///
/// let times = vec![0.0, 1.0, 2.0, 4.0];
/// let concs = vec![0.0, 10.0, 8.0, 4.0];
///
/// let cumulative = auc_cumulative(&times, &concs, AUCMethod::Linear);
/// // cumulative = [0.0, 5.0, 14.0, 26.0]
/// ```
pub fn auc_cumulative(times: &[f64], concentrations: &[f64], method: AUCMethod) -> Vec<f64> {
    assert_eq!(
        times.len(),
        concentrations.len(),
        "times and concentrations must have the same length"
    );

    let mut cumulative = Vec::with_capacity(times.len());
    let mut auc = 0.0;

    cumulative.push(0.0);

    for i in 1..times.len() {
        auc += auc_segment(
            times[i - 1],
            concentrations[i - 1],
            times[i],
            concentrations[i],
            method,
        );
        cumulative.push(auc);
    }

    cumulative
}

/// Calculate cumulative AUC at specific target times
///
/// This is the function designed for PMcore's bestdose integration.
/// It calculates cumulative AUC from the first time point and extracts
/// values at the specified target times.
///
/// # Arguments
///
/// * `dense_times` - Dense time grid (must include or span all `target_times`)
/// * `dense_concentrations` - Concentration predictions at `dense_times`
/// * `target_times` - Times where AUC values should be extracted
/// * `method` - Integration method
///
/// # Returns
///
/// Vector of AUC values at `target_times`.
///
/// # Examples
///
/// ```rust
/// use pharmsol::nca::{auc_at_times, AUCMethod};
///
/// // Dense grid from simulation
/// let dense_times: Vec<f64> = (0..=48).map(|x| x as f64 * 0.5).collect();
/// let dense_concs: Vec<f64> = dense_times.iter().map(|&t| 10.0 * (-0.1 * t).exp()).collect();
///
/// // Extract AUC at specific observation times
/// let target_times = vec![12.0, 24.0];
/// let aucs = auc_at_times(&dense_times, &dense_concs, &target_times, AUCMethod::LinUpLogDown);
/// ```
pub fn auc_at_times(
    dense_times: &[f64],
    dense_concentrations: &[f64],
    target_times: &[f64],
    method: AUCMethod,
) -> Vec<f64> {
    assert_eq!(
        dense_times.len(),
        dense_concentrations.len(),
        "dense_times and dense_concentrations must have the same length"
    );

    let tolerance = 1e-10;
    let mut target_aucs = Vec::with_capacity(target_times.len());
    let mut auc = 0.0;
    let mut target_idx = 0;

    for i in 1..dense_times.len() {
        // Update cumulative AUC
        auc += auc_segment(
            dense_times[i - 1],
            dense_concentrations[i - 1],
            dense_times[i],
            dense_concentrations[i],
            method,
        );

        // Check if current time matches next target time
        while target_idx < target_times.len()
            && (dense_times[i] - target_times[target_idx]).abs() < tolerance
        {
            target_aucs.push(auc);
            target_idx += 1;
        }
    }

    // Fill any remaining targets that might be at the last time point
    while target_idx < target_times.len() {
        target_aucs.push(auc);
        target_idx += 1;
    }

    target_aucs
}

/// Linear interpolation of concentration at a given time
///
/// # Arguments
///
/// * `times` - Time points (must be sorted)
/// * `concentrations` - Concentration values
/// * `target_time` - Time at which to interpolate
///
/// # Returns
///
/// Interpolated concentration value
fn interpolate_concentration(times: &[f64], concentrations: &[f64], target_time: f64) -> f64 {
    if times.is_empty() {
        return 0.0;
    }

    // Handle edge cases
    if target_time <= times[0] {
        return concentrations[0];
    }
    if target_time >= times[times.len() - 1] {
        return concentrations[times.len() - 1];
    }

    // Find bracketing indices
    let upper_idx = times.iter().position(|&t| t >= target_time).unwrap_or(1);
    let lower_idx = upper_idx.saturating_sub(1);

    let t1 = times[lower_idx];
    let t2 = times[upper_idx];
    let c1 = concentrations[lower_idx];
    let c2 = concentrations[upper_idx];

    // Linear interpolation
    if (t2 - t1).abs() < 1e-10 {
        c1
    } else {
        c1 + (c2 - c1) * (target_time - t1) / (t2 - t1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_auc_segment_linear() {
        // Simple rectangle-ish trapezoid
        let auc = auc_segment(0.0, 10.0, 1.0, 10.0, AUCMethod::Linear);
        assert!((auc - 10.0).abs() < 1e-10);

        // Typical trapezoid
        let auc = auc_segment(0.0, 10.0, 1.0, 8.0, AUCMethod::Linear);
        assert!((auc - 9.0).abs() < 1e-10); // (10 + 8) / 2 * 1
    }

    #[test]
    fn test_auc_segment_lin_up_log_down() {
        // Ascending - should use linear
        let auc_up = auc_segment(0.0, 5.0, 1.0, 10.0, AUCMethod::LinUpLogDown);
        let auc_lin = auc_segment(0.0, 5.0, 1.0, 10.0, AUCMethod::Linear);
        assert!((auc_up - auc_lin).abs() < 1e-10);

        // Descending - should use log-linear
        let auc_down = auc_segment(0.0, 10.0, 1.0, 5.0, AUCMethod::LinUpLogDown);
        // Log-linear: (10 - 5) * 1 / ln(10/5) = 5 / ln(2) ≈ 7.21
        let expected = 5.0 / (10.0_f64 / 5.0).ln();
        assert!((auc_down - expected).abs() < 1e-10);
    }

    #[test]
    fn test_auc_segment_edge_cases() {
        // Zero time interval
        let auc = auc_segment(1.0, 10.0, 1.0, 8.0, AUCMethod::Linear);
        assert!((auc - 0.0).abs() < 1e-10);

        // Negative time interval
        let auc = auc_segment(2.0, 10.0, 1.0, 8.0, AUCMethod::Linear);
        assert!((auc - 0.0).abs() < 1e-10);

        // Zero concentrations
        let auc = auc_segment(0.0, 0.0, 1.0, 0.0, AUCMethod::LinUpLogDown);
        assert!((auc - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_auc_last() {
        let times = vec![0.0, 1.0, 2.0, 4.0, 8.0, 12.0];
        let concs = vec![0.0, 10.0, 8.0, 4.0, 1.0, 0.0]; // Last non-zero at t=8

        let auc = auc_last(&times, &concs, AUCMethod::Linear);

        // Manual calculation:
        // 0-1: (0 + 10) / 2 * 1 = 5
        // 1-2: (10 + 8) / 2 * 1 = 9
        // 2-4: (8 + 4) / 2 * 2 = 12
        // 4-8: (4 + 1) / 2 * 4 = 10
        // Total = 36
        assert!((auc - 36.0).abs() < 1e-10);
    }

    #[test]
    fn test_auc_cumulative() {
        let times = vec![0.0, 1.0, 2.0];
        let concs = vec![0.0, 10.0, 10.0];

        let cumulative = auc_cumulative(&times, &concs, AUCMethod::Linear);

        assert_eq!(cumulative.len(), 3);
        assert!((cumulative[0] - 0.0).abs() < 1e-10);
        assert!((cumulative[1] - 5.0).abs() < 1e-10); // (0 + 10) / 2 * 1
        assert!((cumulative[2] - 15.0).abs() < 1e-10); // 5 + (10 + 10) / 2 * 1
    }

    #[test]
    fn test_auc_interval() {
        let times = vec![0.0, 1.0, 2.0, 4.0, 8.0];
        let concs = vec![0.0, 10.0, 8.0, 4.0, 2.0];

        // Full interval
        let auc_full = auc_interval(&times, &concs, 0.0, 8.0, AUCMethod::Linear);
        let auc_last_val = auc_last(&times, &concs, AUCMethod::Linear);
        assert!((auc_full - auc_last_val).abs() < 1e-10);

        // Partial interval 1-4
        let auc_partial = auc_interval(&times, &concs, 1.0, 4.0, AUCMethod::Linear);
        // 1-2: (10 + 8) / 2 * 1 = 9
        // 2-4: (8 + 4) / 2 * 2 = 12
        // Total = 21
        assert!((auc_partial - 21.0).abs() < 1e-10);
    }
}
