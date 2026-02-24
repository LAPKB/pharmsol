//! Pure AUC (Area Under the Curve) calculation primitives
//!
//! This module provides standalone functions for computing AUC, AUMC, and related
//! quantities on raw `&[f64]` slices. These are the building blocks used by
//! [`ObservationProfile`](crate::data::observation::ObservationProfile), NCA analysis,
//! and any downstream code (e.g., PMcore best-dose) that needs trapezoidal integration.
//!
//! # Design
//!
//! All functions in this module are **pure math** — no dependency on data structures,
//! no BLQ filtering, no error types beyond what the caller can check. They accept
//! raw slices and an [`AUCMethod`], and return `f64`.
//!
//! # Example
//!
//! ```rust
//! use pharmsol::data::auc::{auc, auc_interval, aumc, interpolate_linear};
//! use pharmsol::prelude::AUCMethod;
//!
//! let times = [0.0, 1.0, 2.0, 4.0, 8.0];
//! let concs = [0.0, 10.0, 8.0, 4.0, 2.0];
//!
//! let total = auc(&times, &concs, &AUCMethod::Linear).unwrap();
//! let partial = auc_interval(&times, &concs, 1.0, 4.0, &AUCMethod::Linear).unwrap();
//! let moment = aumc(&times, &concs, &AUCMethod::Linear).unwrap();
//! let c_at_3 = interpolate_linear(&times, &concs, 3.0);
//! ```

use crate::data::event::AUCMethod;
use crate::data::observation_error::ObservationError;

// ============================================================================
// Segment-level helpers (private)
// ============================================================================

/// Check if log-linear method should be used for this segment
#[inline]
fn use_log_linear(c1: f64, c2: f64) -> bool {
    c2 < c1 && c1 > 0.0 && c2 > 0.0 && ((c1 / c2) - 1.0).abs() >= 1e-10
}

/// Linear trapezoidal AUC for a single segment
#[inline]
fn auc_linear(c1: f64, c2: f64, dt: f64) -> f64 {
    (c1 + c2) / 2.0 * dt
}

/// Log-linear AUC for a single segment (assumes c1 > c2 > 0)
#[inline]
fn auc_log(c1: f64, c2: f64, dt: f64) -> f64 {
    (c1 - c2) * dt / (c1 / c2).ln()
}

/// Linear trapezoidal AUMC for a single segment
#[inline]
fn aumc_linear(t1: f64, c1: f64, t2: f64, c2: f64, dt: f64) -> f64 {
    (t1 * c1 + t2 * c2) / 2.0 * dt
}

/// Log-linear AUMC for a single segment (PKNCA formula)
#[inline]
fn aumc_log(t1: f64, c1: f64, t2: f64, c2: f64, dt: f64) -> f64 {
    let k = (c1 / c2).ln() / dt;
    (t1 * c1 - t2 * c2) / k + (c1 - c2) / (k * k)
}

// ============================================================================
// Public segment functions
// ============================================================================

/// Calculate AUC for a single segment between two time points
///
/// For [`AUCMethod::LinLog`], this falls back to linear because segment-level
/// calculation cannot know Tmax context. Use [`auc`] or
/// [`auc_segment_with_tmax`] for proper LinLog handling.
#[inline]
pub fn auc_segment(
    t1: f64,
    c1: f64,
    t2: f64,
    c2: f64,
    method: &AUCMethod,
) -> Result<f64, ObservationError> {
    let dt = t2 - t1;
    if dt <= 0.0 {
        return Err(ObservationError::InvalidTimeSequence);
    }

    Ok(match method {
        AUCMethod::Linear | AUCMethod::LinLog => auc_linear(c1, c2, dt),
        AUCMethod::LinUpLogDown => {
            if use_log_linear(c1, c2) {
                auc_log(c1, c2, dt)
            } else {
                auc_linear(c1, c2, dt)
            }
        }
    })
}

/// Calculate AUC for a segment with Tmax context (for LinLog method)
///
/// This is the fully-aware version: for `LinLog`, it uses linear trapezoidal
/// before/at Tmax, and log-linear for descending portions after Tmax.
#[inline]
pub fn auc_segment_with_tmax(
    t1: f64,
    c1: f64,
    t2: f64,
    c2: f64,
    tmax: f64,
    method: &AUCMethod,
) -> Result<f64, ObservationError> {
    let dt = t2 - t1;
    if dt <= 0.0 {
        return Err(ObservationError::InvalidTimeSequence);
    }

    Ok(match method {
        AUCMethod::Linear => auc_linear(c1, c2, dt),
        AUCMethod::LinUpLogDown => {
            if use_log_linear(c1, c2) {
                auc_log(c1, c2, dt)
            } else {
                auc_linear(c1, c2, dt)
            }
        }
        AUCMethod::LinLog => {
            if t2 <= tmax || !use_log_linear(c1, c2) {
                auc_linear(c1, c2, dt)
            } else {
                auc_log(c1, c2, dt)
            }
        }
    })
}

/// Calculate AUMC for a segment with Tmax context (for LinLog method)
#[inline]
pub fn aumc_segment_with_tmax(
    t1: f64,
    c1: f64,
    t2: f64,
    c2: f64,
    tmax: f64,
    method: &AUCMethod,
) -> Result<f64, ObservationError> {
    let dt = t2 - t1;
    if dt <= 0.0 {
        return Err(ObservationError::InvalidTimeSequence);
    }

    Ok(match method {
        AUCMethod::Linear => aumc_linear(t1, c1, t2, c2, dt),
        AUCMethod::LinUpLogDown => {
            if use_log_linear(c1, c2) {
                aumc_log(t1, c1, t2, c2, dt)
            } else {
                aumc_linear(t1, c1, t2, c2, dt)
            }
        }
        AUCMethod::LinLog => {
            if t2 <= tmax || !use_log_linear(c1, c2) {
                aumc_linear(t1, c1, t2, c2, dt)
            } else {
                aumc_log(t1, c1, t2, c2, dt)
            }
        }
    })
}

// ============================================================================
// Full-profile functions (public API)
// ============================================================================

/// Calculate AUC (Area Under the Curve) over an entire profile
///
/// Computes ∫ C(t) dt from the first to the last time point using the
/// specified trapezoidal method. Tmax is auto-detected for `LinLog`.
///
/// # Arguments
/// * `times` - Sorted time points
/// * `values` - Concentration values (parallel to `times`)
/// * `method` - Trapezoidal rule variant
///
/// # Panics
/// Panics if `times.len() != values.len()`.
///
/// # Example
/// ```rust
/// use pharmsol::data::auc::auc;
/// use pharmsol::prelude::AUCMethod;
///
/// let times = [0.0, 1.0, 2.0, 4.0];
/// let concs = [0.0, 10.0, 8.0, 4.0];
/// let result = auc(&times, &concs, &AUCMethod::Linear).unwrap();
/// // (0+10)/2*1 + (10+8)/2*1 + (8+4)/2*2 = 5 + 9 + 12 = 26
/// assert!((result - 26.0).abs() < 1e-10);
/// ```
pub fn auc(times: &[f64], values: &[f64], method: &AUCMethod) -> Result<f64, ObservationError> {
    assert_eq!(
        times.len(),
        values.len(),
        "times and values must have equal length"
    );

    if times.len() < 2 {
        return Ok(0.0);
    }

    // Auto-detect tmax for LinLog
    let tmax = tmax_from_arrays(times, values);

    let mut total = 0.0;
    for i in 1..times.len() {
        total += auc_segment_with_tmax(
            times[i - 1],
            values[i - 1],
            times[i],
            values[i],
            tmax,
            method,
        )?;
    }
    Ok(total)
}

/// Calculate partial AUC over a specific time interval
///
/// Computes ∫ C(t) dt from `start` to `end`, using linear interpolation
/// at interval boundaries if they don't coincide with data points.
///
/// # Arguments
/// * `times` - Sorted time points
/// * `values` - Concentration values (parallel to `times`)
/// * `start` - Start time of interval
/// * `end` - End time of interval
/// * `method` - Trapezoidal rule variant
///
/// # Example
/// ```rust
/// use pharmsol::data::auc::auc_interval;
/// use pharmsol::prelude::AUCMethod;
///
/// let times = [0.0, 1.0, 2.0, 4.0, 8.0];
/// let concs = [0.0, 10.0, 8.0, 4.0, 2.0];
/// let partial = auc_interval(&times, &concs, 1.0, 4.0, &AUCMethod::Linear).unwrap();
/// // (10+8)/2*1 + (8+4)/2*2 = 9 + 12 = 21
/// assert!((partial - 21.0).abs() < 1e-10);
/// ```
pub fn auc_interval(
    times: &[f64],
    values: &[f64],
    start: f64,
    end: f64,
    method: &AUCMethod,
) -> Result<f64, ObservationError> {
    assert_eq!(
        times.len(),
        values.len(),
        "times and values must have equal length"
    );

    if end <= start || times.len() < 2 {
        return Ok(0.0);
    }

    // Auto-detect tmax for LinLog (same as auc())
    let tmax = tmax_from_arrays(times, values);

    let mut total = 0.0;

    for i in 1..times.len() {
        let t1 = times[i - 1];
        let t2 = times[i];

        // Skip segments entirely outside the interval
        if t2 <= start || t1 >= end {
            continue;
        }

        let seg_start = t1.max(start);
        let seg_end = t2.min(end);

        let c1 = if t1 < start {
            interpolate_linear(times, values, start)
        } else {
            values[i - 1]
        };

        let c2 = if t2 > end {
            interpolate_linear(times, values, end)
        } else {
            values[i]
        };

        total += auc_segment_with_tmax(seg_start, c1, seg_end, c2, tmax, method)?;
    }

    Ok(total)
}

/// Calculate AUMC (Area Under the first Moment Curve) over an entire profile
///
/// Computes ∫ t·C(t) dt from the first to the last time point.
/// Used for Mean Residence Time calculation: MRT = AUMC / AUC.
///
/// # Arguments
/// * `times` - Sorted time points
/// * `values` - Concentration values (parallel to `times`)
/// * `method` - Trapezoidal rule variant
pub fn aumc(times: &[f64], values: &[f64], method: &AUCMethod) -> Result<f64, ObservationError> {
    assert_eq!(
        times.len(),
        values.len(),
        "times and values must have equal length"
    );

    if times.len() < 2 {
        return Ok(0.0);
    }

    let tmax = tmax_from_arrays(times, values);

    let mut total = 0.0;
    for i in 1..times.len() {
        total += aumc_segment_with_tmax(
            times[i - 1],
            values[i - 1],
            times[i],
            values[i],
            tmax,
            method,
        )?;
    }
    Ok(total)
}

/// Linear interpolation of a value at a given time
///
/// Returns the linearly interpolated concentration at `time`.
/// Clamps to the first or last value if `time` is outside the data range.
///
/// # Arguments
/// * `times` - Sorted time points
/// * `values` - Values (parallel to `times`)
/// * `time` - Time at which to interpolate
///
/// # Example
/// ```rust
/// use pharmsol::data::auc::interpolate_linear;
///
/// let times = [0.0, 2.0, 4.0];
/// let values = [0.0, 10.0, 6.0];
/// assert!((interpolate_linear(&times, &values, 1.0) - 5.0).abs() < 1e-10);
/// assert!((interpolate_linear(&times, &values, 3.0) - 8.0).abs() < 1e-10);
/// ```
pub fn interpolate_linear(times: &[f64], values: &[f64], time: f64) -> f64 {
    assert_eq!(
        times.len(),
        values.len(),
        "times and values must have equal length"
    );

    if times.is_empty() {
        return 0.0;
    }

    if time <= times[0] {
        return values[0];
    }

    let last = times.len() - 1;
    if time >= times[last] {
        return values[last];
    }

    let upper_idx = times.iter().position(|&t| t >= time).unwrap_or(last);
    let lower_idx = upper_idx.saturating_sub(1);

    let t1 = times[lower_idx];
    let t2 = times[upper_idx];
    let v1 = values[lower_idx];
    let v2 = values[upper_idx];

    if (t2 - t1).abs() < 1e-10 {
        v1
    } else {
        v1 + (v2 - v1) * (time - t1) / (t2 - t1)
    }
}

// ============================================================================
// Internal helpers
// ============================================================================

/// Find tmax (time of maximum value) from parallel arrays
fn tmax_from_arrays(times: &[f64], values: &[f64]) -> f64 {
    values
        .iter()
        .enumerate()
        .fold((0, f64::NEG_INFINITY), |(max_i, max_v), (i, &v)| {
            if v > max_v {
                (i, v)
            } else {
                (max_i, max_v)
            }
        })
        .0
        .min(times.len() - 1)
        .pipe(|idx| times[idx])
}

/// Helper trait for pipe syntax
trait Pipe: Sized {
    fn pipe<R>(self, f: impl FnOnce(Self) -> R) -> R {
        f(self)
    }
}
impl<T> Pipe for T {}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::observation_error::ObservationError;

    #[test]
    fn test_auc_segment_linear() {
        let result = auc_segment(0.0, 10.0, 1.0, 8.0, &AUCMethod::Linear).unwrap();
        assert!((result - 9.0).abs() < 1e-10); // (10 + 8) / 2 * 1
    }

    #[test]
    fn test_auc_segment_log_down() {
        let result = auc_segment(0.0, 10.0, 1.0, 5.0, &AUCMethod::LinUpLogDown).unwrap();
        let expected = 5.0 / (10.0_f64 / 5.0).ln();
        assert!((result - expected).abs() < 1e-10);
    }

    #[test]
    fn test_auc_segment_ascending_linuplogdown() {
        // Ascending — should use linear even with LinUpLogDown
        let result = auc_segment(0.0, 5.0, 1.0, 10.0, &AUCMethod::LinUpLogDown).unwrap();
        let expected = (5.0 + 10.0) / 2.0 * 1.0;
        assert!((result - expected).abs() < 1e-10);
    }

    #[test]
    fn test_auc_segment_zero_dt() {
        let result = auc_segment(1.0, 10.0, 1.0, 8.0, &AUCMethod::Linear);
        assert!(matches!(result, Err(ObservationError::InvalidTimeSequence)));
    }

    #[test]
    fn test_auc_full_profile_linear() {
        let times = [0.0, 1.0, 2.0, 4.0, 8.0, 12.0];
        let concs = [0.0, 10.0, 8.0, 4.0, 2.0, 1.0];

        let result = auc(&times, &concs, &AUCMethod::Linear).unwrap();
        // Manual calculation:
        // 0-1: (0 + 10) / 2 * 1 = 5
        // 1-2: (10 + 8) / 2 * 1 = 9
        // 2-4: (8 + 4) / 2 * 2 = 12
        // 4-8: (4 + 2) / 2 * 4 = 12
        // 8-12: (2 + 1) / 2 * 4 = 6
        // Total = 44
        assert!((result - 44.0).abs() < 1e-10);
    }

    #[test]
    fn test_auc_single_point() {
        let times = [1.0];
        let concs = [10.0];
        assert_eq!(auc(&times, &concs, &AUCMethod::Linear).unwrap(), 0.0);
    }

    #[test]
    fn test_auc_empty() {
        let times: [f64; 0] = [];
        let concs: [f64; 0] = [];
        assert_eq!(auc(&times, &concs, &AUCMethod::Linear).unwrap(), 0.0);
    }

    #[test]
    fn test_auc_interval_exact_boundaries() {
        let times = [0.0, 1.0, 2.0, 4.0, 8.0];
        let concs = [0.0, 10.0, 8.0, 4.0, 2.0];

        let result = auc_interval(&times, &concs, 1.0, 4.0, &AUCMethod::Linear).unwrap();
        // 1-2: (10+8)/2*1 = 9
        // 2-4: (8+4)/2*2 = 12
        // Total = 21
        assert!((result - 21.0).abs() < 1e-10);
    }

    #[test]
    fn test_auc_interval_interpolated_boundaries() {
        let times = [0.0, 2.0, 4.0];
        let concs = [0.0, 10.0, 6.0];

        // Interval [1, 3] requires interpolation at both boundaries
        let result = auc_interval(&times, &concs, 1.0, 3.0, &AUCMethod::Linear).unwrap();
        // C(1) = interpolate(0,0, 2,10, t=1) = 5.0
        // C(3) = interpolate(2,10, 4,6, t=3) = 8.0
        // AUC from 1-2: (5+10)/2*1 = 7.5
        // AUC from 2-3: (10+8)/2*1 = 9.0
        // Total = 16.5
        assert!((result - 16.5).abs() < 1e-10);
    }

    #[test]
    fn test_auc_interval_outside_range() {
        let times = [1.0, 2.0, 4.0];
        let concs = [10.0, 8.0, 4.0];

        // Entirely before data
        assert_eq!(
            auc_interval(&times, &concs, 0.0, 0.5, &AUCMethod::Linear).unwrap(),
            0.0
        );
        // Entirely after data
        assert_eq!(
            auc_interval(&times, &concs, 5.0, 10.0, &AUCMethod::Linear).unwrap(),
            0.0
        );
    }

    #[test]
    fn test_auc_interval_reversed() {
        let times = [0.0, 1.0, 2.0];
        let concs = [0.0, 10.0, 8.0];
        // end <= start should return 0
        assert_eq!(
            auc_interval(&times, &concs, 2.0, 1.0, &AUCMethod::Linear).unwrap(),
            0.0
        );
    }

    #[test]
    fn test_aumc_linear() {
        let times = [0.0, 1.0, 2.0];
        let concs = [0.0, 10.0, 8.0];

        let result = aumc(&times, &concs, &AUCMethod::Linear).unwrap();
        // Segment 0-1: (0*0 + 1*10)/2 * 1 = 5
        // Segment 1-2: (1*10 + 2*8)/2 * 1 = 13
        // Total = 18
        assert!((result - 18.0).abs() < 1e-10);
    }

    #[test]
    fn test_interpolate_linear_within() {
        let times = [0.0, 2.0, 4.0];
        let values = [0.0, 10.0, 6.0];

        assert!((interpolate_linear(&times, &values, 1.0) - 5.0).abs() < 1e-10);
        assert!((interpolate_linear(&times, &values, 3.0) - 8.0).abs() < 1e-10);
    }

    #[test]
    fn test_interpolate_linear_at_boundary() {
        let times = [0.0, 2.0, 4.0];
        let values = [0.0, 10.0, 6.0];

        assert!((interpolate_linear(&times, &values, 0.0) - 0.0).abs() < 1e-10);
        assert!((interpolate_linear(&times, &values, 4.0) - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_interpolate_linear_clamped() {
        let times = [1.0, 3.0];
        let values = [5.0, 15.0];

        // Before first point — clamp to first value
        assert_eq!(interpolate_linear(&times, &values, 0.0), 5.0);
        // After last point — clamp to last value
        assert_eq!(interpolate_linear(&times, &values, 5.0), 15.0);
    }

    #[test]
    fn test_linlog_uses_linear_before_tmax() {
        // tmax at t=1, concs: [0, 10, 8, 4]

        // Before tmax: linear
        let seg_before =
            auc_segment_with_tmax(0.0, 0.0, 1.0, 10.0, 1.0, &AUCMethod::LinLog).unwrap();
        let expected_linear = (0.0 + 10.0) / 2.0 * 1.0;
        assert!((seg_before - expected_linear).abs() < 1e-10);

        // After tmax with descending: log-linear
        let seg_after =
            auc_segment_with_tmax(1.0, 10.0, 2.0, 8.0, 1.0, &AUCMethod::LinLog).unwrap();
        // Should NOT be simple linear
        let linear_val = (10.0 + 8.0) / 2.0 * 1.0;
        // LinLog after tmax with descending should differ
        // Actually for c1>c2>0, log gives different result
        let log_val = (10.0 - 8.0) * 1.0 / (10.0_f64 / 8.0).ln();
        assert!((seg_after - log_val).abs() < 1e-10);
        assert!((seg_after - linear_val).abs() > 1e-5);
    }

    #[test]
    fn test_auc_matches_known_values() {
        // Same profile used in nca::calc tests
        let times = [0.0, 1.0, 2.0, 4.0, 8.0, 12.0];
        let concs = [0.0, 10.0, 8.0, 4.0, 2.0, 1.0];

        let linear = auc(&times, &concs, &AUCMethod::Linear).unwrap();
        assert!((linear - 44.0).abs() < 1e-10);

        let linuplogdown = auc(&times, &concs, &AUCMethod::LinUpLogDown).unwrap();
        // LinUpLogDown should give a different (smaller) result for the descending part
        assert!(linuplogdown < linear);
        assert!(linuplogdown > 0.0);
    }

    #[test]
    fn test_tmax_from_arrays() {
        let times = [0.0, 1.0, 2.0, 4.0];
        let concs = [0.0, 10.0, 8.0, 4.0];
        assert_eq!(tmax_from_arrays(&times, &concs), 1.0);
    }

    #[test]
    fn test_tmax_from_arrays_first_occurrence() {
        // When max occurs at multiple points, should take first
        let times = [0.0, 1.0, 2.0, 3.0];
        let concs = [5.0, 10.0, 10.0, 5.0];
        assert_eq!(tmax_from_arrays(&times, &concs), 1.0);
    }
}
