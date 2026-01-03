//! Primary NCA parameters: Cmax, Tmax, Clast, Tlast
//!
//! This module provides functions for calculating the fundamental concentration
//! parameters from pharmacokinetic data.
//!
//! # Parameters
//!
//! | Parameter | Description |
//! |-----------|-------------|
//! | Cmax | Maximum observed concentration |
//! | Tmax | Time of first occurrence of Cmax |
//! | Clast | Last concentration above LOQ (or > 0) |
//! | Tlast | Time of Clast |
//! | Cmin | Minimum observed concentration (in interval) |
//! | Tmin | Time of minimum concentration |

/// Result of Cmax/Tmax calculation
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CmaxTmax {
    /// Maximum observed concentration
    pub cmax: f64,
    /// Time of maximum concentration
    pub tmax: f64,
    /// Index in the original arrays
    pub index: usize,
}

/// Result of Clast/Tlast calculation
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ClastTlast {
    /// Last measurable concentration (> 0 or > LOQ)
    pub clast: f64,
    /// Time of last measurable concentration
    pub tlast: f64,
    /// Index in the original arrays
    pub index: usize,
}

/// Result of Cmin/Tmin calculation
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CminTmin {
    /// Minimum observed concentration
    pub cmin: f64,
    /// Time of minimum concentration
    pub tmin: f64,
    /// Index in the original arrays
    pub index: usize,
}

/// Calculate Cmax and Tmax from concentration-time data
///
/// Returns the maximum concentration and the time at which it first occurs.
/// If there are multiple time points with the same maximum concentration,
/// the first occurrence is returned (controlled by `first_tmax` parameter).
///
/// # Arguments
///
/// * `times` - Time points (must be sorted in ascending order)
/// * `concentrations` - Concentration values at each time point
/// * `first_tmax` - If true, returns first occurrence of Cmax; if false, returns last
///
/// # Returns
///
/// `Some(CmaxTmax)` if data is valid, `None` if arrays are empty or different lengths.
///
/// # Examples
///
/// ```rust
/// use pharmsol::nca::cmax_tmax;
///
/// let times = vec![0.0, 1.0, 2.0, 4.0, 8.0];
/// let concs = vec![0.0, 8.5, 10.0, 6.0, 2.0];
///
/// let result = cmax_tmax(&times, &concs, true).unwrap();
/// assert_eq!(result.cmax, 10.0);
/// assert_eq!(result.tmax, 2.0);
/// ```
pub fn cmax_tmax(times: &[f64], concentrations: &[f64], first_tmax: bool) -> Option<CmaxTmax> {
    if times.is_empty() || times.len() != concentrations.len() {
        return None;
    }

    let (index, &cmax) = if first_tmax {
        // Find first occurrence of maximum
        let max_val = concentrations
            .iter()
            .copied()
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))?;
        concentrations
            .iter()
            .enumerate()
            .find(|(_, &c)| (c - max_val).abs() < 1e-10)?
    } else {
        // Find last occurrence of maximum
        let max_val = concentrations
            .iter()
            .copied()
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))?;
        concentrations
            .iter()
            .enumerate()
            .rev()
            .find(|(_, &c)| (c - max_val).abs() < 1e-10)?
    };

    Some(CmaxTmax {
        cmax,
        tmax: times[index],
        index,
    })
}

/// Calculate Clast and Tlast from concentration-time data
///
/// Returns the last concentration above zero (or a specified LOQ) and its time.
///
/// # Arguments
///
/// * `times` - Time points (must be sorted in ascending order)
/// * `concentrations` - Concentration values at each time point
/// * `loq` - Limit of quantification (default 0.0, meaning any positive value)
///
/// # Returns
///
/// `Some(ClastTlast)` if there is at least one concentration above LOQ, `None` otherwise.
///
/// # Examples
///
/// ```rust
/// use pharmsol::nca::clast_tlast;
///
/// let times = vec![0.0, 1.0, 2.0, 4.0, 8.0, 12.0];
/// let concs = vec![0.0, 10.0, 8.0, 4.0, 1.0, 0.0];
///
/// let result = clast_tlast(&times, &concs, 0.0).unwrap();
/// assert_eq!(result.clast, 1.0);
/// assert_eq!(result.tlast, 8.0);
/// ```
pub fn clast_tlast(times: &[f64], concentrations: &[f64], loq: f64) -> Option<ClastTlast> {
    if times.is_empty() || times.len() != concentrations.len() {
        return None;
    }

    // Find last index where concentration > loq
    let index = concentrations.iter().rposition(|&c| c > loq)?;

    Some(ClastTlast {
        clast: concentrations[index],
        tlast: times[index],
        index,
    })
}

/// Calculate Cmin and Tmin from concentration-time data
///
/// Returns the minimum concentration and the time at which it first occurs.
/// This is typically used for trough concentration analysis in multiple-dose studies.
///
/// # Arguments
///
/// * `times` - Time points (must be sorted in ascending order)
/// * `concentrations` - Concentration values at each time point
/// * `exclude_zero` - If true, excludes zero concentrations from consideration
///
/// # Returns
///
/// `Some(CminTmin)` if data is valid, `None` if arrays are empty or different lengths.
///
/// # Examples
///
/// ```rust
/// use pharmsol::nca::cmin_tmin;
///
/// let times = vec![0.0, 1.0, 2.0, 4.0, 8.0, 12.0];
/// let concs = vec![0.0, 10.0, 8.0, 4.0, 2.0, 1.0];
///
/// // Excluding zeros
/// let result = cmin_tmin(&times, &concs, true).unwrap();
/// assert_eq!(result.cmin, 1.0);
/// assert_eq!(result.tmin, 12.0);
///
/// // Including zeros
/// let result = cmin_tmin(&times, &concs, false).unwrap();
/// assert_eq!(result.cmin, 0.0);
/// assert_eq!(result.tmin, 0.0);
/// ```
pub fn cmin_tmin(times: &[f64], concentrations: &[f64], exclude_zero: bool) -> Option<CminTmin> {
    if times.is_empty() || times.len() != concentrations.len() {
        return None;
    }

    let (index, &cmin) = if exclude_zero {
        concentrations
            .iter()
            .enumerate()
            .filter(|(_, &c)| c > 0.0)
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))?
    } else {
        concentrations
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))?
    };

    Some(CminTmin {
        cmin,
        tmin: times[index],
        index,
    })
}

/// Calculate the first non-zero concentration and its time (Tfirst, Cfirst)
///
/// Useful for determining lag time or the start of absorption.
///
/// # Arguments
///
/// * `times` - Time points (must be sorted in ascending order)
/// * `concentrations` - Concentration values at each time point
///
/// # Returns
///
/// Tuple of (time, concentration) for the first positive concentration, or None if all are zero.
pub fn first_positive(times: &[f64], concentrations: &[f64]) -> Option<(f64, f64)> {
    if times.is_empty() || times.len() != concentrations.len() {
        return None;
    }

    for (i, &c) in concentrations.iter().enumerate() {
        if c > 0.0 {
            return Some((times[i], c));
        }
    }

    None
}

/// Calculate average concentration over an interval
///
/// Cav = AUC / (t_end - t_start)
///
/// # Arguments
///
/// * `auc` - AUC over the interval
/// * `t_start` - Start time of interval
/// * `t_end` - End time of interval
///
/// # Returns
///
/// Average concentration, or 0.0 if interval is invalid.
pub fn cav(auc: f64, t_start: f64, t_end: f64) -> f64 {
    let dt = t_end - t_start;
    if dt <= 0.0 {
        0.0
    } else {
        auc / dt
    }
}

/// Calculate peak-to-trough ratio (PTR)
///
/// PTR = Cmax / Cmin
///
/// # Arguments
///
/// * `cmax` - Maximum concentration
/// * `cmin` - Minimum concentration (must be > 0)
///
/// # Returns
///
/// Peak-to-trough ratio, or None if Cmin <= 0.
pub fn peak_trough_ratio(cmax: f64, cmin: f64) -> Option<f64> {
    if cmin <= 0.0 {
        None
    } else {
        Some(cmax / cmin)
    }
}

/// Calculate swing (fluctuation)
///
/// Swing = (Cmax - Cmin) / Cmin
///
/// # Arguments
///
/// * `cmax` - Maximum concentration
/// * `cmin` - Minimum concentration (must be > 0)
///
/// # Returns
///
/// Swing value, or None if Cmin <= 0.
pub fn swing(cmax: f64, cmin: f64) -> Option<f64> {
    if cmin <= 0.0 {
        None
    } else {
        Some((cmax - cmin) / cmin)
    }
}

/// Calculate percent fluctuation
///
/// %Fluctuation = (Cmax - Cmin) / Cav × 100
///
/// # Arguments
///
/// * `cmax` - Maximum concentration
/// * `cmin` - Minimum concentration
/// * `cav` - Average concentration (must be > 0)
///
/// # Returns
///
/// Percent fluctuation, or None if Cav <= 0.
pub fn percent_fluctuation(cmax: f64, cmin: f64, cav: f64) -> Option<f64> {
    if cav <= 0.0 {
        None
    } else {
        Some((cmax - cmin) / cav * 100.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cmax_tmax() {
        let times = vec![0.0, 1.0, 2.0, 4.0, 8.0];
        let concs = vec![0.0, 8.0, 10.0, 6.0, 2.0];

        let result = cmax_tmax(&times, &concs, true).unwrap();
        assert_eq!(result.cmax, 10.0);
        assert_eq!(result.tmax, 2.0);
        assert_eq!(result.index, 2);
    }

    #[test]
    fn test_cmax_tmax_multiple_max() {
        let times = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let concs = vec![0.0, 10.0, 8.0, 10.0, 5.0];

        // First occurrence
        let result = cmax_tmax(&times, &concs, true).unwrap();
        assert_eq!(result.tmax, 1.0);

        // Last occurrence
        let result = cmax_tmax(&times, &concs, false).unwrap();
        assert_eq!(result.tmax, 3.0);
    }

    #[test]
    fn test_clast_tlast() {
        let times = vec![0.0, 1.0, 2.0, 4.0, 8.0, 12.0];
        let concs = vec![0.0, 10.0, 8.0, 4.0, 1.0, 0.0];

        let result = clast_tlast(&times, &concs, 0.0).unwrap();
        assert_eq!(result.clast, 1.0);
        assert_eq!(result.tlast, 8.0);
        assert_eq!(result.index, 4);
    }

    #[test]
    fn test_clast_tlast_with_loq() {
        let times = vec![0.0, 1.0, 2.0, 4.0, 8.0, 12.0];
        let concs = vec![0.0, 10.0, 8.0, 4.0, 0.5, 0.0];

        // With LOQ = 1.0, should skip the 0.5 value
        let result = clast_tlast(&times, &concs, 1.0).unwrap();
        assert_eq!(result.clast, 4.0);
        assert_eq!(result.tlast, 4.0);
    }

    #[test]
    fn test_cmin_tmin() {
        let times = vec![0.0, 1.0, 2.0, 4.0, 8.0];
        let concs = vec![5.0, 10.0, 8.0, 4.0, 2.0];

        let result = cmin_tmin(&times, &concs, false).unwrap();
        assert_eq!(result.cmin, 2.0);
        assert_eq!(result.tmin, 8.0);
    }

    #[test]
    fn test_cmin_tmin_exclude_zero() {
        let times = vec![0.0, 1.0, 2.0, 4.0, 8.0];
        let concs = vec![0.0, 10.0, 8.0, 4.0, 2.0];

        // Excluding zero
        let result = cmin_tmin(&times, &concs, true).unwrap();
        assert_eq!(result.cmin, 2.0);

        // Including zero
        let result = cmin_tmin(&times, &concs, false).unwrap();
        assert_eq!(result.cmin, 0.0);
    }

    #[test]
    fn test_cav() {
        // AUC = 100, interval = 10 -> Cav = 10
        assert!((cav(100.0, 0.0, 10.0) - 10.0).abs() < 1e-10);

        // Invalid interval
        assert_eq!(cav(100.0, 10.0, 5.0), 0.0);
    }

    #[test]
    fn test_fluctuation_metrics() {
        let cmax = 10.0;
        let cmin = 2.0;
        let cav_val = 5.0;

        assert!((peak_trough_ratio(cmax, cmin).unwrap() - 5.0).abs() < 1e-10);
        assert!((swing(cmax, cmin).unwrap() - 4.0).abs() < 1e-10);
        assert!((percent_fluctuation(cmax, cmin, cav_val).unwrap() - 160.0).abs() < 1e-10);
    }
}
