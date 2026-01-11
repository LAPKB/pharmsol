//! Statistical distribution functions for likelihood calculations.
//!
//! This module provides numerically stable implementations of probability
//! distribution functions used in pharmacometric likelihood calculations.
//!
//! All functions operate in log-space for numerical stability.

use crate::ErrorModelError;
use statrs::distribution::{ContinuousCDF, Normal};

// ln(2π) = ln(2) + ln(π) ≈ 1.8378770664093453
pub(crate) const LOG_2PI: f64 = 1.8378770664093453_f64;

/// Log of the probability density function of the normal distribution.
///
/// This is numerically stable and avoids underflow for extreme values.
///
/// # Formula
/// ```text
/// log(φ(x; μ, σ)) = -0.5 * ln(2π) - ln(σ) - (x - μ)² / (2σ²)
/// ```
///
/// # Parameters
/// - `obs`: Observed value
/// - `pred`: Predicted value (mean)
/// - `sigma`: Standard deviation
///
/// # Returns
/// The log probability density
#[inline(always)]
pub fn lognormpdf(obs: f64, pred: f64, sigma: f64) -> f64 {
    let diff = obs - pred;
    -0.5 * LOG_2PI - sigma.ln() - (diff * diff) / (2.0 * sigma * sigma)
}

/// Log of the cumulative distribution function of the normal distribution.
///
/// Used for BLOQ (below limit of quantification) observations where the
/// likelihood is the probability of observing a value ≤ LOQ.
///
/// # Parameters
/// - `obs`: Observed value (typically the LOQ)
/// - `pred`: Predicted value (mean)
/// - `sigma`: Standard deviation
///
/// # Returns
/// The log of the CDF value, or an error if numerical issues occur
///
/// # Numerical Stability
/// For extremely small CDF values (z < -37), uses an asymptotic approximation
/// to avoid underflow to zero.
#[inline(always)]
pub fn lognormcdf(obs: f64, pred: f64, sigma: f64) -> Result<f64, ErrorModelError> {
    let norm = Normal::new(pred, sigma).map_err(|_| ErrorModelError::NegativeSigma)?;
    let cdf = norm.cdf(obs);
    if cdf <= 0.0 {
        // For extremely small CDF values, use an approximation
        // log(Φ(x)) ≈ log(φ(x)) - log(-x) for large negative x
        // where x = (obs - pred) / sigma
        let z = (obs - pred) / sigma;
        if z < -37.0 {
            // Below this, cdf is essentially 0, use asymptotic approximation
            Ok(lognormpdf(obs, pred, sigma) - z.abs().ln())
        } else {
            Err(ErrorModelError::NegativeSigma) // Indicates numerical issue
        }
    } else {
        Ok(cdf.ln())
    }
}

/// Log of the survival function (1 - CDF) of the normal distribution.
///
/// Used for ALOQ (above limit of quantification) observations where the
/// likelihood is the probability of observing a value > LOQ.
///
/// # Parameters
/// - `obs`: Observed value (typically the LOQ)
/// - `pred`: Predicted value (mean)
/// - `sigma`: Standard deviation
///
/// # Returns
/// The log of the survival function value, or an error if numerical issues occur
///
/// # Numerical Stability
/// For extremely small survival function values (z > 37), uses an asymptotic
/// approximation to avoid underflow to zero.
#[inline(always)]
pub fn lognormccdf(obs: f64, pred: f64, sigma: f64) -> Result<f64, ErrorModelError> {
    let norm = Normal::new(pred, sigma).map_err(|_| ErrorModelError::NegativeSigma)?;
    let sf = 1.0 - norm.cdf(obs);
    if sf <= 0.0 {
        let z = (obs - pred) / sigma;
        if z > 37.0 {
            // Use asymptotic approximation for upper tail
            Ok(lognormpdf(obs, pred, sigma) - z.ln())
        } else {
            Err(ErrorModelError::NegativeSigma)
        }
    } else {
        Ok(sf.ln())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lognormpdf_standard_normal() {
        // At mean, log PDF should be -0.5 * ln(2π) - ln(σ)
        let log_pdf = lognormpdf(0.0, 0.0, 1.0);
        let expected = -0.5 * LOG_2PI;
        assert!(
            (log_pdf - expected).abs() < 1e-10,
            "lognormpdf at mean should be -0.5*ln(2π)"
        );
    }

    #[test]
    fn test_lognormpdf_matches_exp_pdf() {
        let obs = 1.5;
        let pred = 1.0;
        let sigma = 0.5;

        let log_pdf = lognormpdf(obs, pred, sigma);
        let pdf = log_pdf.exp();

        // Manual calculation
        let diff = obs - pred;
        let expected_pdf = (1.0 / (sigma * (2.0 * std::f64::consts::PI).sqrt()))
            * (-diff * diff / (2.0 * sigma * sigma)).exp();

        assert!(
            (pdf - expected_pdf).abs() < 1e-10,
            "exp(lognormpdf) should match manual PDF calculation"
        );
    }

    #[test]
    fn test_lognormcdf_basic() {
        // CDF at mean should be 0.5, so log should be ln(0.5)
        let log_cdf = lognormcdf(0.0, 0.0, 1.0).unwrap();
        let expected = 0.5_f64.ln();
        assert!(
            (log_cdf - expected).abs() < 1e-10,
            "lognormcdf at mean should be ln(0.5)"
        );
    }

    #[test]
    fn test_lognormccdf_basic() {
        // SF at mean should be 0.5, so log should be ln(0.5)
        let log_sf = lognormccdf(0.0, 0.0, 1.0).unwrap();
        let expected = 0.5_f64.ln();
        assert!(
            (log_sf - expected).abs() < 1e-10,
            "lognormccdf at mean should be ln(0.5)"
        );
    }

    #[test]
    fn test_lognormcdf_extreme() {
        // Very far in the tail - should still return finite value
        let result = lognormcdf(-40.0, 0.0, 1.0);
        assert!(result.is_ok(), "lognormcdf should handle extreme values");
        assert!(
            result.unwrap().is_finite(),
            "lognormcdf should return finite value"
        );
    }

    #[test]
    fn test_lognormccdf_extreme() {
        // Very far in the upper tail
        let result = lognormccdf(40.0, 0.0, 1.0);
        assert!(result.is_ok(), "lognormccdf should handle extreme values");
        assert!(
            result.unwrap().is_finite(),
            "lognormccdf should return finite value"
        );
    }
}
