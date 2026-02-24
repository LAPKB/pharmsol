//! Single-dose to steady-state prediction via superposition
//!
//! Given a single-dose concentration-time profile and a dosing interval (τ),
//! predict the steady-state profile by summing shifted copies of the single-dose
//! profile, using the terminal phase (λz) to extrapolate beyond the observed data.
//!
//! This is a standard NCA technique for dose selection and steady-state prediction
//! without requiring actual multiple-dose study data.
//!
//! # Usage
//!
//! The simplest way is via the [`Superposition`] trait on [`Subject`] or [`Occasion`]:
//!
//! ```rust,ignore
//! use pharmsol::prelude::*;
//! use pharmsol::nca::{NCAOptions, Superposition};
//!
//! // Full chain: NCA → λz → superposition
//! let result = subject.superposition(12.0, &NCAOptions::default(), None)?;
//! println!("Predicted Cmax_ss: {:.2}", result.cmax_ss);
//!
//! // If you already have an NCA result, skip the recomputation:
//! let nca = subject.nca(&NCAOptions::default())?;
//! let result = subject.superposition_from_nca(&nca, 12.0, None)?;
//! ```

use crate::data::event::BLQRule;
use crate::data::observation::ObservationProfile;
use crate::nca::error::NCAError;
use crate::nca::traits::NCA;
use crate::nca::types::{NCAOptions, NCAResult};
use crate::{Occasion, Subject};
use serde::{Deserialize, Serialize};

/// Result of a superposition prediction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SuperpositionResult {
    /// Time points at steady state (within one dosing interval)
    pub times: Vec<f64>,
    /// Predicted concentrations at steady state
    pub concentrations: Vec<f64>,
    /// Predicted Cmax at steady state
    pub cmax_ss: f64,
    /// Time of predicted Cmax at steady state
    pub tmax_ss: f64,
    /// Predicted Cmin at steady state (trough)
    pub cmin_ss: f64,
    /// Predicted AUC over one dosing interval at steady state
    pub auc_tau_ss: f64,
    /// Predicted average concentration
    pub cavg_ss: f64,
    /// Number of doses summed to reach steady state
    pub n_doses: usize,
    /// Predicted accumulation ratio (AUC_tau_ss / AUC_tau_single)
    pub accumulation_ratio: f64,
}

/// Predict steady-state concentrations by superposition of a single-dose profile
///
/// This is the core algorithm. It is **crate-internal** — callers should use
/// the [`Superposition`] trait on [`Subject`] or [`Occasion`] instead.
///
/// # Arguments
/// * `profile` - Single-dose observation profile
/// * `lambda_z` - Terminal elimination rate constant (from NCA)
/// * `tau` - Dosing interval
/// * `n_eval_points` - Number of evaluation points within [0, τ] (default: use observed times)
///
/// # Returns
/// `None` if `lambda_z` is not positive or profile is empty
pub(crate) fn predict(
    profile: &ObservationProfile,
    lambda_z: f64,
    tau: f64,
    n_eval_points: Option<usize>,
) -> Option<SuperpositionResult> {
    if lambda_z <= 0.0 || !lambda_z.is_finite() || tau <= 0.0 || profile.is_empty() {
        return None;
    }

    let clast = profile.clast();
    let tlast = profile.tlast();

    // Generate evaluation times within [0, tau]
    let eval_times: Vec<f64> = match n_eval_points {
        Some(n) if n >= 2 => (0..n).map(|i| i as f64 * tau / (n - 1) as f64).collect(),
        _ => {
            // Use observed times that fall within [0, tau], plus tau itself
            let mut times: Vec<f64> = profile
                .times
                .iter()
                .copied()
                .filter(|&t| t >= 0.0 && t <= tau)
                .collect();
            if times.is_empty() || (times.last().unwrap() - tau).abs() > 1e-10 {
                times.push(tau);
            }
            if times[0] > 0.0 {
                times.insert(0, 0.0);
            }
            times
        }
    };

    // Tolerance for convergence: stop when dose contribution < this fraction of current total
    let tolerance = 1e-10;
    let max_doses = 1000; // Safety limit

    let mut ss_concentrations = vec![0.0_f64; eval_times.len()];
    let mut n_doses = 0;

    for dose_k in 0..max_doses {
        let mut max_contribution = 0.0_f64;

        for (i, &t) in eval_times.iter().enumerate() {
            // Time since this dose: t + k * tau
            let t_since_dose = t + dose_k as f64 * tau;
            let conc = concentration_at_time(profile, clast, tlast, lambda_z, t_since_dose);
            ss_concentrations[i] += conc;
            max_contribution = max_contribution.max(conc);
        }

        n_doses = dose_k + 1;

        // Check convergence: if the maximum contribution from this dose is negligible
        if dose_k > 0
            && max_contribution
                < tolerance * ss_concentrations.iter().cloned().fold(0.0_f64, f64::max)
        {
            break;
        }
    }

    // Compute derived parameters
    let (cmax_idx, cmax_ss) = ss_concentrations
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, &v)| (i, v))
        .unwrap_or((0, 0.0));

    let tmax_ss = eval_times[cmax_idx];

    let cmin_ss = ss_concentrations
        .iter()
        .copied()
        .filter(|&c| c > 0.0)
        .min_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap_or(0.0);

    // AUC_tau using trapezoidal rule
    let auc_tau_ss = trapezoidal_auc(&eval_times, &ss_concentrations);

    let cavg_ss = if tau > 0.0 { auc_tau_ss / tau } else { 0.0 };

    // Single-dose AUC over tau for accumulation ratio
    let single_dose_auc_tau =
        trapezoidal_auc_from_profile(profile, clast, tlast, lambda_z, tau, &eval_times);
    let accumulation_ratio = if single_dose_auc_tau > 0.0 {
        auc_tau_ss / single_dose_auc_tau
    } else {
        f64::NAN
    };

    Some(SuperpositionResult {
        times: eval_times,
        concentrations: ss_concentrations,
        cmax_ss,
        tmax_ss,
        cmin_ss,
        auc_tau_ss,
        cavg_ss,
        n_doses,
        accumulation_ratio,
    })
}

/// Get concentration at a specific time from the profile, with extrapolation
fn concentration_at_time(
    profile: &ObservationProfile,
    clast: f64,
    tlast: f64,
    lambda_z: f64,
    time: f64,
) -> f64 {
    if time < 0.0 {
        return 0.0;
    }

    if time <= tlast {
        // Within observation range: interpolate
        profile.interpolate(time)
    } else {
        // Beyond observed data: extrapolate using terminal phase
        clast * (-lambda_z * (time - tlast)).exp()
    }
}

/// Simple trapezoidal AUC (linear method), computed directly for internally-sorted eval_times.
fn trapezoidal_auc(times: &[f64], concentrations: &[f64]) -> f64 {
    times
        .windows(2)
        .zip(concentrations.windows(2))
        .map(|(t, c)| (c[0] + c[1]) / 2.0 * (t[1] - t[0]))
        .sum()
}

/// Single-dose AUC over [0, tau] from profile with extrapolation
fn trapezoidal_auc_from_profile(
    profile: &ObservationProfile,
    clast: f64,
    tlast: f64,
    lambda_z: f64,
    tau: f64,
    eval_times: &[f64],
) -> f64 {
    let concs: Vec<f64> = eval_times
        .iter()
        .map(|&t| concentration_at_time(profile, clast, tlast, lambda_z, t.min(tau)))
        .collect();
    trapezoidal_auc(eval_times, &concs)
}

/// Convenience wrapper: run superposition using an existing [`NCAResult`].
///
/// Extracts `lambda_z` from the terminal phase and delegates to [`predict()`].
/// This is **crate-internal** — callers should use
/// [`Superposition::superposition_from_nca`] on [`Subject`] or [`Occasion`] instead.
///
/// # Errors
/// Returns [`NCAError::LambdaZFailed`] if the NCA result has no terminal phase.
pub(crate) fn predict_from_nca(
    profile: &ObservationProfile,
    nca_result: &NCAResult,
    tau: f64,
    n_eval_points: Option<usize>,
) -> Result<SuperpositionResult, NCAError> {
    let lambda_z = nca_result
        .terminal
        .as_ref()
        .map(|t| t.lambda_z)
        .ok_or_else(|| NCAError::LambdaZFailed {
            reason: "λz not estimable; cannot perform superposition".to_string(),
        })?;

    predict(profile, lambda_z, tau, n_eval_points).ok_or_else(|| NCAError::InvalidParameter {
        param: "superposition".to_string(),
        value: "prediction returned None (check lambda_z and tau)".to_string(),
    })
}

/// Extension trait for running superposition directly from a [`Subject`] or [`Occasion`]
///
/// Chains NCA → λz extraction → superposition in a single call,
/// or accepts a pre-computed [`NCAResult`] to avoid redundant NCA runs.
///
/// # Example
///
/// ```rust,ignore
/// use pharmsol::prelude::*;
/// use pharmsol::nca::{NCAOptions, Superposition};
///
/// let subject = Subject::builder("pt1")
///     .bolus(0.0, 100.0, 0)
///     .observation(0.0, 10.0, 0)
///     .observation(4.0, 6.0, 0)
///     .observation(12.0, 3.0, 0)
///     .observation(24.0, 0.9, 0)
///     .build();
///
/// // Full chain (NCA computed internally)
/// let ss = subject.superposition(12.0, &NCAOptions::default(), None)?;
/// println!("Cmax_ss: {:.2}, Cmin_ss: {:.2}", ss.cmax_ss, ss.cmin_ss);
///
/// // Reuse an existing NCA result
/// let nca = subject.nca(&NCAOptions::default())?;
/// let ss = subject.superposition_from_nca(&nca, 12.0, None)?;
/// ```
pub trait Superposition {
    /// Predict steady-state profile via superposition
    ///
    /// Performs NCA to estimate λz, then runs superposition to predict
    /// the steady-state concentration-time profile.
    ///
    /// # Arguments
    /// * `tau` - Dosing interval
    /// * `options` - NCA options (used for λz estimation)
    /// * `n_eval_points` - Number of evaluation points (None = use observed times)
    fn superposition(
        &self,
        tau: f64,
        options: &NCAOptions,
        n_eval_points: Option<usize>,
    ) -> Result<SuperpositionResult, NCAError>;

    /// Predict steady-state profile using a pre-computed [`NCAResult`]
    ///
    /// Skips the NCA step, useful when you already have an NCA result and
    /// want to avoid redundant computation.
    ///
    /// # Arguments
    /// * `nca_result` - Pre-computed NCA result containing terminal phase parameters
    /// * `tau` - Dosing interval
    /// * `n_eval_points` - Number of evaluation points (None = use observed times)
    fn superposition_from_nca(
        &self,
        nca_result: &NCAResult,
        tau: f64,
        n_eval_points: Option<usize>,
    ) -> Result<SuperpositionResult, NCAError>;
}

impl Superposition for Subject {
    fn superposition(
        &self,
        tau: f64,
        options: &NCAOptions,
        n_eval_points: Option<usize>,
    ) -> Result<SuperpositionResult, NCAError> {
        let nca_result = self.nca(options)?;
        self.superposition_from_nca(&nca_result, tau, n_eval_points)
    }

    fn superposition_from_nca(
        &self,
        nca_result: &NCAResult,
        tau: f64,
        n_eval_points: Option<usize>,
    ) -> Result<SuperpositionResult, NCAError> {
        let occ = self
            .occasions()
            .first()
            .ok_or_else(|| NCAError::InvalidParameter {
                param: "occasion".to_string(),
                value: "no occasions found".to_string(),
            })?;
        occ.superposition_from_nca(nca_result, tau, n_eval_points)
    }
}

impl Superposition for Occasion {
    fn superposition(
        &self,
        tau: f64,
        options: &NCAOptions,
        n_eval_points: Option<usize>,
    ) -> Result<SuperpositionResult, NCAError> {
        use crate::nca::traits::NCA;
        let nca_result = self.nca(options)?;
        self.superposition_from_nca(&nca_result, tau, n_eval_points)
    }

    fn superposition_from_nca(
        &self,
        nca_result: &NCAResult,
        tau: f64,
        n_eval_points: Option<usize>,
    ) -> Result<SuperpositionResult, NCAError> {
        let profile = ObservationProfile::from_occasion(self, 0, &BLQRule::Exclude)?;
        predict_from_nca(&profile, nca_result, tau, n_eval_points)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::builder::SubjectBuilderExt;
    use crate::data::event::BLQRule;
    use crate::Subject;

    #[test]
    fn test_superposition_basic() {
        // Simple exponential decay: C = 10 * exp(-0.1 * t)
        let subject = Subject::builder("test")
            .bolus(0.0, 100.0, 0)
            .observation(0.0, 10.0, 0)
            .observation(1.0, 9.048, 0) // 10 * exp(-0.1)
            .observation(2.0, 8.187, 0) // 10 * exp(-0.2)
            .observation(4.0, 6.703, 0) // 10 * exp(-0.4)
            .observation(8.0, 4.493, 0) // 10 * exp(-0.8)
            .observation(12.0, 3.012, 0) // 10 * exp(-1.2)
            .observation(24.0, 0.907, 0) // 10 * exp(-2.4)
            .build();

        let occ = &subject.occasions()[0];
        let profile = ObservationProfile::from_occasion(occ, 0, &BLQRule::Exclude).unwrap();

        let lambda_z = 0.1;
        let tau = 12.0;
        let result = predict(&profile, lambda_z, tau, Some(25)).unwrap();

        assert!(
            result.cmax_ss > 10.0,
            "SS Cmax should be > single dose Cmax due to accumulation"
        );
        assert!(result.cmin_ss > 0.0, "SS Cmin should be positive");
        assert!(
            result.accumulation_ratio > 1.0,
            "Accumulation ratio should be > 1"
        );
        assert!(
            result.n_doses > 1,
            "Should require multiple doses to converge"
        );
    }

    #[test]
    fn test_superposition_invalid_inputs() {
        let subject = Subject::builder("test")
            .bolus(0.0, 100.0, 0)
            .observation(0.0, 10.0, 0)
            .observation(1.0, 5.0, 0)
            .build();

        let occ = &subject.occasions()[0];
        let profile = ObservationProfile::from_occasion(occ, 0, &BLQRule::Exclude).unwrap();

        assert!(predict(&profile, -0.1, 12.0, None).is_none());
        assert!(predict(&profile, 0.1, 0.0, None).is_none());
        assert!(predict(&profile, 0.0, 12.0, None).is_none());
    }

    #[test]
    fn test_superposition_theoretical_accumulation() {
        // For a one-compartment IV model with first-order elimination:
        // Theoretical accumulation factor = 1 / (1 - exp(-λz * τ))
        let lambda_z: f64 = 0.1;
        let tau: f64 = 8.0;
        let theoretical_af = 1.0 / (1.0 - (-lambda_z * tau).exp());

        let subject = Subject::builder("test")
            .bolus(0.0, 100.0, 0)
            .observation(0.0, 10.0, 0)
            .observation(1.0, 9.048, 0)
            .observation(2.0, 8.187, 0)
            .observation(4.0, 6.703, 0)
            .observation(8.0, 4.493, 0)
            .observation(12.0, 3.012, 0)
            .observation(24.0, 0.907, 0)
            .build();

        let occ = &subject.occasions()[0];
        let profile = ObservationProfile::from_occasion(occ, 0, &BLQRule::Exclude).unwrap();

        let result = predict(&profile, lambda_z, tau, Some(50)).unwrap();

        // Accumulation ratio should be close to theoretical
        let tol = 0.05; // 5% tolerance for interpolation effects
        assert!(
            (result.accumulation_ratio - theoretical_af).abs() / theoretical_af < tol,
            "Accumulation ratio {:.3} should be close to theoretical {:.3}",
            result.accumulation_ratio,
            theoretical_af
        );
    }
}
