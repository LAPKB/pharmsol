//! Residual error models for parametric algorithms (SAEM, FOCE, etc.)
//!
//! This module provides error model implementations that use the **prediction**
//! (model output) rather than the **observation** for computing residual error.
//!
//! # Conceptual Difference from [`crate::ErrorModel`]
//!
//! - [`crate::ErrorModel`] (in `error_model.rs`): Represents **measurement/assay noise**.
//!   Sigma is computed from the **observation** using polynomial characterization.
//!   Used by non-parametric algorithms (NPAG, NPOD, etc.).
//!
//! - [`ResidualErrorModel`] (this module): Represents **residual unexplained variability**
//!   in population models. Sigma is computed from the **prediction**.
//!   Used by parametric algorithms (SAEM, FOCE, etc.).
//!
//! # R saemix Correspondence
//!
//! The error model in saemix (func_aux.R):
//! ```R
//! error.typ <- function(f, ab) {
//!     g <- cutoff(sqrt(ab[1]^2 + ab[2]^2 * f^2))
//!     return(g)
//! }
//! ```
//!
//! | saemix parameter | This implementation |
//! |------------------|---------------------|
//! | `ab[1]` (a)      | `Constant::a` or `Combined::a` |
//! | `ab[2]` (b)      | `Proportional::b` or `Combined::b` |
//!
//! # Error Model Types
//!
//! - **Constant**: σ = a (independent of prediction)
//! - **Proportional**: σ = b * |f| (scales with prediction)
//! - **Combined**: σ = sqrt(a² + b²*f²) (most flexible, default in saemix)
//! - **Exponential**: σ for log-transformed data

use serde::{Deserialize, Serialize};

/// Residual error model for parametric estimation algorithms.
///
/// Unlike [`crate::ErrorModel`] which uses observations, this uses
/// the model **prediction** to compute the standard deviation.
///
/// # Usage in SAEM
///
/// The error model affects:
/// 1. **Likelihood computation** in E-step: L(y|f) = N(y; f, σ²)
/// 2. **Residual weighting** in M-step: weighted_res² = (y-f)²/σ²
///
/// # Examples
///
/// ```rust
/// use pharmsol::ResidualErrorModel;
///
/// // Constant (additive) error: σ = 0.5
/// let constant = ResidualErrorModel::Constant { a: 0.5 };
/// assert!((constant.sigma(100.0) - 0.5).abs() < 1e-10);
///
/// // Proportional error: σ = 0.1 * |f|
/// let proportional = ResidualErrorModel::Proportional { b: 0.1 };
/// assert!((proportional.sigma(100.0) - 10.0).abs() < 1e-10);
///
/// // Combined error: σ = sqrt(0.5² + 0.1² * f²)
/// let combined = ResidualErrorModel::Combined { a: 0.5, b: 0.1 };
/// // For f=100: σ = sqrt(0.25 + 100) = sqrt(100.25) ≈ 10.01
/// ```
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum ResidualErrorModel {
    /// Constant (additive) error model
    ///
    /// σ = a
    ///
    /// Error is independent of the predicted value.
    /// Appropriate when measurement error is constant regardless of concentration.
    Constant {
        /// Additive error standard deviation
        a: f64,
    },

    /// Proportional error model
    ///
    /// σ = b * |f|
    ///
    /// Error scales linearly with the prediction.
    /// Appropriate when measurement error is a constant percentage of the value.
    ///
    /// Note: Uses |f| to handle negative predictions gracefully.
    Proportional {
        /// Proportional coefficient (e.g., 0.1 = 10% CV)
        b: f64,
    },

    /// Combined (additive + proportional) error model
    ///
    /// σ = sqrt(a² + b² * f²)
    ///
    /// This is the standard saemix error model from func_aux.R:
    /// ```R
    /// g <- cutoff(sqrt(ab[1]^2 + ab[2]^2 * f^2))
    /// ```
    ///
    /// The combined model:
    /// - Dominates at low concentrations (a term)
    /// - Scales proportionally at high concentrations (b term)
    Combined {
        /// Additive component (a)
        a: f64,
        /// Proportional component (b)
        b: f64,
    },

    /// Exponential error model (for log-transformed data)
    ///
    /// σ = σ_exp (constant on log scale)
    ///
    /// When data is analyzed on the log scale:
    /// ```text
    /// log(Y) = log(f) + ε, where ε ~ N(0, σ²)
    /// ```
    ///
    /// This corresponds to multiplicative error on the original scale.
    Exponential {
        /// Error standard deviation on log scale
        sigma: f64,
    },
}

impl Default for ResidualErrorModel {
    fn default() -> Self {
        // Default to constant error with σ = 1.0
        ResidualErrorModel::Constant { a: 1.0 }
    }
}

impl ResidualErrorModel {
    /// Create a constant (additive) error model
    ///
    /// # Arguments
    /// * `a` - Standard deviation (must be positive)
    pub fn constant(a: f64) -> Self {
        ResidualErrorModel::Constant { a }
    }

    /// Create a proportional error model
    ///
    /// # Arguments
    /// * `b` - Proportional coefficient (e.g., 0.1 for 10% CV)
    pub fn proportional(b: f64) -> Self {
        ResidualErrorModel::Proportional { b }
    }

    /// Create a combined (additive + proportional) error model
    ///
    /// # Arguments
    /// * `a` - Additive component
    /// * `b` - Proportional component
    pub fn combined(a: f64, b: f64) -> Self {
        ResidualErrorModel::Combined { a, b }
    }

    /// Create an exponential error model
    ///
    /// # Arguments
    /// * `sigma` - Standard deviation on log scale
    pub fn exponential(sigma: f64) -> Self {
        ResidualErrorModel::Exponential { sigma }
    }

    /// Compute sigma (standard deviation) for a given prediction
    ///
    /// # Arguments
    /// * `prediction` - The model prediction (f)
    ///
    /// # Returns
    /// The standard deviation σ at this prediction value.
    /// Returns a cutoff minimum to avoid numerical issues with very small σ.
    pub fn sigma(&self, prediction: f64) -> f64 {
        let raw_sigma = match self {
            ResidualErrorModel::Constant { a } => *a,
            ResidualErrorModel::Proportional { b } => b * prediction.abs(),
            ResidualErrorModel::Combined { a, b } => {
                (a.powi(2) + b.powi(2) * prediction.powi(2)).sqrt()
            }
            ResidualErrorModel::Exponential { sigma } => *sigma,
        };

        // Apply cutoff to prevent division by zero in likelihood
        // R saemix uses cutoff function with default .Machine$double.eps
        raw_sigma.max(f64::EPSILON.sqrt())
    }

    /// Compute variance for a given prediction
    ///
    /// # Arguments
    /// * `prediction` - The model prediction (f)
    ///
    /// # Returns
    /// The variance σ² at this prediction value.
    pub fn variance(&self, prediction: f64) -> f64 {
        let sigma = self.sigma(prediction);
        sigma.powi(2)
    }

    /// Compute the weighted residual for M-step sigma updates
    ///
    /// For the M-step in SAEM, we compute the normalized residual:
    /// - For constant/additive: (y - f)² (unweighted)
    /// - For proportional: (y - f)² / f² (weighted by prediction)
    /// - For combined: (y - f)² / (a² + b²*f²) (using current sigma params)
    ///
    /// This matches R saemix's approach in main_mstep.R where for proportional
    /// error: `resk <- sum((yobs - fk)**2 / cutoff(fk**2, .Machine$double.eps))`
    ///
    /// # Arguments
    /// * `observation` - The observed value (y)
    /// * `prediction` - The model prediction (f)
    ///
    /// # Returns
    /// The weighted squared residual for sigma estimation.
    pub fn weighted_squared_residual(&self, observation: f64, prediction: f64) -> f64 {
        let residual = observation - prediction;
        let residual_sq = residual * residual;

        match self {
            ResidualErrorModel::Constant { .. } => {
                // Constant error: unweighted residuals
                // new_sigma² = Σ(y - f)² / n
                residual_sq
            }
            ResidualErrorModel::Proportional { .. } => {
                // Proportional error: weight by 1/f²
                // new_sigma² = Σ(y - f)²/f² / n = b² (the proportional coefficient)
                // This matches R saemix: resk <- sum((yobs - fk)**2 / cutoff(fk**2, ...))
                let pred_sq = prediction.powi(2).max(f64::EPSILON);
                residual_sq / pred_sq
            }
            ResidualErrorModel::Combined { a, b } => {
                // Combined error: weight by current variance estimate
                // This is more complex - use current sigma² = a² + b²*f²
                let variance = (a.powi(2) + b.powi(2) * prediction.powi(2)).max(f64::EPSILON);
                residual_sq / variance
            }
            ResidualErrorModel::Exponential { .. } => {
                // Exponential: residuals on log scale
                // This should be computed differently for log-transformed data
                residual_sq
            }
        }
    }

    /// Compute log-likelihood contribution for a single observation
    ///
    /// Assuming normal distribution:
    /// ```text
    /// log L(y|f,σ) = -0.5 * [log(2π) + log(σ²) + (y-f)²/σ²]
    /// ```
    ///
    /// # Arguments
    /// * `observation` - The observed value (y)
    /// * `prediction` - The model prediction (f)
    ///
    /// # Returns
    /// The log-likelihood contribution.
    pub fn log_likelihood(&self, observation: f64, prediction: f64) -> f64 {
        let sigma = self.sigma(prediction);
        let residual = observation - prediction;
        let normalized_residual = residual / sigma;

        -0.5 * (std::f64::consts::TAU.ln() + 2.0 * sigma.ln() + normalized_residual.powi(2))
    }

    /// Update the error model parameters based on M-step sufficient statistics
    ///
    /// In SAEM, the residual error is estimated in the M-step. This method
    /// updates the appropriate parameter based on the new estimate.
    ///
    /// # Arguments
    /// * `new_sigma` - The new sigma estimate from M-step
    ///
    /// # Returns
    /// A new error model with updated parameters.
    pub fn with_updated_sigma(self, new_sigma: f64) -> Self {
        match self {
            ResidualErrorModel::Constant { .. } => ResidualErrorModel::Constant { a: new_sigma },
            ResidualErrorModel::Proportional { .. } => {
                ResidualErrorModel::Proportional { b: new_sigma }
            }
            ResidualErrorModel::Combined { a: _, b } => {
                // For combined model, we update the additive component
                // and keep the proportional component fixed
                // This is a simplification - full estimation would estimate both
                ResidualErrorModel::Combined { a: new_sigma, b }
            }
            ResidualErrorModel::Exponential { .. } => {
                ResidualErrorModel::Exponential { sigma: new_sigma }
            }
        }
    }

    /// Get the primary sigma parameter value
    ///
    /// For Constant: returns a
    /// For Proportional: returns b
    /// For Combined: returns a (additive component)
    /// For Exponential: returns sigma
    pub fn primary_parameter(&self) -> f64 {
        match self {
            ResidualErrorModel::Constant { a } => *a,
            ResidualErrorModel::Proportional { b } => *b,
            ResidualErrorModel::Combined { a, .. } => *a,
            ResidualErrorModel::Exponential { sigma } => *sigma,
        }
    }

    /// Check if this is a proportional error model
    pub fn is_proportional(&self) -> bool {
        matches!(self, ResidualErrorModel::Proportional { .. })
    }

    /// Check if this is a constant (additive) error model
    pub fn is_constant(&self) -> bool {
        matches!(self, ResidualErrorModel::Constant { .. })
    }

    /// Check if this is a combined error model
    pub fn is_combined(&self) -> bool {
        matches!(self, ResidualErrorModel::Combined { .. })
    }

    /// Check if this is an exponential error model
    pub fn is_exponential(&self) -> bool {
        matches!(self, ResidualErrorModel::Exponential { .. })
    }
}

/// Collection of residual error models for multiple output equations
///
/// This mirrors [`crate::ErrorModels`] but for parametric algorithms.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ResidualErrorModels {
    models: Vec<ResidualErrorModel>,
}

impl ResidualErrorModels {
    /// Create an empty collection
    pub fn new() -> Self {
        Self { models: vec![] }
    }

    /// Add an error model for a specific output equation
    pub fn add(mut self, outeq: usize, model: ResidualErrorModel) -> Self {
        if outeq >= self.models.len() {
            self.models.resize(outeq + 1, ResidualErrorModel::default());
        }
        self.models[outeq] = model;
        self
    }

    /// Get the error model for a specific output equation
    pub fn get(&self, outeq: usize) -> Option<&ResidualErrorModel> {
        self.models.get(outeq)
    }

    /// Get a mutable reference to the error model for a specific output equation
    pub fn get_mut(&mut self, outeq: usize) -> Option<&mut ResidualErrorModel> {
        self.models.get_mut(outeq)
    }

    /// Compute sigma for a specific output equation and prediction
    pub fn sigma(&self, outeq: usize, prediction: f64) -> Option<f64> {
        self.models.get(outeq).map(|m| m.sigma(prediction))
    }

    /// Number of error models
    pub fn len(&self) -> usize {
        self.models.len()
    }

    /// Check if collection is empty
    pub fn is_empty(&self) -> bool {
        self.models.is_empty()
    }

    /// Iterate over (outeq, model) pairs
    pub fn iter(&self) -> impl Iterator<Item = (usize, &ResidualErrorModel)> {
        self.models.iter().enumerate()
    }

    /// Compute log-likelihood for a single observation given its prediction
    ///
    /// # Arguments
    /// * `outeq` - Output equation index
    /// * `observation` - The observed value (y)
    /// * `prediction` - The model prediction (f)
    ///
    /// # Returns
    /// The log-likelihood contribution, or None if outeq is invalid.
    pub fn log_likelihood(&self, outeq: usize, observation: f64, prediction: f64) -> Option<f64> {
        self.models
            .get(outeq)
            .map(|m| m.log_likelihood(observation, prediction))
    }

    /// Compute total log-likelihood for multiple observation-prediction pairs
    ///
    /// # Arguments
    /// * `obs_pred_pairs` - Iterator of (outeq, observation, prediction) tuples
    ///
    /// # Returns
    /// The sum of log-likelihood contributions. Returns `f64::NEG_INFINITY` if any
    /// outeq is invalid.
    pub fn total_log_likelihood<I>(&self, obs_pred_pairs: I) -> f64
    where
        I: IntoIterator<Item = (usize, f64, f64)>,
    {
        let mut total = 0.0;
        for (outeq, obs, pred) in obs_pred_pairs {
            match self.log_likelihood(outeq, obs, pred) {
                Some(ll) => total += ll,
                None => return f64::NEG_INFINITY,
            }
        }
        total
    }

    /// Update all models with a new sigma estimate
    pub fn update_sigma(&mut self, new_sigma: f64) {
        for model in &mut self.models {
            *model = model.with_updated_sigma(new_sigma);
        }
    }
}

/// Convert from [`ErrorModels`] to [`ResidualErrorModels`]
///
/// This allows backward compatibility when users have existing `ErrorModels`
/// configurations that need to be used with parametric algorithms.
///
/// # Conversion Mapping
///
/// | pharmsol ErrorModel | ResidualErrorModel |
/// |---------------------|-------------------|
/// | `Additive { lambda, .. }` | `Constant { a: lambda }` |
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constant_error() {
        let model = ResidualErrorModel::constant(0.5);
        assert!((model.sigma(0.0) - 0.5).abs() < 1e-10);
        assert!((model.sigma(100.0) - 0.5).abs() < 1e-10);
        assert!((model.sigma(-50.0) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_proportional_error() {
        let model = ResidualErrorModel::proportional(0.1);
        assert!((model.sigma(100.0) - 10.0).abs() < 1e-10);
        assert!((model.sigma(50.0) - 5.0).abs() < 1e-10);
        // Uses absolute value, so negative predictions work
        assert!((model.sigma(-100.0) - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_combined_error() {
        let model = ResidualErrorModel::combined(0.5, 0.1);
        // At f=0: sigma = sqrt(0.25 + 0) = 0.5
        assert!((model.sigma(0.0) - 0.5).abs() < 1e-10);
        // At f=100: sigma = sqrt(0.25 + 100) = sqrt(100.25)
        assert!((model.sigma(100.0) - 100.25_f64.sqrt()).abs() < 1e-10);
    }

    #[test]
    fn test_weighted_residual() {
        let model = ResidualErrorModel::constant(1.0);
        // Constant error: unweighted residual = (obs - pred)²
        let wr = model.weighted_squared_residual(5.0, 3.0);
        assert!((wr - 4.0).abs() < 1e-10); // (5-3)² = 4

        let prop_model = ResidualErrorModel::proportional(0.1);
        // Proportional: weighted by 1/pred², NOT 1/sigma²
        // At pred=10, residual = 12-10 = 2, weighted = (2)²/(10)² = 4/100 = 0.04
        let wr2 = prop_model.weighted_squared_residual(12.0, 10.0);
        assert!((wr2 - 0.04).abs() < 1e-10);
    }

    #[test]
    fn test_sigma_cutoff() {
        let model = ResidualErrorModel::proportional(0.1);
        // At prediction = 0, raw sigma would be 0, but cutoff prevents this
        let sigma = model.sigma(0.0);
        assert!(sigma > 0.0);
        assert!(sigma >= f64::EPSILON.sqrt());
    }

    #[test]
    fn test_log_likelihood() {
        let model = ResidualErrorModel::constant(1.0);
        // Standard normal: log L = -0.5 * (log(2π) + 0 + z²)
        let ll = model.log_likelihood(1.0, 0.0);
        let expected = -0.5 * (std::f64::consts::TAU.ln() + 1.0);
        assert!((ll - expected).abs() < 1e-10);
    }

    #[test]
    fn test_residual_error_models_collection() {
        let models = ResidualErrorModels::new()
            .add(0, ResidualErrorModel::constant(0.5))
            .add(1, ResidualErrorModel::proportional(0.1));

        assert_eq!(models.len(), 2);
        assert!(models.get(0).unwrap().is_constant());
        assert!(models.get(1).unwrap().is_proportional());
        assert!((models.sigma(0, 100.0).unwrap() - 0.5).abs() < 1e-10);
        assert!((models.sigma(1, 100.0).unwrap() - 10.0).abs() < 1e-10);
    }
}
