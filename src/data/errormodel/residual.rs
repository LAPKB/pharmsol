use super::{ErrorModel, ErrorModelError, ErrorModels};
use serde::{Deserialize, Serialize};

/// Residual error model for parametric estimation algorithms.
///
/// Unlike [`super::AssayErrorModel`] which uses observations, this uses
/// the model **prediction** to compute the standard deviation.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum ResidualErrorModel {
    /// Constant (additive) error model: σ = a
    Constant {
        /// Additive error standard deviation
        a: f64,
    },

    /// Proportional error model: σ = b * |f|
    Proportional {
        /// Proportional coefficient (e.g., 0.1 = 10% CV)
        b: f64,
    },

    /// Combined (additive + proportional) error model: σ = sqrt(a² + b² * f²)
    Combined {
        /// Additive component (a)
        a: f64,
        /// Proportional component (b)
        b: f64,
    },

    /// Exponential error model (for log-transformed data): σ = sigma (constant on log scale)
    Exponential {
        /// Error standard deviation on log scale
        sigma: f64,
    },
}

impl Default for ResidualErrorModel {
    fn default() -> Self {
        ResidualErrorModel::Constant { a: 1.0 }
    }
}

impl ErrorModel for ResidualErrorModel {
    /// Compute sigma from a prediction value.
    ///
    /// Returns a cutoff minimum to avoid numerical issues with very small σ.
    fn sigma(&self, prediction: f64) -> Result<f64, ErrorModelError> {
        let raw_sigma = match self {
            ResidualErrorModel::Constant { a } => *a,
            ResidualErrorModel::Proportional { b } => b * prediction.abs(),
            ResidualErrorModel::Combined { a, b } => {
                (a.powi(2) + b.powi(2) * prediction.powi(2)).sqrt()
            }
            ResidualErrorModel::Exponential { sigma } => *sigma,
        };

        // Apply cutoff to prevent division by zero in likelihood
        Ok(raw_sigma.max(f64::EPSILON.sqrt()))
    }
}

impl ResidualErrorModel {
    /// Create a constant (additive) error model
    pub fn constant(a: f64) -> Self {
        ResidualErrorModel::Constant { a }
    }

    /// Create a proportional error model
    pub fn proportional(b: f64) -> Self {
        ResidualErrorModel::Proportional { b }
    }

    /// Create a combined (additive + proportional) error model
    pub fn combined(a: f64, b: f64) -> Self {
        ResidualErrorModel::Combined { a, b }
    }

    /// Create an exponential error model
    pub fn exponential(sigma: f64) -> Self {
        ResidualErrorModel::Exponential { sigma }
    }

    /// Compute the weighted residual for M-step sigma updates
    pub fn weighted_squared_residual(&self, observation: f64, prediction: f64) -> f64 {
        let residual = observation - prediction;
        let residual_sq = residual * residual;

        match self {
            ResidualErrorModel::Constant { .. } => residual_sq,
            ResidualErrorModel::Proportional { .. } => {
                let pred_sq = prediction.powi(2).max(f64::EPSILON);
                residual_sq / pred_sq
            }
            ResidualErrorModel::Combined { a, b } => {
                let variance = (a.powi(2) + b.powi(2) * prediction.powi(2)).max(f64::EPSILON);
                residual_sq / variance
            }
            ResidualErrorModel::Exponential { .. } => residual_sq,
        }
    }

    /// Compute log-likelihood contribution for a single observation
    pub fn log_likelihood(&self, observation: f64, prediction: f64) -> f64 {
        // sigma() returns Result but never errors for ResidualErrorModel
        let sigma = ErrorModel::sigma(self, prediction).unwrap_or(f64::EPSILON.sqrt());
        let residual = observation - prediction;
        let normalized_residual = residual / sigma;

        -0.5 * (std::f64::consts::TAU.ln() + 2.0 * sigma.ln() + normalized_residual.powi(2))
    }

    /// Update the error model parameters based on M-step sufficient statistics
    pub fn with_updated_sigma(self, new_sigma: f64) -> Self {
        match self {
            ResidualErrorModel::Constant { .. } => ResidualErrorModel::Constant { a: new_sigma },
            ResidualErrorModel::Proportional { .. } => {
                ResidualErrorModel::Proportional { b: new_sigma }
            }
            ResidualErrorModel::Combined { a: _, b } => {
                ResidualErrorModel::Combined { a: new_sigma, b }
            }
            ResidualErrorModel::Exponential { .. } => {
                ResidualErrorModel::Exponential { sigma: new_sigma }
            }
        }
    }

    /// Get the primary sigma parameter value
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

// ── ResidualErrorModels: type alias + extension methods ──────────────────

/// Collection of residual error models for multiple output equations.
///
/// This is a type alias for [`ErrorModels<ResidualErrorModel>`].
pub type ResidualErrorModels = ErrorModels<ResidualErrorModel>;

/// Extension trait providing residual-error-specific operations on [`ResidualErrorModels`].
pub trait ResidualErrorModelsExt {
    /// Compute log-likelihood for a single observation given its prediction.
    fn log_likelihood(&self, outeq: usize, observation: f64, prediction: f64) -> Option<f64>;

    /// Compute total log-likelihood for multiple observation-prediction pairs.
    fn total_log_likelihood<I>(&self, obs_pred_pairs: I) -> f64
    where
        I: IntoIterator<Item = (usize, f64, f64)>;

    /// Update all models with a new sigma estimate.
    fn update_sigma(&mut self, new_sigma: f64);
}

impl ResidualErrorModelsExt for ResidualErrorModels {
    fn log_likelihood(&self, outeq: usize, observation: f64, prediction: f64) -> Option<f64> {
        self.get(outeq)
            .ok()
            .map(|m| m.log_likelihood(observation, prediction))
    }

    fn total_log_likelihood<I>(&self, obs_pred_pairs: I) -> f64
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

    fn update_sigma(&mut self, new_sigma: f64) {
        // Collect outeqs first to avoid borrow conflict
        let outeqs: Vec<usize> = self.iter().map(|(i, _)| i).collect();
        for outeq in outeqs {
            if let Ok(model) = self.get(outeq) {
                let updated = model.with_updated_sigma(new_sigma);
                if let Ok(m) = self.get_mut(outeq) {
                    *m = updated;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constant_error() {
        let model = ResidualErrorModel::constant(0.5);
        assert!((ErrorModel::sigma(&model, 0.0).unwrap() - 0.5).abs() < 1e-10);
        assert!((ErrorModel::sigma(&model, 100.0).unwrap() - 0.5).abs() < 1e-10);
        assert!((ErrorModel::sigma(&model, -50.0).unwrap() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_proportional_error() {
        let model = ResidualErrorModel::proportional(0.1);
        assert!((ErrorModel::sigma(&model, 100.0).unwrap() - 10.0).abs() < 1e-10);
        assert!((ErrorModel::sigma(&model, 50.0).unwrap() - 5.0).abs() < 1e-10);
        assert!((ErrorModel::sigma(&model, -100.0).unwrap() - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_combined_error() {
        let model = ResidualErrorModel::combined(0.5, 0.1);
        assert!((ErrorModel::sigma(&model, 0.0).unwrap() - 0.5).abs() < 1e-10);
        assert!((ErrorModel::sigma(&model, 100.0).unwrap() - 100.25_f64.sqrt()).abs() < 1e-10);
    }

    #[test]
    fn test_weighted_residual() {
        let model = ResidualErrorModel::constant(1.0);
        let wr = model.weighted_squared_residual(5.0, 3.0);
        assert!((wr - 4.0).abs() < 1e-10);

        let prop_model = ResidualErrorModel::proportional(0.1);
        let wr2 = prop_model.weighted_squared_residual(12.0, 10.0);
        assert!((wr2 - 0.04).abs() < 1e-10);
    }

    #[test]
    fn test_sigma_cutoff() {
        let model = ResidualErrorModel::proportional(0.1);
        let sigma = ErrorModel::sigma(&model, 0.0).unwrap();
        assert!(sigma > 0.0);
        assert!(sigma >= f64::EPSILON.sqrt());
    }

    #[test]
    fn test_log_likelihood() {
        let model = ResidualErrorModel::constant(1.0);
        let ll = model.log_likelihood(1.0, 0.0);
        let expected = -0.5 * (std::f64::consts::TAU.ln() + 1.0);
        assert!((ll - expected).abs() < 1e-10);
    }

    #[test]
    fn test_residual_error_models_collection() {
        let models = ResidualErrorModels::new()
            .add(0, ResidualErrorModel::constant(0.5))
            .unwrap()
            .add(1, ResidualErrorModel::proportional(0.1))
            .unwrap();

        assert_eq!(models.len(), 2);
        assert!(models.get(0).unwrap().is_constant());
        assert!(models.get(1).unwrap().is_proportional());
        assert!((ErrorModels::sigma(&models, 0, 100.0).unwrap() - 0.5).abs() < 1e-10);
        assert!((ErrorModels::sigma(&models, 1, 100.0).unwrap() - 10.0).abs() < 1e-10);
    }
}
