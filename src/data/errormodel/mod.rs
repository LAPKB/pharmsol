//! Error model types and trait for pharmacometric analyses.
//!
//! This module provides the [`ErrorModel`] trait that unifies the two error model
//! families in pharmsol:
//!
//! - [`AssayErrorModel`]: Observation-based sigma for non-parametric algorithms (NPAG, NPOD)
//! - [`ResidualErrorModel`]: Prediction-based sigma for parametric algorithms (SAEM, FOCE)
//!
//! Both implement the same core operation: computing σ from a numeric value.
//! The distinction of *which* value (observation vs prediction) is a call-site concern.
//!
//! The [`ErrorModels`] generic collection provides a single container parameterized
//! by the error model type.

mod assay;
mod residual;

// Re-export everything from submodules
pub use assay::*;
pub use residual::*;

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Trait for error models that compute sigma (standard deviation) from a numeric value.
///
/// Both assay error models (observation-based) and residual error models (prediction-based)
/// implement this trait. The caller decides which value to pass:
///
/// ```text
/// // Non-parametric: sigma from observation
/// let sigma = model.sigma(observation);
///
/// // Parametric: sigma from prediction
/// let sigma = model.sigma(prediction);
/// ```
pub trait ErrorModel: Clone + Send + Sync {
    /// Compute σ (standard deviation) for a given value.
    ///
    /// The interpretation of `value` depends on the algorithm:
    /// - For assay error models: pass the observation
    /// - For residual error models: pass the prediction
    fn sigma(&self, value: f64) -> Result<f64, ErrorModelError>;

    /// Compute variance (σ²) for a given value.
    fn variance(&self, value: f64) -> Result<f64, ErrorModelError> {
        let s = self.sigma(value)?;
        Ok(s.powi(2))
    }
}

/// Errors that can occur during error model operations.
#[derive(Error, Debug, Clone)]
pub enum ErrorModelError {
    #[error("The computed standard deviation is negative")]
    NegativeSigma,
    #[error("The computed standard deviation is zero")]
    ZeroSigma,
    #[error("The computed standard deviation is non-finite")]
    NonFiniteSigma,
    #[error("The output equation index {0} is invalid")]
    InvalidOutputEquation(usize),
    #[error("The output equation number {0} already exists")]
    ExistingOutputEquation(usize),
    #[error("No error model defined for output equation index {0}")]
    NoneErrorModel(usize),
    #[error("The prediction does not have an observation associated with it")]
    MissingObservation,
}

/// Generic collection of error models indexed by output equation.
///
/// Each output equation can have its own error model. This is the shared container
/// used by both [`AssayErrorModels`] and [`ResidualErrorModels`] (which are type aliases).
///
/// # Type Parameters
/// - `M`: The error model type (e.g., `AssayErrorModel` or `ResidualErrorModel`)
///
/// # Examples
/// ```ignore
/// use pharmsol::{ErrorModels, AssayErrorModel, ErrorPoly};
///
/// let models = ErrorModels::new()
///     .add(0, AssayErrorModel::additive(ErrorPoly::new(0.0, 0.1, 0.0, 0.0), 0.0))?;
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorModels<M> {
    models: Vec<Option<M>>,
}

impl<M> Default for ErrorModels<M> {
    fn default() -> Self {
        Self::new()
    }
}

impl<M> ErrorModels<M> {
    /// Create an empty collection.
    pub fn new() -> Self {
        Self { models: vec![] }
    }

    /// Add an error model for a specific output equation.
    ///
    /// # Errors
    /// Returns [`ErrorModelError::ExistingOutputEquation`] if a model is already set for `outeq`.
    pub fn add(mut self, outeq: usize, model: M) -> Result<Self, ErrorModelError> {
        if outeq >= self.models.len() {
            self.models.resize_with(outeq + 1, || None);
        }
        if self.models[outeq].is_some() {
            return Err(ErrorModelError::ExistingOutputEquation(outeq));
        }
        self.models[outeq] = Some(model);
        Ok(self)
    }

    /// Get a reference to the error model for a specific output equation.
    pub fn get(&self, outeq: usize) -> Result<&M, ErrorModelError> {
        self.models
            .get(outeq)
            .and_then(|m| m.as_ref())
            .ok_or(ErrorModelError::NoneErrorModel(outeq))
    }

    /// Get a mutable reference to the error model for a specific output equation.
    pub fn get_mut(&mut self, outeq: usize) -> Result<&mut M, ErrorModelError> {
        self.models
            .get_mut(outeq)
            .and_then(|m| m.as_mut())
            .ok_or(ErrorModelError::NoneErrorModel(outeq))
    }

    /// Returns the number of output equation slots (including empty ones).
    pub fn len(&self) -> usize {
        self.models.len()
    }

    /// Returns `true` if there are no output equation slots.
    pub fn is_empty(&self) -> bool {
        self.models.is_empty()
    }

    /// Iterate over populated (outeq, model) pairs.
    pub fn iter(&self) -> impl Iterator<Item = (usize, &M)> {
        self.models
            .iter()
            .enumerate()
            .filter_map(|(i, m)| m.as_ref().map(|m| (i, m)))
    }

    /// Iterate over populated (outeq, model) pairs mutably.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = (usize, &mut M)> {
        self.models
            .iter_mut()
            .enumerate()
            .filter_map(|(i, m)| m.as_mut().map(|m| (i, m)))
    }
}

impl<M: ErrorModel> ErrorModels<M> {
    /// Compute sigma for a specific output equation and value.
    pub fn sigma(&self, outeq: usize, value: f64) -> Result<f64, ErrorModelError> {
        self.get(outeq)?.sigma(value)
    }

    /// Compute variance for a specific output equation and value.
    pub fn variance(&self, outeq: usize, value: f64) -> Result<f64, ErrorModelError> {
        self.get(outeq)?.variance(value)
    }
}

impl<M> IntoIterator for ErrorModels<M> {
    type Item = (usize, M);
    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.models
            .into_iter()
            .enumerate()
            .filter_map(|(i, m)| m.map(|m| (i, m)))
            .collect::<Vec<_>>()
            .into_iter()
    }
}

impl<'a, M> IntoIterator for &'a ErrorModels<M> {
    type Item = (usize, &'a M);
    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.models
            .iter()
            .enumerate()
            .filter_map(|(i, m)| m.as_ref().map(|m| (i, m)))
            .collect::<Vec<_>>()
            .into_iter()
    }
}

impl<'a, M> IntoIterator for &'a mut ErrorModels<M> {
    type Item = (usize, &'a mut M);
    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.models
            .iter_mut()
            .enumerate()
            .filter_map(|(i, m)| m.as_mut().map(|m| (i, m)))
            .collect::<Vec<_>>()
            .into_iter()
    }
}
