#![allow(dead_code)]

//! A minimal [`ObservationModel`] for integration tests.
//!
//! pharmsol ships no observation model; this one exists only so the SDE
//! particle filter can be tested. It is a normal distribution with a constant
//! sigma, with censored observations scored by the normal CDF (BLOQ) or
//! survival function (ALOQ).

use std::hash::{Hash, Hasher};

use pharmsol::prelude::*;
use statrs::distribution::{ContinuousCDF, Normal};

/// `N(prediction, sigma²)` for every listed output.
#[derive(Debug, Clone)]
pub struct NormalObservation {
    labels: Vec<OutputLabel>,
    sigma: f64,
}

impl NormalObservation {
    /// Normal observations of `label` with standard deviation `sigma`.
    pub fn new(label: impl Into<OutputLabel>, sigma: f64) -> Self {
        Self {
            labels: vec![label.into()],
            sigma,
        }
    }

    /// The same model with a different sigma.
    pub fn with_sigma(mut self, sigma: f64) -> Self {
        self.sigma = sigma;
        self
    }

    /// Sum of the log-densities of every observed prediction.
    pub fn score(&self, predictions: &impl PointPredictions) -> Result<f64, PharmsolError> {
        predictions
            .point()
            .iter()
            .filter_map(|prediction| {
                prediction.observation().map(|observation| {
                    self.log_density(
                        prediction.outeq(),
                        observation,
                        prediction.prediction(),
                        prediction.censoring(),
                        prediction.errorpoly(),
                    )
                })
            })
            .sum()
    }
}

impl ObservationModel for NormalObservation {
    fn log_density(
        &self,
        _outeq: &OutputLabel,
        observation: f64,
        prediction: f64,
        censoring: Censor,
        _errorpoly: Option<ErrorPoly>,
    ) -> Result<f64, PharmsolError> {
        let sigma = self.sigma;
        let log_density = match censoring {
            Censor::None => {
                let z = (observation - prediction) / sigma;
                -0.5 * (2.0 * std::f64::consts::PI).ln() - sigma.ln() - 0.5 * z * z
            }
            Censor::BLOQ | Censor::ALOQ => {
                let normal = Normal::new(prediction, sigma)
                    .map_err(|_| PharmsolError::NonFiniteLikelihood(f64::NAN))?;
                match censoring {
                    Censor::BLOQ => normal.cdf(observation).ln(),
                    _ => normal.sf(observation).ln(),
                }
            }
        };
        if log_density.is_finite() {
            Ok(log_density)
        } else {
            Err(PharmsolError::NonFiniteLikelihood(log_density))
        }
    }

    fn labels(&self) -> Box<dyn Iterator<Item = &OutputLabel> + '_> {
        Box::new(self.labels.iter())
    }

    fn validate(&self) -> Result<(), PharmsolError> {
        if self.sigma > 0.0 && self.sigma.is_finite() {
            Ok(())
        } else {
            Err(PharmsolError::OtherError(format!(
                "sigma must be positive and finite, got {}",
                self.sigma
            )))
        }
    }

    fn cache_key(&self) -> Option<u64> {
        let mut hasher = ahash::AHasher::default();
        "NormalObservation".hash(&mut hasher);
        self.sigma.to_bits().hash(&mut hasher);
        self.labels.hash(&mut hasher);
        Some(hasher.finish())
    }
}
