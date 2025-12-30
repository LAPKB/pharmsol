use crate::simulator::likelihood::progress::ProgressTracker;
use crate::Censor;
use crate::ErrorModelError;
use crate::{
    data::error_model::ErrorModels, Data, Equation, ErrorPoly, Observation, PharmsolError,
    Predictions,
};
use ndarray::{Array2, Axis, ShapeBuilder};
use rayon::prelude::*;
use statrs::distribution::ContinuousCDF;
use statrs::distribution::Normal;

mod progress;

// ln(2π) = ln(2) + ln(π) ≈ 1.8378770664093453
const LOG_2PI: f64 = 1.8378770664093453_f64;

/// Container for predictions associated with a single subject.
///
/// This struct holds all predictions for a subject along with the corresponding
/// observations and time points.
#[derive(Debug, Clone, Default)]
pub struct SubjectPredictions {
    predictions: Vec<Prediction>,
}

impl Predictions for SubjectPredictions {
    fn squared_error(&self) -> f64 {
        self.predictions
            .iter()
            .filter_map(|p| p.observation.map(|obs| (obs - p.prediction).powi(2)))
            .sum()
    }
    fn get_predictions(&self) -> Vec<Prediction> {
        self.predictions.clone()
    }
    fn log_likelihood(&self, error_models: &ErrorModels) -> Result<f64, PharmsolError> {
        SubjectPredictions::log_likelihood(self, error_models)
    }
}

impl SubjectPredictions {
    /// Calculate the log-likelihood of the predictions given an error model.
    ///
    /// This sums the log-likelihood of each prediction to get the joint log-likelihood.
    /// This is numerically more stable than computing the product of likelihoods,
    /// especially for many observations or extreme values.
    ///
    /// # Parameters
    /// - `error_models`: The error models to use for calculating the likelihood
    ///
    /// # Returns
    /// The sum of all individual prediction log-likelihoods
    pub fn log_likelihood(&self, error_models: &ErrorModels) -> Result<f64, PharmsolError> {
        if self.predictions.is_empty() {
            return Ok(0.0); // log(0) for empty predictions
        }

        let log_liks: Result<Vec<f64>, _> = self
            .predictions
            .iter()
            .filter(|p| p.observation.is_some())
            .map(|p| p.log_likelihood(error_models))
            .collect();

        log_liks.map(|lls| lls.iter().sum())
    }

    /// Add a new prediction to the collection.
    ///
    /// This updates both the main predictions vector and the flat vectors.
    ///
    /// # Parameters
    /// - `prediction`: The prediction to add
    pub fn add_prediction(&mut self, prediction: Prediction) {
        self.predictions.push(prediction.clone());
    }

    /// Get a reference to a vector of predictions.
    ///
    /// # Returns
    /// Vector of observation values
    pub fn predictions(&self) -> &Vec<Prediction> {
        &self.predictions
    }

    /// Return a flat vector of predictions.
    pub fn flat_predictions(&self) -> Vec<f64> {
        self.predictions
            .iter()
            .map(|p| p.prediction)
            .collect::<Vec<f64>>()
    }

    /// Return a flat vector of predictions.
    pub fn flat_times(&self) -> Vec<f64> {
        self.predictions
            .iter()
            .map(|p| p.time)
            .collect::<Vec<f64>>()
    }

    /// Return a flat vector of observations.
    pub fn flat_observations(&self) -> Vec<Option<f64>> {
        self.predictions
            .iter()
            .map(|p| p.observation)
            .collect::<Vec<Option<f64>>>()
    }
}

/// Log of the probability density function of the normal distribution.
///
/// This is numerically stable and avoids underflow for extreme values.
/// Returns: -0.5 * ln(2π) - ln(σ) - (obs - pred)² / (2σ²)
#[inline(always)]
fn lognormpdf(obs: f64, pred: f64, sigma: f64) -> f64 {
    let diff = obs - pred;
    -0.5 * LOG_2PI - sigma.ln() - (diff * diff) / (2.0 * sigma * sigma)
}

/// Log of the cumulative distribution function of the normal distribution.
///
/// Uses the error function for numerical stability.
#[inline(always)]
fn lognormcdf(obs: f64, pred: f64, sigma: f64) -> Result<f64, ErrorModelError> {
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
#[inline(always)]
fn lognormccdf(obs: f64, pred: f64, sigma: f64) -> Result<f64, ErrorModelError> {
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

impl From<Vec<Prediction>> for SubjectPredictions {
    fn from(predictions: Vec<Prediction>) -> Self {
        Self {
            predictions: predictions.iter().cloned().collect(),
        }
    }
}

/// Container for predictions across a population of subjects.
///
/// This struct holds predictions for multiple subjects organized in a 2D array.
pub struct PopulationPredictions {
    /// 2D array of subject predictions
    pub subject_predictions: Array2<SubjectPredictions>,
}

impl Default for PopulationPredictions {
    fn default() -> Self {
        Self {
            subject_predictions: Array2::default((0, 0)),
        }
    }
}

impl From<Array2<SubjectPredictions>> for PopulationPredictions {
    fn from(subject_predictions: Array2<SubjectPredictions>) -> Self {
        Self {
            subject_predictions,
        }
    }
}

/// Calculate the log-likelihood matrix for all subjects and support points.
///
/// This function computes log-likelihoods directly in log-space, which is numerically
/// more stable than computing likelihoods and then taking logarithms. This is especially
/// important when dealing with many observations or extreme parameter values that could
/// cause the regular likelihood to underflow to zero.
///
/// # Parameters
/// - `equation`: The equation to use for simulation
/// - `subjects`: The subject data
/// - `support_points`: The support points to evaluate
/// - `error_model`: The error model to use
/// - `progress`: Whether to show a progress bar
/// - `cache`: Whether to use caching
///
/// # Returns
/// A 2D array of log-likelihoods with shape (n_subjects, n_support_points)
pub fn psi(
    equation: &impl Equation,
    subjects: &Data,
    support_points: &Array2<f64>,
    error_models: &ErrorModels,
    progress: bool,
    cache: bool,
) -> Result<Array2<f64>, PharmsolError> {
    let mut log_psi: Array2<f64> = Array2::default((subjects.len(), support_points.nrows()).f());

    let subjects = subjects.subjects();

    let progress_tracker = if progress {
        let total = subjects.len() * support_points.nrows();
        println!(
            "Simulating {} subjects with {} support points each...",
            subjects.len(),
            support_points.nrows()
        );
        Some(ProgressTracker::new(total))
    } else {
        None
    };

    let result: Result<(), PharmsolError> = log_psi
        .axis_iter_mut(Axis(0))
        .into_par_iter()
        .enumerate()
        .try_for_each(|(i, mut row)| {
            row.axis_iter_mut(Axis(0))
                .into_par_iter()
                .enumerate()
                .try_for_each(|(j, mut element)| {
                    let subject = subjects.get(i).unwrap();
                    match equation.estimate_log_likelihood(
                        subject,
                        support_points.row(j).to_vec().as_ref(),
                        error_models,
                        cache,
                    ) {
                        Ok(log_likelihood) => {
                            element.fill(log_likelihood);
                            if let Some(ref tracker) = progress_tracker {
                                tracker.inc();
                            }
                        }
                        Err(e) => return Err(e),
                    };
                    Ok(())
                })
        });

    if let Some(tracker) = progress_tracker {
        tracker.finish();
    }

    result?;
    Ok(log_psi)
}

/// Prediction holds an observation and its prediction
#[derive(Debug, Clone)]
pub struct Prediction {
    pub(crate) time: f64,
    pub(crate) observation: Option<f64>,
    pub(crate) prediction: f64,
    pub(crate) outeq: usize,
    pub(crate) errorpoly: Option<ErrorPoly>,
    pub(crate) state: Vec<f64>,
    pub(crate) occasion: usize,
    pub(crate) censoring: Censor,
}

impl Prediction {
    /// Get the time point of this prediction.
    pub fn time(&self) -> f64 {
        self.time
    }

    /// Get the observed value.
    pub fn observation(&self) -> Option<f64> {
        self.observation
    }

    /// Get the predicted value.
    pub fn prediction(&self) -> f64 {
        self.prediction
    }

    /// Set the predicted value
    pub(crate) fn set_prediction(&mut self, prediction: f64) {
        self.prediction = prediction;
    }

    /// Get the output equation index.
    pub fn outeq(&self) -> usize {
        self.outeq
    }

    /// Get the error polynomial coefficients, if available.
    pub fn errorpoly(&self) -> Option<ErrorPoly> {
        self.errorpoly
    }

    /// Calculate the raw prediction error (prediction - observation).
    pub fn prediction_error(&self) -> Option<f64> {
        self.observation.map(|obs| self.prediction - obs)
    }

    /// Calculate the percentage error as (prediction - observation)/observation * 100.
    pub fn percentage_error(&self) -> Option<f64> {
        self.observation
            .map(|obs| ((self.prediction - obs) / obs) * 100.0)
    }

    /// Calculate the absolute error |prediction - observation|.
    pub fn absolute_error(&self) -> Option<f64> {
        self.observation.map(|obs| (self.prediction - obs).abs())
    }

    /// Calculate the squared error (prediction - observation)².
    pub fn squared_error(&self) -> Option<f64> {
        self.observation.map(|obs| (self.prediction - obs).powi(2))
    }

    /// Calculate the log-likelihood of this prediction given an error model.
    ///
    /// This method is numerically stable and avoids underflow issues that can occur
    /// with the standard likelihood calculation for extreme values.
    ///
    /// Returns an error if the observation is missing or if the log-likelihood is non-finite.
    #[inline]
    pub fn log_likelihood(&self, error_models: &ErrorModels) -> Result<f64, PharmsolError> {
        if self.observation.is_none() {
            return Err(PharmsolError::MissingObservation);
        }

        let sigma = error_models.sigma(self)?;
        let obs = self.observation.unwrap();

        let log_lik = match self.censoring {
            Censor::None => lognormpdf(obs, self.prediction, sigma),
            Censor::BLOQ => lognormcdf(obs, self.prediction, sigma)?,
            Censor::ALOQ => lognormccdf(obs, self.prediction, sigma)?,
        };

        if log_lik.is_finite() {
            Ok(log_lik)
        } else {
            Err(PharmsolError::NonFiniteLikelihood(log_lik))
        }
    }

    /// Get the state vector at this prediction point
    pub fn state(&self) -> &Vec<f64> {
        &self.state
    }

    /// Get the occasion index
    pub fn occasion(&self) -> usize {
        self.occasion
    }

    /// Get a mutable reference to the occasion index
    pub fn mut_occasion(&mut self) -> &mut usize {
        &mut self.occasion
    }

    /// Get the censoring status
    pub fn censoring(&self) -> Censor {
        self.censoring
    }

    /// Create an [Observation] from this prediction
    pub fn to_observation(&self) -> Observation {
        Observation::new(
            self.time,
            self.observation,
            self.outeq,
            self.errorpoly,
            self.occasion,
            self.censoring,
        )
    }
}

impl Default for Prediction {
    fn default() -> Self {
        Self {
            time: 0.0,
            observation: None,
            prediction: 0.0,
            outeq: 0,
            errorpoly: None,
            state: vec![],
            occasion: 0,
            censoring: Censor::None,
        }
    }
}

// Implement display for Prediction
impl std::fmt::Display for Prediction {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let obs_str = match self.observation {
            Some(obs) => format!("{:.4}", obs),
            None => "NA".to_string(),
        };
        write!(
            f,
            "Time: {:.2}\tObs: {:.4}\tPred: {:.4}\tOuteq: {:.2}",
            self.time, obs_str, self.prediction, self.outeq
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::error_model::{ErrorModel, ErrorPoly};
    use crate::data::event::Observation;
    use crate::Censor;

    #[test]
    fn empty_predictions_have_neutral_log_likelihood() {
        let preds = SubjectPredictions::default();
        let errors = ErrorModels::new();
        assert_eq!(preds.log_likelihood(&errors).unwrap(), 0.0); // log(1) = 0
    }

    #[test]
    fn log_likelihood_combines_observations() {
        let mut preds = SubjectPredictions::default();
        let obs = Observation::new(0.0, Some(1.0), 0, None, 0, Censor::None);
        preds.add_prediction(obs.to_prediction(1.0, vec![]));

        let error_model = ErrorModel::additive(ErrorPoly::new(1.0, 0.0, 0.0, 0.0), 0.0);
        let errors = ErrorModels::new().add(0, error_model).unwrap();

        // log_likelihood should be finite and negative (log of a probability < 1)
        let ll = preds.log_likelihood(&errors).unwrap();
        assert!(ll.is_finite());
    }

    #[test]
    fn test_log_likelihood_basic() {
        // Create a prediction with an observation
        let prediction = Prediction {
            time: 1.0,
            observation: Some(10.0),
            prediction: 10.5,
            outeq: 0,
            errorpoly: None,
            state: vec![10.5],
            occasion: 0,
            censoring: Censor::None,
        };

        // Create error model with additive error
        let error_models = ErrorModels::new()
            .add(
                0,
                ErrorModel::additive(ErrorPoly::new(0.0, 1.0, 0.0, 0.0), 0.0),
            )
            .unwrap();

        let log_lik = prediction.log_likelihood(&error_models).unwrap();

        // log_likelihood should be finite and negative (log of probability < 1)
        assert!(log_lik.is_finite());
        assert!(log_lik < 0.0, "log_likelihood should be negative");
    }

    #[test]
    fn test_log_likelihood_numerical_stability() {
        // Test with values that would cause very small likelihood
        let prediction = Prediction {
            time: 1.0,
            observation: Some(10.0),
            prediction: 30.0, // Far from observation (20 sigma away with sigma=1)
            outeq: 0,
            errorpoly: None,
            state: vec![30.0],
            occasion: 0,
            censoring: Censor::None,
        };

        // Using c0=1.0 (constant error term) to ensure sigma=1 regardless of observation
        let error_models = ErrorModels::new()
            .add(
                0,
                ErrorModel::additive(ErrorPoly::new(1.0, 0.0, 0.0, 0.0), 0.0),
            )
            .unwrap();

        // log_likelihood should give a finite (very negative) value
        let log_lik = prediction.log_likelihood(&error_models).unwrap();

        assert!(log_lik.is_finite(), "log_likelihood should be finite");
        assert!(
            log_lik < -100.0,
            "log_likelihood should be very negative for large mismatch"
        );
    }

    #[test]
    fn test_log_likelihood_extreme_underflow() {
        // Test with truly extreme values where regular likelihood would underflow to 0
        let prediction = Prediction {
            time: 1.0,
            observation: Some(10.0),
            prediction: 50.0, // 40 sigma away - regular pdf ≈ 10^{-350}
            outeq: 0,
            errorpoly: None,
            state: vec![50.0],
            occasion: 0,
            censoring: Censor::None,
        };

        // Using c0=1.0 (constant error term) to ensure sigma=1 regardless of observation
        let error_models = ErrorModels::new()
            .add(
                0,
                ErrorModel::additive(ErrorPoly::new(1.0, 0.0, 0.0, 0.0), 0.0),
            )
            .unwrap();

        // log_likelihood should still work even for extreme values
        let log_lik = prediction.log_likelihood(&error_models).unwrap();

        assert!(
            log_lik.is_finite(),
            "log_likelihood should be finite even for extreme values"
        );
        assert!(log_lik < -100.0, "log_likelihood should be very negative");

        // For 40 sigma away: log_lik ≈ -0.5*ln(2π) - ln(1) - (40)^2/2 ≈ -800
        assert!(
            log_lik < -700.0 && log_lik > -900.0,
            "log_likelihood ({}) should be approximately -800 for 40 sigma away",
            log_lik
        );
    }

    #[test]
    fn test_subject_predictions_log_likelihood() {
        let predictions = vec![
            Prediction {
                time: 1.0,
                observation: Some(10.0),
                prediction: 10.1,
                outeq: 0,
                errorpoly: None,
                state: vec![10.1],
                occasion: 0,
                censoring: Censor::None,
            },
            Prediction {
                time: 2.0,
                observation: Some(8.0),
                prediction: 8.2,
                outeq: 0,
                errorpoly: None,
                state: vec![8.2],
                occasion: 0,
                censoring: Censor::None,
            },
        ];

        let subject_predictions = SubjectPredictions::from(predictions);
        let error_models = ErrorModels::new()
            .add(
                0,
                ErrorModel::additive(ErrorPoly::new(0.0, 1.0, 0.0, 0.0), 0.0),
            )
            .unwrap();

        let log_lik = subject_predictions.log_likelihood(&error_models).unwrap();

        // log_likelihood should be finite and negative
        assert!(log_lik.is_finite());
        assert!(log_lik < 0.0, "Combined log_likelihood should be negative");
    }

    #[test]
    fn test_lognormpdf_direct() {
        // Test the helper function directly
        let obs = 0.0;
        let pred = 0.0;
        let sigma = 1.0;

        let log_pdf = lognormpdf(obs, pred, sigma);

        // For standard normal at x=0: pdf = 1/sqrt(2*pi) ≈ 0.3989
        // log(pdf) ≈ -0.9189
        let expected = -0.5 * LOG_2PI; // -0.5 * ln(2π)
        assert!(
            (log_pdf - expected).abs() < 1e-12,
            "lognormpdf at x=0 should equal -0.5*ln(2π), got {} expected {}",
            log_pdf,
            expected
        );
    }
}
