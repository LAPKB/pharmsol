use crate::simulator::likelihood::progress::ProgressTracker;
use crate::{
    data::error_model::ErrorModels, Data, Equation, ErrorPoly, Observation, PharmsolError,
    Predictions,
};
use ndarray::{Array2, Axis, ShapeBuilder};
use rayon::prelude::*;
use statrs::distribution::ContinuousCDF;
use statrs::distribution::Normal;

mod progress;

const FRAC_1_SQRT_2PI: f64 =
    std::f64::consts::FRAC_2_SQRT_PI * std::f64::consts::FRAC_1_SQRT_2 / 2.0;

/// Container for predictions associated with a single subject.
///
/// This struct holds all predictions for a subject along with the corresponding
/// observations and time points.
#[derive(Debug, Clone, Default)]
pub struct SubjectPredictions {
    predictions: Vec<Prediction>,
    flat_predictions: Vec<f64>,
    flat_observations: Vec<f64>,
    flat_time: Vec<f64>,
}

impl Predictions for SubjectPredictions {
    fn squared_error(&self) -> f64 {
        self.predictions
            .iter()
            .map(|p| (p.observation - p.prediction).powi(2))
            .sum()
    }
    fn get_predictions(&self) -> Vec<Prediction> {
        self.predictions.clone()
    }
}

impl SubjectPredictions {
    /// Calculate the likelihood of the predictions given an error model.
    ///
    /// This multiplies the likelihood of each prediction to get the joint likelihood.
    ///
    /// # Parameters
    /// - `error_model`: The error model to use for calculating the likelihood
    ///
    /// # Returns
    /// The product of all individual prediction likelihoods
    pub fn likelihood(&self, error_models: &ErrorModels) -> Result<f64, PharmsolError> {
        match self.predictions.is_empty() {
            true => Ok(0.0),
            false => self
                .predictions
                .iter()
                .map(|p| p.likelihood(error_models))
                .collect::<Result<Vec<f64>, _>>()
                .map(|likelihoods| likelihoods.iter().product())
                .map_err(PharmsolError::from),
        }
    }

    /// Add a new prediction to the collection.
    ///
    /// This updates both the main predictions vector and the flat vectors.
    ///
    /// # Parameters
    /// - `prediction`: The prediction to add
    pub fn add_prediction(&mut self, prediction: Prediction) {
        self.predictions.push(prediction.clone());
        self.flat_observations.push(prediction.observation);
        self.flat_predictions.push(prediction.prediction);
        self.flat_time.push(prediction.time);
    }

    /// Get a vector of all observation values.
    ///
    /// # Returns
    /// Vector of observation values
    pub fn flat_observations(&self) -> Vec<f64> {
        self.flat_observations.to_vec()
    }

    /// Get a vector of all prediction values.
    ///
    /// # Returns
    /// Vector of prediction values
    pub fn flat_predictions(&self) -> Vec<f64> {
        self.flat_predictions.to_vec()
    }

    /// Get a vector of all time points.
    ///
    /// # Returns
    /// Vector of time points
    pub fn flat_time(&self) -> Vec<f64> {
        self.flat_time.to_vec()
    }
}

/// Probability density function of the normal distribution
#[inline(always)]
fn normpdf(obs: f64, pred: f64, sigma: f64) -> f64 {
    (FRAC_1_SQRT_2PI / sigma) * (-((obs - pred) * (obs - pred)) / (2.0 * sigma * sigma)).exp()
}
#[inline(always)]
fn normcdf(obs: f64, pred: f64, sigma: f64) -> f64 {
    let norm = Normal::new(pred, sigma).expect("σ must be > 0");
    norm.cdf(obs)
}

impl From<Vec<Prediction>> for SubjectPredictions {
    fn from(predictions: Vec<Prediction>) -> Self {
        Self {
            flat_predictions: predictions.iter().map(|p| p.prediction).collect(),
            flat_observations: predictions.iter().map(|p| p.observation).collect(),
            flat_time: predictions.iter().map(|p| p.time).collect(),
            predictions,
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

/// Calculate the psi matrix for maximum likelihood estimation.
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
/// A 2D array of likelihoods
pub fn psi(
    equation: &impl Equation,
    subjects: &Data,
    support_points: &Array2<f64>,
    error_models: &ErrorModels,
    progress: bool,
    cache: bool,
) -> Result<Array2<f64>, PharmsolError> {
    let mut psi: Array2<f64> = Array2::default((subjects.len(), support_points.nrows()).f());
    let subjects = subjects.get_subjects();

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

    let result: Result<(), PharmsolError> = psi
        .axis_iter_mut(Axis(0))
        .into_par_iter()
        .enumerate()
        .try_for_each(|(i, mut row)| {
            row.axis_iter_mut(Axis(0))
                .into_par_iter()
                .enumerate()
                .try_for_each(|(j, mut element)| {
                    let subject = subjects.get(i).unwrap();
                    match equation.estimate_likelihood(
                        subject,
                        support_points.row(j).to_vec().as_ref(),
                        error_models,
                        cache,
                    ) {
                        Ok(likelihood) => {
                            element.fill(likelihood);
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
    Ok(psi)
}

/// Prediction holds an observation and its prediction
#[derive(Debug, Clone)]
pub struct Prediction {
    pub(crate) time: f64,
    pub(crate) observation: f64,
    pub(crate) prediction: f64,
    pub(crate) outeq: usize,
    pub(crate) errorpoly: Option<ErrorPoly>,
    pub(crate) state: Vec<f64>,
}

impl Prediction {
    /// Get the time point of this prediction.
    pub fn time(&self) -> f64 {
        self.time
    }

    /// Get the observed value.
    pub fn observation(&self) -> f64 {
        self.observation
    }

    /// Get the predicted value.
    pub fn prediction(&self) -> f64 {
        self.prediction
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
    pub fn prediction_error(&self) -> f64 {
        self.prediction - self.observation
    }

    /// Calculate the percentage error as (prediction - observation)/observation * 100.
    pub fn percentage_error(&self) -> f64 {
        ((self.prediction - self.observation) / self.observation) * 100.0
    }

    /// Calculate the absolute error |prediction - observation|.
    pub fn absolute_error(&self) -> f64 {
        (self.prediction - self.observation).abs()
    }

    /// Calculate the squared error (prediction - observation)².
    pub fn squared_error(&self) -> f64 {
        (self.prediction - self.observation).powi(2)
    }

    /// Calculate the likelihood of this prediction given an error model.
    pub fn likelihood(&self, error_models: &ErrorModels) -> Result<f64, PharmsolError> {
        let sigma = error_models.sigma(self)?;

        let likelihood = if let Some(lloq) = error_models.get(self.outeq)?.lloq() {
            if self.observation <= lloq {
                normcdf(self.observation, self.prediction, sigma)
            } else {
                normpdf(self.observation, self.prediction, sigma)
            }
        } else {
            normpdf(self.observation, self.prediction, sigma)
        };

        if likelihood.is_finite() {
            return Ok(likelihood);
        } else if likelihood == 0.0 {
            return Err(PharmsolError::ZeroLikelihood);
        } else {
            return Err(PharmsolError::NonFiniteLikelihood(likelihood));
        }
    }

    /// Get the state vector at this prediction point
    pub fn state(&self) -> &Vec<f64> {
        &self.state
    }

    /// Create an [Observation] from this prediction
    pub fn to_observation(&self) -> Observation {
        Observation::new(
            self.time,
            self.observation,
            self.outeq,
            self.errorpoly,
            false,
        )
    }
}

impl Default for Prediction {
    fn default() -> Self {
        Self {
            time: 0.0,
            observation: 0.0,
            prediction: 0.0,
            outeq: 0,
            errorpoly: None,
            state: vec![],
        }
    }
}

// Implement display for Prediction
impl std::fmt::Display for Prediction {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "Time: {:.2}\tObs: {:.4}\tPred: {:.15}\tOuteq: {:.2}",
            self.time, self.observation, self.prediction, self.outeq
        )
    }
}
