use crate::{
    data::{error_model::ErrorModel, Observation},
    Data, Equation,
};

use indicatif::{ProgressBar, ProgressStyle};
use ndarray::{Array2, Axis, ShapeBuilder};
use rayon::prelude::*;

const FRAC_1_SQRT_2PI: f64 =
    std::f64::consts::FRAC_2_SQRT_PI * std::f64::consts::FRAC_1_SQRT_2 / 2.0;

/// Container for predictions associated with a single subject.
///
/// This struct holds all predictions for a subject along with the corresponding
/// observations and time points.
#[derive(Debug, Clone, Default)]
pub struct SubjectPredictions {
    pub(crate) predictions: Vec<Prediction>,
    flat_predictions: Vec<f64>,
    flat_observations: Vec<f64>,
    flat_time: Vec<f64>,
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
    pub fn likelihood(&self, error_model: &ErrorModel) -> f64 {
        self.predictions
            .iter()
            .map(|p| p.likelihood(error_model))
            .product()
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
fn normpdf(obs: f64, pred: f64, sigma: f64) -> f64 {
    (FRAC_1_SQRT_2PI / sigma) * (-((obs - pred) * (obs - pred)) / (2.0 * sigma * sigma)).exp()
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
    error_model: &ErrorModel,
    progress: bool,
    cache: bool,
) -> Array2<f64> {
    let mut pred: Array2<f64> = Array2::default((subjects.len(), support_points.nrows()).f());
    let subjects = subjects.get_subjects();
    let pb = match progress {
        true => {
            let pb = ProgressBar::new(pred.ncols() as u64 * pred.nrows() as u64);
            pb.set_style(
                ProgressStyle::with_template(
                    "Simulating subjects...\n[{elapsed_precise}] {bar:40.green} {percent}% ETA:{eta}",
                )
                .unwrap()
                .progress_chars("##-"),
            );
            Some(pb)
        }
        false => None,
    };

    pred.axis_iter_mut(Axis(0))
        .into_par_iter()
        .enumerate()
        .for_each(|(i, mut row)| {
            row.axis_iter_mut(Axis(0))
                .into_par_iter()
                .enumerate()
                .for_each(|(j, mut element)| {
                    let subject = subjects.get(i).unwrap();
                    let likelihood = equation.estimate_likelihood(
                        subject,
                        support_points.row(j).to_vec().as_ref(),
                        error_model,
                        cache,
                    );
                    element.fill(likelihood);
                    if let Some(pb_ref) = pb.as_ref() {
                        pb_ref.inc(1);
                    }
                });
        });
    if let Some(pb_ref) = pb.as_ref() {
        pb_ref.finish();
    }

    pred
}

// Commented code for disk cache
// #[io_cached(
//    disk = true,
//    map_error = r##"|e| anyhow!(e)"##,
//    convert = r#"{ format!("{}{}", subject.id(), spphash(support_point)) }"#,
//    ty = "DiskCache<String, f64>"
//)]

// #[inline(always)]
// #[cached(
//     ty = "UnboundCache<String, f64>",
//     create = "{ UnboundCache::with_capacity(100_000) }",
//     convert = r#"{ format!("{}{}", subject.id(), spphash(support_point)) }"#
// )]
// fn estimate_likelihood(
//     subject: &Subject,
//     equation: &impl Equation,
//     support_point: &Vec<f64>,
//     error_model: &ErrorModel,
// ) -> f64 {
//     let ypred = equation.simulate_subject(subject, support_point, Some(error_model));
//     ypred.1.unwrap()
// }

// fn spphash(spp: &[f64]) -> u64 {
//     spp.iter().fold(0, |acc, x| acc + x.to_bits())
// }

/// Prediction holds an observation and its prediction
#[derive(Debug, Clone)]
pub struct Prediction {
    time: f64,
    pub(crate) observation: f64,
    pub(crate) prediction: f64,
    outeq: usize,
    errorpoly: Option<(f64, f64, f64, f64)>,
    state: Vec<f64>,
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
    pub fn errorpoly(&self) -> Option<(f64, f64, f64, f64)> {
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
    pub fn likelihood(&self, error_model: &ErrorModel) -> f64 {
        let sigma = error_model.estimate_sigma(self);
        normpdf(self.observation, self.prediction, sigma)
    }

    /// Get the state vector at this prediction point.
    pub fn state(&self) -> &Vec<f64> {
        &self.state
    }
}

/// Trait for converting observations to predictions.
pub(crate) trait ToPrediction {
    /// Convert an observation to a prediction object.
    ///
    /// # Parameters
    /// - `pred`: The predicted value
    /// - `state`: The state vector at the prediction time
    ///
    /// # Returns
    /// A new prediction object
    fn to_obs_pred(&self, pred: f64, state: Vec<f64>) -> Prediction;
}

impl ToPrediction for Observation {
    fn to_obs_pred(&self, pred: f64, state: Vec<f64>) -> Prediction {
        Prediction {
            time: self.time(),
            observation: self.value(),
            prediction: pred,
            outeq: self.outeq(),
            errorpoly: self.errorpoly(),
            state,
        }
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
