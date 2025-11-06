use crate::simulator::likelihood::progress::ProgressTracker;
use crate::Censor;
use crate::ErrorModelError;
use crate::{
    data::error_model::ErrorModels, Data, Equation, ErrorPoly, Observation, PharmsolError,
    Predictions,
};
use faer::Mat;
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
                .filter(|p| p.observation.is_some())
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

/// Probability density function of the normal distribution
#[inline(always)]
fn normpdf(obs: f64, pred: f64, sigma: f64) -> f64 {
    (FRAC_1_SQRT_2PI / sigma) * (-((obs - pred) * (obs - pred)) / (2.0 * sigma * sigma)).exp()
}
#[inline(always)]
fn normcdf(obs: f64, pred: f64, sigma: f64) -> Result<f64, ErrorModelError> {
    let norm = Normal::new(pred, sigma).map_err(|_| ErrorModelError::NegativeSigma)?;
    Ok(norm.cdf(obs))
}

impl From<Vec<Prediction>> for SubjectPredictions {
    fn from(predictions: Vec<Prediction>) -> Self {
        Self {
            predictions: predictions.iter().cloned().collect(),
        }
    }
}

/// Matrix structure for storing predictions in a 2D layout.
/// Organized as rows x columns.
#[derive(Clone, Debug, Default)]
pub struct PredictionMatrix<T> {
    data: Vec<Vec<T>>,
    nrows: usize,
    ncols: usize,
}

impl<T: Clone + Default> PredictionMatrix<T> {
    /// Create a new matrix with the given dimensions
    pub fn new(nrows: usize, ncols: usize) -> Self {
        Self {
            data: vec![vec![T::default(); ncols]; nrows],
            nrows,
            ncols,
        }
    }

    /// Get the number of rows
    pub fn nrows(&self) -> usize {
        self.nrows
    }

    /// Get the number of columns
    pub fn ncols(&self) -> usize {
        self.ncols
    }

    /// Check if the matrix is empty
    pub fn is_empty(&self) -> bool {
        self.nrows == 0 || self.ncols == 0
    }

    /// Get a reference to a specific element
    pub fn get(&self, row: usize, col: usize) -> Option<&T> {
        self.data.get(row).and_then(|r| r.get(col))
    }

    /// Get a mutable reference to a specific element
    pub fn get_mut(&mut self, row: usize, col: usize) -> Option<&mut T> {
        self.data.get_mut(row).and_then(|r| r.get_mut(col))
    }

    /// Get a reference to a row
    pub fn row(&self, row: usize) -> Option<&Vec<T>> {
        self.data.get(row)
    }

    /// Get an iterator over rows
    pub fn rows(&self) -> impl Iterator<Item = &Vec<T>> {
        self.data.iter()
    }

    /// Get a column as a vector
    pub fn column(&self, col: usize) -> Vec<&T> {
        self.data.iter().filter_map(|row| row.get(col)).collect()
    }

    /// Append a column to the matrix
    pub fn append_column(&mut self, column: Vec<T>) -> Result<(), PharmsolError> {
        if column.len() != self.nrows {
            return Err(PharmsolError::OtherError(format!(
                "Column length {} does not match matrix rows {}",
                column.len(),
                self.nrows
            )));
        }
        for (row, item) in self.data.iter_mut().zip(column.into_iter()) {
            row.push(item);
        }
        self.ncols += 1;
        Ok(())
    }
}

/// Container for predictions across a population of subjects.
///
/// This struct holds predictions for multiple subjects organized in a 2D matrix.
pub struct PopulationPredictions {
    /// 2D matrix of subject predictions
    pub subject_predictions: PredictionMatrix<SubjectPredictions>,
}

impl Default for PopulationPredictions {
    fn default() -> Self {
        Self {
            subject_predictions: PredictionMatrix::new(0, 0),
        }
    }
}

impl From<PredictionMatrix<SubjectPredictions>> for PopulationPredictions {
    fn from(subject_predictions: PredictionMatrix<SubjectPredictions>) -> Self {
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
/// - `support_points`: The support points to evaluate (rows = support points, cols = parameters)
/// - `error_model`: The error model to use
/// - `progress`: Whether to show a progress bar
/// - `cache`: Whether to use caching
///
/// # Returns
/// A 2D matrix of likelihoods
pub fn psi(
    equation: &impl Equation,
    subjects: &Data,
    support_points: &Mat<f64>,
    error_models: &ErrorModels,
    progress: bool,
    cache: bool,
) -> Result<Mat<f64>, PharmsolError> {
    let nrows = subjects.len();
    let ncols = support_points.nrows();

    let subjects_vec = subjects.subjects();

    let progress_tracker = if progress {
        let total = nrows * ncols;
        println!(
            "Simulating {} subjects with {} support points each...",
            nrows, ncols
        );
        Some(ProgressTracker::new(total))
    } else {
        None
    };

    // Collect results into a flat vector in row-major order
    let mut results: Vec<f64> = vec![0.0; nrows * ncols];

    let result: Result<(), PharmsolError> =
        results
            .par_chunks_mut(ncols)
            .enumerate()
            .try_for_each(|(i, row)| {
                row.par_iter_mut().enumerate().try_for_each(|(j, element)| {
                    let subject = subjects_vec.get(i).unwrap();
                    // Convert Mat row to Vec
                    let support_point: Vec<f64> = (0..support_points.ncols())
                        .map(|k| support_points[(j, k)])
                        .collect();
                    match equation.estimate_likelihood(subject, &support_point, error_models, cache)
                    {
                        Ok(likelihood) => {
                            *element = likelihood;
                            if let Some(ref tracker) = progress_tracker {
                                tracker.inc();
                            }
                            Ok(())
                        }
                        Err(e) => Err(e),
                    }
                })
            });

    if let Some(tracker) = progress_tracker {
        tracker.finish();
    }

    result?;

    // Convert flat vector to faer::Mat
    // faer uses column-major order by default, so we need to transpose or use from_fn
    let psi = Mat::from_fn(nrows, ncols, |i, j| results[i * ncols + j]);

    Ok(psi)
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

    /// Calculate the likelihood of this prediction given an error model.
    ///
    /// Returns an error if the observation is missing or if the likelihood is either zero or non-finite.
    pub fn likelihood(&self, error_models: &ErrorModels) -> Result<f64, PharmsolError> {
        if self.observation.is_none() {
            return Err(PharmsolError::MissingObservation);
        }

        let sigma = error_models.sigma(self)?;

        //TODO: For the BLOQ and ALOQ cases, we should be using the LOQ values, not the observation values.
        let likelihood = match self.censoring {
            Censor::None => normpdf(self.observation.unwrap(), self.prediction, sigma),
            Censor::BLOQ => normcdf(self.observation.unwrap(), self.prediction, sigma)?,
            Censor::ALOQ => 1.0 - normcdf(self.observation.unwrap(), self.prediction, sigma)?,
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
