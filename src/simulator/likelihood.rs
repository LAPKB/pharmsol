use crate::data::{error_model::ErrorModel, Observation};

use ndarray::{Array1, Array2, Axis};
use rayon::prelude::*;

#[derive(Debug, Clone, Default)]
pub struct SubjectPredictions {
    predictions: Vec<Prediction>,
    flat_predictions: Array1<f64>,
    flat_observations: Array1<f64>,
}
impl SubjectPredictions {
    pub fn get_predictions(&self) -> &Vec<Prediction> {
        &self.predictions
    }

    pub(crate) fn likelihood(&self, error_model: &ErrorModel) -> f64 {
        //TODO: This sigma should not be calculated here, we should precalculate it and inject it into the struct
        let sigma: Array1<f64> = self
            .predictions
            .iter()
            .map(|p| error_model.estimate_sigma(p))
            .collect();

        normal_likelihood(&self.flat_predictions, &self.flat_observations, &sigma)
    }

    pub(crate) fn squared_error(&self) -> f64 {
        self.predictions
            .iter()
            .map(|p| (p.observation - p.prediction).powi(2))
            .sum()
    }
}
fn normal_likelihood(
    predictions: &Array1<f64>,
    observations: &Array1<f64>,
    sigma: &Array1<f64>,
) -> f64 {
    const FRAC_1_SQRT_2PI: f64 =
        std::f64::consts::FRAC_2_SQRT_PI * std::f64::consts::FRAC_1_SQRT_2 / 2.0;
    let diff = (observations - predictions).mapv(|x| x.powi(2));
    let two_sigma_sq = (2.0 * sigma).mapv(|x| x.powi(2));
    let aux_vec = FRAC_1_SQRT_2PI * (-&diff / two_sigma_sq).mapv(|x| x.exp()) / sigma;
    aux_vec.product()
}
impl From<Vec<Prediction>> for SubjectPredictions {
    fn from(predictions: Vec<Prediction>) -> Self {
        Self {
            flat_predictions: predictions.iter().map(|p| p.prediction).collect(),
            flat_observations: predictions.iter().map(|p| p.observation).collect(),
            predictions,
        }
    }
}

pub struct PopulationPredictions {
    pub subject_predictions: Array2<SubjectPredictions>,
}

impl Default for PopulationPredictions {
    fn default() -> Self {
        Self {
            subject_predictions: Array2::default((0, 0)),
        }
    }
}

impl PopulationPredictions {
    pub fn get_psi(&self, ep: &ErrorModel) -> Array2<f64> {
        let mut psi = Array2::zeros((
            self.subject_predictions.nrows(),
            self.subject_predictions.ncols(),
        ));
        psi.axis_iter_mut(Axis(0))
            .into_par_iter()
            .enumerate()
            .for_each(|(i, mut row)| {
                row.axis_iter_mut(Axis(0))
                    .into_par_iter()
                    .enumerate()
                    .for_each(|(j, mut element)| {
                        element.fill(self.subject_predictions.get((i, j)).unwrap().likelihood(ep));
                    })
            });
        psi
    }
}

impl From<Array2<SubjectPredictions>> for PopulationPredictions {
    fn from(subject_predictions: Array2<SubjectPredictions>) -> Self {
        Self {
            subject_predictions,
        }
    }
}

/// Prediction holds an observation and its prediction
#[derive(Debug, Clone)]
pub struct Prediction {
    time: f64,
    observation: f64,
    prediction: f64,
    outeq: usize,
    errorpoly: Option<(f64, f64, f64, f64)>,
}

impl Prediction {
    pub fn time(&self) -> f64 {
        self.time
    }
    pub(crate) fn observation(&self) -> f64 {
        self.observation
    }
    pub fn prediction(&self) -> f64 {
        self.prediction
    }
    pub fn outeq(&self) -> usize {
        self.outeq
    }
    pub(crate) fn errorpoly(&self) -> Option<(f64, f64, f64, f64)> {
        self.errorpoly
    }
}

pub(crate) trait ToPrediction {
    fn to_obs_pred(&self, pred: f64) -> Prediction;
}

impl ToPrediction for Observation {
    fn to_obs_pred(&self, pred: f64) -> Prediction {
        Prediction {
            time: self.time(),
            observation: self.value(),
            prediction: pred,
            outeq: self.outeq(),
            errorpoly: self.errorpoly(),
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
