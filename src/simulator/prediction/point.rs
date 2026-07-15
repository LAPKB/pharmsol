//! Single-point prediction data.
//!
//! [`Prediction`] carries observation/prediction pairs and simulation metadata.

use crate::{Censor, ErrorPoly};

/// Prediction holds an observation and its prediction at a single time point.
///
/// This struct exposes the fields consumers need to inspect a simulated point:
/// the time, the (optional) observed value, the noiseless model prediction, the
/// output index, optional [`ErrorPoly`] data, the state vector, the occasion
/// index, and the censoring state.
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

    /// Get the C0-C3 data carried verbatim from the source observation.
    pub fn errorpoly(&self) -> Option<ErrorPoly> {
        self.errorpoly
    }

    /// Get the state vector at this prediction point
    pub fn state(&self) -> &[f64] {
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

impl std::fmt::Display for Prediction {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let obs_str = match self.observation {
            Some(obs) => format!("{:.4}", obs),
            None => "NA".to_string(),
        };
        write!(
            f,
            "Time: {:.2}\tObs: {:.4}\tPred: {:.4}\tOuteq: {}",
            self.time, obs_str, self.prediction, self.outeq
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prediction_exposes_simulation_fields() {
        let errorpoly = Some(ErrorPoly::new(0.1, 0.2, 0.3, 0.4));
        let prediction = Prediction {
            time: 1.0,
            observation: Some(10.0),
            prediction: 12.0,
            outeq: 0,
            errorpoly,
            state: vec![12.0],
            occasion: 0,
            censoring: Censor::None,
        };

        assert_eq!(prediction.time(), 1.0);
        assert_eq!(prediction.observation(), Some(10.0));
        assert_eq!(prediction.prediction(), 12.0);
        assert_eq!(prediction.outeq(), 0);
        assert_eq!(prediction.errorpoly(), errorpoly);
        assert_eq!(prediction.state(), &[12.0]);
    }
}
