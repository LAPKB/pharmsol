//! Population-level log-likelihood matrix computation.
//!
//! This module provides functions for computing log-likelihood matrices
//! across populations of subjects and parameter support points.

use ndarray::{Array2, Axis, ShapeBuilder};
use rayon::prelude::*;

use crate::data::error_model::AssayErrorModels;
use crate::{Data, Equation, PharmsolError};

use super::progress::ProgressTracker;

/// Options for log-likelihood matrix computation.
///
/// Contains flags for wether or not to show a progress bar, printed to STDOUT.
///
/// Cache behavior is controlled globally via [`crate::simulator::cache`].
#[derive(Debug, Clone)]
pub struct LikelihoodMatrixOptions {
    /// Show a progress bar during computation
    pub show_progress: bool,
}

impl Default for LikelihoodMatrixOptions {
    fn default() -> Self {
        Self {
            show_progress: false,
        }
    }
}

impl LikelihoodMatrixOptions {
    /// Create new options with default values
    pub fn new() -> Self {
        Self::default()
    }

    /// Enable progress bar display
    pub fn with_progress(mut self) -> Self {
        self.show_progress = true;
        self
    }

    /// Disable progress bar display
    pub fn without_progress(mut self) -> Self {
        self.show_progress = false;
        self
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
/// - `support_points`: The support points to evaluate (rows = support points, cols = parameters)
/// - `error_models`: The error models to use (observation-based sigma)
/// - `options`: Computation options (progress bar, caching)
///
/// # Returns
/// A 2D array of log-likelihoods with shape (n_subjects, n_support_points)
///
/// # Example
/// ```ignore
/// use pharmsol::prelude::simulator::{log_likelihood_matrix, LikelihoodMatrixOptions};
///
/// let log_liks = log_likelihood_matrix(
///     &equation,
///     &data,
///     &support_points,
///     &error_models,
///     LikelihoodMatrixOptions::new().with_progress(),
/// )?;
/// ```
pub fn log_likelihood_matrix(
    equation: &impl Equation,
    subjects: &Data,
    support_points: &Array2<f64>,
    error_models: &AssayErrorModels,
    options: LikelihoodMatrixOptions,
) -> Result<Array2<f64>, PharmsolError> {
    let mut log_psi: Array2<f64> = Array2::default((subjects.len(), support_points.nrows()).f());

    let subjects_vec = subjects.subjects();

    let progress_tracker = if options.show_progress {
        let total = subjects_vec.len() * support_points.nrows();
        println!(
            "Computing log-likelihood matrix: {} subjects × {} support points...",
            subjects_vec.len(),
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
                    let subject = subjects_vec.get(i).unwrap();
                    match equation.estimate_log_likelihood(
                        subject,
                        &support_points.row(j).to_vec(),
                        error_models,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_likelihood_matrix_options_builder() {
        let opts = LikelihoodMatrixOptions::new().with_progress();

        assert!(opts.show_progress);

        let opts2 = LikelihoodMatrixOptions::new().without_progress();

        assert!(!opts2.show_progress);
    }

    #[test]
    fn test_default_options() {
        let opts = LikelihoodMatrixOptions::default();
        assert!(!opts.show_progress);
    }
}
