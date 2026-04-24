//! Population-level log-likelihood matrix computation.
//!
//! This module provides functions for computing log-likelihood matrices
//! across populations of subjects and parameter parameters.

use ndarray::{Array2, Axis, ShapeBuilder};
use rayon::prelude::*;

use crate::data::error_model::AssayErrorModels;
use crate::{Data, Equation, PharmsolError};

use super::progress::ProgressTracker;

/// Calculate the log-likelihood matrix for all subjects and parameters.
///
/// This function computes log-likelihoods directly in log-space, which is numerically
/// more stable than computing likelihoods and then taking logarithms. This is especially
/// important when dealing with many observations or extreme parameter values that could
/// cause the regular likelihood to underflow to zero.
///
/// # Parameters
/// - `equation`: The equation to use for simulation
/// - `subjects`: The subject data
/// - `parameters`: The parameters to evaluate (rows = parameters, cols = parameters)
/// - `error_models`: The error models to use (observation-based sigma)
/// - `progress`: Whether to display a progress bar during computation`
///
/// # Returns
/// A 2D array of log-likelihoods with shape (n_subjects, n_parameters)
///
/// # Example
/// ```ignore
/// use pharmsol::prelude::simulator::log_likelihood_matrix;
///
/// let log_liks = log_likelihood_matrix(
///     &equation,
///     &data,
///     &parameters,
///     &error_models,
///     false
/// )?;
/// ```
pub fn log_likelihood_matrix(
    equation: &impl Equation,
    subjects: &Data,
    parameters: &Array2<f64>,
    error_models: &AssayErrorModels,
    progress: bool,
) -> Result<Array2<f64>, PharmsolError> {
    let mut log_psi: Array2<f64> = Array2::default((subjects.len(), parameters.nrows()).f());

    let subjects_vec = subjects.subjects();

    let progress_tracker = if progress {
        let total = subjects_vec.len() * parameters.nrows();
        println!(
            "Computing log-likelihood matrix: {} subjects × {} parameters...",
            subjects_vec.len(),
            parameters.nrows()
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
                        &parameters.row(j).to_vec(),
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

/// Calculate the log-likelihood matrix (deprecated signature with boolean flags).
///
/// Deprecated: Use [log_likelihood_matrix] with [LikelihoodMatrixOptions] instead.
///
/// This function is provided for backward compatibility with the old log_psi API.
#[deprecated(
    since = "0.23.0",
    note = "Use log_likelihood_matrix() with LikelihoodMatrixOptions instead"
)]
pub fn log_psi(
    equation: &impl Equation,
    subjects: &Data,
    parameters: &Array2<f64>,
    error_models: &AssayErrorModels,
    progress: bool,
) -> Result<Array2<f64>, PharmsolError> {
    log_likelihood_matrix(equation, subjects, parameters, error_models, progress)
}

/// Calculate the likelihood matrix (deprecated).
///
/// Deprecated: Use [log_likelihood_matrix] instead. This function exponentiates
/// the log-likelihood matrix, which can cause numerical underflow for many observations
/// or extreme parameter values.
///
/// This function is provided for backward compatibility with the old psi API.
#[deprecated(
    since = "0.23.0",
    note = "Use log_likelihood_matrix() instead and exponentiate if needed"
)]

pub fn psi(
    equation: &impl Equation,
    subjects: &Data,
    parameters: &Array2<f64>,
    error_models: &AssayErrorModels,
    progress: bool,
) -> Result<Array2<f64>, PharmsolError> {
    let log_psi_matrix =
        log_likelihood_matrix(equation, subjects, parameters, error_models, progress)?;

    // Exponentiate to get likelihoods (may underflow to 0 for extreme values)
    Ok(log_psi_matrix.mapv(f64::exp))
}
