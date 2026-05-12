//! Population-level log-likelihood matrix computation.
//!
//! This module provides functions for computing log-likelihood matrices
//! across populations of subjects and parameter support points.

use ndarray::{Array2, Axis, ShapeBuilder};
use rayon::prelude::*;

use crate::data::error_model::AssayErrorModels;
use crate::{Data, Equation, PharmsolError};

use super::progress::ProgressTracker;

/// Calculate the log-likelihood matrix for all subjects and support points.
///
/// This function computes log-likelihoods directly in log-space, which is numerically
/// more stable than computing likelihoods and then taking logarithms. This is especially
/// important when dealing with many observations or extreme parameter values that could
/// cause the regular likelihood to underflow to zero.
///
/// `support_points` must already be a dense matrix in model order. If the
/// incoming columns are in an external named order, validate that order once
/// with [`crate::ParameterOrder`] and reorder before calling this function.
///
/// # Parameters
/// - `equation`: The equation to use for simulation
/// - `subjects`: The subject data
/// - `support_points`: The support points to evaluate (rows = support points, cols = parameters)
/// - `error_models`: The error models to use (observation-based sigma)
/// - `progress`: Whether to display a progress bar during computation
///
/// # Returns
/// A 2D array of log-likelihoods with shape (n_subjects, n_support_points)
///
/// # Example
/// ```ignore
/// use ndarray::array;
/// use pharmsol::{ParameterOrder, prelude::simulator::log_likelihood_matrix};
///
/// let order = ParameterOrder::with_model(&equation, ["ka", "ke"])?;
/// let support_points_in_source_order = array![[0.1, 0.3], [0.2, 0.4]];
/// let support_points = order.matrix(support_points_in_source_order)?;
///
/// let log_liks = log_likelihood_matrix(
///     &equation,
///     &data,
///     &support_points,
///     &error_models,
///     false
/// )?;
/// ```
pub fn log_likelihood_matrix(
    equation: &impl Equation,
    subjects: &Data,
    support_points: &Array2<f64>,
    error_models: &AssayErrorModels,
    progress: bool,
) -> Result<Array2<f64>, PharmsolError> {
    let n_support_points = support_points.nrows();
    let mut log_psi: Array2<f64> = Array2::default((subjects.len(), n_support_points).f());
    let subject_slice = subjects.subjects_slice();
    let support_point_rows = support_points
        .axis_iter(Axis(0))
        .map(|row| row.to_vec())
        .collect::<Vec<_>>();

    let progress_tracker = if progress {
        let total = subject_slice.len() * n_support_points;
        println!(
            "Computing log-likelihood matrix: {} subjects × {} support points...",
            subject_slice.len(),
            n_support_points
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
            let subject = &subject_slice[i];

            for (element, support_point) in row.iter_mut().zip(support_point_rows.iter()) {
                *element = equation.estimate_log_likelihood(
                    subject,
                    support_point.as_slice(),
                    error_models,
                )?;
                if let Some(ref tracker) = progress_tracker {
                    tracker.inc();
                }
            }

            Ok(())
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
    support_points: &Array2<f64>,
    error_models: &AssayErrorModels,
    progress: bool,
) -> Result<Array2<f64>, PharmsolError> {
    log_likelihood_matrix(equation, subjects, support_points, error_models, progress)
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
    support_points: &Array2<f64>,
    error_models: &AssayErrorModels,
    progress: bool,
) -> Result<Array2<f64>, PharmsolError> {
    let log_psi_matrix =
        log_likelihood_matrix(equation, subjects, support_points, error_models, progress)?;

    // Exponentiate to get likelihoods (may underflow to 0 for extreme values)
    Ok(log_psi_matrix.mapv(f64::exp))
}

#[cfg(test)]
mod tests {
    use super::log_likelihood_matrix;
    use crate::data::builder::SubjectBuilderExt;
    use crate::data::error_model::{AssayErrorModel, ErrorPoly};
    use crate::{fa, lag, metadata, AssayErrorModels, Data, ModelKind, ODE, ParameterOrder};
    use ndarray::array;

    fn likelihood_named_order_ode() -> ODE {
        ODE::new(
            |_x, _p, _t, _dx, _b, _rateiv, _cov| {},
            |_p, _t, _cov| lag! {},
            |_p, _t, _cov| fa! {},
            |_p, _t, _cov, x| {
                x[0] = 0.0;
            },
            |_x, p, _t, _cov, y| {
                y[0] = p[0] - p[1];
            },
        )
        .with_nstates(1)
        .with_ndrugs(0)
        .with_nout(1)
        .with_metadata(
            metadata::new("likelihood_named_order")
                .kind(ModelKind::Ode)
                .parameters(["v", "ke"])
                .states(["central"])
                .outputs(["cp"]),
        )
        .expect("attach metadata")
    }

    fn likelihood_error_models() -> AssayErrorModels {
        AssayErrorModels::empty()
            .add(
                0,
                AssayErrorModel::additive(ErrorPoly::new(0.0, 1.0, 0.0, 0.0), 0.0),
            )
            .expect("add error model")
    }

    #[test]
    fn parameter_order_feeds_likelihood_matrix_once() {
        let equation = likelihood_named_order_ode();
        let data = Data::from(
            crate::Subject::builder("likelihood-named-order")
                .observation(0.0, 9.5, "cp")
                .build(),
        );
        let error_models = likelihood_error_models();
        let manual_support_points = array![[10.0, 0.5], [20.0, 0.7]];
        let source_order_support_points = array![[0.5, 10.0], [0.7, 20.0]];
        let order = ParameterOrder::with_model(&equation, ["ke", "v"]).unwrap();
        let reordered_support_points = order.matrix(source_order_support_points.clone()).unwrap();

        assert_eq!(reordered_support_points, manual_support_points);

        let manual = log_likelihood_matrix(
            &equation,
            &data,
            &manual_support_points,
            &error_models,
            false,
        )
        .unwrap();
        let reordered = log_likelihood_matrix(
            &equation,
            &data,
            &reordered_support_points,
            &error_models,
            false,
        )
        .unwrap();
        let unreordered = log_likelihood_matrix(
            &equation,
            &data,
            &source_order_support_points,
            &error_models,
            false,
        )
        .unwrap();

        assert_eq!(manual, reordered);
        assert_ne!(manual, unreordered);
    }
}
