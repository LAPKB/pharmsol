//! Prediction containers and the parallel prediction driver.
//!
//! An [`Equation`] turns a subject and a parameter vector into predictions:
//!
//! - [`Prediction`]: one observation paired with its predicted value
//! - [`SubjectPredictions`]: a deterministic subject's predictions
//! - [`ParticlePredictions`]: a stochastic subject's particle predictions
//! - [`PopulationPredictions`]: subject predictions laid out by subject and
//!   support point
//! - [`PointPredictions`]: one prediction per observation, whatever the
//!   container
//!
//! [`map_predictions`] simulates every subject at every support point in
//! parallel and hands each result to a closure. pharmsol does not score
//! observations: likelihoods and error models live downstream, and
//! [`map_predictions`] is the building block they compute likelihood
//! matrices with. The one exception is the [`SDE`](crate::SDE) particle
//! filter, which needs an observation model while it simulates; see
//! [`ObservationModel`](crate::ObservationModel).

mod particles;
mod prediction;
mod subject;

use std::borrow::Cow;
use std::sync::Arc;

pub use particles::ParticlePredictions;
pub use prediction::Prediction;
pub use subject::{PopulationPredictions, SubjectPredictions};

use ndarray::{Array2, ArrayView1, Axis, ShapeBuilder, Zip};

use crate::{Data, Equation, PharmsolError, Subject};

/// Prediction container returned by an [`Equation`], reduced to one
/// prediction per observation.
///
/// Deterministic equations return [`SubjectPredictions`], which already are
/// one prediction per observation and are borrowed. [`SDE`](crate::SDE)
/// returns [`ParticlePredictions`], whose point is the (weighted) particle
/// mean, built on each call.
///
/// The trait has no likelihood: a stochastic model's likelihood is not a
/// function of any one set of predictions.
pub trait PointPredictions {
    /// One prediction per observation.
    fn point(&self) -> Cow<'_, [Prediction]>;
}

/// Simulate every subject at every support point, in parallel, and map each
/// result through `f`.
///
/// This is the building block for likelihood matrices: `f` receives the
/// subject and the simulation result for one (subject, support point) cell
/// and returns that cell's value, e.g. the log-likelihood of the subject's
/// observations under an error model.
///
/// `support_points` is a dense matrix in model order, one support point per
/// row. If its columns come in an external named order, validate that order
/// once with [`crate::ParameterOrder`] and reorder before calling this
/// function.
///
/// Each cell is simulated with [`Equation::predict_shared`], so cached
/// predictions are handed to `f` without copying them.
///
/// # Returns
///
/// An `n_subjects × n_support_points` matrix: row `i` is subject `i`, column
/// `j` is support point `j`. It is stored in column-major (Fortran) order,
/// and subjects are processed in parallel, each over all support points.
///
/// # Errors
///
/// Only for shape problems: a support-point matrix with no columns (no
/// parameters) while there are support points. A cell whose simulation fails
/// is not an error here; `f` receives the `Err` and decides what it means,
/// e.g. a log-likelihood of negative infinity.
///
/// # Example
///
/// ```rust
/// use ndarray::array;
/// use pharmsol::prelude::*;
/// use pharmsol::map_predictions;
///
/// let model = analytical! {
///     name: "one_cmt_iv",
///     params: [ke, v],
///     states: [central],
///     outputs: [cp],
///     routes: [
///         infusion(iv) -> central,
///     ],
///     structure: one_compartment,
///     out: |x, _p, _t, _cov, y| {
///         y[cp] = x[central] / v;
///     },
/// };
/// let data = Data::new(vec![Subject::builder("1")
///     .infusion(0.0, 500.0, "iv", 0.5)
///     .observation(1.0, 2.0, "cp")
///     .build()]);
/// let support_points = array![[1.0, 190.0], [1.1, 200.0]];
///
/// // Sum of squared errors of every subject at every support point.
/// let sse = map_predictions(&model, &data, &support_points, |_subject, result| {
///     result.map_or(f64::INFINITY, |predictions| predictions.squared_error())
/// })?;
/// assert_eq!(sse.dim(), (1, 2));
/// # Ok::<(), PharmsolError>(())
/// ```
pub fn map_predictions<E, T, F>(
    equation: &E,
    subjects: &Data,
    support_points: &Array2<f64>,
    f: F,
) -> Result<Array2<T>, PharmsolError>
where
    E: Equation,
    T: Send,
    F: Fn(&Subject, Result<Arc<E::P>, PharmsolError>) -> T + Sync,
{
    let n_support_points = support_points.nrows();
    if n_support_points > 0 && support_points.ncols() == 0 {
        return Err(PharmsolError::OtherError(format!(
            "support points have {n_support_points} rows but no parameter columns"
        )));
    }

    let subject_slice = subjects.subjects_slice();
    let support_point_rows = support_points
        .axis_iter(Axis(0))
        .map(|row| row.to_vec())
        .collect::<Vec<_>>();

    // The result is allocated once, column-major, and filled in place: one
    // parallel task per subject (row), each over every support point.
    // `T` need not implement `Default` or `Clone`, so the cells start
    // uninitialized.
    // (`Zip` rather than `axis_iter_mut().into_par_iter()`, which would
    // also require `T: Sync`.)
    let mut cells = Array2::<T>::uninit((subject_slice.len(), n_support_points).f());
    Zip::from(cells.rows_mut())
        .and(ArrayView1::from(subject_slice))
        .par_for_each(|mut row, subject| {
            for (cell, support_point) in row.iter_mut().zip(&support_point_rows) {
                cell.write(f(
                    subject,
                    equation.predict_shared(subject, support_point.as_slice()),
                ));
            }
        });
    // SAFETY: `cells` has one row per subject and one column per support
    // point, and the loop above wrote every cell of every row. A panic in
    // `f` propagates out of `for_each` before this point, leaking (not
    // reading) the cells already written.
    Ok(unsafe { cells.assume_init() })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::builder::SubjectBuilderExt;
    use crate::{fa, lag, metadata, Cache, ModelKind, ParameterOrder, ODE};
    use ndarray::array;

    fn named_order_ode() -> ODE {
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
            metadata::new("named_order")
                .kind(ModelKind::Ode)
                .parameters(["v", "ke"])
                .states(["central"])
                .outputs(["cp"]),
        )
        .expect("attach metadata")
    }

    fn first_prediction(
        _subject: &Subject,
        result: Result<Arc<SubjectPredictions>, PharmsolError>,
    ) -> f64 {
        result.unwrap().predictions()[0].prediction()
    }

    #[test]
    fn rows_are_subjects_and_columns_are_support_points() {
        let equation = named_order_ode();
        let data = Data::new(vec![
            Subject::builder("a").observation(0.0, 1.0, "cp").build(),
            Subject::builder("b").missing_observation(0.0, "cp").build(),
        ]);
        let support_points = array![[10.0, 0.5], [20.0, 0.7], [30.0, 1.0]];

        let matrix = map_predictions(&equation, &data, &support_points, first_prediction).unwrap();

        assert_eq!(matrix.dim(), (2, 3));
        assert!(!matrix.is_standard_layout(), "column-major like before");
        for i in 0..2 {
            assert_eq!(matrix.row(i).to_vec(), vec![9.5, 19.3, 29.0]);
        }

        let ids = map_predictions(&equation, &data, &support_points, |subject, _| {
            subject.id().to_string()
        })
        .unwrap();
        assert_eq!(ids[[0, 2]], "a");
        assert_eq!(ids[[1, 0]], "b");
    }

    #[test]
    fn parameter_order_feeds_map_predictions_once() {
        let equation = named_order_ode();
        let data = Data::from(
            Subject::builder("named-order")
                .observation(0.0, 9.5, "cp")
                .build(),
        );
        let manual_support_points = array![[10.0, 0.5], [20.0, 0.7]];
        let source_order_support_points = array![[0.5, 10.0], [0.7, 20.0]];
        let order = ParameterOrder::with_model(&equation, ["ke", "v"]).unwrap();
        let reordered_support_points = order.matrix(source_order_support_points.clone()).unwrap();

        assert_eq!(reordered_support_points, manual_support_points);

        let manual =
            map_predictions(&equation, &data, &manual_support_points, first_prediction).unwrap();
        let reordered = map_predictions(
            &equation,
            &data,
            &reordered_support_points,
            first_prediction,
        )
        .unwrap();
        let unreordered = map_predictions(
            &equation,
            &data,
            &source_order_support_points,
            first_prediction,
        )
        .unwrap();

        assert_eq!(manual, reordered);
        assert_ne!(manual, unreordered);
    }

    #[test]
    fn simulation_errors_reach_the_closure() {
        let equation = named_order_ode();
        // An observation for an output the model does not declare.
        let data = Data::new(vec![Subject::builder("bad")
            .observation(0.0, 1.0, "not_an_output")
            .build()]);
        let matrix = map_predictions(&equation, &data, &array![[1.0, 2.0]], |_, result| {
            result.map_or(f64::NEG_INFINITY, |predictions| predictions.squared_error())
        })
        .unwrap();
        assert_eq!(matrix[[0, 0]], f64::NEG_INFINITY);
    }

    #[test]
    fn shape_problems_are_errors() {
        let equation = named_order_ode();
        let data = Data::new(vec![Subject::builder("a")
            .observation(0.0, 1.0, "cp")
            .build()]);

        let no_columns = Array2::<f64>::zeros((2, 0));
        assert!(map_predictions(&equation, &data, &no_columns, first_prediction).is_err());

        let no_support_points = Array2::<f64>::zeros((0, 2));
        let empty =
            map_predictions(&equation, &data, &no_support_points, first_prediction).unwrap();
        assert_eq!(empty.dim(), (1, 0));
    }

    #[test]
    fn cached_predictions_are_shared_not_copied() {
        let equation = named_order_ode();
        let data = Data::new(vec![Subject::builder("a")
            .observation(0.0, 1.0, "cp")
            .build()]);
        let support_points = array![[10.0, 0.5]];

        let first = map_predictions(&equation, &data, &support_points, |_, result| {
            result.unwrap()
        })
        .unwrap();
        let second = map_predictions(&equation, &data, &support_points, |_, result| {
            result.unwrap()
        })
        .unwrap();
        assert!(Arc::ptr_eq(&first[[0, 0]], &second[[0, 0]]));

        let uncached = equation.disable_cache();
        let third = map_predictions(&uncached, &data, &support_points, |_, result| {
            result.unwrap()
        })
        .unwrap();
        assert!(!Arc::ptr_eq(&first[[0, 0]], &third[[0, 0]]));
        assert_eq!(
            first[[0, 0]].flat_predictions(),
            third[[0, 0]].flat_predictions()
        );
    }
}
