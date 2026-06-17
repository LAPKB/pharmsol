use crate::core::State;
use crate::data::error_model::AssayErrorModels;
use crate::data::{Covariates, Infusion};
use crate::simulator::likelihood::Prediction;
use crate::{Observation, PharmsolError};

/// How to advance a model's state through time.
///
/// This is the trait backend authors implement. It captures the integration
/// mechanism — analytical closed form, numerical ODE integration, stochastic
/// stepping, etc. — without coupling to the event loop or prediction pipeline.
///
/// # Batch vs per-interval solving
///
/// Most backends use per-interval solving: the event loop calls [`solve`]
/// between events. If your solver prefers to handle all events internally
/// (like diffsol which does adaptive stepping across events), override
/// [`is_batch`] to return `true` and provide your own
/// [`Simulate::simulate_subject`](super::Simulate::simulate_subject).
///
/// # Example (analytical backend)
///
/// ```ignore
/// impl Solver for MyModel {
///     type State = V;
///
///     fn solve(&self, x: &mut V, params: &[f64], covariates: &Covariates,
///              infusions: &[Infusion], ti: f64, tf: f64) -> Result<(), Error> {
///         let dt = tf - ti;
///         *x = (self.eq)(x, &params_vector(params), dt, &rateiv(infusions, ti, tf), covariates);
///         Ok(())
///     }
///
///     fn process_observation(&self, state: &V, params: &[f64],
///                            observation: &Observation, error_models: Option<&AssayErrorModels>,
///                            covariates: &Covariates) -> Result<(Prediction, Option<f64>), Error> {
///         let mut y = V::zeros(self.nout(), NalgebraContext);
///         (self.output_fn)(state, &params_vector(params), observation.time(), covariates, &mut y);
///         let ix = observation.outeq_index().unwrap();
///         let pred = observation.to_prediction(y[ix], state.as_slice().to_vec());
///         let lik = error_models.map(|em| pred.log_likelihood(em).map(f64::exp)).transpose()?;
///         Ok((pred, lik))
///     }
///     // ...
/// }
/// ```
pub trait Solver {
    /// The state vector type this solver operates on.
    type State: State;

    /// Advance the system state from `ti` to `tf`.
    ///
    /// # Parameters
    /// * `state` — current state at `ti`, mutated to state at `tf` on return
    /// * `params` — model parameters in model order
    /// * `covariates` — time-varying covariates for this occasion
    /// * `infusions` — active infusion events in this interval
    /// * `ti` — start time (inclusive)
    /// * `tf` — end time (exclusive)
    fn solve(
        &self,
        _state: &mut Self::State,
        _params: &[f64],
        _covariates: &Covariates,
        _infusions: &[Infusion],
        _ti: f64,
        _tf: f64,
    ) -> Result<(), PharmsolError> {
        unimplemented!(
            "solve() is not used by batch-mode solvers; \
             set is_batch() to false or implement solve()"
        )
    }

    /// Compute a prediction (and optionally a likelihood component) from the
    /// current state at an observation time point.
    fn process_observation(
        &self,
        _state: &Self::State,
        _params: &[f64],
        _observation: &Observation,
        _error_models: Option<&AssayErrorModels>,
        _covariates: &Covariates,
    ) -> Result<(Prediction, Option<f64>), PharmsolError> {
        unimplemented!(
            "process_observation() is not used by batch-mode solvers; \
             set is_batch() to false or implement process_observation()"
        )
    }

    /// Create the initial state vector for the start of an occasion.
    ///
    /// For `occasion_index == 0`, this should call the model's init closure.
    /// For subsequent occasions, it should be zero (carry-over handled
    /// elsewhere).
    fn initial_state(
        &self,
        params: &[f64],
        covariates: &Covariates,
        occasion_index: usize,
    ) -> Self::State;

    /// Number of particles. 1 for deterministic (ODE/Analytical), >1 for SDE.
    fn nparticles(&self) -> usize {
        1
    }

    /// Whether this solver prefers batch event handling.
    ///
    /// When `true`, [`Simulate::simulate_subject`](super::Simulate::simulate_subject)
    /// must be implemented manually — the standard event loop won't be used.
    fn is_batch(&self) -> bool {
        false
    }
}
