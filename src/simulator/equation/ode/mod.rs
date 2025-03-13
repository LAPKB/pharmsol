mod closure;
mod diffsol_traits;

use std::collections::HashMap;

use crate::{
    data::{Covariates, Infusion},
    error_model::ErrorModel,
    prelude::simulator::SubjectPredictions,
    simulator::{likelihood::ToPrediction, DiffEq, Fa, Init, Lag, Neqs, Out, M, T, V},
    Observation, Subject,
};
use cached::proc_macro::cached;
use cached::UnboundCache;

use diffsol::{ode_solver::method::OdeSolverMethod, Bdf, OdeSolverState};

use self::diffsol_traits::build_pm_ode;

use super::{Equation, EquationPriv, EquationTypes, State};

/// Relative tolerance used for ODE solving
const RTOL: f64 = 1e-4;
/// Absolute tolerance used for ODE solving
const ATOL: f64 = 1e-4;

/// Model equation implementation using ordinary differential equations (ODEs).
///
/// This struct encapsulates all components needed to define and solve a pharmacometric model
/// using ordinary differential equations. It handles integration of the system from initial
/// conditions through time, applying doses, and computing output equations.
///
/// The ODE implementation uses numerical integration with adaptive step size control to
/// ensure accuracy and efficiency.
///
/// # Example
///
/// ```
/// use pharmsol::simulator::{DiffEq, Lag, Fa, Init, Out, Neqs};
/// use pharmsol::simulator::equation::ODE;
///
/// // Define model components (simplified for example)
/// let diffeq: DiffEq = |state, params, time, out, rateiv, covs| { /* model equations */ };
/// let lag: Lag = |params| std::collections::HashMap::new();
/// let fa: Fa = |params| std::collections::HashMap::new();
/// let init: Init = |params, time, covs, state| { /* initial conditions */ };
/// let out: Out = |state, params, time, covs, out| { /* output equations */ };
/// let neqs = (3, 2); // 3 state equations, 2 output equations
///
/// // Create ODE solver
/// let ode_solver = ODE::new(diffeq, lag, fa, init, out, neqs);
/// ```
#[derive(Clone, Debug)]
pub struct ODE {
    diffeq: DiffEq,
    lag: Lag,
    fa: Fa,
    init: Init,
    out: Out,
    neqs: Neqs,
}

impl ODE {
    /// Creates a new ODE equation model.
    ///
    /// # Parameters
    ///
    /// - `diffeq`: Function defining the differential equations of the model system
    /// - `lag`: Function that computes absorption lag times for different inputs
    /// - `fa`: Function that computes bioavailability fractions for different inputs
    /// - `init`: Function that initializes the state vector at the start of simulation
    /// - `out`: Function that computes output equations from the current state
    /// - `neqs`: Tuple containing (number of state equations, number of output equations)
    ///
    /// # Returns
    ///
    /// A configured ODE solver ready for simulation
    pub fn new(diffeq: DiffEq, lag: Lag, fa: Fa, init: Init, out: Out, neqs: Neqs) -> Self {
        Self {
            diffeq,
            lag,
            fa,
            init,
            out,
            neqs,
        }
    }
}

/// Implementation of the State trait for vector operations in ODE solving.
///
/// This enables the addition of bolus doses directly to state variables.
impl State for V {
    /// Adds a bolus dose to the specified input compartment in the state vector.
    ///
    /// # Arguments
    ///
    /// * `input`: Index of the compartment receiving the dose
    /// * `amount`: Amount of drug to add
    #[inline(always)]
    fn add_bolus(&mut self, input: usize, amount: f64) {
        self[input] += amount;
    }
}

/// Computes a hash value for model parameters.
///
/// This function creates a deterministic hash from parameter values to enable
/// caching of simulation results for identical parameter sets.
///
/// # Arguments
///
/// * `spp`: Slice containing parameter values
///
/// # Returns
///
/// A u64 hash value representing the parameter vector
fn spphash(spp: &[f64]) -> u64 {
    let mut hasher = std::hash::DefaultHasher::new();
    spp.iter().for_each(|&value| {
        // Normalize negative zero to zero, e.g. -0.0 -> 0.0
        let normalized_value = if value == 0.0 && value.is_sign_negative() {
            0.0
        } else {
            value
        };
        // Convert the value to bits and hash it
        let bits = normalized_value.to_bits();
        std::hash::Hash::hash(&bits, &mut hasher);
    });

    std::hash::Hasher::finish(&hasher)
}

/// Cached version of subject predictions.
///
/// This function caches simulation results for each unique combination of subject ID
/// and parameter vector, avoiding redundant computations in optimization algorithms.
///
/// # Arguments
///
/// * `ode`: Reference to the ODE solver
/// * `subject`: Subject data containing dosing and observation information
/// * `support_point`: Parameter vector for the model
///
/// # Returns
///
/// Predictions for all observations in the subject
#[inline(always)]
#[cached(
    ty = "UnboundCache<String, SubjectPredictions>",
    create = "{ UnboundCache::with_capacity(100_000) }",
    convert = r#"{ format!("{}{}", subject.id(), spphash(support_point)) }"#
)]
fn _subject_predictions(
    ode: &ODE,
    subject: &Subject,
    support_point: &Vec<f64>,
) -> SubjectPredictions {
    ode.simulate_subject(subject, support_point, None).0
}

/// Computes likelihood of observed data given model parameters.
///
/// # Arguments
///
/// * `ode`: Reference to the ODE solver
/// * `subject`: Subject data containing observations
/// * `support_point`: Parameter vector for the model
/// * `error_model`: Error model to use for likelihood calculations
/// * `cache`: Whether to use cached predictions
///
/// # Returns
///
/// Log-likelihood of the subject data given the model and parameters
fn _estimate_likelihood(
    ode: &ODE,
    subject: &Subject,
    support_point: &Vec<f64>,
    error_model: &ErrorModel,
    cache: bool,
) -> f64 {
    let ypred = if cache {
        _subject_predictions(ode, subject, support_point)
    } else {
        _subject_predictions_no_cache(ode, subject, support_point)
    };
    ypred.likelihood(error_model)
}

/// Type definitions for ODE equation system.
impl EquationTypes for ODE {
    type S = V;
    type P = SubjectPredictions;
}

/// Private implementation of ODE equation solver.
impl EquationPriv for ODE {
    #[inline(always)]
    fn get_lag(&self, spp: &[f64]) -> Option<HashMap<usize, f64>> {
        Some((self.lag)(&V::from_vec(spp.to_owned())))
    }

    #[inline(always)]
    fn get_fa(&self, spp: &[f64]) -> Option<HashMap<usize, f64>> {
        Some((self.fa)(&V::from_vec(spp.to_owned())))
    }

    #[inline(always)]
    fn get_nstates(&self) -> usize {
        self.neqs.0
    }

    #[inline(always)]
    fn get_nouteqs(&self) -> usize {
        self.neqs.1
    }
    #[inline(always)]
    fn solve(
        &self,
        state: &mut Self::S,
        support_point: &Vec<f64>,
        covariates: &Covariates,
        infusions: &Vec<Infusion>,
        start_time: f64,
        end_time: f64,
    ) {
        if f64::abs(start_time - end_time) < 1e-8 {
            return;
        }
        // dbg!(start_time, end_time);
        let problem = build_pm_ode::<M, _, _>(
            self.diffeq,
            |_p: &V, _t: T| state.clone(),
            V::from_vec(support_point.to_vec()),
            start_time,
            1e-3,
            RTOL,
            ATOL,
            covariates.clone(),
            infusions.clone(),
        )
        .unwrap();

        let mut solver = Bdf::default();
        let st = OdeSolverState::new(&problem, &solver).unwrap();
        solver.set_problem(st, &problem).unwrap();
        while solver.state().unwrap().t <= end_time {
            solver.step().unwrap();
        }
        *state = solver.interpolate(end_time).unwrap();
    }
    #[inline(always)]
    fn process_observation(
        &self,
        support_point: &Vec<f64>,
        observation: &Observation,
        error_model: Option<&ErrorModel>,
        _time: f64,
        covariates: &Covariates,
        x: &mut Self::S,
        likelihood: &mut Vec<f64>,
        output: &mut Self::P,
    ) {
        let mut y = V::zeros(self.get_nouteqs());
        let out = &self.out;
        (out)(
            x,
            &V::from_vec(support_point.clone()),
            observation.time(),
            covariates,
            &mut y,
        );
        let pred = y[observation.outeq()];
        let pred = observation.to_obs_pred(pred, x.as_slice().to_vec());
        if let Some(error_model) = error_model {
            likelihood.push(pred.likelihood(error_model));
        }
        output.add_prediction(pred);
    }

    #[inline(always)]
    fn initial_state(&self, spp: &Vec<f64>, covariates: &Covariates, occasion_index: usize) -> V {
        let init = &self.init;
        let mut x = V::zeros(self.get_nstates());
        if occasion_index == 0 {
            (init)(&V::from_vec(spp.to_vec()), 0.0, covariates, &mut x);
        }
        x
    }
}

/// Implementation of the Equation trait for ODE models.
impl Equation for ODE {
    /// Estimates the likelihood of observed data given a model and parameters.
    ///
    /// # Arguments
    ///
    /// * `subject`: Subject data containing observations
    /// * `support_point`: Parameter vector for the model
    /// * `error_model`: Error model to use for likelihood calculations
    /// * `cache`: Whether to cache likelihood results for reuse
    ///
    /// # Returns
    ///
    /// The log-likelihood of the observed data given the model and parameters
    fn estimate_likelihood(
        &self,
        subject: &Subject,
        support_point: &Vec<f64>,
        error_model: &ErrorModel,
        cache: bool,
    ) -> f64 {
        _estimate_likelihood(self, subject, support_point, error_model, cache)
    }
}

// Test spphash
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_spphash() {
        let spp1 = vec![1.0, 2.0, 3.0];
        let spp2 = vec![1.0, 2.0, 3.0];
        let spp3 = vec![3.0, 2.0, 1.0];
        let spp4 = vec![1.0, 2.0, 3.000001];
        // Equal values should have the same hash
        assert_eq!(spphash(&spp1), spphash(&spp2));
        // Mirrored values should have different hashes
        assert_ne!(spphash(&spp1), spphash(&spp3));
        // Very close values should have different hashes
        // Note: Due to f64 precision this will fail for values that are very close, e.g. 3.0 and 3.0000000000000001
        assert_ne!(spphash(&spp1), spphash(&spp4));
    }
}
