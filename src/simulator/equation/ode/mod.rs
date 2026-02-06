mod closure;

use crate::{
    data::{Covariates, Infusion},
    error_model::AssayErrorModels,
    prelude::simulator::SubjectPredictions,
    simulator::{DiffEq, Fa, Init, Lag, Neqs, Out, M, V},
    Event, Observation, PharmsolError, Subject,
};

use cached::proc_macro::cached;
use cached::UnboundCache;

use crate::simulator::equation::Predictions;
use closure::PMProblem;
use diffsol::{
    error::OdeSolverError, ode_solver::method::OdeSolverMethod, Bdf, NalgebraContext,
    NewtonNonlinearSolver, OdeBuilder, OdeSolverStopReason, Vector, VectorHost,
};
use nalgebra::DVector;

use super::{Equation, EquationPriv, EquationTypes, State};

const RTOL: f64 = 1e-4;
const ATOL: f64 = 1e-4;
#[repr(C)]
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

impl State for V {
    #[inline(always)]
    fn add_bolus(&mut self, input: usize, amount: f64) {
        self[input] += amount;
    }
}

/// Hash support points to a u64 for cache key generation.
/// Uses DefaultHasher for good distribution and collision resistance.
#[inline(always)]
fn spphash(spp: &[f64]) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::hash::DefaultHasher::new();
    for &value in spp {
        // Normalize -0.0 to 0.0 for consistent hashing
        let bits = if value == 0.0 { 0u64 } else { value.to_bits() };
        bits.hash(&mut hasher);
    }
    hasher.finish()
}

/// Hash a subject ID string to u64 for cache key generation.

fn _estimate_likelihood(
    ode: &ODE,
    subject: &Subject,
    support_point: &Vec<f64>,
    error_models: &AssayErrorModels,
    cache: bool,
) -> Result<f64, PharmsolError> {
    let ypred = if cache {
        _subject_predictions(ode, subject, support_point)
    } else {
        _subject_predictions_no_cache(ode, subject, support_point)
    }?;
    Ok(ypred.log_likelihood(error_models)?.exp())
}

#[inline(always)]
#[cached(
    ty = "UnboundCache<(u64, u64), SubjectPredictions>",
    create = "{ UnboundCache::with_capacity(100_000) }",
    convert = r#"{ ((subject.hash()), spphash(support_point)) }"#,
    result = "true"
)]
fn _subject_predictions(
    ode: &ODE,
    subject: &Subject,
    support_point: &Vec<f64>,
) -> Result<SubjectPredictions, PharmsolError> {
    Ok(ode.simulate_subject(subject, support_point, None)?.0)
}

impl EquationTypes for ODE {
    type S = V;
    type P = SubjectPredictions;
}

impl EquationPriv for ODE {
    //#[inline(always)]
    // fn get_lag(&self, spp: &[f64]) -> Option<HashMap<usize, f64>> {
    //     let spp = DVector::from_vec(spp.to_vec());
    //     Some((self.lag)(&spp))
    // }

    // #[inline(always)]
    // fn get_fa(&self, spp: &[f64]) -> Option<HashMap<usize, f64>> {
    //     let spp = DVector::from_vec(spp.to_vec());
    //     Some((self.fa)(&spp))
    // }
    #[inline(always)]
    fn lag(&self) -> &Lag {
        &self.lag
    }

    #[inline(always)]
    fn fa(&self) -> &Fa {
        &self.fa
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
        _state: &mut Self::S,
        _support_point: &Vec<f64>,
        _covariates: &Covariates,
        _infusions: &Vec<Infusion>,
        _start_time: f64,
        _end_time: f64,
    ) -> Result<(), PharmsolError> {
        unimplemented!("solve not implemented for ODE");
    }
    #[inline(always)]
    fn process_observation(
        &self,
        _support_point: &Vec<f64>,
        _observation: &Observation,
        _error_models: Option<&AssayErrorModels>,
        _time: f64,
        _covariates: &Covariates,
        _x: &mut Self::S,
        _likelihood: &mut Vec<f64>,
        _output: &mut Self::P,
    ) -> Result<(), PharmsolError> {
        unimplemented!("process_observation not implemented for ODE");
    }

    #[inline(always)]
    fn initial_state(&self, spp: &Vec<f64>, covariates: &Covariates, occasion_index: usize) -> V {
        let init = &self.init;
        let mut x = V::zeros(self.get_nstates(), NalgebraContext);
        if occasion_index == 0 {
            let spp = DVector::from_vec(spp.clone());
            (init)(&spp.into(), 0.0, covariates, &mut x);
        }
        x
    }
}

impl Equation for ODE {
    fn estimate_likelihood(
        &self,
        subject: &Subject,
        support_point: &Vec<f64>,
        error_models: &AssayErrorModels,
        cache: bool,
    ) -> Result<f64, PharmsolError> {
        _estimate_likelihood(self, subject, support_point, error_models, cache)
    }

    fn estimate_log_likelihood(
        &self,
        subject: &Subject,
        support_point: &Vec<f64>,
        error_models: &AssayErrorModels,
        cache: bool,
    ) -> Result<f64, PharmsolError> {
        let ypred = if cache {
            _subject_predictions(self, subject, support_point)
        } else {
            _subject_predictions_no_cache(self, subject, support_point)
        }?;
        ypred.log_likelihood(error_models)
    }

    fn kind() -> crate::EqnKind {
        crate::EqnKind::ODE
    }

    fn simulate_subject(
        &self,
        subject: &Subject,
        support_point: &Vec<f64>,
        error_models: Option<&AssayErrorModels>,
    ) -> Result<(Self::P, Option<f64>), PharmsolError> {
        let mut output = Self::P::new(self.nparticles());

        // Preallocate likelihood vector
        let event_count: usize = subject.occasions().iter().map(|o| o.events().len()).sum();
        let mut likelihood = Vec::with_capacity(event_count);

        // Cache nstates to avoid repeated method calls
        let nstates = self.get_nstates();

        let state_buffer_size = nstates;
        let output_buffer_size = self.get_nouteqs();

        // Preallocate reusable vectors for bolus computation
        let mut state_with_bolus = V::zeros(state_buffer_size, NalgebraContext);
        let mut state_without_bolus = V::zeros(state_buffer_size, NalgebraContext);
        let zero_vector = V::zeros(state_buffer_size, NalgebraContext);
        let mut bolus_v = V::zeros(state_buffer_size, NalgebraContext);
        let spp_v: V = DVector::from_vec(support_point.clone()).into();

        // Pre-allocate output vector for observations
        let mut y_out = V::zeros(output_buffer_size, NalgebraContext);

        // Iterate over occasions
        for occasion in subject.occasions() {
            let covariates = occasion.covariates();
            let infusions = occasion.infusions_ref();
            let events = occasion.process_events(
                Some((self.fa(), self.lag(), support_point, covariates)),
                true,
            );

            let problem = OdeBuilder::<M>::new()
                .atol(vec![ATOL])
                .rtol(RTOL)
                .t0(occasion.initial_time())
                .h0(1e-3)
                .p(support_point.clone())
                .build_from_eqn(PMProblem::with_params_v(
                    self.diffeq,
                    nstates,
                    support_point.clone(),
                    spp_v.clone(), // Reuse pre-converted V
                    covariates,
                    infusions.as_slice(),
                    self.initial_state(support_point, covariates, occasion.index())
                        .into(),
                ))?;

            let mut solver: Bdf<
                '_,
                PMProblem<DiffEq>,
                NewtonNonlinearSolver<M, diffsol::NalgebraLU<f64>>,
            > = problem.bdf::<diffsol::NalgebraLU<f64>>()?;

            // Iterate over events
            for (index, event) in events.iter().enumerate() {
                // Check if we have a next event
                let next_event = events.get(index + 1);

                // Handle events accordingly
                match event {
                    Event::Bolus(bolus) => {
                        // Reset and reuse the pre-allocated bolus vector
                        bolus_v.fill(0.0);
                        bolus_v[bolus.input()] = bolus.amount();

                        // Reset and reuse the bolus changes vectors
                        state_with_bolus.fill(0.0);
                        state_without_bolus.fill(0.0);

                        // Call the differential equation closure without bolus
                        (self.diffeq)(
                            solver.state().y,
                            &spp_v,
                            event.time(),
                            &mut state_without_bolus,
                            &zero_vector,
                            &zero_vector,
                            covariates,
                        );

                        // Call the differential equation closure with bolus
                        (self.diffeq)(
                            solver.state().y,
                            &spp_v,
                            event.time(),
                            &mut state_with_bolus,
                            &bolus_v,
                            &zero_vector,
                            covariates,
                        );

                        // The difference between the two states is the actual bolus effect
                        // Apply the computed changes to the state using vectorized operations
                        // state_with_bolus now contains (with_bolus - without_bolus) after axpy
                        state_with_bolus.axpy(-1.0, &state_without_bolus, 1.0);

                        // Add the difference to the solver state
                        solver.state_mut().y.axpy(1.0, &state_with_bolus, 1.0);
                    }
                    Event::Infusion(_infusion) => {
                        // Infusions are handled within the ODE function itself
                    }
                    Event::Observation(observation) => {
                        // Reuse pre-allocated output vector
                        y_out.fill(0.0);
                        let out = &self.out;
                        (out)(
                            solver.state().y,
                            &spp_v,
                            observation.time(),
                            covariates,
                            &mut y_out,
                        );
                        let pred = y_out[observation.outeq()];
                        let pred =
                            observation.to_prediction(pred, solver.state().y.as_slice().to_vec());
                        if let Some(error_models) = error_models {
                            likelihood.push(pred.log_likelihood(error_models)?.exp());
                        }
                        output.add_prediction(pred);
                    }
                }

                // Advance to the next event time if it exists
                if let Some(next_event) = next_event {
                    if !event.time().eq(&next_event.time()) {
                        match solver.set_stop_time(next_event.time()) {
                            Ok(_) => loop {
                                let ret = solver.step();
                                match ret {
                                    Ok(OdeSolverStopReason::InternalTimestep) => continue,
                                    Ok(OdeSolverStopReason::TstopReached) => break,
                                    Err(err) => match err {
                                        diffsol::error::DiffsolError::OdeSolverError(
                                            OdeSolverError::StepSizeTooSmall { time },
                                        ) => {
                                            let _time = time;
                                            return Err(PharmsolError::OtherError("The step size of the ODE solver went to zero, this means one of your parameters is getting really close to 0.0 or INFINITE. Check your model".to_string()));
                                        }
                                        _ => {
                                            return Err(PharmsolError::OtherError(
                                                "Unexpected solver error: {:?}".to_string(),
                                            ));
                                        }
                                    },
                                    _ => {
                                        return Err(PharmsolError::OtherError(
                                            "Unexpected solver error: {:?}".to_string(),
                                        ));
                                    }
                                }
                            },
                            Err(e) => {
                                match e {
                                    diffsol::error::DiffsolError::OdeSolverError(
                                        OdeSolverError::StopTimeAtCurrentTime,
                                    ) => {
                                        // If the stop time is at the current state time, we can just continue
                                        continue;
                                    }
                                    _ => {
                                        return Err(PharmsolError::OtherError(
                                            "Unexpected solver error: {:?}".to_string(),
                                        ));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        let ll = error_models.map(|_| likelihood.iter().product::<f64>());
        Ok((output, ll))
    }
}
