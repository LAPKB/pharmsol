mod closure;

use core::panic;

use crate::{
    data::{Covariates, Infusion},
    error_model::ErrorModels,
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

// Hash the support points by converting them to bits and summing them
// The wrapping_add is used to avoid overflow, and prevent panics
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

fn _estimate_likelihood(
    ode: &ODE,
    subject: &Subject,
    support_point: &Vec<f64>,
    error_models: &ErrorModels,
    cache: bool,
) -> Result<f64, PharmsolError> {
    let ypred = if cache {
        _subject_predictions(ode, subject, support_point)
    } else {
        _subject_predictions_no_cache(ode, subject, support_point)
    }?;
    ypred.likelihood(error_models)
}

#[inline(always)]
#[cached(
    ty = "UnboundCache<String, SubjectPredictions>",
    create = "{ UnboundCache::with_capacity(100_000) }",
    convert = r#"{ format!("{}{}", subject.id(), spphash(support_point)) }"#,
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
        _error_models: Option<&ErrorModels>,
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
        error_models: &ErrorModels,
        cache: bool,
    ) -> Result<f64, PharmsolError> {
        _estimate_likelihood(self, subject, support_point, error_models, cache)
    }

    fn kind() -> crate::EqnKind {
        crate::EqnKind::ODE
    }

    fn simulate_subject(
        &self,
        subject: &Subject,
        support_point: &Vec<f64>,
        error_models: Option<&ErrorModels>,
    ) -> Result<(Self::P, Option<f64>), PharmsolError> {
        // let lag = self.get_lag(support_point);
        // let fa = self.get_fa(support_point);
        let mut output = Self::P::new(self.nparticles());
        let mut likelihood = Vec::new();
        // Preallocate bolus vectors
        let mut bolus_vec = vec![0.0; self.get_nstates()];
        let mut state_with_bolus = V::zeros(self.get_nstates(), NalgebraContext);
        let mut state_without_bolus = V::zeros(self.get_nstates(), NalgebraContext);
        let zero_vector = V::zeros(self.get_nstates(), NalgebraContext);
        let spp_v = DVector::from_vec(support_point.clone());
        let spp_v: V = spp_v.into();
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
                .build_from_eqn(PMProblem::new(
                    self.diffeq,
                    self.get_nstates(),
                    support_point.clone(), //TODO: Avoid cloning the support point
                    covariates,
                    infusions,
                    self.initial_state(support_point, covariates, occasion.index())
                        .into(),
                ))?;

            let mut solver: Bdf<
                '_,
                PMProblem<DiffEq>,
                NewtonNonlinearSolver<M, diffsol::NalgebraLU<f64>>,
            > = problem.bdf::<diffsol::NalgebraLU<f64>>()?; // TODO: Result

            for (index, event) in events.iter().enumerate() {
                let next_event = events.get(index + 1);
                //START SIMULATE_EVENT
                match event {
                    // Event::Bolus(bolus) => {
                    //     solver.state_mut().y[bolus.input()] += bolus.amount();
                    // }
                    Event::Bolus(bolus) => {
                        // Reset and reuse the bolus vector
                        bolus_vec.fill(0.0);
                        bolus_vec[bolus.input()] = bolus.amount();

                        // Reset and reuse the bolus changes vector
                        state_with_bolus.fill(0.0);
                        state_without_bolus.fill(0.0);

                        let bolus_v: V = DVector::from_vec(bolus_vec.clone()).into();

                        // Call the differential equation closure without bolus
                        (self.diffeq)(
                            solver.state().y,
                            &spp_v,
                            event.time(),
                            &mut state_without_bolus,
                            zero_vector.clone(), // Zero bolus
                            zero_vector.clone(),
                            covariates,
                        );

                        // Call the differential equation closure with bolus
                        (self.diffeq)(
                            solver.state().y,
                            &spp_v,
                            event.time(),
                            &mut state_with_bolus,
                            bolus_v,
                            zero_vector.clone(),
                            covariates,
                        );

                        // The difference between the two states is the actual bolus effect
                        // Apply the computed changes to the state
                        for i in 0..self.get_nstates() {
                            solver.state_mut().y[i] += state_with_bolus[i] - state_without_bolus[i];
                        }
                    }
                    Event::Infusion(_infusion) => {}
                    Event::Observation(observation) => {
                        //START PROCESS_OBSERVATION
                        let mut y = V::zeros(self.get_nouteqs(), NalgebraContext);
                        let out = &self.out;
                        let spp = DVector::from_vec(support_point.clone()); // TODO: Avoid clone
                        (out)(
                            solver.state().y,
                            &spp.into(),
                            observation.time(),
                            covariates,
                            &mut y,
                        );
                        let pred = y[observation.outeq()];
                        let pred =
                            observation.to_prediction(pred, solver.state().y.as_slice().to_vec());
                        if let Some(error_models) = error_models {
                            likelihood.push(pred.likelihood(error_models)?);
                        }
                        output.add_prediction(pred);
                        //END PROCESS_OBSERVATION
                    }
                }
                // START SOLVE
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
                                            panic!("Unexpected solver error: {:?}", err)
                                        }
                                    },
                                    _ => {
                                        panic!("Unexpected solver return value: {:?}", ret);
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
                                        panic!("Unexpected solver error: {:?}", e)
                                    }
                                }
                            }
                        }
                    }
                }
                //End SOLVE
                //END SIMULATE_EVENT
            }
        }
        let ll = error_models.map(|_| likelihood.iter().product::<f64>());
        Ok((output, ll))
    }
}
