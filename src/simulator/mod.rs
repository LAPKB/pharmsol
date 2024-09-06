pub(crate) mod analytical;
pub mod fitting;
pub(crate) mod likelihood;
mod ode;
mod sde;
use crate::{
    data::{Covariates, Event, Infusion, Subject},
    error_model::ErrorModel,
    simulator::likelihood::{SubjectPredictions, ToPrediction},
};
use likelihood::Prediction;
use ndarray::parallel::prelude::*;
use rand::prelude::*;
use sde::simulate_sde_event;
use std::collections::HashMap;

type T = f64;
type V = nalgebra::DVector<T>;
type M = nalgebra::DMatrix<T>;

/// This closure represents the differential equation of the model:
/// Params:
/// - x: The state vector at time t
/// - p: The parameters of the model; Use the [fetch_params!] macro to extract the parameters
/// - t: The time at which the differential equation is evaluated
/// - dx: A mutable reference to the derivative of the state vector at time t
/// - rateiv: A vector of infusion rates at time t
/// - cov: A reference to the covariates at time t; Use the [fetch_cov!] macro to extract the covariates
/// Example:
/// ```ignore
/// use pharmsol::*;
/// let diff_eq = |x, p, t, dx, rateiv, cov| {
///  fetch_params!(p, ka, ke, v);
///  fetch_cov!(cov, t, wt);
///  dx[0] = -ka * x[0];
///  dx[1] = ka * x[0] - ke * x[1];
/// };
pub type DiffEq = fn(&V, &V, T, &mut V, V, &Covariates);

/// This closure represents an Analytical solution of the model, see [analytical] module for examples.
/// Params:
/// - x: The state vector at time t
/// - p: The parameters of the model; Use the [fetch_params!] macro to extract the parameters
/// - t: The time at which the output equation is evaluated
/// - rateiv: A vector of infusion rates at time t
/// - cov: A reference to the covariates at time t; Use the [fetch_cov!] macro to extract the covariates
/// TODO: Remove covariates. They are not used in the analytical solution
pub type AnalyticalEq = fn(&V, &V, T, V, &Covariates) -> V;

/// This closure represents the drift term of the model:
/// Params:
/// - x: The state vector at time t
/// - p: The parameters of the model; Use the [fetch_params!] macro to extract the parameters
/// - t: The time at which the drift term is evaluated
/// - dx: A mutable reference to the derivative of the state vector at time t
/// - rateiv: A vector of infusion rates at time t
/// - cov: A reference to the covariates at time t; Use the [fetch_cov!] macro to extract the covariates
/// Example:
/// ```ignore
/// use pharmsol::*;
/// let drift = |x, p, t, dx, rateiv, cov| {
/// fetch_params!(p, mka, mke, v);
/// fetch_cov!(cov, t, wt);
/// ka = dx[2];
/// ke = dx[3];
/// dx[0] = -ka * x[0];
/// dx[1] = ka * x[0] - ke * x[1];
/// dx[2] = -dx[2] + mka; // Mean reverting to mka
/// dx[3] = -dx[3] + mke; // Mean reverting to mke
/// };
pub type Drift = DiffEq;

/// This closure represents the diffusion term of the model:
/// Params:
/// - p: The parameters of the model; Use the [fetch_params!] macro to extract the parameters
/// - d: A mutable reference to the diffusion term for each state variable
/// (This vector should have the same length as the x, and dx vectors on the drift closure)
pub type Diffusion = fn(&V, &mut V);
/// This closure represents the initial state of the system:
/// Params:
/// - p: The parameters of the model; Use the [fetch_params!] macro to extract the parameters
/// - t: The time at which the initial state is evaluated; Hardcoded to 0.0
/// - cov: A reference to the covariates at time t; Use the [fetch_cov!] macro to extract the covariates
/// - x: A mutable reference to the state vector at time t
/// Example:
/// ```ignore
/// use pharmsol::*;
/// let init = |p, t, cov, x| {
///  fetch_params!(p, ka, ke, v);
///  fetch_cov!(cov, t, wt);
///  x[0] = 500.0;
///  x[1] = 0.0;
/// };
pub type Init = fn(&V, T, &Covariates, &mut V);

/// This closure represents the output equation of the model:
/// Params:
/// - x: The state vector at time t
/// - p: The parameters of the model; Use the [fetch_params!] macro to extract the parameters
/// - t: The time at which the output equation is evaluated
/// - cov: A reference to the covariates at time t; Use the [fetch_cov!] macro to extract the covariates
/// - y: A mutable reference to the output vector at time t
/// Example:
/// ```ignore
/// use pharmsol::*;
/// let out = |x, p, t, cov, y| {
///   fetch_params!(p, ka, ke, v);
///   y[0] = x[1] / v;
/// };
pub type Out = fn(&V, &V, T, &Covariates, &mut V);

/// This closure represents the secondary equation of the model, secondary equations are used to update
/// the parameter values based on the covariates.
/// Params:
/// - p: The parameters of the model; Use the [fetch_params!] macro to extract the parameters
/// - t: The time at which the secondary equation is evaluated
/// - cov: A reference to the covariates at time t; Use the [fetch_cov!] macro to extract the covariates
/// Example:
/// ```ignore
/// use pharmsol::*;
/// let sec_eq = |p, _t, cov| {
///    fetch_params!(p, ka, ke);
///    fetch_cov!(cov, t, wt);
///    ka = ka * wt;
/// };
/// ```
pub type SecEq = fn(&mut V, T, &Covariates);

/// This closure represents the lag time of the model, the lag term delays the only the boluses going into
/// an specific comparment.
/// Params:
/// - p: The parameters of the model; Use the [fetch_params!] macro to extract the parameters
/// Returns:
/// - A hashmap with the lag times for each comparment, if not presennt it is assumed to be 0.
/// There is a convenience macro [lag!] to create the hashmap
/// Example:
/// ```ignore
/// use pharmsol::*;
/// let lag = |p| {
///    fetch_params!(p, tlag);
///    lag! {0=>tlag, 1=>0.3}
/// };
/// ```
/// This will lag the bolus going into the first compartment by tlag and the bolus going into the
/// second compartment by 0.3
pub type Lag = fn(&V) -> HashMap<usize, T>;

/// This closure represents the fraction absorbed (also called bioavailability or protein binding)
/// of the model, the fa term is used to adjust the amount of drug that is absorbed into the system.
/// Params:
/// - p: The parameters of the model; Use the [fetch_params!] macro to extract the parameters
/// Returns:
/// - A hashmap with the fraction absorbed for each comparment, if not presennt it is assumed to be 1.
/// There is a convenience macro [fa!] to create the hashmap
/// Example:
/// ```ignore
/// use pharmsol::*;
/// let fa = |p| {
///   fetch_params!(p, fa);
///   fa! {0=>fa, 1=>0.3}
/// };
/// ```
/// This will adjust the amount of drug absorbed into the first compartment by fa and the amount of drug
/// absorbed into the second compartment by 0.3
pub type Fa = fn(&V) -> HashMap<usize, T>;

/// The number of states and output equations of the model
/// The first element is the number of states and the second element is the number of output equations
/// This is used to initialize the state vector and the output vector
/// Example:
/// ```ignore
/// let neqs = (2, 1);
/// ```
/// This means that the system of equations has 2 states and there is only 1 output equation.
///
pub type Neqs = (usize, usize);

fn sysresample(q: &[f64]) -> Vec<usize> {
    let mut qc = vec![0.0; q.len()];
    qc[0] = q[0];
    for i in 1..q.len() {
        qc[i] = qc[i - 1] + q[i];
    }
    let m = q.len();
    let mut rng = thread_rng();
    let u: Vec<f64> = (0..m)
        .map(|i| (i as f64 + rng.gen::<f64>()) / m as f64)
        .collect();
    let mut i = vec![0; m];
    let mut k = 0;
    for j in 0..m {
        while qc[k] < u[j] {
            k += 1;
        }
        i[j] = k;
    }
    i
}

#[derive(Debug, Clone)]
pub enum Equation {
    ODE(DiffEq, Lag, Fa, Init, Out, Neqs),
    SDE(Drift, Diffusion, Lag, Fa, Init, Out, Neqs),
    Analytical(AnalyticalEq, SecEq, Lag, Fa, Init, Out, Neqs),
}

impl Equation {
    pub fn new_ode(diffeq: DiffEq, lag: Lag, fa: Fa, init: Init, out: Out, neqs: Neqs) -> Self {
        Equation::ODE(diffeq, lag, fa, init, out, neqs)
    }
    pub fn new_analytical(
        eq: AnalyticalEq,
        seq_eq: SecEq,
        lag: Lag,
        fa: Fa,
        init: Init,
        out: Out,
        neqs: Neqs,
    ) -> Self {
        Equation::Analytical(eq, seq_eq, lag, fa, init, out, neqs)
    }

    pub fn new_sde(
        drift: Drift,
        diffusion: Diffusion,
        lag: Lag,
        fa: Fa,
        init: Init,
        out: Out,
        neqs: Neqs,
    ) -> Self {
        Equation::SDE(drift, diffusion, lag, fa, init, out, neqs)
    }

    pub fn is_sde(&self) -> bool {
        match self {
            Equation::SDE(_, _, _, _, _, _, _) => true,
            _ => false,
        }
    }

    pub fn particle_filter(
        &self,
        subject: &Subject,
        support_point: &Vec<f64>,
        nparticles: usize,
        error_model: &ErrorModel,
    ) -> f64 {
        match self {
            Equation::ODE(_, _, _, _, _, _) => {
                unimplemented!("Particle Filter not implemented for ODE models")
            }
            Equation::Analytical(_, _, _, _, _, _, _) => {
                unimplemented!("Particle Filter not implemented for Analytical models")
            }
            Equation::SDE(drift, difussion, _, _, _, _, _) => {
                // Check for a cache entry
                let init = self.get_init();
                let out = self.get_out();
                let lag = self.get_lag(support_point);
                let fa = self.get_fa(support_point);
                let mut ll = vec![];

                for occasion in subject.occasions() {
                    let covariates = occasion.get_covariates().unwrap();
                    // if occasion == 0, we use the init closure to get the initial state
                    // otherwise we initialize the state vector to zero
                    let mut initial = V::zeros(self.get_nstates());
                    if occasion.index() == 0 {
                        (init)(
                            &V::from_vec(support_point.clone()),
                            0.0,
                            covariates,
                            &mut initial,
                        );
                    }
                    let mut x = Vec::with_capacity(nparticles);
                    for _ in 0..nparticles {
                        x.push(initial.clone());
                    }

                    let mut infusions: Vec<Infusion> = vec![];
                    let events = occasion.get_events(Some(&lag), Some(&fa), true);
                    for (index, event) in events.iter().enumerate() {
                        match event {
                            Event::Bolus(bolus) => {
                                x.par_iter_mut().for_each(|particle| {
                                    particle[bolus.input()] += bolus.amount();
                                });
                            }
                            Event::Infusion(infusion) => {
                                infusions.push(infusion.clone());
                            }
                            Event::Observation(observation) => {
                                let mut pred = Vec::with_capacity(nparticles);
                                for _ in 0..nparticles {
                                    pred.push(0.0);
                                }
                                pred.par_iter_mut().enumerate().for_each(|(i, p)| {
                                    let mut y = V::zeros(self.get_nouteqs());
                                    (out)(
                                        &x[i],
                                        &V::from_vec(support_point.clone()),
                                        observation.time(),
                                        covariates,
                                        &mut y,
                                    );
                                    *p = y[observation.outeq()];
                                });
                                //e = y[t] .- x[:,1]
                                // q = pdf.(Distributions.Normal(0, 0.5), e)
                                let mut q: Vec<f64> = Vec::with_capacity(pred.len());
                                pred.iter().for_each(|p| {
                                    q.push(observation.to_obs_pred(*p).likelihood(error_model))
                                });
                                let sum_q: f64 = q.iter().sum();
                                //py = (1 / Np) * sum(q)
                                let py = sum_q / nparticles as f64;
                                //b[t] = log(py)
                                ll.push(py.ln());
                                //q = q ./ sum(q)
                                let w: Vec<f64> = q.iter().map(|qi| qi / sum_q).collect();
                                //ind = sysresample(q)
                                let i = sysresample(&w);
                                //x = x[ind,:]
                                x = i.iter().map(|&i| x[i].clone()).collect();
                            }
                        }

                        if let Some(next_event) = events.get(index + 1) {
                            x.par_iter_mut().for_each(|particle| {
                                *particle = simulate_sde_event(
                                    drift,
                                    difussion,
                                    particle.clone(),
                                    support_point,
                                    covariates,
                                    &infusions,
                                    event.get_time(),
                                    next_event.get_time(),
                                );
                            });
                        }
                    }
                }
                // let pred: SubjectPredictions = yout.into();
                let likelihood = ll.iter().sum::<f64>().exp();
                // Insert the cache entry
                likelihood
            }
        }
    }

    pub fn simulate_trajectories(
        &self,
        subject: &Subject,
        support_point: &Vec<f64>,
        nparticles: usize,
    ) -> Vec<Vec<f64>> {
        match self {
            Equation::ODE(_, _, _, _, _, _) => {
                unimplemented!("Particle Filter not implemented for ODE models")
            }
            Equation::Analytical(_, _, _, _, _, _, _) => {
                unimplemented!("Particle Filter not implemented for Analytical models")
            }
            Equation::SDE(drift, difussion, _, _, _, _, _) => {
                let init = self.get_init();
                let out = self.get_out();
                let lag = self.get_lag(support_point);
                let fa = self.get_fa(support_point);
                let mut trajectories = vec![];

                for occasion in subject.occasions() {
                    let covariates = occasion.get_covariates().unwrap();
                    // if occasion == 0, we use the init closure to get the initial state
                    // otherwise we initialize the state vector to zero
                    let mut initial = V::zeros(self.get_nstates());
                    if occasion.index() == 0 {
                        (init)(
                            &V::from_vec(support_point.clone()),
                            0.0,
                            covariates,
                            &mut initial,
                        );
                    }
                    let mut x = Vec::with_capacity(nparticles);
                    for _ in 0..nparticles {
                        x.push(initial.clone());
                    }

                    let mut infusions: Vec<Infusion> = vec![];
                    let events = occasion.get_events(Some(&lag), Some(&fa), true);
                    for (index, event) in events.iter().enumerate() {
                        match event {
                            Event::Bolus(bolus) => {
                                x.par_iter_mut().for_each(|particle| {
                                    particle[bolus.input()] += bolus.amount();
                                });
                            }
                            Event::Infusion(infusion) => {
                                infusions.push(infusion.clone());
                            }
                            Event::Observation(observation) => {
                                let mut pred = Vec::with_capacity(nparticles);
                                for _ in 0..nparticles {
                                    pred.push(0.0);
                                }
                                pred.par_iter_mut().enumerate().for_each(|(i, p)| {
                                    let mut y = V::zeros(self.get_nouteqs());
                                    (out)(
                                        &x[i],
                                        &V::from_vec(support_point.clone()),
                                        observation.time(),
                                        covariates,
                                        &mut y,
                                    );
                                    *p = y[observation.outeq()];
                                });
                                trajectories.push(pred.clone());
                            }
                        }

                        if let Some(next_event) = events.get(index + 1) {
                            x.par_iter_mut().for_each(|particle| {
                                *particle = simulate_sde_event(
                                    drift,
                                    difussion,
                                    particle.clone(),
                                    support_point,
                                    covariates,
                                    &infusions,
                                    event.get_time(),
                                    next_event.get_time(),
                                );
                            });
                        }
                    }
                }
                trajectories
            }
        }
    }

    pub fn simulate_subject(
        &self,
        subject: &Subject,
        support_point: &Vec<f64>,
    ) -> SubjectPredictions {
        simulator(&self, subject, support_point)
    }

    #[inline(always)]

    fn simulate_event(
        &self,
        x: &mut V,
        support_point: &[f64],
        covariates: &Covariates,
        infusions: &Vec<Infusion>,
        start_time: T,
        end_time: T,
    ) {
        match self {
            Equation::ODE(eqn, _, _, _, _, _) => ode::simulate_ode_event(
                eqn,
                x,
                support_point,
                covariates,
                infusions,
                start_time,
                end_time,
            ),
            Equation::SDE(_, _, _, _, _, _, _) => {
                unimplemented!("Not Implemented");
            }
            Equation::Analytical(eq, seq_eq, _, _, _, _, _) => {
                analytical::simulate_analytical_event(
                    eq,
                    seq_eq,
                    x,
                    support_point,
                    covariates,
                    infusions,
                    start_time,
                    end_time,
                )
            }
        }
    }
    #[inline(always)]
    fn get_init(&self) -> &Init {
        match self {
            Equation::ODE(_, _, _, init, _, _) => init,
            Equation::SDE(_, _, _, _, init, _, _) => init,
            Equation::Analytical(_, _, _, _, init, _, _) => init,
        }
    }
    #[inline(always)]
    fn get_out(&self) -> &Out {
        match self {
            Equation::ODE(_, _, _, _, out, _) => out,
            Equation::SDE(_, _, _, _, _, out, _) => out,
            Equation::Analytical(_, _, _, _, _, out, _) => out,
        }
    }
    #[inline(always)]
    fn get_lag(&self, spp: &[f64]) -> HashMap<usize, f64> {
        match self {
            Equation::ODE(_, lag, _, _, _, _) => (lag)(&V::from_vec(spp.to_owned())),
            Equation::SDE(_, _, lag, _, _, _, _) => (lag)(&V::from_vec(spp.to_owned())),
            Equation::Analytical(_, _, lag, _, _, _, _) => (lag)(&V::from_vec(spp.to_owned())),
        }
    }
    #[inline(always)]
    fn get_fa(&self, spp: &[f64]) -> HashMap<usize, f64> {
        match self {
            Equation::ODE(_, _, fa, _, _, _) => (fa)(&V::from_vec(spp.to_owned())),
            Equation::SDE(_, _, _, fa, _, _, _) => (fa)(&V::from_vec(spp.to_owned())),
            Equation::Analytical(_, _, _, fa, _, _, _) => (fa)(&V::from_vec(spp.to_owned())),
        }
    }
    #[inline(always)]
    fn get_nstates(&self) -> usize {
        match self {
            Equation::ODE(_, _, _, _, _, (nstates, _)) => *nstates,
            Equation::SDE(_, _, _, _, _, _, (nstates, _)) => *nstates,
            Equation::Analytical(_, _, _, _, _, _, (nstates, _)) => *nstates,
        }
    }
    #[inline(always)]
    fn get_nouteqs(&self) -> usize {
        match self {
            Equation::ODE(_, _, _, _, _, (_, nouteqs)) => *nouteqs,
            Equation::SDE(_, _, _, _, _, _, (_, nouteqs)) => *nouteqs,
            Equation::Analytical(_, _, _, _, _, _, (_, nouteqs)) => *nouteqs,
        }
    }
}

//TODO: This is the one and only simulator, other copies of this (inside the Equation impl) should be removed
pub fn simulator(eq: &Equation, subject: &Subject, support_point: &Vec<f64>) -> SubjectPredictions {
    // Check for a cache entry
    // if cache {
    //     let pred = get_entry(subject, support_point);
    //     if let Some(pred) = pred {
    //         return pred;
    //     }
    // }
    let init = eq.get_init();
    let lag = eq.get_lag(support_point);
    let fa = eq.get_fa(support_point);
    let mut yout = vec![];

    for occasion in subject.occasions() {
        let covariates = occasion.get_covariates().unwrap();

        // if occasion == 0, we use the init closure to get the initial state
        // otherwise we initialize the state vector to zero
        let mut x = V::zeros(eq.get_nstates());
        if occasion.index() == 0 {
            (init)(&V::from_vec(support_point.clone()), 0.0, covariates, &mut x);
        }
        let mut infusions: Vec<Infusion> = vec![];
        let events = occasion.get_events(Some(&lag), Some(&fa), true);
        for (index, event) in events.iter().enumerate() {
            simulate_event(
                eq,
                support_point,
                event,
                events.get(index + 1),
                covariates,
                &mut x,
                &mut infusions,
                &mut yout,
            )
        }
    }
    let pred: SubjectPredictions = yout.into();

    pred
}

#[inline(always)]
fn simulate_event(
    eq: &Equation,
    support_point: &Vec<f64>,
    event: &Event,
    next_event: Option<&Event>,
    covariates: &Covariates,
    x: &mut nalgebra::DVector<f64>,
    infusions: &mut Vec<Infusion>,
    yout: &mut Vec<Prediction>,
) {
    let out = eq.get_out();
    match event {
        Event::Bolus(bolus) => {
            x[bolus.input()] += bolus.amount();
        }
        Event::Infusion(infusion) => {
            infusions.push(infusion.clone());
        }
        Event::Observation(observation) => {
            let mut y = V::zeros(eq.get_nouteqs());
            (out)(
                &x,
                &V::from_vec(support_point.clone()),
                observation.time(),
                covariates,
                &mut y,
            );
            let pred = y[observation.outeq()];

            yout.push(observation.to_obs_pred(pred));
        }
    }

    if let Some(next_event) = next_event {
        eq.simulate_event(
            x,
            support_point,
            covariates,
            &infusions,
            event.get_time(),
            next_event.get_time(),
        );
    }
}
