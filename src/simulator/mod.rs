pub(crate) mod analytical;
mod cache;
pub mod fitting;
pub(crate) mod likelihood;
mod ode;
use ndarray::{parallel::prelude::*, Axis};
use std::collections::HashMap;

use crate::{
    data::{Covariates, Data, Event, Infusion, Subject},
    simulator::likelihood::{PopulationPredictions, SubjectPredictions, ToPrediction},
};

use cache::*;
use ndarray::prelude::*;
use ndarray::Array2;

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

/// This closure represents the initial state of the system:
/// Params:
/// - p: The parameters of the model; Use the [fetch_params!] macro to extract the parameters
/// - t: The time at which the initial state is evaluated; Hardcoded to 0.0
/// - cov: A reference to the covariates at time t; Use the [fetch_cov!] macro to extract the covariates
/// - x: A mutable reference to the state vector at time t
/// Example:
/// ```ignore
/// use pharmsol::*;
/// let init = |p, _t, cov, x| {
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

/// This closure represents an Analytical solution of the model, see [analytical] module for examples.
/// Params:
/// - x: The state vector at time t
/// - p: The parameters of the model; Use the [fetch_params!] macro to extract the parameters
/// - t: The time at which the output equation is evaluated
/// - rateiv: A vector of infusion rates at time t
/// - cov: A reference to the covariates at time t; Use the [fetch_cov!] macro to extract the covariates
/// TODO: Remove covariates. They are not used in the analytical solution
pub type AnalyticalEq = fn(&V, &V, T, V, &Covariates) -> V;

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

#[derive(Debug, Clone)]
pub enum Equation {
    ODE(DiffEq, Lag, Fa, Init, Out, Neqs),
    SDE(DiffEq, DiffEq, Lag, Fa, Init, Out, Neqs),
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

    pub fn simulate_subject(
        &self,
        subject: &Subject,
        support_point: &Vec<f64>,
    ) -> SubjectPredictions {
        let init = self.get_init();
        let out = self.get_out();
        let lag = self.get_lag(support_point);
        let fa = self.get_fa(support_point);
        let mut yout = vec![];

        for occasion in subject.occasions() {
            // Check for a cache entry
            let pred = get_entry(subject, support_point);
            if let Some(pred) = pred {
                return pred;
            }
            let covariates = occasion.get_covariates().unwrap();

            // if occasion == 0, we use the init closure to get the initial state
            // otherwise we initialize the state vector to zero
            let mut x = V::zeros(self.get_nstates());
            if occasion.index() == 0 {
                (init)(&V::from_vec(support_point.clone()), 0.0, covariates, &mut x);
            }
            let mut infusions: Vec<Infusion> = vec![];
            let events = occasion.get_events(Some(&lag), Some(&fa), true);
            for (index, event) in events.iter().enumerate() {
                match event {
                    Event::Bolus(bolus) => {
                        x[bolus.input()] += bolus.amount();
                    }
                    Event::Infusion(infusion) => {
                        infusions.push(infusion.clone());
                    }
                    Event::Observation(observation) => {
                        let mut y = V::zeros(self.get_nouteqs());
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

                if let Some(next_event) = events.get(index + 1) {
                    x = self.simulate_event(
                        x,
                        support_point,
                        covariates,
                        &infusions,
                        event.get_time(),
                        next_event.get_time(),
                    );
                }
            }
        }
        // Insert the cache entry
        let pred: SubjectPredictions = yout.into();
        insert_entry(subject, support_point, pred.clone());
        pred
    }
    #[inline(always)]

    fn simulate_event(
        &self,
        x: V,
        support_point: &[f64],
        covariates: &Covariates,
        infusions: &Vec<Infusion>,
        start_time: T,
        end_time: T,
    ) -> V {
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
            Equation::SDE(_, _, _, _, _, _, _) => unimplemented!("Not Implemented"),
            Equation::Analytical(_, _, lag, _, _, _, _) => (lag)(&V::from_vec(spp.to_owned())),
        }
    }
    #[inline(always)]
    fn get_fa(&self, spp: &[f64]) -> HashMap<usize, f64> {
        match self {
            Equation::ODE(_, _, fa, _, _, _) => (fa)(&V::from_vec(spp.to_owned())),
            Equation::SDE(_, _, _, _, _, _, _) => unimplemented!("Not Implemented"),
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

pub fn get_population_predictions(
    equation: &Equation,
    subjects: &Data,
    support_points: &Array2<f64>,
    _cache: bool,
) -> PopulationPredictions {
    let mut pred = Array2::default((subjects.len(), support_points.nrows()).f());
    pred.axis_iter_mut(Axis(0))
        .into_par_iter()
        .enumerate()
        .for_each(|(i, mut row)| {
            row.axis_iter_mut(Axis(0))
                .into_par_iter()
                .enumerate()
                .for_each(|(j, mut element)| {
                    let subjects = subjects.get_subjects();
                    let subject = subjects.get(i).unwrap();
                    let ypred =
                        equation.simulate_subject(subject, support_points.row(j).to_vec().as_ref());
                    element.fill(ypred);
                });
        });
    pred.into()
}
