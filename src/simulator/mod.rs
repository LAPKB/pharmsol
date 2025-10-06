pub mod equation;
pub(crate) mod likelihood;
use diffsol::{NalgebraMat, NalgebraVec};

use crate::{
    data::{Covariates, Infusion},
    error_model::ErrorModels,
    simulator::likelihood::SubjectPredictions,
};

use std::collections::HashMap;

/// Type alias for floating point values used in simulations
pub type T = f64;
/// Type alias for state vector type used in simulations
pub type V = NalgebraVec<T>;
/// Type alias for matrix type used in simulations
pub type M = NalgebraMat<T>;

/// This closure represents the differential equation of the model.
///
/// # Parameters
/// - `x`: The state vector at time t
/// - `p`: The parameters of the model; Use the [fetch_params!] macro to extract the parameters
/// - `t`: The time at which the differential equation is evaluated
/// - `dx`: A mutable reference to the derivative of the state vector at time t
/// - `bolus`: A vector of bolus amounts at time t
/// - `rateiv`: A vector of infusion rates at time t
/// - `cov`: A reference to the covariates at time t; Use the [fetch_cov!] macro to extract the covariates
///
/// # Example
/// ```ignore
/// use pharmsol::*;
/// let diff_eq = |x, p, t, dx, bolus, rateiv, cov| {
///  fetch_params!(p, ka, ke, v);
///  fetch_cov!(cov, t, wt);
///  dx[0] = -ka * x[0];
///  dx[1] = ka * x[0] - ke * x[1];
/// };
/// ```
pub type DiffEq = fn(&V, &V, T, &mut V, V, V, &Covariates);

/// This closure represents an Analytical solution of the model.
/// See [analytical] module for examples.
///
/// # Parameters
/// - `x`: The state vector at time t
/// - `p`: The parameters of the model; Use the [fetch_params!] macro to extract the parameters
/// - `t`: The time at which the output equation is evaluated
/// - `rateiv`: A vector of infusion rates at time t
/// - `cov`: A reference to the covariates at time t; Use the [fetch_cov!] macro to extract the covariates
///
/// TODO: Remove covariates. They are not used in the analytical solution
pub type AnalyticalEq = fn(&V, &V, T, V, &Covariates) -> V;

/// This closure represents the drift term of a stochastic differential equation model.
///
/// # Parameters
/// - `x`: The state vector at time t
/// - `p`: The parameters of the model; Use the [fetch_params!] macro to extract the parameters
/// - `t`: The time at which the drift term is evaluated
/// - `dx`: A mutable reference to the derivative of the state vector at time t
/// - `rateiv`: A vector of infusion rates at time t
/// - `cov`: A reference to the covariates at time t; Use the [fetch_cov!] macro to extract the covariates
///
/// # Example
/// ```ignore
/// use pharmsol::*;
/// let drift = |x, p, t, dx, rateiv, cov| {
///   fetch_params!(p, mka, mke, v);
///   fetch_cov!(cov, t, wt);
///   ka = dx[2];
///   ke = dx[3];
///   dx[0] = -ka * x[0];
///   dx[1] = ka * x[0] - ke * x[1];
///   dx[2] = -dx[2] + mka; // Mean reverting to mka
///   dx[3] = -dx[3] + mke; // Mean reverting to mke
/// };
/// ```
// pub type Drift = DiffEq;
pub type Drift = fn(&V, &V, T, &mut V, V, &Covariates);

/// This closure represents the diffusion term of a stochastic differential equation model.
///
/// # Parameters
/// - `p`: The parameters of the model; Use the [fetch_params!] macro to extract the parameters
/// - `d`: A mutable reference to the diffusion term for each state variable
///   (This vector should have the same length as the x, and dx vectors on the drift closure)
pub type Diffusion = fn(&V, &mut V);

/// This closure represents the initial state of the system.
///
/// # Parameters
/// - `p`: The parameters of the model; Use the [fetch_params!] macro to extract the parameters
/// - `t`: The time at which the initial state is evaluated; Hardcoded to 0.0
/// - `cov`: A reference to the covariates at time t; Use the [fetch_cov!] macro to extract the covariates
/// - `x`: A mutable reference to the state vector at time t
///
/// # Example
/// ```ignore
/// use pharmsol::*;
/// let init = |p, t, cov, x| {
///  fetch_params!(p, ka, ke, v);
///  fetch_cov!(cov, t, wt);
///  x[0] = 500.0;
///  x[1] = 0.0;
/// };
/// ```
pub type Init = fn(&V, T, &Covariates, &mut V);

/// This closure represents the output equation of the model.
///
/// # Parameters
/// - `x`: The state vector at time t
/// - `p`: The parameters of the model; Use the [fetch_params!] macro to extract the parameters
/// - `t`: The time at which the output equation is evaluated
/// - `cov`: A reference to the covariates at time t; Use the [fetch_cov!] macro to extract the covariates
/// - `y`: A mutable reference to the output vector at time t
///
/// # Example
/// ```ignore
/// use pharmsol::*;
/// let out = |x, p, t, cov, y| {
///   fetch_params!(p, ka, ke, v);
///   y[0] = x[1] / v;
/// };
/// ```
pub type Out = fn(&V, &V, T, &Covariates, &mut V);

/// This closure represents the secondary equation of the model.
///
/// Secondary equations are used to update the parameter values based on the covariates.
///
/// # Parameters
/// - `p`: The parameters of the model; Use the [fetch_params!] macro to extract the parameters
/// - `t`: The time at which the secondary equation is evaluated
/// - `cov`: A reference to the covariates at time t; Use the [fetch_cov!] macro to extract the covariates
///
/// # Example
/// ```ignore
/// use pharmsol::*;
/// let sec_eq = |p, _t, cov| {
///    fetch_params!(p, ka, ke);
///    fetch_cov!(cov, t, wt);
///    ka = ka * wt;
/// };
/// ```
pub type SecEq = fn(&mut V, T, &Covariates);

/// This closure represents the lag time of the model.
///
/// The lag term delays only the boluses going into a specific compartment.
///
/// # Parameters
/// - `p`: The parameters of the model; Use the [fetch_params!] macro to extract the parameters
///
/// # Returns
/// - A hashmap with the lag times for each compartment. If not present, lag is assumed to be 0.
///
/// There is a convenience macro [lag!] to create the hashmap
///
/// # Example
/// ```ignore
/// use pharmsol::*;
/// let lag = |p, _t, _cov| {
///    fetch_params!(p, tlag);
///    lag! {0=>tlag, 1=>0.3}
/// };
/// ```
/// This will lag the bolus going into the first compartment by tlag and the bolus going into the
/// second compartment by 0.3
pub type Lag = fn(&V, T, &Covariates) -> HashMap<usize, T>;

/// This closure represents the fraction absorbed (also called bioavailability or protein binding)
/// of the model.
///
/// The fa term is used to adjust the amount of drug that is absorbed into the system.
///
/// # Parameters
/// - `p`: The parameters of the model; Use the [fetch_params!] macro to extract the parameters
///
/// # Returns
/// - A hashmap with the fraction absorbed for each compartment. If not present, it is assumed to be 1.
///
/// There is a convenience macro [fa!] to create the hashmap
///
/// # Example
/// ```ignore
/// use pharmsol::*;
/// let fa = |p, _t, _cov| {
///   fetch_params!(p, fa);
///   fa! {0=>fa, 1=>0.3}
/// };
/// ```
/// This will adjust the amount of drug absorbed into the first compartment by fa and the amount of drug
/// absorbed into the second compartment by 0.3
pub type Fa = fn(&V, T, &Covariates) -> HashMap<usize, T>;

/// The number of states and output equations of the model.
///
/// # Components
/// - The first element is the number of states
/// - The second element is the number of output equations
///
/// This is used to initialize the state vector and the output vector.
///
/// # Example
/// ```ignore
/// let neqs = (2, 1);
/// ```
/// This means that the system of equations has 2 states and there is only 1 output equation.
///
pub type Neqs = (usize, usize);
