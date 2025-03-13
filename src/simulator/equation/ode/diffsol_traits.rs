use super::closure::PMClosure;
use crate::data::{Covariates, Infusion};
use diffsol::{
    error::DiffsolError,
    matrix::Matrix,
    ode_solver::{equations::OdeSolverEquations, problem::OdeSolverProblem},
    vector::Vector,
    ConstantClosure, OdeBuilder,
};
use std::rc::Rc;

/// Builds an ODE problem for pharmacometric models using the diffsol library.
///
/// This function creates an ODE solver problem by wrapping the pharmacometric model
/// functions into the appropriate closures and structures expected by diffsol.
///
/// # Type Parameters
///
/// * `M`: Matrix type used for linear algebra operations
/// * `F`: Type of the right-hand side function for the ODEs
/// * `I`: Type of the initial condition function
///
/// # Arguments
///
/// * `rhs`: Right-hand side function of the ODE system
/// * `init`: Initial condition function
/// * `p`: Parameter vector
/// * `t0`: Initial time
/// * `h0`: Initial step size
/// * `rtol`: Relative tolerance for error control
/// * `atol`: Absolute tolerance for error control
/// * `cov`: Covariates that may influence the system
/// * `infusions`: Vector of infusion events to be applied during simulation
///
/// # Returns
///
/// Result containing either the configured ODE solver problem or an error
#[allow(clippy::type_complexity)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_pm_ode<M, F, I>(
    rhs: F,
    init: I,
    p: M::V,
    t0: f64,
    h0: f64,
    rtol: f64,
    atol: f64,
    cov: Covariates,
    infusions: Vec<Infusion>,
) -> Result<
    OdeSolverProblem<OdeSolverEquations<M, PMClosure<M, F>, ConstantClosure<M, I>>>,
    DiffsolError,
>
where
    M: Matrix,
    F: Fn(&M::V, &M::V, M::T, &mut M::V, M::V, &Covariates),
    I: Fn(&M::V, M::T) -> M::V,
{
    let p = Rc::new(p);
    let t0 = M::T::from(t0);
    let y0 = (init)(&p, t0);
    let nstates = y0.len();
    let rhs = PMClosure::new(rhs, nstates, nstates, p.clone(), cov, infusions);
    // let mass = Rc::new(UnitCallable::new(nstates));
    let rhs = Rc::new(rhs);
    let init = ConstantClosure::new(init, p.clone());
    let init = Rc::new(init);
    let eqn = OdeSolverEquations::new(rhs, None, None, init, None, p);
    // let atol = M::V::from_element(nstates, M::T::from(atol));
    OdeBuilder::new()
        .atol(vec![atol])
        .h0(h0)
        .rtol(rtol)
        .t0(t0.into())
        .build_from_eqn(eqn)

    // OdeSolverProblem::new(
    //     eqn,
    //     M::T::from(rtol),
    //     atol,
    //     t0,
    //     M::T::from(h0),
    //     false,
    //     false,
    // )
}
