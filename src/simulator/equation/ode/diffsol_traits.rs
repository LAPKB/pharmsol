use super::closure::PMClosure;
use crate::data::{Covariates, Infusion};
use diffsol::{
    error::DiffsolError,
    matrix::Matrix,
    ode_solver::{equations::OdeSolverEquations, problem::OdeSolverProblem},
    vector::Vector,
    ConstantClosure,
};
use std::rc::Rc;

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
    let atol = M::V::from_element(nstates, M::T::from(atol));
    OdeSolverProblem::new(
        eqn,
        M::T::from(rtol),
        atol,
        t0,
        M::T::from(h0),
        false,
        false,
    )
}
