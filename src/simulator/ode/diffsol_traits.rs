use crate::data::{Covariates, Infusion};
use anyhow::{Ok, Result};
use diffsol::{
    matrix::Matrix,
    ode_solver::{equations::OdeSolverEquations, problem::OdeSolverProblem},
    op::unit::UnitCallable,
    vector::Vector,
};

use std::rc::Rc;

use super::closure::PMClosure;

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
) -> Result<OdeSolverProblem<OdeSolverEquations<M, PMClosure<M, F>, I>>>
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
    let mass = Rc::new(UnitCallable::new(nstates));
    let rhs = Rc::new(rhs);
    let eqn = OdeSolverEquations::new(rhs, mass, None, init, p, false);
    let atol = M::V::from_element(nstates, M::T::from(atol));
    Ok(OdeSolverProblem::new(
        eqn,
        M::T::from(rtol),
        atol,
        t0,
        M::T::from(h0),
    ))
}
