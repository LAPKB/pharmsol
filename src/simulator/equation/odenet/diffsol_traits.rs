use super::closure::ODENetClosure;
use crate::data::{Covariates, Infusion};
use diffsol::{
    error::DiffsolError,
    matrix::Matrix,
    ode_solver::{equations::OdeSolverEquations, problem::OdeSolverProblem},
    vector::Vector,
    ConstantClosure, OdeBuilder,
};
use std::rc::Rc;

#[allow(clippy::type_complexity)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_network_ode<M, I>(
    linear: Vec<M>,
    init: I,
    p: M::V,
    t0: f64,
    h0: f64,
    rtol: f64,
    atol: f64,
    cov: Covariates,
    infusions: Vec<Infusion>,
    nl: M::V,
) -> Result<
    OdeSolverProblem<OdeSolverEquations<M, ODENetClosure<M>, ConstantClosure<M, I>>>,
    DiffsolError,
>
where
    M: Matrix,
    I: Fn(&M::V, M::T) -> M::V,
{
    let p = Rc::new(p);
    let t0 = M::T::from(t0);
    let y0 = (init)(&p, t0);
    let nstates = y0.len();
    let rhs = ODENetClosure::new(linear, nstates, nstates, p.clone(), cov, infusions, nl);
    // let mass = Rc::new(UnitCallable::new(nstates));
    let rhs = Rc::new(rhs);
    let init = ConstantClosure::new(init, p.clone());
    let init = Rc::new(init);
    let eqn = OdeSolverEquations::new(rhs, None, None, init, None, p);
    OdeBuilder::new()
        .atol(vec![atol])
        .h0(h0)
        .rtol(rtol)
        .build_from_eqn(eqn)
}
