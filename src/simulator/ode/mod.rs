mod closure;
mod diffsol_traits;

use crate::{
    data::{Covariates, Infusion},
    simulator::{DiffEq, M, T, V},
};

use diffsol::{ode_solver::method::OdeSolverMethod, Bdf};

use self::diffsol_traits::build_pm_ode;

const RTOL: f64 = 1e-4;
const ATOL: f64 = 1e-4;

#[inline(always)]
pub(crate) fn simulate_ode_event(
    diffeq: &DiffEq,
    x: &mut V,
    support_point: &[f64],
    cov: &Covariates,
    infusions: &[Infusion],
    ti: f64,
    tf: f64,
) {
    if ti == tf {
        return;
    }
    let problem = build_pm_ode::<M, _, _>(
        *diffeq,
        |_p: &V, _t: T| x.clone(),
        V::from_vec(support_point.to_vec()),
        ti,
        1e-3,
        RTOL,
        ATOL,
        cov.clone(),
        infusions.to_owned(),
    )
    .unwrap();
    let mut solver = Bdf::default();
    let sol = solver.solve(&problem, tf).unwrap();
    *x = sol.0.last().unwrap().clone()
}
