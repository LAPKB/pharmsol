mod EM;

use std::process::abort;

use nalgebra::DVector;

use crate::{
    data::{Covariates, Infusion},
    simulator::{Diffusion, Drift, M, T, V},
};

#[inline(always)]
pub(crate) fn simulate_sde_event(
    drift: &Drift,
    difussion: &Diffusion,
    x: V,
    support_point: &[f64],
    cov: &Covariates,
    infusions: &[Infusion],
    ti: f64,
    tf: f64,
) -> V {
    if ti == tf {
        return x;
    }
    dbg!(ti, tf);
    dbg!(&x);
    let mut sde = EM::SDE::new(
        drift.clone(),
        difussion.clone(),
        DVector::from_column_slice(support_point),
        x,
    );
    let solution = sde.solve(ti, tf, 1);
    let a: V = solution.last().unwrap().clone().into();
    dbg!(&a);
    abort();
    a
}
