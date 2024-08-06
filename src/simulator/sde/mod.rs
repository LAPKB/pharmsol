mod em;

use nalgebra::DVector;

use crate::{
    data::{Covariates, Infusion},
    simulator::{Diffusion, Drift, V},
};

#[inline(always)]
pub(crate) fn simulate_sde_event(
    drift: &Drift,
    difussion: &Diffusion,
    x: V,
    support_point: &[f64],
    _cov: &Covariates,
    _infusions: &[Infusion],
    ti: f64,
    tf: f64,
) -> V {
    if ti == tf {
        return x;
    }

    let mut sde = em::SDE::new(
        drift.clone(),
        difussion.clone(),
        DVector::from_column_slice(support_point),
        x,
    );
    let solution = sde.solve(ti, tf, 10);
    solution.last().unwrap().clone().into()
}
