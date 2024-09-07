use crate::{
    data::Covariates,
    simulator::{Diffusion, Drift},
};
use nalgebra::DVector;
use rand::prelude::*;
use rand_distr::{Distribution, Normal};
/// Structure to hold the SDE system parameters and state
#[derive(Clone)]
pub struct EM {
    drift: Drift,
    diffusion: Diffusion,
    params: DVector<f64>,
    state: DVector<f64>,
}

impl EM {
    /// Creates a new SDE system
    pub fn new(
        drift: Drift,
        diffusion: Diffusion,
        params: DVector<f64>,
        initial_state: DVector<f64>,
    ) -> Self {
        Self {
            drift,
            diffusion,
            params,
            state: initial_state,
        }
    }

    /// Performs one step of the Euler-Maruyama method
    fn euler_maruyama_step(&mut self, time: f64, dt: f64) {
        let n = self.state.len();
        let mut drift_term = DVector::zeros(n);
        (self.drift)(
            &self.state.clone(),
            &self.params.clone(),
            time,
            &mut drift_term,
            DVector::zeros(0),
            &Covariates::default(),
        );

        let mut diffusion_term = DVector::zeros(n);
        (self.diffusion)(&self.params.clone(), &mut diffusion_term);

        // Create a seeded RNG
        let mut rng = StdRng::seed_from_u64(0);
        // let mut rng = thread_rng();

        for i in 0..n {
            let normal_dist = Normal::new(0.0, 1.0).unwrap();
            self.state[i] +=
                drift_term[i] * dt + diffusion_term[i] * normal_dist.sample(&mut rng) * dt.sqrt();
        }
    }

    /// Solves the SDE system over the given time period with the specified time step
    pub fn solve(&mut self, mut time: f64, tf: f64, steps: usize) -> Vec<DVector<f64>> {
        let dt = (tf - time) / steps as f64;
        let mut solution = Vec::with_capacity(steps);
        for _ in 0..steps {
            self.euler_maruyama_step(time, dt);
            time += dt;
            solution.push(self.state.clone());
        }
        solution
    }
}
