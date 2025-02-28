use crate::{
    data::Covariates,
    simulator::{Diffusion, Drift},
    Infusion,
};
use nalgebra::DVector;
use rand::rng;
use rand_distr::{Distribution, Normal};
/// Structure to hold the SDE system parameters and state
#[derive(Clone)]
pub struct EM {
    drift: Drift,
    diffusion: Diffusion,
    params: DVector<f64>,
    state: DVector<f64>,
    cov: Covariates,
    infusions: Vec<Infusion>,
    rtol: f64,
    atol: f64,
    max_step: f64,
    min_step: f64,
}

impl EM {
    /// Creates a new SDE system
    pub fn new(
        drift: Drift,
        diffusion: Diffusion,
        params: DVector<f64>,
        initial_state: DVector<f64>,
        cov: Covariates,
        infusions: Vec<Infusion>,
        rtol: f64,
        atol: f64,
    ) -> Self {
        Self {
            drift,
            diffusion,
            params,
            state: initial_state,
            cov,
            infusions,
            rtol,
            atol,
            max_step: 0.1,  // Can be made configurable
            min_step: 1e-6, // Can be made configurable
        }
    }

    fn calculate_error(&self, y1: &DVector<f64>, y2: &DVector<f64>) -> f64 {
        let n = y1.len();
        let mut err = 0.0f64;

        for i in 0..n {
            let tol = self.atol + self.rtol * self.state[i].abs();
            let e = (y1[i] - y2[i]).abs() / tol;
            err = err.max(e);
        }
        err
    }

    fn compute_new_step(&self, dt: f64, error: f64, safety: f64) -> f64 {
        let mut new_dt = dt * safety * (1.0 / error).powf(0.5);
        new_dt = new_dt.clamp(self.min_step, self.max_step);
        new_dt
    }

    fn euler_maruyama_step(&self, time: f64, dt: f64, state: &mut DVector<f64>) {
        let n = state.len();
        let mut rateiv = DVector::from_vec(vec![0.0, 0.0, 0.0]);
        //TODO: This should be pre-calculated
        for infusion in &self.infusions {
            if time >= infusion.time() && time <= infusion.duration() + infusion.time() {
                rateiv[infusion.input()] += infusion.amount() / infusion.duration();
            }
        }
        let mut drift_term = DVector::zeros(n);
        (self.drift)(
            state,
            &self.params,
            time,
            &mut drift_term,
            rateiv,
            &self.cov,
        );

        let mut diffusion_term = DVector::zeros(n);
        (self.diffusion)(&self.params, &mut diffusion_term);

        let mut rng = rng();
        let normal_dist = Normal::new(0.0, 1.0).unwrap();

        for i in 0..n {
            state[i] +=
                drift_term[i] * dt + diffusion_term[i] * normal_dist.sample(&mut rng) * dt.sqrt();
        }
    }

    pub fn solve(&mut self, t0: f64, tf: f64) -> (Vec<f64>, Vec<DVector<f64>>) {
        let mut t = t0;
        let mut dt = self.max_step;
        let safety = 0.9;
        let mut times = vec![t0];
        let mut solution = vec![self.state.clone()];

        while t < tf {
            let mut y1 = self.state.clone();
            let mut y2 = self.state.clone();

            // Single step
            self.euler_maruyama_step(t, dt, &mut y1);

            // Two half steps
            self.euler_maruyama_step(t, dt / 2.0, &mut y2);
            self.euler_maruyama_step(t + dt / 2.0, dt / 2.0, &mut y2);

            let error = self.calculate_error(&y1, &y2);

            if error <= 1.0 {
                t += dt;
                self.state = y2; // Use more accurate solution
                times.push(t);
                solution.push(self.state.clone());
                dt = self.compute_new_step(dt, error, safety);
                dt = dt.min(tf - t); // Don't step beyond tf
            } else {
                dt = self.compute_new_step(dt, error, safety);
            }
        }

        (times, solution)
    }
}
