use nalgebra::DVector;
#[cfg(test)]
use rand::rng;
use rand::Rng;
use rand_distr::{Distribution, Normal};

/// Implementation of the Euler-Maruyama method for solving stochastic differential equations.
///
/// This structure holds the SDE system parameters and state, providing a numerical method
/// for approximating solutions to stochastic differential equations with adaptive step size
/// control for improved accuracy.
pub struct EM<D, G>
where
    D: Fn(f64, &DVector<f64>, &mut DVector<f64>),
    G: Fn(f64, &DVector<f64>, &mut DVector<f64>),
{
    drift: D,
    diffusion: G,
    state: DVector<f64>,
    rtol: f64,
    atol: f64,
    max_step: f64,
    min_step: f64,
}

impl<D, G> EM<D, G>
where
    D: Fn(f64, &DVector<f64>, &mut DVector<f64>),
    G: Fn(f64, &DVector<f64>, &mut DVector<f64>),
{
    /// Creates a new SDE solver using the Euler-Maruyama method.
    ///
    /// # Arguments
    ///
    /// * `drift` - Function defining the deterministic component of the SDE
    /// * `diffusion` - Function defining the stochastic component of the SDE
    /// * `params` - Vector of model parameters
    /// * `initial_state` - Initial state vector of the system
    /// * `cov` - Covariates that may influence the system dynamics
    /// * `infusions` - Vector of infusion events to be applied during simulation
    /// * `rtol` - Relative tolerance for adaptive step size control
    /// * `atol` - Absolute tolerance for adaptive step size control
    ///
    /// # Returns
    ///
    /// A new instance of the Euler-Maruyama solver configured with the given parameters.
    pub fn new(drift: D, diffusion: G, initial_state: DVector<f64>, rtol: f64, atol: f64) -> Self {
        Self {
            drift,
            diffusion,
            state: initial_state,
            rtol,
            atol,
            max_step: 0.1,
            min_step: 1e-6,
        }
    }

    /// Calculates the error between two approximations for adaptive step size control.
    ///
    /// # Arguments
    ///
    /// * `y1` - First approximation of the solution
    /// * `y2` - Second approximation of the solution (typically more accurate)
    ///
    /// # Returns
    ///
    /// The maximum normalized error between the two approximations.
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

    /// Computes a new step size based on the current error.
    ///
    /// # Arguments
    ///
    /// * `dt` - Current step size
    /// * `error` - Current error estimate
    /// * `safety` - Safety factor to prevent overly aggressive step size changes
    ///
    /// # Returns
    ///
    /// The adjusted step size for the next iteration.
    fn compute_new_step(&self, dt: f64, error: f64, safety: f64) -> f64 {
        if error == 0.0 {
            return self.max_step;
        }
        let new_dt = dt * safety * (1.0 / error).powf(0.5);
        new_dt.clamp(self.min_step, self.max_step)
    }

    /// Performs a single Euler-Maruyama integration step.
    ///
    /// # Arguments
    ///
    /// * `time` - Current simulation time
    /// * `dt` - Step size
    /// * `state` - Current state of the system (modified in-place)
    fn sample_brownian_increment<R: Rng + ?Sized>(&self, dt: f64, rng: &mut R) -> DVector<f64> {
        let normal_dist = Normal::new(0.0, dt.sqrt()).expect("positive integration step");
        DVector::from_fn(self.state.len(), |_, _| normal_dist.sample(rng))
    }

    fn euler_maruyama_step(
        &self,
        time: f64,
        dt: f64,
        state: &mut DVector<f64>,
        brownian_increment: &DVector<f64>,
    ) {
        let n = state.len();
        let mut drift_term = DVector::zeros(n);
        (self.drift)(time, state, &mut drift_term);

        let mut diffusion_term = DVector::zeros(n);
        (self.diffusion)(time, state, &mut diffusion_term);

        for i in 0..n {
            state[i] += drift_term[i] * dt + diffusion_term[i] * brownian_increment[i];
        }
    }

    /// Solves the SDE system over the specified time interval.
    ///
    /// Uses adaptive step size control to balance accuracy and performance.
    ///
    /// # Arguments
    ///
    /// * `t0` - Starting time
    /// * `tf` - Ending time
    ///
    /// # Returns
    ///
    /// A tuple containing:
    /// * Vector of time points where solutions were computed
    /// * Vector of state vectors corresponding to each time point
    #[cfg(test)]
    pub fn solve(&mut self, t0: f64, tf: f64) -> (Vec<f64>, Vec<DVector<f64>>) {
        let mut rng = rng();
        self.solve_impl(t0, tf, &mut rng)
    }

    /// Solve using only random draws from the supplied generator.
    pub fn solve_with_rng<R: Rng + ?Sized>(
        &mut self,
        t0: f64,
        tf: f64,
        rng: &mut R,
    ) -> (Vec<f64>, Vec<DVector<f64>>) {
        self.solve_impl(t0, tf, rng)
    }

    fn solve_impl<R: Rng + ?Sized>(
        &mut self,
        t0: f64,
        tf: f64,
        rng: &mut R,
    ) -> (Vec<f64>, Vec<DVector<f64>>) {
        let mut t = t0;
        let mut dt = self.max_step;
        let safety = 0.9;
        let mut times = vec![t0];
        let mut solution = vec![self.state.clone()];

        while t < tf {
            dt = dt.min(tf - t);
            let mut y1 = self.state.clone();
            let mut y2 = self.state.clone();

            let first_half_increment = self.sample_brownian_increment(dt / 2.0, rng);
            let second_half_increment = self.sample_brownian_increment(dt / 2.0, rng);
            let full_increment = &first_half_increment + &second_half_increment;

            self.euler_maruyama_step(t, dt, &mut y1, &full_increment);
            self.euler_maruyama_step(t, dt / 2.0, &mut y2, &first_half_increment);
            self.euler_maruyama_step(t + dt / 2.0, dt / 2.0, &mut y2, &second_half_increment);

            let error = self.calculate_error(&y1, &y2);
            if error <= 1.0 {
                t += dt;
                self.state = y2;
                times.push(t);
                solution.push(self.state.clone());
                dt = self.compute_new_step(dt, error, safety);
            } else {
                dt = self.compute_new_step(dt, error, safety);
            }
        }

        (times, solution)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    type TestFn = fn(f64, &DVector<f64>, &mut DVector<f64>);

    fn test_drift(_time: f64, state: &DVector<f64>, out: &mut DVector<f64>) {
        out[0] = 0.25 * state[0];
    }

    fn test_diffusion(_time: f64, state: &DVector<f64>, out: &mut DVector<f64>) {
        out[0] = 0.5 + 0.1 * state[0];
    }

    fn solver(initial_state: DVector<f64>) -> EM<TestFn, TestFn> {
        EM::new(test_drift, test_diffusion, initial_state, 1e-2, 1e-2)
    }

    #[test]
    fn full_step_increment_is_sum_of_seeded_half_step_increments() {
        let solver = solver(DVector::from_vec(vec![2.0]));
        let mut rng = StdRng::seed_from_u64(7);
        let first = solver.sample_brownian_increment(0.05, &mut rng);
        let second = solver.sample_brownian_increment(0.05, &mut rng);
        let full = &first + &second;

        let mut full_state = DVector::from_vec(vec![2.0]);
        solver.euler_maruyama_step(0.0, 0.1, &mut full_state, &full);
        let expected = 2.0 + 0.25 * 2.0 * 0.1 + (0.5 + 0.1 * 2.0) * (first[0] + second[0]);
        assert_eq!(full[0].to_bits(), (first[0] + second[0]).to_bits());
        assert!((full_state[0] - expected).abs() <= f64::EPSILON);
    }

    #[test]
    fn unseeded_solve_clamps_a_short_interval() {
        let drift = |_time: f64, _state: &DVector<f64>, out: &mut DVector<f64>| out[0] = 1.0;
        let diffusion = |_time: f64, _state: &DVector<f64>, out: &mut DVector<f64>| out[0] = 0.0;
        let mut solver = EM::new(drift, diffusion, DVector::from_vec(vec![0.0]), 1e-2, 1e-2);

        let (times, states) = solver.solve(0.0, 0.025);

        assert_eq!(times, vec![0.0, 0.025]);
        assert!((states.last().unwrap()[0] - 0.025).abs() < 1e-15);
    }

    #[test]
    fn seeded_adaptive_transition_is_reproducible() {
        let mut first = solver(DVector::from_vec(vec![2.0]));
        let mut second = solver(DVector::from_vec(vec![2.0]));
        let mut first_rng = StdRng::seed_from_u64(19);
        let mut second_rng = StdRng::seed_from_u64(19);

        let first_transition = first.solve_with_rng(0.0, 0.25, &mut first_rng);
        let second_transition = second.solve_with_rng(0.0, 0.25, &mut second_rng);

        assert_eq!(first_transition, second_transition);
        assert_eq!(first_transition.0.last(), Some(&0.25));
    }
}
