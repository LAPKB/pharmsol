use nalgebra::DVector;
use rand::rng;
use rand_distr::{Distribution, Normal};

/// Step size strategy for the Euler-Maruyama SDE solver.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SdeStepSize {
    /// Advance the solver in fixed-size increments of `dt`.
    Fixed(f64),
    /// Divide each solved interval (i.e. the span between two consecutive
    /// bolus/observation events; infusions are applied continuously within
    /// the drift function and don't create extra intervals) into exactly
    /// `n` equal steps, regardless of the interval's length.
    EventSteps(usize),
    /// Adapt the step size between `min_step` and `max_step` so that the
    /// estimated local error stays within `rtol`/`atol`.
    Adaptive {
        rtol: f64,
        atol: f64,
        min_step: f64,
        max_step: f64,
    },
}

impl SdeStepSize {
    /// Convenience constructor for the adaptive strategy using the crate's
    /// historical default `min_step`/`max_step` bounds.
    pub fn adaptive(rtol: f64, atol: f64) -> Self {
        SdeStepSize::Adaptive {
            rtol,
            atol,
            min_step: 1e-6,
            max_step: 0.1,
        }
    }
}

impl Default for SdeStepSize {
    fn default() -> Self {
        SdeStepSize::adaptive(1e-2, 1e-2)
    }
}

/// Implementation of the Euler-Maruyama method for solving stochastic differential equations.
///
/// This structure holds the SDE system parameters and state, providing a numerical method
/// for approximating solutions to stochastic differential equations, using either a fixed
/// or an adaptive step size (see [`SdeStepSize`]).
pub struct EM<D, G>
where
    D: Fn(f64, &DVector<f64>, &mut DVector<f64>),
    G: Fn(f64, &DVector<f64>, &mut DVector<f64>),
{
    drift: D,
    diffusion: G,
    state: DVector<f64>,
    step_size: SdeStepSize,
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
    /// * `initial_state` - Initial state vector of the system
    /// * `step_size` - Fixed or adaptive step size configuration
    ///
    /// # Returns
    ///
    /// A new instance of the Euler-Maruyama solver configured with the given parameters.
    pub fn new(
        drift: D,
        diffusion: G,
        initial_state: DVector<f64>,
        step_size: SdeStepSize,
    ) -> Self {
        Self {
            drift,
            diffusion,
            state: initial_state,
            step_size,
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
    fn calculate_error(&self, y1: &DVector<f64>, y2: &DVector<f64>, rtol: f64, atol: f64) -> f64 {
        let n = y1.len();
        let mut err = 0.0f64;

        for i in 0..n {
            let tol = atol + rtol * self.state[i].abs();
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
    #[allow(clippy::too_many_arguments)]
    fn compute_new_step(
        &self,
        dt: f64,
        error: f64,
        safety: f64,
        min_step: f64,
        max_step: f64,
    ) -> f64 {
        let mut new_dt = dt * safety * (1.0 / error).powf(0.5);
        new_dt = new_dt.clamp(min_step, max_step);
        new_dt
    }

    /// Performs a single Euler-Maruyama integration step.
    ///
    /// # Arguments
    ///
    /// * `time` - Current simulation time
    /// * `dt` - Step size
    /// * `state` - Current state of the system (modified in-place)
    fn euler_maruyama_step(&self, time: f64, dt: f64, state: &mut DVector<f64>) {
        let n = state.len();
        let mut drift_term = DVector::zeros(n);
        (self.drift)(time, state, &mut drift_term);

        let mut diffusion_term = DVector::zeros(n);
        (self.diffusion)(time, state, &mut diffusion_term);

        let mut rng = rng();
        let normal_dist = Normal::new(0.0, 1.0).unwrap();

        for i in 0..n {
            state[i] +=
                drift_term[i] * dt + diffusion_term[i] * normal_dist.sample(&mut rng) * dt.sqrt();
        }
    }

    /// Solves the SDE system over the specified time interval, using either a
    /// fixed step size or adaptive step size control, depending on how the
    /// solver was configured (see [`SdeStepSize`]).
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
    pub fn solve(&mut self, t0: f64, tf: f64) -> (Vec<f64>, Vec<DVector<f64>>) {
        match self.step_size {
            SdeStepSize::Fixed(dt) => self.solve_fixed(t0, tf, dt),
            SdeStepSize::EventSteps(n) => {
                let dt = (tf - t0) / n.max(1) as f64;
                self.solve_fixed(t0, tf, dt)
            }
            SdeStepSize::Adaptive {
                rtol,
                atol,
                min_step,
                max_step,
            } => self.solve_adaptive(t0, tf, rtol, atol, min_step, max_step),
        }
    }

    /// Advances the system from `t0` to `tf` in fixed increments of `dt`,
    /// with no local error control (the final step is truncated to land exactly on `tf`).
    fn solve_fixed(&mut self, t0: f64, tf: f64, dt: f64) -> (Vec<f64>, Vec<DVector<f64>>) {
        let mut t = t0;
        let mut times = vec![t0];
        let mut solution = vec![self.state.clone()];

        while t < tf {
            let step = dt.min(tf - t);
            let mut next = self.state.clone();
            self.euler_maruyama_step(t, step, &mut next);
            self.state = next;
            t += step;
            times.push(t);
            solution.push(self.state.clone());
        }

        (times, solution)
    }

    /// Advances the system from `t0` to `tf`, adapting the step size so that the
    /// estimated local error (via step-doubling) stays within `rtol`/`atol`.
    #[allow(clippy::too_many_arguments)]
    fn solve_adaptive(
        &mut self,
        t0: f64,
        tf: f64,
        rtol: f64,
        atol: f64,
        min_step: f64,
        max_step: f64,
    ) -> (Vec<f64>, Vec<DVector<f64>>) {
        let mut t = t0;
        let mut dt = max_step;
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

            let error = self.calculate_error(&y1, &y2, rtol, atol);

            if error <= 1.0 {
                t += dt;
                self.state = y2; // Use more accurate solution
                times.push(t);
                solution.push(self.state.clone());
                dt = self.compute_new_step(dt, error, safety, min_step, max_step);
                dt = dt.min(tf - t); // Don't step beyond tf
            } else {
                dt = self.compute_new_step(dt, error, safety, min_step, max_step);
            }
        }

        (times, solution)
    }
}
