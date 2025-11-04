use crate::{
    data::Covariates,
    simulator::{Diffusion, Drift},
    Infusion,
};
use nalgebra::DVector;
use rand::rng;
use rand_distr::{Distribution, Normal};
use std::collections::BTreeMap;

/// Pre-computed infusion schedule for efficient rate lookups during integration.
///
/// Instead of iterating through all infusions at every time step, this structure
/// maintains a sorted list of time points where infusion rates change and the
/// corresponding rate vectors for efficient binary search lookups.
#[derive(Clone, Debug)]
struct InfusionSchedule {
    /// Sorted time points where infusion rates change
    time_points: Vec<f64>,
    /// Infusion rate vector for each time interval
    rates: Vec<DVector<f64>>,
}

impl InfusionSchedule {
    /// Creates a new infusion schedule from a list of infusions.
    ///
    /// # Arguments
    ///
    /// * `infusions` - Vector of infusion events
    /// * `nstates` - Number of states in the system
    /// * `t_start` - Start time of simulation
    /// * `t_end` - End time of simulation
    ///
    /// # Returns
    ///
    /// A pre-computed schedule for O(log n) rate lookups
    fn new(infusions: &[Infusion], nstates: usize, t_start: f64, t_end: f64) -> Self {
        // Build a map of time points to rate changes
        let mut events: BTreeMap<String, DVector<f64>> = BTreeMap::new();

        for inf in infusions {
            let start = inf.time();
            let end = start + inf.duration();
            let rate = inf.amount() / inf.duration();
            let input = inf.input();

            // Add rate at start time
            if start >= t_start && start <= t_end {
                let key = format!("{:.15}", start);
                events.entry(key).or_insert_with(|| DVector::zeros(nstates))[input] += rate;
            }

            // Subtract rate at end time
            if end >= t_start && end <= t_end {
                let key = format!("{:.15}", end);
                events.entry(key).or_insert_with(|| DVector::zeros(nstates))[input] -= rate;
            }
        }

        // Convert to sorted vectors with cumulative rates
        let mut time_points = Vec::new();
        let mut rates = Vec::new();
        let mut current_rate = DVector::zeros(nstates);

        // Add initial zero rate
        time_points.push(t_start);
        rates.push(current_rate.clone());

        // Process events in chronological order
        for (time_str, rate_change) in events {
            let time: f64 = time_str.parse().unwrap();
            current_rate += rate_change;
            time_points.push(time);
            rates.push(current_rate.clone());
        }

        Self { time_points, rates }
    }

    /// Gets the infusion rate vector at a given time.
    ///
    /// Uses binary search for O(log n) lookup complexity.
    ///
    /// # Arguments
    ///
    /// * `time` - The time at which to get infusion rates
    /// * `out` - Output vector to store the rates
    fn get_rate(&self, time: f64, out: &mut DVector<f64>) {
        // Binary search for the appropriate interval
        let idx = match self
            .time_points
            .binary_search_by(|t| t.partial_cmp(&time).unwrap_or(std::cmp::Ordering::Equal))
        {
            Ok(i) => i,
            Err(i) => i.saturating_sub(1),
        };

        let idx = idx.min(self.rates.len() - 1);
        out.copy_from(&self.rates[idx]);
    }
}

/// Implementation of the Euler-Maruyama method for solving stochastic differential equations.
///
/// This structure holds the SDE system parameters and state, providing a numerical method
/// for approximating solutions to stochastic differential equations with adaptive step size
/// control for improved accuracy.
#[derive(Clone)]
pub struct EM {
    drift: Drift,
    diffusion: Diffusion,
    params: DVector<f64>,
    state: DVector<f64>,
    cov: Covariates,
    infusion_schedule: InfusionSchedule,
    rateiv_buffer: DVector<f64>,
    rtol: f64,
    atol: f64,
    max_step: f64,
    min_step: f64,
}

impl EM {
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
    /// * `t0` - Start time for infusion schedule computation
    /// * `tf` - End time for infusion schedule computation
    ///
    /// # Returns
    ///
    /// A new instance of the Euler-Maruyama solver configured with the given parameters.
    pub fn new(
        drift: Drift,
        diffusion: Diffusion,
        params: DVector<f64>,
        initial_state: DVector<f64>,
        cov: Covariates,
        infusions: Vec<Infusion>,
        rtol: f64,
        atol: f64,
        t0: f64,
        tf: f64,
    ) -> Self {
        let nstates = initial_state.len();
        let infusion_schedule = InfusionSchedule::new(&infusions, nstates, t0, tf);
        let rateiv_buffer = DVector::zeros(nstates);

        Self {
            drift,
            diffusion,
            params,
            state: initial_state,
            cov,
            infusion_schedule,
            rateiv_buffer,
            rtol,
            atol,
            max_step: 0.1,  // Can be made configurable
            min_step: 1e-6, // Can be made configurable
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
        let mut new_dt = dt * safety * (1.0 / error).powf(0.5);
        new_dt = new_dt.clamp(self.min_step, self.max_step);
        new_dt
    }

    /// Performs a single Euler-Maruyama integration step.
    ///
    /// # Arguments
    ///
    /// * `time` - Current simulation time
    /// * `dt` - Step size
    /// * `state` - Current state of the system (modified in-place)
    fn euler_maruyama_step(&mut self, time: f64, dt: f64, state: &mut DVector<f64>) {
        let n = state.len();

        // Get pre-computed infusion rates at this time (O(log n) instead of O(n))
        self.infusion_schedule
            .get_rate(time, &mut self.rateiv_buffer);

        let mut drift_term = DVector::zeros(n).into();
        (self.drift)(
            &state.clone().into(),
            &self.params.clone().into(),
            time,
            &mut drift_term,
            self.rateiv_buffer.clone().into(),
            &self.cov,
        );

        let mut diffusion_term = DVector::zeros(n).into();
        (self.diffusion)(&self.params.clone().into(), &mut diffusion_term);

        let mut rng = rng();
        let normal_dist = Normal::new(0.0, 1.0).unwrap();

        for i in 0..n {
            state[i] +=
                drift_term[i] * dt + diffusion_term[i] * normal_dist.sample(&mut rng) * dt.sqrt();
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
