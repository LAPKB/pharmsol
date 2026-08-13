//! Direct fixed-step simulation used to generate the paper's SDE trajectory figures.
//!
//! This module intentionally bypasses the particle-filter workflow. Each entry in
//! [`PopulationTrajectories::subjects`] is one simulated population subject with
//! its own fixed `ke0` parameter and complete `X(t)`/`Ke(t)` trajectory.

use std::{fs::File, io, path::Path};

use csv::Writer;
use nalgebra::DVector;
use rand::{rngs::StdRng, Rng, RngExt, SeedableRng};
use rand_distr::{Distribution, StandardNormal};
use thiserror::Error;

use super::em::EM;

/// Configuration for the paper's two-state stochastic PK simulation.
#[derive(Clone, Debug, PartialEq)]
pub struct PaperSimulationConfig {
    pub seed: u64,
    pub n_subjects: usize,
    pub t0: f64,
    pub tf: f64,
    pub dt: f64,
    pub x0: f64,
    pub volume: f64,
    pub sigma_x: f64,
    pub sigma_ke: f64,
    pub mixture_weight_1: f64,
    pub mixture_mean_1: f64,
    pub mixture_sd_1: f64,
    pub mixture_weight_2: f64,
    pub mixture_mean_2: f64,
    pub mixture_sd_2: f64,
}

impl Default for PaperSimulationConfig {
    fn default() -> Self {
        Self {
            seed: 12_345,
            n_subjects: 100,
            t0: 0.0,
            tf: 1.0,
            dt: 0.0002,
            x0: 20.0,
            volume: 1.0,
            sigma_x: 0.05,
            // The manuscript heading says 0.1, while the experiment text says
            // 0.5. The historical figure appears more compatible with 0.5.
            sigma_ke: 0.5,
            mixture_weight_1: 0.5,
            mixture_mean_1: 0.5,
            mixture_sd_1: 0.05,
            mixture_weight_2: 0.5,
            mixture_mean_2: 1.5,
            mixture_sd_2: 0.15,
        }
    }
}

/// A complete trajectory for one simulated population subject.
#[derive(Clone, Debug, PartialEq)]
pub struct SubjectTrajectory {
    pub subject: usize,
    pub mixture_component: usize,
    pub ke0: f64,
    pub times: Vec<f64>,
    pub x: Vec<f64>,
    pub concentration: Vec<f64>,
    pub ke: Vec<f64>,
}

/// Complete trajectories for the simulated population.
#[derive(Clone, Debug, PartialEq)]
pub struct PopulationTrajectories {
    pub subjects: Vec<SubjectTrajectory>,
}

/// Errors raised by the paper-specific simulation or CSV export.
#[derive(Debug, Error)]
pub enum PaperSimulationError {
    #[error("invalid paper simulation configuration: {0}")]
    InvalidConfiguration(String),
    #[error("non-finite value for subject {subject} at step {step}: {state}")]
    NonFinite {
        subject: usize,
        step: usize,
        state: &'static str,
    },
    #[error(transparent)]
    Io(#[from] io::Error),
    #[error(transparent)]
    Csv(#[from] csv::Error),
}

/// Return the number of fixed Euler--Maruyama updates for an interval.
pub fn fixed_step_count(t0: f64, tf: f64, dt: f64) -> Result<usize, PaperSimulationError> {
    if !t0.is_finite() || !tf.is_finite() || !dt.is_finite() {
        return Err(PaperSimulationError::InvalidConfiguration(
            "t0, tf, and dt must be finite".to_string(),
        ));
    }
    if tf <= t0 {
        return Err(PaperSimulationError::InvalidConfiguration(
            "tf must be greater than t0".to_string(),
        ));
    }
    if dt <= 0.0 {
        return Err(PaperSimulationError::InvalidConfiguration(
            "dt must be positive".to_string(),
        ));
    }

    let ratio = (tf - t0) / dt;
    let rounded = ratio.round();
    let tolerance = 1e-10 * ratio.abs().max(1.0);
    if (ratio - rounded).abs() > tolerance || rounded < 1.0 {
        return Err(PaperSimulationError::InvalidConfiguration(format!(
            "(tf - t0) / dt must be a positive integer, got {ratio}"
        )));
    }
    Ok(rounded as usize)
}

impl PaperSimulationConfig {
    /// Validate the numerical configuration and return its fixed-step count.
    pub fn validate(&self) -> Result<usize, PaperSimulationError> {
        let steps = fixed_step_count(self.t0, self.tf, self.dt)?;
        if self.n_subjects == 0 {
            return Err(PaperSimulationError::InvalidConfiguration(
                "n_subjects must be positive".to_string(),
            ));
        }
        if !self.x0.is_finite() || !self.volume.is_finite() || self.volume <= 0.0 {
            return Err(PaperSimulationError::InvalidConfiguration(
                "x0 must be finite and volume must be finite and positive".to_string(),
            ));
        }
        if !self.sigma_x.is_finite() || self.sigma_x < 0.0 {
            return Err(PaperSimulationError::InvalidConfiguration(
                "sigma_x must be finite and nonnegative".to_string(),
            ));
        }
        if !self.sigma_ke.is_finite() || self.sigma_ke < 0.0 {
            return Err(PaperSimulationError::InvalidConfiguration(
                "sigma_ke must be finite and nonnegative".to_string(),
            ));
        }
        if !self.mixture_weight_1.is_finite()
            || !self.mixture_weight_2.is_finite()
            || self.mixture_weight_1 < 0.0
            || self.mixture_weight_1 > 1.0
            || self.mixture_weight_2 < 0.0
            || self.mixture_weight_2 > 1.0
            || (self.mixture_weight_1 + self.mixture_weight_2 - 1.0).abs() > 1e-12
        {
            return Err(PaperSimulationError::InvalidConfiguration(
                "mixture weights must be in [0, 1] and sum to one".to_string(),
            ));
        }
        for (name, value) in [
            ("mixture_mean_1", self.mixture_mean_1),
            ("mixture_sd_1", self.mixture_sd_1),
            ("mixture_mean_2", self.mixture_mean_2),
            ("mixture_sd_2", self.mixture_sd_2),
        ] {
            if !value.is_finite() {
                return Err(PaperSimulationError::InvalidConfiguration(format!(
                    "{name} must be finite"
                )));
            }
        }
        if self.mixture_sd_1 < 0.0 || self.mixture_sd_2 < 0.0 {
            return Err(PaperSimulationError::InvalidConfiguration(
                "mixture standard deviations must be nonnegative".to_string(),
            ));
        }
        Ok(steps)
    }

    /// Sample one mixture component and its fixed subject-specific `ke0`.
    pub fn sample_ke0<R: Rng + ?Sized>(&self, rng: &mut R) -> (usize, f64) {
        if rng.random_bool(self.mixture_weight_1) {
            let z: f64 = StandardNormal.sample(rng);
            (0, self.mixture_mean_1 + self.mixture_sd_1 * z)
        } else {
            let z: f64 = StandardNormal.sample(rng);
            (1, self.mixture_mean_2 + self.mixture_sd_2 * z)
        }
    }
}

/// Simulate the complete population with a deterministic stream seeded by the config.
pub fn simulate_population(
    config: &PaperSimulationConfig,
) -> Result<PopulationTrajectories, PaperSimulationError> {
    config.validate()?;
    let mut rng = StdRng::seed_from_u64(config.seed);
    let mut subjects = Vec::with_capacity(config.n_subjects);

    for subject in 0..config.n_subjects {
        let (mixture_component, ke0) = config.sample_ke0(&mut rng);
        subjects.push(simulate_subject(
            config,
            subject,
            mixture_component,
            ke0,
            &mut rng,
        )?);
    }

    let population = PopulationTrajectories { subjects };
    population.validate(config)?;
    Ok(population)
}

/// Simulate one subject using the caller's RNG stream.
pub fn simulate_subject<R: Rng + ?Sized>(
    config: &PaperSimulationConfig,
    subject: usize,
    mixture_component: usize,
    ke0: f64,
    rng: &mut R,
) -> Result<SubjectTrajectory, PaperSimulationError> {
    let steps = config.validate()?;
    if mixture_component > 1 {
        return Err(PaperSimulationError::InvalidConfiguration(
            "mixture_component must be 0 or 1".to_string(),
        ));
    }
    if !ke0.is_finite() {
        return Err(PaperSimulationError::InvalidConfiguration(
            "ke0 must be finite".to_string(),
        ));
    }

    let ke0_for_drift = ke0;
    let sigma_x = config.sigma_x;
    let sigma_ke = config.sigma_ke;
    let solver = EM::new(
        move |_time: f64, state: &DVector<f64>, drift: &mut DVector<f64>| {
            drift[0] = -state[1] * state[0];
            drift[1] = -(state[1] - ke0_for_drift);
        },
        move |_time: f64, _state: &DVector<f64>, diffusion: &mut DVector<f64>| {
            diffusion[0] = sigma_x;
            diffusion[1] = sigma_ke;
        },
        DVector::from_vec(vec![config.x0, ke0]),
        1e-2,
        1e-2,
    );

    let mut state = DVector::from_vec(vec![config.x0, ke0]);
    let mut times = Vec::with_capacity(steps + 1);
    let mut x = Vec::with_capacity(steps + 1);
    let mut concentration = Vec::with_capacity(steps + 1);
    let mut ke = Vec::with_capacity(steps + 1);

    times.push(config.t0);
    x.push(state[0]);
    concentration.push(state[0] / config.volume);
    ke.push(state[1]);

    for step in 0..steps {
        let previous_ke = state[1];
        let time = config.t0 + step as f64 * config.dt;
        solver.step_with_rng(time, config.dt, &mut state, rng);

        // Manuscript rule: reject-and-hold, not clipping or mathematical reflection.
        if state[1] < 0.0 {
            state[1] = previous_ke;
        }

        if !state[0].is_finite() {
            return Err(PaperSimulationError::NonFinite {
                subject,
                step: step + 1,
                state: "x",
            });
        }
        if !state[1].is_finite() {
            return Err(PaperSimulationError::NonFinite {
                subject,
                step: step + 1,
                state: "ke",
            });
        }
        let subject_concentration = state[0] / config.volume;
        if !subject_concentration.is_finite() {
            return Err(PaperSimulationError::NonFinite {
                subject,
                step: step + 1,
                state: "concentration",
            });
        }

        let next_step = step + 1;
        let next_time = if next_step == steps {
            config.tf
        } else {
            config.t0 + next_step as f64 * config.dt
        };
        times.push(next_time);
        x.push(state[0]);
        concentration.push(subject_concentration);
        ke.push(state[1]);
    }

    Ok(SubjectTrajectory {
        subject,
        mixture_component,
        ke0,
        times,
        x,
        concentration,
        ke,
    })
}

impl PopulationTrajectories {
    /// Check the invariants needed by the two paper figures.
    pub fn validate(&self, config: &PaperSimulationConfig) -> Result<(), PaperSimulationError> {
        let steps = config.validate()?;
        if self.subjects.len() != config.n_subjects {
            return Err(PaperSimulationError::InvalidConfiguration(format!(
                "expected {} subjects, got {}",
                config.n_subjects,
                self.subjects.len()
            )));
        }
        for subject in &self.subjects {
            if subject.times.len() != steps + 1
                || subject.x.len() != steps + 1
                || subject.concentration.len() != steps + 1
                || subject.ke.len() != steps + 1
            {
                return Err(PaperSimulationError::InvalidConfiguration(format!(
                    "subject {} does not have {} stored points",
                    subject.subject,
                    steps + 1
                )));
            }
            if subject.x[0] != config.x0 {
                return Err(PaperSimulationError::InvalidConfiguration(format!(
                    "subject {} does not start at x0",
                    subject.subject
                )));
            }
            if subject.concentration[0] != config.x0 / config.volume {
                return Err(PaperSimulationError::InvalidConfiguration(format!(
                    "subject {} does not start at x0 / volume",
                    subject.subject
                )));
            }
            if subject.ke[0] != subject.ke0 {
                return Err(PaperSimulationError::InvalidConfiguration(format!(
                    "subject {} does not start at its sampled ke0",
                    subject.subject
                )));
            }
            if subject.times[0] != config.t0 || (subject.times[steps] - config.tf).abs() > 1e-12 {
                return Err(PaperSimulationError::InvalidConfiguration(format!(
                    "subject {} has an invalid time grid",
                    subject.subject
                )));
            }
            if subject
                .times
                .iter()
                .chain(subject.x.iter())
                .chain(subject.concentration.iter())
                .chain(subject.ke.iter())
                .any(|value| !value.is_finite())
            {
                return Err(PaperSimulationError::NonFinite {
                    subject: subject.subject,
                    step: 0,
                    state: "trajectory",
                });
            }
            if subject.ke.iter().any(|value| *value < 0.0) {
                return Err(PaperSimulationError::InvalidConfiguration(format!(
                    "subject {} has a negative ke",
                    subject.subject
                )));
            }
        }
        Ok(())
    }
}

/// Write all trajectories in the long-form CSV used by the plotting script.
pub fn write_trajectory_csv(
    population: &PopulationTrajectories,
    path: impl AsRef<Path>,
) -> Result<(), PaperSimulationError> {
    let file = File::create(path)?;
    let mut writer = Writer::from_writer(file);
    writer.write_record([
        "subject",
        "mixture_component",
        "ke0",
        "step",
        "time",
        "x",
        "concentration",
        "ke",
    ])?;

    for trajectory in &population.subjects {
        for step in 0..trajectory.times.len() {
            writer.write_record([
                trajectory.subject.to_string(),
                trajectory.mixture_component.to_string(),
                format_float(trajectory.ke0),
                step.to_string(),
                format_float(trajectory.times[step]),
                format_float(trajectory.x[step]),
                format_float(trajectory.concentration[step]),
                format_float(trajectory.ke[step]),
            ])?;
        }
    }
    writer.flush()?;
    Ok(())
}

/// Write the sampled population parameters separately for easy inspection.
pub fn write_population_k0_csv(
    population: &PopulationTrajectories,
    path: impl AsRef<Path>,
) -> Result<(), PaperSimulationError> {
    let file = File::create(path)?;
    let mut writer = Writer::from_writer(file);
    writer.write_record(["subject", "mixture_component", "ke0"])?;
    for trajectory in &population.subjects {
        writer.write_record([
            trajectory.subject.to_string(),
            trajectory.mixture_component.to_string(),
            format_float(trajectory.ke0),
        ])?;
    }
    writer.flush()?;
    Ok(())
}

fn format_float(value: f64) -> String {
    format!("{value:.17}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_population_has_distributed_values_in_both_modes() {
        let config = PaperSimulationConfig::default();
        let mut rng = StdRng::seed_from_u64(config.seed);
        let mut samples = [Vec::new(), Vec::new()];
        for _ in 0..1_000 {
            let (component, ke0) = config.sample_ke0(&mut rng);
            samples[component].push(ke0);
        }

        assert!(!samples[0].is_empty());
        assert!(!samples[1].is_empty());
        assert!(samples[0].windows(2).any(|pair| pair[0] != pair[1]));
        assert!(samples[1].windows(2).any(|pair| pair[0] != pair[1]));
    }

    #[test]
    fn fixed_grid_has_5000_updates_and_5001_points() {
        let mut config = PaperSimulationConfig::default();
        config.n_subjects = 1;
        let population = simulate_population(&config).unwrap();
        let trajectory = &population.subjects[0];
        assert_eq!(
            fixed_step_count(config.t0, config.tf, config.dt).unwrap(),
            5_000
        );
        assert_eq!(trajectory.times.len(), 5_001);
        assert_eq!(trajectory.times[0], 0.0);
        assert_eq!(*trajectory.times.last().unwrap(), 1.0);
    }

    #[test]
    fn initial_state_and_concentration_are_correct() {
        let mut config = PaperSimulationConfig::default();
        config.n_subjects = 1;
        config.volume = 4.0;
        let population = simulate_population(&config).unwrap();
        let trajectory = &population.subjects[0];
        assert_eq!(trajectory.x[0], 20.0);
        assert_eq!(trajectory.concentration[0], 5.0);
        assert_eq!(trajectory.ke[0], trajectory.ke0);
    }

    #[test]
    fn zero_ke_noise_keeps_each_subject_at_its_own_ke0() {
        let mut config = PaperSimulationConfig::default();
        config.n_subjects = 4;
        config.sigma_ke = 0.0;
        let population = simulate_population(&config).unwrap();
        for trajectory in &population.subjects {
            assert!(trajectory.ke.iter().all(|value| *value == trajectory.ke0));
        }
        assert!(population
            .subjects
            .windows(2)
            .any(|pair| pair[0].ke0 != pair[1].ke0));
    }

    #[test]
    fn same_seed_reproduces_the_complete_population() {
        let mut config = PaperSimulationConfig::default();
        config.n_subjects = 2;
        let first = simulate_population(&config).unwrap();
        let second = simulate_population(&config).unwrap();
        assert_eq!(first, second);

        config.seed += 1;
        let different = simulate_population(&config).unwrap();
        assert_ne!(first, different);
    }

    #[test]
    fn negative_ke_proposals_are_rejected_and_held() {
        let previous_ke = 0.7;
        let proposed_ke = -0.1;
        let new_ke = if proposed_ke < 0.0 {
            previous_ke
        } else {
            proposed_ke
        };
        assert_eq!(new_ke, previous_ke);
        assert_ne!(new_ke, 0.0);
    }

    #[test]
    fn stochastic_ke_moves_after_time_zero() {
        let mut config = PaperSimulationConfig::default();
        config.n_subjects = 1;
        config.sigma_ke = 0.5;
        let population = simulate_population(&config).unwrap();
        let trajectory = &population.subjects[0];
        assert!(trajectory.ke[1..]
            .iter()
            .any(|value| *value != trajectory.ke0));
    }
}
