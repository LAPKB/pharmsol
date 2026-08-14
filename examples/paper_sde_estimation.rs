//! Generate the population data and fixed-grid NPMLE distributions for Figures 1, 2, and 5.
//!
//! This example deliberately reuses the `population_ke0.csv` produced for
//! Figures 3 and 4. It keeps the expensive estimation workflow separate from
//! the trajectory example:
//!
//! ```text
//! cargo run --release --example paper_sde_estimation
//! /tmp/pharmsol-paper-venv/bin/python paper/plot_population_distributions.py
//!
//! The default is the practical reduced fixed-grid run (201 support points,
//! 256 particles). Add `--manuscript` to request the manuscript's 1000 × 1000
//! Experiment 2 calculation; it is substantially more expensive with the
//! fixed-step paper particle filter.
//! ```
//!
//! PMcore was inspected for this branch. Its adaptive NPAG runner expects to
//! own the model/data likelihood evaluation and is not a dependency of this
//! paper-only package. This example therefore uses the permitted fixed-grid
//! NPMLE fallback: the likelihood matrix is computed on a fixed one-dimensional
//! K0 grid and the mixture weights are optimized by the stable mixture EM
//! iteration. Experiment 1 uses an exact scalar Gaussian likelihood. Experiment
//! 2 uses a seeded fixed-step particle filter with the manuscript's reject-and-
//! hold Ke boundary.

use std::{collections::BTreeMap, env, path::PathBuf};

use csv::{Reader, Writer};
use rand::{rngs::StdRng, RngExt, SeedableRng};
use rand_distr::{Distribution, StandardNormal};
use rayon::prelude::*;

const OBSERVATION_TIMES: [f64; 5] = [0.2, 0.4, 0.6, 0.8, 1.0];
const X0: f64 = 20.0;
const VOLUME: f64 = 1.0;
const SIGMA_X: f64 = 0.05;
const SIGMA_KE_EXPERIMENT_1: f64 = 0.0;
const SIGMA_KE_EXPERIMENT_2: f64 = 0.5;
const PF_DT: f64 = 0.002;
const K0_LOWER: f64 = 0.3;
const K0_UPPER: f64 = 3.0;
const N_SUPPORT_POINTS: usize = 1000;
const N_PARTICLES: usize = 1000;
const SIGMA_Y: f64 = 0.5;
const EXPERIMENT_1_SEED: u64 = 20_260_814;
const EXPERIMENT_2_SEED: u64 = 20_260_815;
const FIGURE_5_SAMPLE_SEED: u64 = 20_260_816;

#[derive(Clone, Copy, Debug)]
struct RunConfig {
    n_subjects: usize,
    n_support_points: usize,
    n_particles: usize,
    output_dir: &'static str,
}

impl RunConfig {
    fn full() -> Self {
        Self {
            n_subjects: 100,
            n_support_points: N_SUPPORT_POINTS,
            n_particles: N_PARTICLES,
            output_dir: "paper/output",
        }
    }

    fn smoke() -> Self {
        Self {
            n_subjects: 10,
            n_support_points: 50,
            n_particles: 50,
            output_dir: "paper/output_estimation_smoke",
        }
    }

    fn reduced() -> Self {
        Self {
            n_subjects: 100,
            n_support_points: 201,
            n_particles: 256,
            output_dir: "paper/output",
        }
    }
}

#[derive(Clone, Debug)]
struct SubjectObservations {
    subject: usize,
    true_ke0: f64,
    observations: Vec<Observation>,
}

#[derive(Clone, Copy, Debug)]
struct Observation {
    time: f64,
    value: f64,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = env::args().skip(1).collect::<Vec<_>>();
    let smoke = args.iter().any(|arg| arg == "--smoke");
    let manuscript = args.iter().any(|arg| arg == "--manuscript");
    let reduced = args.iter().any(|arg| arg == "--reduced");
    let config = if smoke {
        RunConfig::smoke()
    } else if manuscript {
        RunConfig::full()
    } else if reduced {
        RunConfig::reduced()
    } else {
        RunConfig::reduced()
    };
    let output_dir = PathBuf::from(config.output_dir);
    std::fs::create_dir_all(&output_dir)?;

    println!("Paper population estimation");
    println!(
        "  mode: {}",
        if smoke {
            "smoke"
        } else if manuscript {
            "manuscript"
        } else {
            "reduced"
        }
    );
    println!("  subjects: {}", config.n_subjects);
    println!("  support points: {}", config.n_support_points);
    println!("  particle count (Experiment 2): {}", config.n_particles);
    println!("  PF EM delta: {}", PF_DT);
    println!("  K0 search region: [{}, {}]", K0_LOWER, K0_UPPER);
    println!("  sigma_X: {}", SIGMA_X);
    println!("  sigma_Ke Experiment 1: {}", SIGMA_KE_EXPERIMENT_1);
    println!("  sigma_Ke Experiment 2: {}", SIGMA_KE_EXPERIMENT_2);
    println!("  sigma_Y: {}", SIGMA_Y);
    println!("  Experiment 1 seed: {}", EXPERIMENT_1_SEED);
    println!("  Experiment 2 seed: {}", EXPERIMENT_2_SEED);
    println!("  Figure 5 sample seed: {}", FIGURE_5_SAMPLE_SEED);
    println!(
        "  sigma_Y evidence: fixed additive 0.5 used by pharmsol's existing SDE PF test; the old PMcore SDE paper fit used a separate 1.0 + 0.15*y assay model"
    );

    let population_path = PathBuf::from("paper/output/population_ke0.csv");
    let mut population = read_population(&population_path)?;
    if population.len() < config.n_subjects {
        return Err(format!(
            "{} contains {} subjects, but {} are required",
            population_path.display(),
            population.len(),
            config.n_subjects
        )
        .into());
    }
    population.truncate(config.n_subjects);
    validate_population(&population, config.n_subjects)?;

    write_initial_distribution(
        &population,
        &output_dir.join("figure1_initial_k0_distribution.csv"),
    )?;

    let experiment_1 = generate_experiment_1(&population);
    let experiment_2 = generate_experiment_2(&population);
    validate_observations(&experiment_1, config.n_subjects)?;
    validate_observations(&experiment_2, config.n_subjects)?;
    write_observations(
        &experiment_1,
        &output_dir.join("experiment1_observations.csv"),
    )?;
    write_observations(
        &experiment_2,
        &output_dir.join("experiment2_observations.csv"),
    )?;

    let support_points = make_support_grid(config.n_support_points);
    println!("Computing exact Experiment 1 likelihood matrix...");
    let experiment_1_log_likelihood = experiment_1
        .par_iter()
        .map(|subject| {
            support_points
                .iter()
                .map(|&ke0| exact_experiment_1_log_likelihood(subject, ke0))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    validate_log_likelihood_matrix(
        &experiment_1_log_likelihood,
        population.len(),
        support_points.len(),
    )?;

    println!("Computing fixed-step Experiment 2 particle likelihood matrix...");
    let experiment_2_log_likelihood = experiment_2
        .par_iter()
        .map(|subject| {
            support_points
                .iter()
                .enumerate()
                .map(|(support_index, &ke0)| {
                    particle_log_likelihood(
                        subject,
                        ke0,
                        config.n_particles,
                        EXPERIMENT_2_SEED,
                        support_index,
                    )
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    validate_log_likelihood_matrix(
        &experiment_2_log_likelihood,
        population.len(),
        support_points.len(),
    )?;

    let (weights_1, objective_1, iterations_1) = fit_fixed_grid(&experiment_1_log_likelihood);
    let (weights_2, objective_2, iterations_2) = fit_fixed_grid(&experiment_2_log_likelihood);
    validate_weights(&weights_1)?;
    validate_weights(&weights_2)?;

    write_likelihood_matrix(
        &experiment_1_log_likelihood,
        &output_dir.join("experiment1_log_likelihood.csv"),
    )?;
    write_likelihood_matrix(
        &experiment_2_log_likelihood,
        &output_dir.join("experiment2_log_likelihood.csv"),
    )?;
    write_distribution(
        &support_points,
        &weights_1,
        &output_dir.join("figure2_fml_sigma_ke_0.csv"),
    )?;
    write_distribution(
        &support_points,
        &weights_2,
        &output_dir.join("figure5_fml_sigma_ke_0_5.csv"),
    )?;

    let sample_1 = sample_distribution(&support_points, &weights_1, FIGURE_5_SAMPLE_SEED);
    let sample_2 = sample_distribution(
        &support_points,
        &weights_2,
        FIGURE_5_SAMPLE_SEED.wrapping_add(1),
    );
    write_sample(&sample_1, &output_dir.join("figure2_fml_sample_n100.csv"))?;
    write_sample(&sample_2, &output_dir.join("figure5_fml_sample_n100.csv"))?;

    println!("Experiment 1 fixed-grid NPMLE:");
    println!("  iterations: {}", iterations_1);
    println!("  log likelihood: {:.6}", objective_1);
    println!(
        "  retained non-negligible mass points: {}",
        count_nonzero(&weights_1)
    );
    print_mode_masses(&support_points, &weights_1);
    println!("Experiment 2 fixed-grid NPMLE approximation:");
    println!("  iterations: {}", iterations_2);
    println!("  log likelihood: {:.6}", objective_2);
    println!(
        "  retained non-negligible mass points: {}",
        count_nonzero(&weights_2)
    );
    print_mode_masses(&support_points, &weights_2);
    println!("Outputs written to {}", output_dir.display());

    Ok(())
}

fn read_population(path: &PathBuf) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
    let mut reader = Reader::from_path(path)?;
    let headers = reader.headers()?.clone();
    let ke0_index = headers
        .iter()
        .position(|header| header == "ke0")
        .ok_or("population CSV has no ke0 column")?;
    let mut values = Vec::new();
    for row in reader.records() {
        let row = row?;
        let ke0: f64 = row
            .get(ke0_index)
            .ok_or("population CSV row has no ke0 value")?
            .parse()?;
        values.push(ke0);
    }
    Ok(values)
}

fn validate_population(
    population: &[f64],
    expected_subjects: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    if population.len() != expected_subjects {
        return Err(format!(
            "expected {expected_subjects} population values, got {}",
            population.len()
        )
        .into());
    }
    if population.iter().any(|value| !value.is_finite()) {
        return Err("population contains a non-finite ke0".into());
    }
    if population
        .iter()
        .all(|value| *value == 0.5 || *value == 1.5)
    {
        return Err("population collapsed to the two mixture means".into());
    }
    if !population.iter().any(|value| *value < 0.9) || !population.iter().any(|value| *value > 1.1)
    {
        return Err("population does not contain both broad mixture modes".into());
    }
    Ok(())
}

fn generate_experiment_1(population: &[f64]) -> Vec<SubjectObservations> {
    population
        .iter()
        .enumerate()
        .map(|(subject, &ke0)| {
            let mut rng = StdRng::seed_from_u64(subject_seed(EXPERIMENT_1_SEED, subject));
            let mut x = X0;
            let mut previous_time = 0.0;
            let observations = OBSERVATION_TIMES
                .iter()
                .map(|&time| {
                    let h = time - previous_time;
                    let (mean_factor, process_variance) = ou_transition(ke0, h);
                    let process_sd = (SIGMA_X.powi(2) * process_variance).sqrt();
                    let z_process: f64 = StandardNormal.sample(&mut rng);
                    x = mean_factor * x + process_sd * z_process;
                    let z_observation: f64 = StandardNormal.sample(&mut rng);
                    let value = x / VOLUME + SIGMA_Y * z_observation;
                    previous_time = time;
                    Observation { time, value }
                })
                .collect();
            SubjectObservations {
                subject,
                true_ke0: ke0,
                observations,
            }
        })
        .collect()
}

fn generate_experiment_2(population: &[f64]) -> Vec<SubjectObservations> {
    let steps_per_observation = (0.2 / PF_DT).round() as usize;
    population
        .iter()
        .enumerate()
        .map(|(subject, &ke0)| {
            let mut rng = StdRng::seed_from_u64(subject_seed(EXPERIMENT_2_SEED, subject));
            let mut x = X0;
            let mut ke = ke0;
            let mut observations = Vec::with_capacity(OBSERVATION_TIMES.len());
            for step in 0..(steps_per_observation * OBSERVATION_TIMES.len()) {
                let previous_x = x;
                let previous_ke = ke;
                let z_x: f64 = StandardNormal.sample(&mut rng);
                let z_ke: f64 = StandardNormal.sample(&mut rng);
                let sqrt_dt = PF_DT.sqrt();
                x = previous_x + (-previous_ke * previous_x) * PF_DT + SIGMA_X * sqrt_dt * z_x;
                let proposed_ke = previous_ke - (previous_ke - ke0) * PF_DT
                    + SIGMA_KE_EXPERIMENT_2 * sqrt_dt * z_ke;
                ke = if proposed_ke < 0.0 {
                    previous_ke
                } else {
                    proposed_ke
                };

                if (step + 1) % steps_per_observation == 0 {
                    let z_observation: f64 = StandardNormal.sample(&mut rng);
                    observations.push(Observation {
                        time: (step + 1) as f64 * PF_DT,
                        value: x / VOLUME + SIGMA_Y * z_observation,
                    });
                }
            }
            SubjectObservations {
                subject,
                true_ke0: ke0,
                observations,
            }
        })
        .collect()
}

fn ou_transition(ke0: f64, h: f64) -> (f64, f64) {
    let factor = (-ke0 * h).exp();
    let variance_factor = if ke0.abs() < 1e-12 {
        h
    } else {
        (1.0 - (-2.0 * ke0 * h).exp()) / (2.0 * ke0)
    };
    (factor, variance_factor)
}

fn exact_experiment_1_log_likelihood(subject: &SubjectObservations, ke0: f64) -> f64 {
    let mut mean = X0;
    let mut variance = 0.0;
    let mut previous_time = 0.0;
    let mut log_likelihood = 0.0;
    for observation in &subject.observations {
        let h = observation.time - previous_time;
        let (factor, variance_factor) = ou_transition(ke0, h);
        mean *= factor;
        variance = factor.powi(2) * variance + SIGMA_X.powi(2) * variance_factor;
        let total_variance = variance + SIGMA_Y.powi(2);
        let residual = observation.value - mean / VOLUME;
        log_likelihood += -0.5
            * ((2.0 * std::f64::consts::PI * total_variance).ln()
                + residual.powi(2) / total_variance);

        let observation_gain = variance / total_variance;
        mean += observation_gain * residual * VOLUME;
        variance *= 1.0 - observation_gain;
        previous_time = observation.time;
    }
    log_likelihood
}

fn particle_log_likelihood(
    subject: &SubjectObservations,
    ke0: f64,
    n_particles: usize,
    seed: u64,
    support_index: usize,
) -> f64 {
    let mut rng = StdRng::seed_from_u64(particle_seed(seed, subject.subject, support_index));
    let mut x = vec![X0; n_particles];
    let mut ke = vec![ke0; n_particles];
    let steps_per_observation = (0.2 / PF_DT).round() as usize;
    let sqrt_dt = PF_DT.sqrt();
    let mut log_likelihood = 0.0;

    for (observation_index, observation) in subject.observations.iter().enumerate() {
        for _ in 0..steps_per_observation {
            for particle in 0..n_particles {
                let previous_x = x[particle];
                let previous_ke = ke[particle];
                let z_x: f64 = StandardNormal.sample(&mut rng);
                let z_ke: f64 = StandardNormal.sample(&mut rng);
                x[particle] =
                    previous_x + (-previous_ke * previous_x) * PF_DT + SIGMA_X * sqrt_dt * z_x;
                let proposed_ke = previous_ke - (previous_ke - ke0) * PF_DT
                    + SIGMA_KE_EXPERIMENT_2 * sqrt_dt * z_ke;
                ke[particle] = if proposed_ke < 0.0 {
                    previous_ke
                } else {
                    proposed_ke
                };
            }
        }

        let mut log_weights = Vec::with_capacity(n_particles);
        for &particle_x in &x {
            let residual = observation.value - particle_x / VOLUME;
            log_weights.push(
                -0.5 * ((2.0 * std::f64::consts::PI * SIGMA_Y.powi(2)).ln()
                    + residual.powi(2) / SIGMA_Y.powi(2)),
            );
        }
        let log_mean = logsumexp(&log_weights) - (n_particles as f64).ln();
        log_likelihood += log_mean;
        systematic_resample(&mut x, &mut ke, &log_weights, &mut rng);

        if observation_index + 1 == subject.observations.len() {
            break;
        }
    }
    log_likelihood
}

fn systematic_resample(x: &mut [f64], ke: &mut [f64], log_weights: &[f64], rng: &mut StdRng) {
    let n = x.len();
    let max_log_weight = log_weights
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let weights = log_weights
        .iter()
        .map(|weight| (*weight - max_log_weight).exp())
        .collect::<Vec<_>>();
    let total = weights.iter().sum::<f64>();
    let mut cumulative = Vec::with_capacity(n);
    let mut running = 0.0;
    for weight in weights {
        running += weight / total;
        cumulative.push(running);
    }

    let u0 = rng.random::<f64>() / n as f64;
    let old_x = x.to_vec();
    let old_ke = ke.to_vec();
    let mut index = 0;
    for draw in 0..n {
        let target = u0 + draw as f64 / n as f64;
        while index + 1 < n && target > cumulative[index] {
            index += 1;
        }
        x[draw] = old_x[index];
        ke[draw] = old_ke[index];
    }
}

fn make_support_grid(n_points: usize) -> Vec<f64> {
    assert!(n_points >= 2, "support grid needs at least two points");
    (0..n_points)
        .map(|index| K0_LOWER + (K0_UPPER - K0_LOWER) * index as f64 / (n_points - 1) as f64)
        .collect()
}

fn fit_fixed_grid(log_likelihoods: &[Vec<f64>]) -> (Vec<f64>, f64, usize) {
    let n_subjects = log_likelihoods.len();
    let n_support = log_likelihoods[0].len();
    let mut weights = vec![1.0 / n_support as f64; n_support];
    let mut iterations = 0;

    for iteration in 0..50_000 {
        iterations = iteration + 1;
        let mut new_weights = vec![0.0; n_support];
        for row in log_likelihoods {
            let terms = row
                .iter()
                .zip(weights.iter())
                .map(|(&log_likelihood, &weight)| {
                    if weight > 0.0 {
                        log_likelihood + weight.ln()
                    } else {
                        f64::NEG_INFINITY
                    }
                })
                .collect::<Vec<_>>();
            let denominator = logsumexp(&terms);
            for (new_weight, term) in new_weights.iter_mut().zip(terms.iter()) {
                *new_weight += (term - denominator).exp();
            }
        }
        for weight in &mut new_weights {
            *weight /= n_subjects as f64;
        }
        let max_change = weights
            .iter()
            .zip(new_weights.iter())
            .map(|(old, new)| (old - new).abs())
            .fold(0.0, f64::max);
        weights = new_weights;
        if max_change < 1e-9 {
            break;
        }
    }

    let objective = log_likelihoods
        .iter()
        .map(|row| {
            row.iter()
                .zip(weights.iter())
                .map(|(&log_likelihood, &weight)| {
                    if weight > 0.0 {
                        log_likelihood + weight.ln()
                    } else {
                        f64::NEG_INFINITY
                    }
                })
                .collect::<Vec<_>>()
        })
        .map(|terms| logsumexp(&terms))
        .sum();
    (weights, objective, iterations)
}

fn logsumexp(values: &[f64]) -> f64 {
    let maximum = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    maximum
        + values
            .iter()
            .map(|value| (value - maximum).exp())
            .sum::<f64>()
            .ln()
}

fn validate_observations(
    observations: &[SubjectObservations],
    expected_subjects: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    if observations.len() != expected_subjects {
        return Err(format!(
            "expected {expected_subjects} subjects, got {}",
            observations.len()
        )
        .into());
    }
    for subject in observations {
        if subject.observations.len() != OBSERVATION_TIMES.len() {
            return Err(format!(
                "subject {} has the wrong observation count",
                subject.subject
            )
            .into());
        }
        for (observation, expected_time) in subject.observations.iter().zip(OBSERVATION_TIMES) {
            if (observation.time - expected_time).abs() > 1e-12 || !observation.value.is_finite() {
                return Err(format!("invalid observation for subject {}", subject.subject).into());
            }
        }
    }
    Ok(())
}

fn validate_log_likelihood_matrix(
    matrix: &[Vec<f64>],
    n_subjects: usize,
    n_support: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    if matrix.len() != n_subjects || matrix.iter().any(|row| row.len() != n_support) {
        return Err("likelihood matrix has unexpected dimensions".into());
    }
    if matrix.iter().flatten().any(|value| !value.is_finite()) {
        return Err("likelihood matrix contains a non-finite value".into());
    }
    Ok(())
}

fn validate_weights(weights: &[f64]) -> Result<(), Box<dyn std::error::Error>> {
    if weights
        .iter()
        .any(|weight| !weight.is_finite() || *weight < 0.0)
    {
        return Err("estimated weights are not finite and nonnegative".into());
    }
    let sum = weights.iter().sum::<f64>();
    if (sum - 1.0).abs() > 1e-8 {
        return Err(format!("estimated weights sum to {sum}, not 1").into());
    }
    Ok(())
}

fn count_nonzero(weights: &[f64]) -> usize {
    weights.iter().filter(|weight| **weight > 1e-6).count()
}

fn print_mode_masses(support: &[f64], weights: &[f64]) {
    let lower = support
        .iter()
        .zip(weights.iter())
        .filter(|(support, _)| **support < 0.9)
        .map(|(_, weight)| weight)
        .sum::<f64>();
    let upper = support
        .iter()
        .zip(weights.iter())
        .filter(|(support, _)| **support > 1.1)
        .map(|(_, weight)| weight)
        .sum::<f64>();
    println!(
        "  mass below 0.9: {:.6}; mass above 1.1: {:.6}",
        lower, upper
    );
}

fn write_initial_distribution(
    population: &[f64],
    path: &PathBuf,
) -> Result<(), Box<dyn std::error::Error>> {
    let bin_width = 0.01;
    let mut bins = BTreeMap::<i64, usize>::new();
    for &ke0 in population {
        let bin = (ke0 / bin_width).floor() as i64;
        *bins.entry(bin).or_default() += 1;
    }
    let mut writer = Writer::from_path(path)?;
    writer.write_record(["k0_bin", "count", "relative_frequency"])?;
    for (bin, count) in bins {
        writer.write_record([
            format!("{:.17}", (bin as f64 + 0.5) * bin_width),
            count.to_string(),
            format!("{:.17}", count as f64 / population.len() as f64),
        ])?;
    }
    writer.flush()?;
    Ok(())
}

fn write_observations(
    subjects: &[SubjectObservations],
    path: &PathBuf,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut writer = Writer::from_path(path)?;
    writer.write_record(["subject", "time", "observation", "true_ke0"])?;
    for subject in subjects {
        for observation in &subject.observations {
            writer.write_record([
                subject.subject.to_string(),
                format!("{:.17}", observation.time),
                format!("{:.17}", observation.value),
                format!("{:.17}", subject.true_ke0),
            ])?;
        }
    }
    writer.flush()?;
    Ok(())
}

fn write_likelihood_matrix(
    matrix: &[Vec<f64>],
    path: &PathBuf,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut writer = Writer::from_path(path)?;
    for row in matrix {
        writer.write_record(row.iter().map(|value| format!("{value:.17}")))?;
    }
    writer.flush()?;
    Ok(())
}

fn write_distribution(
    support: &[f64],
    weights: &[f64],
    path: &PathBuf,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut writer = Writer::from_path(path)?;
    writer.write_record(["ke0", "weight"])?;
    for (&ke0, &weight) in support.iter().zip(weights.iter()) {
        if weight > 1e-10 {
            writer.write_record([format!("{ke0:.17}"), format!("{weight:.17}")])?;
        }
    }
    writer.flush()?;
    Ok(())
}

fn sample_distribution(support: &[f64], weights: &[f64], seed: u64) -> Vec<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut sample = Vec::with_capacity(100);
    for _ in 0..100 {
        let draw = rng.random::<f64>();
        let mut cumulative = 0.0;
        let mut selected = *support.last().unwrap();
        for (&ke0, &weight) in support.iter().zip(weights.iter()) {
            cumulative += weight;
            if draw <= cumulative {
                selected = ke0;
                break;
            }
        }
        sample.push(selected);
    }
    sample
}

fn write_sample(sample: &[f64], path: &PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    let mut writer = Writer::from_path(path)?;
    writer.write_record(["sample_index", "ke0"])?;
    for (index, &ke0) in sample.iter().enumerate() {
        writer.write_record([index.to_string(), format!("{ke0:.17}")])?;
    }
    writer.flush()?;
    Ok(())
}

fn subject_seed(base: u64, subject: usize) -> u64 {
    base.wrapping_add((subject as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15))
}

fn particle_seed(base: u64, subject: usize, support_index: usize) -> u64 {
    subject_seed(base, subject)
        .wrapping_add((support_index as u64 + 1).wrapping_mul(0xD1B5_4A32_D192_ED03))
}
