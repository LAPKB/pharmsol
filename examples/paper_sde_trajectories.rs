//! Generate the paper's 100-subject SDE trajectory data.
//!
//! Run with:
//!
//! ```text
//! cargo run --example paper_sde_trajectories
//! ```
//!
//! Rust performs the seeded simulation and writes CSV. Use
//! `paper/plot_sde_population.py` for the two figures.

use std::{env, fs, path::PathBuf};

use pharmsol::simulator::equation::sde::paper::{
    fixed_step_count, simulate_population, write_population_k0_csv, write_trajectory_csv,
    PaperSimulationConfig,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = PaperSimulationConfig::default();
    let output_dir = env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("paper/output"));
    fs::create_dir_all(&output_dir)?;

    let steps = fixed_step_count(config.t0, config.tf, config.dt)?;
    println!("Paper SDE population simulation");
    println!("  seed: {}", config.seed);
    println!("  subjects: {}", config.n_subjects);
    println!("  t0: {}", config.t0);
    println!("  tf: {}", config.tf);
    println!("  dt: {}", config.dt);
    println!("  steps: {}", steps);
    println!("  stored points per subject: {}", steps + 1);
    println!("  X0: {}", config.x0);
    println!("  V: {}", config.volume);
    println!("  sigma_X: {}", config.sigma_x);
    println!("  sigma_Ke: {}", config.sigma_ke);
    println!(
        "  mixture 1: weight={}, mean={}, sd={}",
        config.mixture_weight_1, config.mixture_mean_1, config.mixture_sd_1
    );
    println!(
        "  mixture 2: weight={}, mean={}, sd={}",
        config.mixture_weight_2, config.mixture_mean_2, config.mixture_sd_2
    );
    println!("  Ke boundary: reject negative proposal and hold previous value");
    println!("  output directory: {}", output_dir.display());
    println!(
        "  note: manuscript sigma_Ke is inconsistent (heading 0.1, experiment text 0.5); using {}",
        config.sigma_ke
    );

    let population = simulate_population(&config)?;
    let component_0 = population
        .subjects
        .iter()
        .filter(|subject| subject.mixture_component == 0)
        .count();
    let component_1 = population.subjects.len() - component_0;
    let min_ke0 = population
        .subjects
        .iter()
        .map(|subject| subject.ke0)
        .fold(f64::INFINITY, f64::min);
    let max_ke0 = population
        .subjects
        .iter()
        .map(|subject| subject.ke0)
        .fold(f64::NEG_INFINITY, f64::max);
    let mean_ke0 = population
        .subjects
        .iter()
        .map(|subject| subject.ke0)
        .sum::<f64>()
        / population.subjects.len() as f64;

    println!("  mixture component 0 subjects (mean 0.5): {}", component_0);
    println!("  mixture component 1 subjects (mean 1.5): {}", component_1);
    println!("  sampled Ke0 min: {:.6}", min_ke0);
    println!("  sampled Ke0 max: {:.6}", max_ke0);
    println!("  sampled Ke0 mean: {:.6}", mean_ke0);

    let trajectories_path = output_dir.join("sde_population_trajectories.csv");
    let population_path = output_dir.join("population_ke0.csv");
    write_trajectory_csv(&population, &trajectories_path)?;
    write_population_k0_csv(&population, &population_path)?;
    println!("  trajectory CSV: {}", trajectories_path.display());
    println!("  population CSV: {}", population_path.display());

    Ok(())
}
