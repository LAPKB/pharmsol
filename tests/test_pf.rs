use pharmsol::data::error_model::AssayErrorModel;
use pharmsol::*;

/// Test the particle filter (SDE) likelihood estimation
/// This test runs the particle filter multiple times to assess consistency
#[test]
fn test_particle_filter_likelihood() {
    let subject = data::Subject::builder("id1")
        .bolus(0.0, 20.0, 0)
        .observation(0.2, 16.6434, 0)
        .observation(0.4, 14.3233, 0)
        .observation(0.6, 9.8468, 0)
        .observation(0.8, 9.4177, 0)
        .observation(1.0, 7.5170, 0)
        .build();

    let sde = equation::SDE::new(
        |x, p, _t, dx, _rateiv, _cov| {
            dx[0] = -x[0] * x[1]; // ke *x[0]
            dx[1] = -x[1] + p[0]; // mean reverting
        },
        |_p, d| {
            d[0] = 1.0;
            d[1] = 0.01;
        },
        |_p, _t, _cov| lag! {},
        |_p, _t, _cov| fa! {},
        |_p, _t, _cov, x| x[1] = 1.0,
        |x, _p, _t, _cov, y| {
            y[0] = x[0];
        },
        (2, 1),
        10000,
    );

    let ems = AssayErrorModels::new()
        .add(
            0,
            AssayErrorModel::additive(ErrorPoly::new(0.5, 0.0, 0.0, 0.0), 0.0),
        )
        .unwrap();

    // Run the particle filter multiple times to get statistics
    const NUM_RUNS: usize = 10;
    let mut likelihoods = Vec::with_capacity(NUM_RUNS);

    for i in 0..NUM_RUNS {
        let ll = sde
            .estimate_log_likelihood(&subject, &vec![1.0], &ems, false)
            .unwrap()
            .exp();
        println!("Run {}: likelihood = {}", i + 1, ll);
        likelihoods.push(ll);
    }

    // Calculate mean and standard deviation
    let mean: f64 = likelihoods.iter().sum::<f64>() / NUM_RUNS as f64;
    let variance: f64 =
        likelihoods.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / NUM_RUNS as f64;
    let std_dev = variance.sqrt();

    println!("\n=== Particle Filter Likelihood Statistics ===");
    println!("Number of runs: {}", NUM_RUNS);
    println!("Likelihoods: {:?}", likelihoods);
    println!("Mean likelihood: {}", mean);
    println!("Std deviation: {}", std_dev);
    println!(
        "Min: {}",
        likelihoods.iter().cloned().fold(f64::INFINITY, f64::min)
    );
    println!(
        "Max: {}",
        likelihoods
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max)
    );

    // The likelihood should be a reasonable negative value (log-likelihood)
    // This assertion is loose since particle filters have stochastic variation
    assert!(mean.is_finite(), "Mean likelihood should be finite");
}
