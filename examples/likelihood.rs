const FRAC_1_SQRT_2PI: f64 =
    std::f64::consts::FRAC_2_SQRT_PI * std::f64::consts::FRAC_1_SQRT_2 / 2.0;

fn main() {
    let predictions = vec![1.0, 2.0, 3.0];
    let observations = vec![1.1, 1.9, 3.2];
    let sigma = vec![0.1, 0.2, 0.3];

    let result1 = normal_likelihood1(&predictions, &observations, &sigma);
    let result2 = normal_likelihood2(&predictions, &observations, &sigma);

    println!("Result 1: {}", result1);
    println!("Result 2: {}", result2);
}

fn normal_likelihood1(predictions: &[f64], observations: &[f64], sigma: &[f64]) -> f64 {
    const FRAC_1_SQRT_2PI: f64 =
        std::f64::consts::FRAC_2_SQRT_PI * std::f64::consts::FRAC_1_SQRT_2 / 2.0;

    predictions
        .iter()
        .zip(observations.iter())
        .zip(sigma.iter())
        .map(|((pred, obs), sig)| {
            let diff = (obs - pred).powi(2);
            let two_sigma_sq = 2.0 * sig * sig;
            let exponent = (-diff / two_sigma_sq).exp();
            FRAC_1_SQRT_2PI * exponent / sig
        })
        .product()
}

fn normal_likelihood2(predictions: &[f64], observations: &[f64], sigma: &[f64]) -> f64 {
    predictions
        .iter()
        .zip(observations.iter())
        .zip(sigma.iter())
        .map(|((pred, obs), sig)| normpdf(*obs, *pred, *sig))
        .product()
}

/// Probability density function
fn normpdf(obs: f64, pred: f64, sigma: f64) -> f64 {
    (FRAC_1_SQRT_2PI / sigma) * (-((obs - pred) * (obs - pred)) / (2.0 * sigma * sigma)).exp()
}
