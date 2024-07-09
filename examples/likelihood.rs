use ndarray::Array1;
const FRAC_1_SQRT_2PI: f64 =
    std::f64::consts::FRAC_2_SQRT_PI * std::f64::consts::FRAC_1_SQRT_2 / 2.0;
fn main() {
    use ndarray::array;

    let predictions = array![1.0, 2.0, 3.0];
    let observations = array![1.1, 1.9, 3.2];
    let sigma = array![0.1, 0.2, 0.3];

    let result1 = normal_likelihood1(&predictions, &observations, &sigma);
    let result2 = normal_likelihood2(&predictions, &observations, &sigma);

    println!("Result 1: {}", result1);
    println!("Result 2: {}", result2);
}

fn normal_likelihood1(
    predictions: &Array1<f64>,
    observations: &Array1<f64>,
    sigma: &Array1<f64>,
) -> f64 {
    const FRAC_1_SQRT_2PI: f64 =
        std::f64::consts::FRAC_2_SQRT_PI * std::f64::consts::FRAC_1_SQRT_2 / 2.0;
    let diff = (observations - predictions).mapv(|x| x.powi(2));
    let two_sigma_sq = sigma.mapv(|x| 2.0 * x * x);
    let exponent = (-&diff / two_sigma_sq).mapv(|x| x.exp());
    let aux_vec = FRAC_1_SQRT_2PI * exponent / sigma;
    aux_vec.product()
}

fn normal_likelihood2(
    predictions: &Array1<f64>,
    observations: &Array1<f64>,
    sigma: &Array1<f64>,
) -> f64 {
    let aux_vec = predictions
        .iter()
        .zip(observations.iter())
        .zip(sigma.iter())
        .map(|((pred, obs), sig)| normpdf(*obs, *pred, *sig))
        .collect::<Array1<f64>>();
    aux_vec.product()
}

/// Probability density function
fn normpdf(obs: f64, pred: f64, sigma: f64) -> f64 {
    (FRAC_1_SQRT_2PI / sigma) * (-((obs - pred) * (obs - pred)) / (2.0 * sigma * sigma)).exp()
}
