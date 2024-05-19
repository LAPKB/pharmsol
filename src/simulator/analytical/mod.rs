use crate::{data::Covariates, simulator::*};
use nalgebra::{DVector, Matrix2, Vector2};

#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn simulate_analytical_event(
    eq: &AnalyticalEq,
    seq_eq: &SecEq,

    x: V,
    support_point: &[f64],
    cov: &Covariates,
    infusions: &Vec<Infusion>,
    ti: f64,
    tf: f64,
) -> V {
    if ti == tf {
        return x;
    }
    let mut support_point = V::from_vec(support_point.to_owned());
    let mut rateiv = V::from_vec(vec![0.0, 0.0, 0.0]);
    //TODO: This should be pre-calculated
    for infusion in infusions {
        if tf >= infusion.time() && tf <= infusion.duration() + infusion.time() {
            rateiv[infusion.input()] = infusion.amount() / infusion.duration();
        }
    }
    (seq_eq)(&mut support_point, cov);
    (eq)(&x, &support_point, tf - ti, rateiv, cov)
}

///
/// Analytical for one compartment
/// Assumptions:
///   - p is a vector of length 1 with the value of the elimination constant
///   - rateiv is a vector of length 1 with the value of the infusion rate (only one drug)
///   - x is a vector of length 1
///   - covariates are not used
///

pub fn one_compartment(x: &V, p: &V, t: T, rateiv: V, _cov: &Covariates) -> V {
    let mut xout = x.clone();
    let ke = p[0];

    xout[0] = x[0] * (-ke * t).exp() + rateiv[0] / ke * (1.0 - (-ke * t).exp());
    // dbg!(t, &rateiv, x, &xout);
    xout
}

///
/// Analytical for one compartment with absorption
/// Assumptions:
///   - p is a vector of length 2 with ke and ka in that order
///   - rateiv is a vector of length 1 with the value of the infusion rate (only one drug)
///   - x is a vector of length 2
///   - covariates are not used
///

pub fn one_compartment_with_absorption(x: &V, p: &V, t: T, rateiv: V, _cov: &Covariates) -> V {
    let mut xout = x.clone();
    let ka = p[0];
    let ke = p[1];

    xout[0] = x[0] * (-ka * t).exp();

    xout[1] = x[1] * (-ke * t).exp()
        + rateiv[0] / ke * (1.0 - (-ke * t).exp())
        + ((ka * x[0]) / (ka - ke)) * ((-ke * t).exp() - (-ka * t).exp());

    xout
}

///
/// Analytical for two compartment
/// Assumptions:
///   - p is a vector of length 3 with ke, kcp and kpc in that order
///   - rateiv is a vector of length 1 with the value of the infusion rate (only one drug)
///   - x is a vector of length 2
///   - covariates are not used
///
pub fn two_compartments(x: &V, p: &V, t: T, rateiv: V, _cov: &Covariates) -> V {
    let ke = p[0];
    let kcp = p[1];
    let kpc = p[2];

    let sqrt = (ke + kcp + kpc).powi(2) - 4.0 * ke * kpc;
    if sqrt < 0.0 {
        panic!("Imaginary solutions, program stopped!");
    }
    let sqrt = sqrt.sqrt();
    let l1 = (ke + kcp + kpc + sqrt) / 2.0;
    let l2 = (ke + kcp + kpc - sqrt) / 2.0;
    let exp_l1_t = (-l1 * t).exp();
    let exp_l2_t = (-l2 * t).exp();
    let non_zero_matrix = Matrix2::new(
        (l1 - kpc) * exp_l1_t + (kpc - l2) * exp_l2_t,
        -kpc * exp_l1_t + kpc * exp_l2_t,
        kcp * exp_l1_t + kcp * exp_l2_t,
        (l1 - ke - kcp) * exp_l1_t + (-ke + kcp - l2) * exp_l2_t,
    );

    let non_zero = (non_zero_matrix * x) / (l1 - l2);

    let infusion_vector = Vector2::new(
        ((l1 - kpc) / l1) * (1.0 - exp_l1_t) + ((kpc - l2) / l2) * (1.0 - exp_l2_t),
        (-kpc / l1) * (1.0 - exp_l1_t) + (kpc / l2) * (1.0 - exp_l2_t),
    );

    let infusion = infusion_vector * (rateiv[0] / (l1 - l2));

    let result_vector = non_zero + infusion;

    // Convert Vector2 to DVector
    DVector::from_vec(vec![result_vector[0], result_vector[1]])
}

///
/// Analytical for two compartment with absorption
/// Assumptions:
///   - p is a vector of length 4 with ke, ka, kcp and kpc in that order
///   - rateiv is a vector of length 1 with the value of the infusion rate (only one drug)
///   - x is a vector of length 2
///   - covariates are not used
///
pub fn two_compartments_with_absorption(x: &V, p: &V, t: T, rateiv: V, _cov: &Covariates) -> V {
    let ke = p[0];
    let ka = p[1];
    let kcp = p[2];
    let kpc = p[3];
    let mut xout = x.clone();

    let sqrt = (ke + kcp + kpc).powi(2) - 4.0 * ke * kpc;
    if sqrt < 0.0 {
        panic!("Imaginary solutions, program stopped!");
    }
    let sqrt = sqrt.sqrt();
    let l1 = (ke + kcp + kpc + sqrt) / 2.0;
    let l2 = (ke + kcp + kpc - sqrt) / 2.0;

    let exp_l1_t = (-l1 * t).exp();
    let exp_l2_t = (-l2 * t).exp();

    let non_zero_matrix = Matrix2::new(
        (l1 - kpc) * exp_l1_t + (kpc - l2) * exp_l2_t,
        -kpc * exp_l1_t + kpc * exp_l2_t,
        kcp * exp_l1_t + kcp * exp_l2_t,
        (l1 - ke - kcp) * exp_l1_t + (-ke + kcp - l2) * exp_l2_t,
    );

    let non_zero = (non_zero_matrix * Vector2::new(x[1], x[2])) / (l1 - l2);

    let infusion_vector = Vector2::new(
        ((l1 - kpc) / l1) * (1.0 - exp_l1_t) + ((kpc - l2) / l2) * (1.0 - exp_l2_t),
        (-kpc / l1) * (1.0 - exp_l1_t) + (kpc / l2) * (1.0 - exp_l2_t),
    );

    let infusion = infusion_vector * (rateiv[0] / (l1 - l2));

    let exp_ka_t = (-ka * t).exp();

    let absorption_vector = Vector2::new(
        ((l1 - kpc) / (ka - l1)) * (exp_l1_t - exp_ka_t)
            + ((kpc - l2) / (ka - l2)) * (exp_l2_t - exp_ka_t),
        (-kpc / (ka - l1)) * (exp_l1_t - exp_ka_t) + (kpc / (ka - l2)) * (exp_l2_t - exp_ka_t),
    );

    let absorption = absorption_vector * (ka * x[0] / (l1 - l2));

    let aux = non_zero + infusion + absorption;

    xout[0] = x[0] * exp_ka_t;
    xout[1] = aux[0];
    xout[2] = aux[1];

    xout
}
