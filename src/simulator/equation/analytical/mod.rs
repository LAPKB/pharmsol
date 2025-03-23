use crate::{
    data::Covariates, simulator::*, Equation, EquationPriv, EquationTypes, Observation, Subject,
};
use cached::proc_macro::cached;
use cached::UnboundCache;
use nalgebra::{DVector, Matrix2, Matrix3, Vector2, Vector3};

/// Model equation using analytical solutions.
///
/// This implementation uses closed-form analytical solutions for the model
/// equations rather than numerical integration.
#[derive(Clone, Debug)]
pub struct Analytical {
    eq: AnalyticalEq,
    seq_eq: SecEq,
    lag: Lag,
    fa: Fa,
    init: Init,
    out: Out,
    neqs: Neqs,
}

impl Analytical {
    /// Create a new Analytical equation model.
    ///
    /// # Parameters
    /// - `eq`: The analytical equation function
    /// - `seq_eq`: The secondary equation function
    /// - `lag`: The lag time function
    /// - `fa`: The fraction absorbed function
    /// - `init`: The initial state function
    /// - `out`: The output equation function
    /// - `neqs`: The number of states and output equations
    pub fn new(
        eq: AnalyticalEq,
        seq_eq: SecEq,
        lag: Lag,
        fa: Fa,
        init: Init,
        out: Out,
        neqs: Neqs,
    ) -> Self {
        Self {
            eq,
            seq_eq,
            lag,
            fa,
            init,
            out,
            neqs,
        }
    }
}

impl EquationTypes for Analytical {
    type S = V;
    type P = SubjectPredictions;
}

impl EquationPriv for Analytical {
    // #[inline(always)]
    // fn get_init(&self) -> &Init {
    //     &self.init
    // }

    // #[inline(always)]
    // fn get_out(&self) -> &Out {
    //     &self.out
    // }

    #[inline(always)]
    fn get_lag(&self, spp: &[f64]) -> Option<HashMap<usize, f64>> {
        Some((self.lag)(&V::from_vec(spp.to_owned())))
    }

    #[inline(always)]
    fn get_fa(&self, spp: &[f64]) -> Option<HashMap<usize, f64>> {
        Some((self.fa)(&V::from_vec(spp.to_owned())))
    }

    #[inline(always)]
    fn get_nstates(&self) -> usize {
        self.neqs.0
    }

    #[inline(always)]
    fn get_nouteqs(&self) -> usize {
        self.neqs.1
    }
    #[inline(always)]
    fn solve(
        &self,
        x: &mut Self::S,
        support_point: &Vec<f64>,
        covariates: &Covariates,
        infusions: &Vec<Infusion>,
        ti: f64,
        tf: f64,
    ) {
        if ti == tf {
            return;
        }
        let mut support_point = V::from_vec(support_point.to_owned());
        let mut rateiv = V::from_vec(vec![0.0, 0.0, 0.0]);
        //TODO: This should be pre-calculated
        for infusion in infusions {
            if tf >= infusion.time() && tf <= infusion.duration() + infusion.time() {
                rateiv[infusion.input()] += infusion.amount() / infusion.duration();
            }
        }
        (self.seq_eq)(&mut support_point, tf, covariates);
        *x = (self.eq)(x, &support_point, tf - ti, rateiv, covariates);
    }
    #[inline(always)]
    fn process_observation(
        &self,
        support_point: &Vec<f64>,
        observation: &Observation,
        error_model: Option<&ErrorModel>,
        _time: f64,
        covariates: &Covariates,
        x: &mut Self::S,
        likelihood: &mut Vec<f64>,
        output: &mut Self::P,
    ) {
        let mut y = V::zeros(self.get_nouteqs());
        let out = &self.out;
        (out)(
            x,
            &V::from_vec(support_point.clone()),
            observation.time(),
            covariates,
            &mut y,
        );
        let pred = y[observation.outeq()];
        let pred = observation.to_obs_pred(pred, x.as_slice().to_vec());
        if let Some(error_model) = error_model {
            likelihood.push(pred.likelihood(error_model));
        }
        output.add_prediction(pred);
    }
    #[inline(always)]
    fn initial_state(&self, spp: &Vec<f64>, covariates: &Covariates, occasion_index: usize) -> V {
        let init = &self.init;
        let mut x = V::zeros(self.get_nstates());
        if occasion_index == 0 {
            (init)(&V::from_vec(spp.to_vec()), 0.0, covariates, &mut x);
        }
        x
    }
}

impl Equation for Analytical {
    fn estimate_likelihood(
        &self,
        subject: &Subject,
        support_point: &Vec<f64>,
        error_model: &ErrorModel,
        cache: bool,
    ) -> f64 {
        _estimate_likelihood(self, subject, support_point, error_model, cache)
    }
}
fn spphash(spp: &[f64]) -> u64 {
    spp.iter().fold(0, |acc, x| acc + x.to_bits())
}
#[inline(always)]
#[cached(
    ty = "UnboundCache<String, SubjectPredictions>",
    create = "{ UnboundCache::with_capacity(100_000) }",
    convert = r#"{ format!("{}{}", subject.id(), spphash(support_point)) }"#
)]
fn _subject_predictions(
    ode: &Analytical,
    subject: &Subject,
    support_point: &Vec<f64>,
) -> SubjectPredictions {
    ode.simulate_subject(subject, support_point, None).0
}

fn _estimate_likelihood(
    ode: &Analytical,
    subject: &Subject,
    support_point: &Vec<f64>,
    error_model: &ErrorModel,
    cache: bool,
) -> f64 {
    let ypred = if cache {
        _subject_predictions(ode, subject, support_point)
    } else {
        _subject_predictions_no_cache(ode, subject, support_point)
    };
    ypred.likelihood(error_model)
}

/// Analytical solution for one compartment model.
///
/// # Assumptions
/// - `p` is a vector of length 1 with the value of the elimination constant
/// - `rateiv` is a vector of length 1 with the value of the infusion rate (only one drug)
/// - `x` is a vector of length 1
/// - covariates are not used
pub fn one_compartment(x: &V, p: &V, t: T, rateiv: V, _cov: &Covariates) -> V {
    let mut xout = x.clone();
    let ke = p[0];

    xout[0] = x[0] * (-ke * t).exp() + rateiv[0] / ke * (1.0 - (-ke * t).exp());
    // dbg!(t, &rateiv, x, &xout);
    xout
}

/// Analytical solution for one compartment model with first-order absorption.
///
/// # Assumptions
/// - `p` is a vector of length 2 with ke and ka in that order
/// - `rateiv` is a vector of length 1 with the value of the infusion rate (only one drug)
/// - `x` is a vector of length 2
/// - covariates are not used
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

/// Analytical solution for two compartment model.
///
/// # Assumptions
/// - `p` is a vector of length 3 with ke, kcp and kpc in that order
/// - `rateiv` is a vector of length 1 with the value of the infusion rate (only one drug)
/// - `x` is a vector of length 2
/// - covariates are not used
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
        -kcp * exp_l1_t + kcp * exp_l2_t,
        (l1 - ke - kcp) * exp_l1_t + (ke + kcp - l2) * exp_l2_t,
    );

    let non_zero = (non_zero_matrix * x) / (l1 - l2);

    let infusion_vector = Vector2::new(
        ((l1 - kpc) / l1) * (1.0 - exp_l1_t) + ((kpc - l2) / l2) * (1.0 - exp_l2_t),
        (-kcp / l1) * (1.0 - exp_l1_t) + (kcp / l2) * (1.0 - exp_l2_t),
    );

    let infusion = infusion_vector * (rateiv[0] / (l1 - l2));

    let result_vector = non_zero + infusion;

    // Convert Vector2 to DVector
    DVector::from_vec(vec![result_vector[0], result_vector[1]])
}

/// Analytical solution for two compartment model with first-order absorption.
///
/// # Assumptions
/// - `p` is a vector of length 4 with ke, ka, kcp and kpc in that order
/// - `rateiv` is a vector of length 1 with the value of the infusion rate (only one drug)
/// - `x` is a vector of length 3
/// - covariates are not used
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
        -kcp * exp_l1_t + kcp * exp_l2_t,
        (l1 - ke - kcp) * exp_l1_t + (ke + kcp - l2) * exp_l2_t,
    );

    let non_zero = (non_zero_matrix * Vector2::new(x[1], x[2])) / (l1 - l2);

    let infusion_vector = Vector2::new(
        ((l1 - kpc) / l1) * (1.0 - exp_l1_t) + ((kpc - l2) / l2) * (1.0 - exp_l2_t),
        (-kcp / l1) * (1.0 - exp_l1_t) + (kcp / l2) * (1.0 - exp_l2_t),
    );

    let infusion = infusion_vector * (rateiv[0] / (l1 - l2));

    let exp_ka_t = (-ka * t).exp();

    let absorption_vector = Vector2::new(
        ((l1 - kpc) / (ka - l1)) * (exp_l1_t - exp_ka_t)
            + ((kpc - l2) / (ka - l2)) * (exp_l2_t - exp_ka_t),
        (-kcp / (ka - l1)) * (exp_l1_t - exp_ka_t) + (kcp / (ka - l2)) * (exp_l2_t - exp_ka_t),
    );

    let absorption = absorption_vector * (ka * x[0] / (l1 - l2));

    let aux = non_zero + infusion + absorption;

    xout[0] = x[0] * exp_ka_t;
    xout[1] = aux[0];
    xout[2] = aux[1];

    xout
}

///
/// Analytical for three compartments
/// Assumptions:
///   - p is a vector of length 5 with k10, k12, k13, k21 and k31 in that order
///   - rateiv is a vector of length 1 with the value of the infusion rate (only one drug)
///   - x is a vector of length 3
///   - covariates are not used
///
pub fn three_compartments(x: &V, p: &V, t: T, rateiv: V, _cov: &Covariates) -> V {
    let k10 = p[0];
    let k12 = p[1];
    let k13 = p[2];
    let k21 = p[3];
    let k31 = p[4];

    let a = k10 + k12 + k13 + k21 + k31;
    let b = k10 * k21 + k13 * k21 + k10 * k31 + k12 * k31 + k21 * k31;
    let c = k10 * k21 * k31;

    let m = (3.0 * b - a.powi(2)) / 3.0;
    let n = (2.0 * a.powi(3) - 9.0 * a * b + 27.0 * c) / 27.0;
    let q = (n.powi(2)) / 4.0 + (m.powi(3)) / 27.0;

    if q > 0.0 {
        panic!("Imaginary solutions, program stopped!");
    }

    let alpha = (-1.0 * q).sqrt();
    let beta = -1.0 * n / 2.0;
    let gamma = (beta.powi(2) + alpha.powi(2)).sqrt();
    let theta = alpha.atan2(beta);

    let l1 = a / 3.0 + gamma.powf(1.0 / 3.0)*((theta / 3.0).cos() + 3.0_f64.sqrt() * (theta / 3.0).sin());
    let l2 = a / 3.0 + gamma.powf(1.0 / 3.0)*((theta / 3.0).cos() - 3.0_f64.sqrt() * (theta / 3.0).sin());
    let l3 = a / 3.0 - (2.0 * gamma.powf(1.0 / 3.0) * (theta / 3.0).cos());

    let exp_l1_t = (-l1 * t).exp();
    let exp_l2_t = (-l2 * t).exp();
    let exp_l3_t = (-l3 * t).exp();

    let c1 = (k21 -l1) * (k31 - l1) / ((l2 - l1) * (l3 - l1));
    let c2 = (k21 -l2) * (k31 - l2) / ((l1 - l2) * (l3 - l2));
    let c3 = (k21 -l3) * (k31 - l3) / ((l1 - l3) * (l2 - l3));
    let c4 = k21 * (k31 - l1) / ((l2 - l1) * (l3 - l1));
    let c5 = k21 * (k31 - l2) / ((l1 - l2) * (l3 - l2));
    let c6 = k21 * (k31 - l3) / ((l1 - l3) * (l2 - l3));
    let c7 = k31 * (k21 - l1) / ((l2 - l1) * (l3 - l1));
    let c8 = k31 * (k21 - l2) / ((l1 - l2) * (l3 - l2));
    let c9 = k31 * (k21 - l3) / ((l1 - l3) * (l2 - l3));
    let c10 = k12 * (k31 - l1) / ((l2 - l1) * (l3 - l1));
    let c11 = k12 * (k31 - l2) / ((l1 - l2) * (l3 - l2));
    let c12 = k12 * (k31 - l3) / ((l1 - l3) * (l2 - l3));
    let c13 = ((k10 + k12 + k13 - l1) * (k31 - l1) - (k13 * k31)) / ((l2 - l1) * (l3 - l1));
    let c14 = ((k10 + k12 + k13 - l2) * (k31 - l2) - (k13 * k31)) / ((l1 - l2) * (l3 - l2));
    let c15 = ((k10 + k12 + k13 - l3) * (k31 - l3) - (k13 * k31)) / ((l1 - l3) * (l2 - l3));
    let c16 = k12 * k31 / ((l2 - l1) * (l3 - l1));   
    let c17 = k12 * k31 / ((l1 - l2) * (l3 - l2));
    let c18 = k12 * k31 / ((l1 - l3) * (l2 - l3));
    let c19 = k13 * (k21 - l1) / ((l2 - l1) * (l3 - l1));
    let c20 = k13 * (k21 - l2) / ((l1 - l2) * (l3 - l2));
    let c21 = k13 * (k21 - l3) / ((l1 - l3) * (l2 - l3));
    let c22 = k21 * k13 / ((l2 - l1) * (l3 - l1));
    let c23 = k21 * k13 / ((l1 - l2) * (l3 - l2));
    let c24 = k21 * k13 / ((l1 - l3) * (l2 - l3));
    let c25 = ((k10 + k12 + k13 - l1) * (k21 - l1) - (k12 * k21)) / ((l2 - l1) * (l3 - l1));
    let c26 = ((k10 + k12 + k13 - l2) * (k21 - l2) - (k12 * k21)) / ((l1 - l2) * (l3 - l2));
    let c27 = ((k10 + k12 + k13 - l3) * (k21 - l3) - (k12 * k21)) / ((l1 - l3) * (l2 - l3));

    let non_zero_matrix = Matrix3::new(
        c1 * exp_l1_t + c2 * exp_l2_t + c3 * exp_l3_t,
        c4 * exp_l1_t + c5 * exp_l2_t + c6 * exp_l3_t,
        c7 * exp_l1_t + c8 * exp_l2_t + c9 * exp_l3_t,
        c10 * exp_l1_t + c11 * exp_l2_t + c12 * exp_l3_t,
        c13 * exp_l1_t + c14 * exp_l2_t + c15 * exp_l3_t,
        c16 * exp_l1_t + c17 * exp_l2_t + c18 * exp_l3_t,
        c19 * exp_l1_t + c20 * exp_l2_t + c21 * exp_l3_t,
        c22 * exp_l1_t + c23 * exp_l2_t + c24 * exp_l3_t,
        c25 * exp_l1_t + c26 * exp_l2_t + c27 * exp_l3_t,
    );

    let non_zero = non_zero_matrix * x;

    let infusion_vector = Vector3::new(
        ((1.0 - exp_l1_t) * c1 / l1) + ((1.0 - exp_l2_t) * c2 / l2) + ((1.0 - exp_l3_t) * c3 / l3),
        ((1.0 - exp_l1_t) * c10 / l1) + ((1.0 - exp_l2_t) * c11 / l2) + ((1.0 - exp_l3_t) * c12 / l3),
        ((1.0 - exp_l1_t) * c19 / l1) + ((1.0 - exp_l2_t) * c20 / l2) + ((1.0 - exp_l3_t) * c21 / l3),
    );

    let infusion = infusion_vector * rateiv[0];

    let result_vector = non_zero + infusion;

    // Convert Vector2 to DVector
    DVector::from_vec(vec![result_vector[0], result_vector[1], result_vector[2]])
}

///
/// Analytical for three compartments with absorption
/// Assumptions:
///   - p is a vector of length 6 with ka, k10, k12, k13, k21 and k31 in that order
///   - rateiv is a vector of length 1 with the value of the infusion rate (only one drug)
///   - x is a vector of length 4
///   - covariates are not used
///
pub fn three_compartments_with_absorption(x: &V, p: &V, t: T, rateiv: V, _cov: &Covariates) -> V {
    let ka = p[0];
    let k10 = p[1];
    let k12 = p[2];
    let k13 = p[3];
    let k21 = p[4];
    let k31 = p[5];
    let mut xout = x.clone();

    let a = k10 + k12 + k13 + k21 + k31;
    let b = k10 * k21 + k13 * k21 + k10 * k31 + k12 * k31 + k21 * k31;
    let c = k10 * k21 * k31;

    let m = (3.0 * b - a.powi(2)) / 3.0;
    let n = (2.0 * a.powi(3) - 9.0 * a * b + 27.0 * c) / 27.0;
    let q = (n.powi(2)) / 4.0 + (m.powi(3)) / 27.0;

    if q > 0.0 {
        panic!("Imaginary solutions, program stopped!");
    }

    let alpha = (-1.0 * q).sqrt();
    let beta = -1.0 * n / 2.0;
    let gamma = (beta.powi(2) + alpha.powi(2)).sqrt();
    let theta = alpha.atan2(beta);

    let l1 = a / 3.0 + gamma.powf(1.0 / 3.0)*((theta / 3.0).cos() + 3.0_f64.sqrt() * (theta / 3.0).sin());
    let l2 = a / 3.0 + gamma.powf(1.0 / 3.0)*((theta / 3.0).cos() - 3.0_f64.sqrt() * (theta / 3.0).sin());
    let l3 = a / 3.0 - (2.0 * gamma.powf(1.0 / 3.0) * (theta / 3.0).cos());

    let exp_l1_t = (-l1 * t).exp();
    let exp_l2_t = (-l2 * t).exp();
    let exp_l3_t = (-l3 * t).exp();

    let c1 = (k21 -l1) * (k31 - l1) / ((l2 - l1) * (l3 - l1));
    let c2 = (k21 -l2) * (k31 - l2) / ((l1 - l2) * (l3 - l2));
    let c3 = (k21 -l3) * (k31 - l3) / ((l1 - l3) * (l2 - l3));
    let c4 = k21 * (k31 - l1) / ((l2 - l1) * (l3 - l1));
    let c5 = k21 * (k31 - l2) / ((l1 - l2) * (l3 - l2));
    let c6 = k21 * (k31 - l3) / ((l1 - l3) * (l2 - l3));
    let c7 = k31 * (k21 - l1) / ((l2 - l1) * (l3 - l1));
    let c8 = k31 * (k21 - l2) / ((l1 - l2) * (l3 - l2));
    let c9 = k31 * (k21 - l3) / ((l1 - l3) * (l2 - l3));
    let c10 = k12 * (k31 - l1) / ((l2 - l1) * (l3 - l1));
    let c11 = k12 * (k31 - l2) / ((l1 - l2) * (l3 - l2));
    let c12 = k12 * (k31 - l3) / ((l1 - l3) * (l2 - l3));
    let c13 = ((k10 + k12 + k13 - l1) * (k31 - l1) - (k13 * k31)) / ((l2 - l1) * (l3 - l1));
    let c14 = ((k10 + k12 + k13 - l2) * (k31 - l2) - (k13 * k31)) / ((l1 - l2) * (l3 - l2));
    let c15 = ((k10 + k12 + k13 - l3) * (k31 - l3) - (k13 * k31)) / ((l1 - l3) * (l2 - l3));
    let c16 = k12 * k31 / ((l2 - l1) * (l3 - l1));   
    let c17 = k12 * k31 / ((l1 - l2) * (l3 - l2));
    let c18 = k12 * k31 / ((l1 - l3) * (l2 - l3));
    let c19 = k13 * (k21 - l1) / ((l2 - l1) * (l3 - l1));
    let c20 = k13 * (k21 - l2) / ((l1 - l2) * (l3 - l2));
    let c21 = k13 * (k21 - l3) / ((l1 - l3) * (l2 - l3));
    let c22 = k21 * k13 / ((l2 - l1) * (l3 - l1));
    let c23 = k21 * k13 / ((l1 - l2) * (l3 - l2));
    let c24 = k21 * k13 / ((l1 - l3) * (l2 - l3));
    let c25 = ((k10 + k12 + k13 - l1) * (k21 - l1) - (k12 * k21)) / ((l2 - l1) * (l3 - l1));
    let c26 = ((k10 + k12 + k13 - l2) * (k21 - l2) - (k12 * k21)) / ((l1 - l2) * (l3 - l2));
    let c27 = ((k10 + k12 + k13 - l3) * (k21 - l3) - (k12 * k21)) / ((l1 - l3) * (l2 - l3));

    let non_zero_matrix = Matrix3::new(
        c1 * exp_l1_t + c2 * exp_l2_t + c3 * exp_l3_t,
        c4 * exp_l1_t + c5 * exp_l2_t + c6 * exp_l3_t,
        c7 * exp_l1_t + c8 * exp_l2_t + c9 * exp_l3_t,
        c10 * exp_l1_t + c11 * exp_l2_t + c12 * exp_l3_t,
        c13 * exp_l1_t + c14 * exp_l2_t + c15 * exp_l3_t,
        c16 * exp_l1_t + c17 * exp_l2_t + c18 * exp_l3_t,
        c19 * exp_l1_t + c20 * exp_l2_t + c21 * exp_l3_t,
        c22 * exp_l1_t + c23 * exp_l2_t + c24 * exp_l3_t,
        c25 * exp_l1_t + c26 * exp_l2_t + c27 * exp_l3_t,
    );

    let non_zero = non_zero_matrix * Vector3::new(x[1], x[2], x[3]);

    let infusion_vector = Vector3::new(
        ((1.0 - exp_l1_t) * c1 / l1) + ((1.0 - exp_l2_t) * c2 / l2) + ((1.0 - exp_l3_t) * c3 / l3),
        ((1.0 - exp_l1_t) * c10 / l1) + ((1.0 - exp_l2_t) * c11 / l2) + ((1.0 - exp_l3_t) * c12 / l3),
        ((1.0 - exp_l1_t) * c19 / l1) + ((1.0 - exp_l2_t) * c20 / l2) + ((1.0 - exp_l3_t) * c21 / l3),
    );

    let infusion = infusion_vector * rateiv[0];

    let exp_ka_t = (-ka * t).exp();

    let absorption_vector = Vector3::new(
        (exp_l1_t - exp_ka_t) * c1 / (ka - l1) + (exp_l2_t - exp_ka_t) * c2 / (ka - l2) + (exp_l3_t - exp_ka_t) * c3 / (ka - l3),
        (exp_l1_t - exp_ka_t) * c10 / (ka - l1) + (exp_l2_t - exp_ka_t) * c11 / (ka - l2) + (exp_l3_t - exp_ka_t) * c12 / (ka - l3),
        (exp_l1_t - exp_ka_t) * c19 / (ka - l1) + (exp_l2_t - exp_ka_t) * c20 / (ka - l2) + (exp_l3_t - exp_ka_t) * c21 / (ka - l3),
    );

    let absorption = absorption_vector * ka * x[0];

    let aux = non_zero + infusion + absorption;

    xout[0] = x[0] * exp_ka_t;
    xout[1] = aux[0];
    xout[2] = aux[1];
    xout[3] = aux[2];

    xout
}

#[cfg(test)]
mod tests {
    use crate::*;

    enum SubjectInfo {
        InfusionDosing,
        OralInfusionDosage
    }

    impl SubjectInfo {
        fn get_subject(&self) -> Subject {
            match self {
                
                SubjectInfo::InfusionDosing => Subject::builder("id1")
                    .bolus(0.0, 100.0, 0)
                    .infusion(24.0, 150.0, 0, 3.0)
                    .observation(0.0, 0.0, 0)
                    .observation(1.0, 0.0, 0)
                    .observation(2.0, 0.0, 0)
                    .observation(4.0, 0.0, 0)
                    .observation(8.0, 0.0, 0)
                    .observation(12.0, 0.0, 0)
                    .observation(24.0, 0.0, 0)
                    .observation(25.0, 0.0, 0)
                    .observation(26.0, 0.0, 0)
                    .observation(27.0, 0.0, 0)
                    .observation(28.0, 0.0, 0)
                    .observation(32.0, 0.0, 0)
                    .observation(36.0, 0.0, 0)
                    .build(),

                SubjectInfo::OralInfusionDosage => Subject::builder("id1")
                    .bolus(0.0, 100.0, 1)
                    .infusion(24.0, 150.0, 0, 3.0)
                    .bolus(48.0, 100.0, 0)
                    .observation(0.0, 0.0, 0)
                    .observation(1.0, 0.0, 0)
                    .observation(2.0, 0.0, 0)
                    .observation(4.0, 0.0, 0)
                    .observation(8.0, 0.0, 0)
                    .observation(12.0, 0.0, 0)
                    .observation(24.0, 0.0, 0)
                    .observation(25.0, 0.0, 0)
                    .observation(26.0, 0.0, 0)
                    .observation(27.0, 0.0, 0)
                    .observation(28.0, 0.0, 0)
                    .observation(32.0, 0.0, 0)
                    .observation(36.0, 0.0, 0)
                    .observation(48.0, 0.0, 0)
                    .observation(49.0, 0.0, 0)
                    .observation(50.0, 0.0, 0)
                    .observation(52.0, 0.0, 0)
                    .observation(56.0, 0.0, 0)
                    .observation(60.0, 0.0, 0)
                    .build(),
            }
        }
    }

    #[test]
    fn test_one_compartment() {

        let infusion_dosing = SubjectInfo::InfusionDosing;
        let subject = infusion_dosing.get_subject();

        let ode = equation::ODE::new(
            |x, p, _t, dx, rateiv, _cov| {
                fetch_params!(p, ke, _v);
    
                dx[0] = - ke * x[0] + rateiv[0];
            },
            |_p| lag! {},
            |_p| fa! {},
            |_p, _t, _cov, _x| {},
            |x, p, _t, _cov, y| {
                fetch_params!(p, _ke, v);
                y[0] = x[0] / v;
            },
            (1, 1),
        );

        let analytical = equation::Analytical::new(
            one_compartment,
            |_p, _t, _cov| {},
            |_p| lag! {},
            |_p| fa! {},
            |_p, _t, _cov, _x| {},
            |x, p, _t, _cov, y| {
                fetch_params!(p, _ke, v);
                y[0] = x[0] / v;
            },
            (1, 1),
        );

        let op_ode = ode.estimate_predictions(&subject, &vec![0.1, 1.0]);
        let op_analytical = analytical.estimate_predictions(&subject, &vec![0.1, 1.0]);

        let pred_ode= &op_ode.flat_predictions()[..];
        let pred_analytical = &op_analytical.flat_predictions()[..];
        
        println!("ode: {:?}", pred_ode);
        println!("analitycal: {:?}", pred_analytical);
        
        for (&od, &an) in pred_ode.iter().zip(pred_analytical.iter()) {
            let error = (od - an).abs() / od;  
            assert!(error < 0.1, "error = {}, ode value = {}, analitycal value = {}", error, od, an);
        }
    }

    #[test]
    fn test_one_compartment_with_absorption() {

        let oral_infusion_dosing = SubjectInfo::OralInfusionDosage;
        let subject = oral_infusion_dosing.get_subject();

        let ode = equation::ODE::new(
            |x, p, _t, dx, rateiv, _cov| {
                fetch_params!(p, ka, ke, _v);
    
                dx[0] = - ka * x[0];
                dx[1] = ka * x[0] - ke * x[1] + rateiv[0];
            },
            |_p| lag! {},
            |_p| fa! {},
            |_p, _t, _cov, _x| {},
            |x, p, _t, _cov, y| {
                fetch_params!(p, _ka, _ke, v);
                y[0] = x[1] / v;
            },
            (2, 1),
        );

        let analytical = equation::Analytical::new(
            one_compartment_with_absorption,
            |_p, _t, _cov| {},
            |_p| lag! {},
            |_p| fa! {},
            |_p, _t, _cov, _x| {},
            |x, p, _t, _cov, y| {
                fetch_params!(p, _ka, _ke, v);
                y[0] = x[1] / v;
            },
            (2, 1),
        );

        let op_ode = ode.estimate_predictions(&subject, &vec![1.0, 0.1, 1.0]);
        let op_analytical = analytical.estimate_predictions(&subject, &vec![1.0, 0.1, 1.0]);

        let pred_ode= &op_ode.flat_predictions()[..];
        let pred_analytical = &op_analytical.flat_predictions()[..];

        println!("ode: {:?}", pred_ode);
        println!("analitycal: {:?}", pred_analytical);
        
        for (&od, &an) in pred_ode.iter().zip(pred_analytical.iter()) {
            let error = (od - an).abs() / od;  
            assert!(error < 0.1, "error = {}, ode value = {}, analitycal value = {}", error, od, an);
        }
    }

    #[test]
    fn test_two_compartments() {

        let infusion_dosing = SubjectInfo::InfusionDosing;
        let subject = infusion_dosing.get_subject();

        let ode = equation::ODE::new(
            |x, p, _t, dx, rateiv, _cov| {
                fetch_params!(p, ke, kcp, kpc, _v);
    
                dx[0] = rateiv[0] - ke * x[0] - kcp * x[0] + kpc * x[1];
                dx[1] = kcp * x[0] - kpc * x[1];
            },
            |_p| lag! {},
            |_p| fa! {},
            |_p, _t, _cov, _x| {},
            |x, p, _t, _cov, y| {
                fetch_params!(p, _ke, _kcp, _kpc, v);
                y[0] = x[0] / v;
            },
            (2, 1),
        );

        let analytical = equation::Analytical::new(
            two_compartments,
            |_p, _t, _cov| {},
            |_p| lag! {},
            |_p| fa! {},
            |_p, _t, _cov, _x| {},
            |x, p, _t, _cov, y| {
                fetch_params!(p, _ke, _kcp, _kpc, v);
                y[0] = x[0] / v;
            },
            (2, 1),
        );

        let op_ode = ode.estimate_predictions(&subject, &vec![0.1, 3.0, 1.0, 1.0]);
        let op_analytical = analytical.estimate_predictions(&subject, &vec![0.1, 3.0, 1.0, 1.0]);

        let pred_ode= &op_ode.flat_predictions()[..];
        let pred_analytical = &op_analytical.flat_predictions()[..];

        println!("ode: {:?}", pred_ode);
        println!("analitycal: {:?}", pred_analytical);
        
        for (&od, &an) in pred_ode.iter().zip(pred_analytical.iter()) {
            let error = (od - an).abs() / od;  
            assert!(error < 0.1, "error = {}, ode value = {}, analitycal value = {}", error, od, an);
        }
    }

    #[test]
    fn test_two_compartments_with_absorption() {

        let oral_infusion_dosing = SubjectInfo::OralInfusionDosage;
        let subject = oral_infusion_dosing.get_subject();

        let ode = equation::ODE::new(
            |x, p, _t, dx, rateiv, _cov| {
                fetch_params!(p, ke, ka, kcp, kpc, _v);
    
                dx[0] = - ka * x[0];
                dx[1] = rateiv[0] - ke * x[1] + ka * x[0] - kcp * x[1] + kpc * x[2];
                dx[2] = kcp * x[1] - kpc * x[2];
            },
            |_p| lag! {},
            |_p| fa! {},
            |_p, _t, _cov, _x| {},
            |x, p, _t, _cov, y| {
                fetch_params!(p, _ke, _ka, _kcp, _kpc, v);
                y[0] = x[1] / v;
            },
            (3, 1),
        );

        let analytical = equation::Analytical::new(
            two_compartments_with_absorption,
            |_p, _t, _cov| {},
            |_p| lag! {},
            |_p| fa! {},
            |_p, _t, _cov, _x| {},
            |x, p, _t, _cov, y| {
                fetch_params!(p, _ke, _ka, _kcp, _kpc, v);
                y[0] = x[1] / v;
            },
            (3, 1),
        );

        let op_ode = ode.estimate_predictions(&subject, &vec![0.1, 1.0, 3.0, 1.0, 1.0]);
        let op_analytical = analytical.estimate_predictions(&subject, &vec![0.1, 1.0, 3.0, 1.0, 1.0]);

        let pred_ode= &op_ode.flat_predictions()[..];
        let pred_analytical = &op_analytical.flat_predictions()[..];

        println!("ode: {:?}", pred_ode);
        println!("analitycal: {:?}", pred_analytical);
        
        for (&od, &an) in pred_ode.iter().zip(pred_analytical.iter()) {
            let error = (od - an).abs() / od;  
            assert!(error < 0.1, "error = {}, ode value = {}, analitycal value = {}", error, od, an);
        }
    }

    #[test]
    fn test_three_compartments() {

        let infusion_dosing = SubjectInfo::InfusionDosing;
        let subject = infusion_dosing.get_subject();

        let ode = equation::ODE::new(
            |x, p, _t, dx, rateiv, _cov| {
                fetch_params!(p, k10, k12, k13, k21, k31, _v);
    
                dx[0] = rateiv[0] - (k10 + k12 + k13) * x[0] + k21 * x[1] + k31 * x[2];
                dx[1] = k12 * x[0] - k21 * x[1];
                dx[2] = k13 * x[0] - k31 * x[2];
            },
            |_p| lag! {},
            |_p| fa! {},
            |_p, _t, _cov, _x| {},
            |x, p, _t, _cov, y| {
                fetch_params!(p, _k10, _k12, _k13, _k21, _k31, v);
                y[0] = x[0] / v;
            },
            (3, 1),
        );

        let analytical = equation::Analytical::new(
            three_compartments,
            |_p, _t, _cov| {},
            |_p| lag! {},
            |_p| fa! {},
            |_p, _t, _cov, _x| {},
            |x, p, _t, _cov, y| {
                fetch_params!(p, _k10, _k12, _k13, _k21, _k31, v);
                y[0] = x[0] / v;
            },
            (3, 1),
        );

        let op_ode = ode.estimate_predictions(&subject, &vec![0.1, 3.0, 2.0, 1.0, 0.5, 1.0]);
        let pred_ode= &op_ode.flat_predictions()[..];

        let op_analytical = analytical.estimate_predictions(&subject, &vec![0.1, 3.0, 2.0, 1.0, 0.5, 1.0]);
        let pred_analytical = &op_analytical.flat_predictions()[..];

        println!("ode: {:?}", pred_ode);
        println!("analitycal: {:?}", pred_analytical);

        for (&od, &an) in pred_ode.iter().zip(pred_analytical.iter()) {
            let error = (od - an).abs() / od;  
            assert!(error < 0.1, "error = {}, ode value = {}, analitycal value = {}", error, od, an);
        }
    }

    #[test]
    fn test_three_compartments_with_absorption() {

        let oral_infusion_dosing = SubjectInfo::OralInfusionDosage;
        let subject = oral_infusion_dosing.get_subject();

        let ode = equation::ODE::new(
            |x, p, _t, dx, rateiv, _cov| {
                fetch_params!(p, ka, k10, k12, k13, k21, k31, _v);
    
                dx[0] = - ka * x[0];
                dx[1] = rateiv[0] - (k10 + k12 + k13) * x[1] + ka * x[0] + k21 * x[2] + k31 * x[3];
                dx[2] = k12 * x[1] - k21 * x[2];
                dx[3] = k13 * x[1] - k31 * x[3];
            },
            |_p| lag! {},
            |_p| fa! {},
            |_p, _t, _cov, _x| {},
            |x, p, _t, _cov, y| {
                fetch_params!(p, _ka, _k10, _k12, _k13, _k21, _k31, v);
                y[0] = x[1] / v;
            },
            (4, 1),
        );

        let analytical = equation::Analytical::new(
            three_compartments_with_absorption,
            |_p, _t, _cov| {},
            |_p| lag! {},
            |_p| fa! {},
            |_p, _t, _cov, _x| {},
            |x, p, _t, _cov, y| {
                fetch_params!(p, _ka, _k10, _k12, _k13, _k21, _k31, v);
                y[0] = x[1] / v;
            },
            (4, 1),
        );

        let op_ode = ode.estimate_predictions(&subject, &vec![1.0, 0.1, 3.0, 2.0, 1.0, 0.5, 1.0]);
        let op_analytical = analytical.estimate_predictions(&subject, &vec![1.0, 0.1, 3.0, 2.0, 1.0, 0.5, 1.0]);

        let pred_ode= &op_ode.flat_predictions()[..];
        let pred_analytical = &op_analytical.flat_predictions()[..];

        println!("ode: {:?}", pred_ode);
        println!("analitycal: {:?}", pred_analytical);
        
        for (&od, &an) in pred_ode.iter().zip(pred_analytical.iter()) {
            let error = (od - an).abs() / od;  
            assert!(error < 0.1, "error = {}, ode value = {}, analitycal value = {}", error, od, an);
        }
    }

}