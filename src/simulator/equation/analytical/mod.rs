use crate::{
    data::Covariates, simulator::*, Equation, EquationPriv, EquationTypes, Observation, Subject,
};
use cached::proc_macro::cached;
use cached::UnboundCache;
use nalgebra::{DVector, Matrix2, Vector2};
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
///   - p is a vector of length xxxx with ke, ka, kcp and kpc in that order
///   - rateiv is a vector of length 1 with the value of the infusion rate (only one drug)
///   - x is a vector of length xxxxx
///   - covariates are not used
///
// pub fn three_compartments(x: &V, p: &V, t: T, rateiv: V, _cov: &Covariates) -> V {}

///
/// Analytical for three compartments with absorption
/// Assumptions:
///   - p is a vector of length xxxxx with ke, ka, kcp and kpc in that order
///   - rateiv is a vector of length 1 with the value of the infusion rate (only one drug)
///   - x is a vector of length xxxxx
///   - covariates are not used
///
// pub fn three_compartments_with_absorption(x: &V, p: &V, t: T, rateiv: V, _cov: &Covariates) -> V {}

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
            assert!(error < 0.2, "error = {}, ode value = {}, analitycal value = {}", error, od, an);
        }
    }

    #[test]
    fn test_one_compartment_with_absorption() {

        let oral_dosing = SubjectInfo::OralInfusionDosage;
        let subject = oral_dosing.get_subject();

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
            assert!(error < 0.2, "error = {}, ode value = {}, analitycal value = {}", error, od, an);
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
            assert!(error < 0.2, "error = {}, ode value = {}, analitycal value = {}", error, od, an);
        }
    }

    #[test]
    fn test_two_compartments_with_absorption() {

        let oral_dosing = SubjectInfo::OralInfusionDosage;
        let subject = oral_dosing.get_subject();

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
            assert!(error < 0.2, "error = {}, ode value = {}, analitycal value = {}", error, od, an);
        }
    }

}
