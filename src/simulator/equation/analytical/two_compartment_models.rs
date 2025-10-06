use crate::{data::Covariates, simulator::*};
use diffsol::VectorCommon;
use nalgebra::{DVector, Matrix2, Vector2};

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

    let non_zero = (non_zero_matrix * x.inner()) / (l1 - l2);

    let infusion_vector = Vector2::new(
        ((l1 - kpc) / l1) * (1.0 - exp_l1_t) + ((kpc - l2) / l2) * (1.0 - exp_l2_t),
        (-kcp / l1) * (1.0 - exp_l1_t) + (kcp / l2) * (1.0 - exp_l2_t),
    );

    let infusion = infusion_vector * (rateiv[0] / (l1 - l2));

    let result_vector = non_zero + infusion;

    // Convert Vector2 to DVector
    DVector::from_vec(vec![result_vector[0], result_vector[1]]).into()
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

#[cfg(test)]
mod tests {
    use super::{two_compartments, two_compartments_with_absorption};
    use crate::*;
    use approx::assert_relative_eq;

    enum SubjectInfo {
        InfusionDosing,
        OralInfusionDosage,
    }

    impl SubjectInfo {
        fn get_subject(&self) -> Subject {
            match self {
                SubjectInfo::InfusionDosing => Subject::builder("id1")
                    .bolus(0.0, 100.0, 0)
                    .infusion(24.0, 150.0, 0, 3.0)
                    .missing_observation(0.0, 0)
                    .missing_observation(1.0, 0)
                    .missing_observation(2.0, 0)
                    .missing_observation(4.0, 0)
                    .missing_observation(8.0, 0)
                    .missing_observation(12.0, 0)
                    .missing_observation(24.0, 0)
                    .missing_observation(25.0, 0)
                    .missing_observation(26.0, 0)
                    .missing_observation(27.0, 0)
                    .missing_observation(28.0, 0)
                    .missing_observation(32.0, 0)
                    .missing_observation(36.0, 0)
                    .build(),

                SubjectInfo::OralInfusionDosage => Subject::builder("id1")
                    .bolus(0.0, 100.0, 1)
                    .infusion(24.0, 150.0, 0, 3.0)
                    .bolus(48.0, 100.0, 0)
                    .missing_observation(0.0, 0)
                    .missing_observation(1.0, 0)
                    .missing_observation(2.0, 0)
                    .missing_observation(4.0, 0)
                    .missing_observation(8.0, 0)
                    .missing_observation(12.0, 0)
                    .missing_observation(24.0, 0)
                    .missing_observation(25.0, 0)
                    .missing_observation(26.0, 0)
                    .missing_observation(27.0, 0)
                    .missing_observation(28.0, 0)
                    .missing_observation(32.0, 0)
                    .missing_observation(36.0, 0)
                    .missing_observation(48.0, 0)
                    .missing_observation(49.0, 0)
                    .missing_observation(50.0, 0)
                    .missing_observation(52.0, 0)
                    .missing_observation(56.0, 0)
                    .missing_observation(60.0, 0)
                    .build(),
            }
        }
    }

    #[test]
    fn test_two_compartments() {
        let infusion_dosing = SubjectInfo::InfusionDosing;
        let subject = infusion_dosing.get_subject();

        let ode = equation::ODE::new(
            |x, p, _t, dx, b, rateiv, _cov| {
                fetch_params!(p, ke, kcp, kpc, _v);

                dx[0] = rateiv[0] - ke * x[0] - kcp * x[0] + kpc * x[1] + b[0];
                dx[1] = kcp * x[0] - kpc * x[1] + b[1];
            },
            |_p, _t, _cov| lag! {},
            |_p, _t, _cov| fa! {},
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
            |_p, _t, _cov| lag! {},
            |_p, _t, _cov| fa! {},
            |_p, _t, _cov, _x| {},
            |x, p, _t, _cov, y| {
                fetch_params!(p, _ke, _kcp, _kpc, v);
                y[0] = x[0] / v;
            },
            (2, 1),
        );

        let op_ode = ode
            .estimate_predictions(&subject, &vec![0.1, 3.0, 1.0, 1.0])
            .unwrap();
        let op_analytical = analytical
            .estimate_predictions(&subject, &vec![0.1, 3.0, 1.0, 1.0])
            .unwrap();

        let pred_ode = &op_ode.flat_predictions()[..];
        let pred_analytical = &op_analytical.flat_predictions()[..];

        println!("ode: {:?}", pred_ode);
        println!("analitycal: {:?}", pred_analytical);

        for (&od, &an) in pred_ode.iter().zip(pred_analytical.iter()) {
            assert_relative_eq!(od, an, max_relative = 1e-4, epsilon = 1.0);
        }
    }

    #[test]
    fn test_two_compartments_with_absorption() {
        let oral_infusion_dosing = SubjectInfo::OralInfusionDosage;
        let subject = oral_infusion_dosing.get_subject();

        let ode = equation::ODE::new(
            |x, p, _t, dx, b, rateiv, _cov| {
                fetch_params!(p, ke, ka, kcp, kpc, _v);

                dx[0] = -ka * x[0] + b[0];
                dx[1] = rateiv[0] - ke * x[1] + ka * x[0] - kcp * x[1] + kpc * x[2] + b[1];
                dx[2] = kcp * x[1] - kpc * x[2] + b[2];
            },
            |_p, _t, _cov| lag! {},
            |_p, _t, _cov| fa! {},
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
            |_p, _t, _cov| lag! {},
            |_p, _t, _cov| fa! {},
            |_p, _t, _cov, _x| {},
            |x, p, _t, _cov, y| {
                fetch_params!(p, _ke, _ka, _kcp, _kpc, v);
                y[0] = x[1] / v;
            },
            (3, 1),
        );

        let op_ode = ode
            .estimate_predictions(&subject, &vec![0.1, 1.0, 3.0, 1.0, 1.0])
            .unwrap();
        let op_analytical = analytical
            .estimate_predictions(&subject, &vec![0.1, 1.0, 3.0, 1.0, 1.0])
            .unwrap();

        let pred_ode = &op_ode.flat_predictions()[..];
        let pred_analytical = &op_analytical.flat_predictions()[..];

        println!("ode: {:?}", pred_ode);
        println!("analitycal: {:?}", pred_analytical);

        for (&od, &an) in pred_ode.iter().zip(pred_analytical.iter()) {
            assert_relative_eq!(od, an, max_relative = 1e-3, epsilon = 1e-3,);
        }
    }
}
