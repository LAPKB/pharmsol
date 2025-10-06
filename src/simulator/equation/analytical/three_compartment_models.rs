use crate::{data::Covariates, simulator::*};
use diffsol::VectorCommon;
use nalgebra::{DVector, Matrix3, Vector3};

///
/// Analytical for three compartments
/// Assumptions:
///   - p is a vector of length 5 with k10, k12, k13, k21 and k31 in that order.
///     k10 represents the elimination rate constant from compartment 1 (central).
///     The remaining parameters of the form kij represent the transfer rate constants from the ith to the jth compartment
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

    let l1 = a / 3.0
        + gamma.powf(1.0 / 3.0) * ((theta / 3.0).cos() + 3.0_f64.sqrt() * (theta / 3.0).sin());
    let l2 = a / 3.0
        + gamma.powf(1.0 / 3.0) * ((theta / 3.0).cos() - 3.0_f64.sqrt() * (theta / 3.0).sin());
    let l3 = a / 3.0 - (2.0 * gamma.powf(1.0 / 3.0) * (theta / 3.0).cos());

    let exp_l1_t = (-l1 * t).exp();
    let exp_l2_t = (-l2 * t).exp();
    let exp_l3_t = (-l3 * t).exp();

    let c1 = (k21 - l1) * (k31 - l1) / ((l2 - l1) * (l3 - l1));
    let c2 = (k21 - l2) * (k31 - l2) / ((l1 - l2) * (l3 - l2));
    let c3 = (k21 - l3) * (k31 - l3) / ((l1 - l3) * (l2 - l3));
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

    let non_zero = non_zero_matrix * x.inner();

    let infusion_vector = Vector3::new(
        ((1.0 - exp_l1_t) * c1 / l1) + ((1.0 - exp_l2_t) * c2 / l2) + ((1.0 - exp_l3_t) * c3 / l3),
        ((1.0 - exp_l1_t) * c10 / l1)
            + ((1.0 - exp_l2_t) * c11 / l2)
            + ((1.0 - exp_l3_t) * c12 / l3),
        ((1.0 - exp_l1_t) * c19 / l1)
            + ((1.0 - exp_l2_t) * c20 / l2)
            + ((1.0 - exp_l3_t) * c21 / l3),
    );

    let infusion = infusion_vector * rateiv[0];

    let result_vector = non_zero + infusion;

    // Convert Vector2 to DVector
    DVector::from_vec(vec![result_vector[0], result_vector[1], result_vector[2]]).into()
}

///
/// Analytical solution for three compartment model with first-order absorption.
/// Assumptions:
///   - p is a vector of length 6 with ka, k10, k12, k13, k21 and k31 in that order.
///     ka represents the absorption rate constant
///     k10 represents the elimination rate constant from compartment 1 (central).
///     The remaining parameters of the form kij represent the transfer rate constants from the ith to the jth compartment
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

    let l1 = a / 3.0
        + gamma.powf(1.0 / 3.0) * ((theta / 3.0).cos() + 3.0_f64.sqrt() * (theta / 3.0).sin());
    let l2 = a / 3.0
        + gamma.powf(1.0 / 3.0) * ((theta / 3.0).cos() - 3.0_f64.sqrt() * (theta / 3.0).sin());
    let l3 = a / 3.0 - (2.0 * gamma.powf(1.0 / 3.0) * (theta / 3.0).cos());

    let exp_l1_t = (-l1 * t).exp();
    let exp_l2_t = (-l2 * t).exp();
    let exp_l3_t = (-l3 * t).exp();

    let c1 = (k21 - l1) * (k31 - l1) / ((l2 - l1) * (l3 - l1));
    let c2 = (k21 - l2) * (k31 - l2) / ((l1 - l2) * (l3 - l2));
    let c3 = (k21 - l3) * (k31 - l3) / ((l1 - l3) * (l2 - l3));
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
        ((1.0 - exp_l1_t) * c10 / l1)
            + ((1.0 - exp_l2_t) * c11 / l2)
            + ((1.0 - exp_l3_t) * c12 / l3),
        ((1.0 - exp_l1_t) * c19 / l1)
            + ((1.0 - exp_l2_t) * c20 / l2)
            + ((1.0 - exp_l3_t) * c21 / l3),
    );

    let infusion = infusion_vector * rateiv[0];

    let exp_ka_t = (-ka * t).exp();

    let absorption_vector = Vector3::new(
        (exp_l1_t - exp_ka_t) * c1 / (ka - l1)
            + (exp_l2_t - exp_ka_t) * c2 / (ka - l2)
            + (exp_l3_t - exp_ka_t) * c3 / (ka - l3),
        (exp_l1_t - exp_ka_t) * c10 / (ka - l1)
            + (exp_l2_t - exp_ka_t) * c11 / (ka - l2)
            + (exp_l3_t - exp_ka_t) * c12 / (ka - l3),
        (exp_l1_t - exp_ka_t) * c19 / (ka - l1)
            + (exp_l2_t - exp_ka_t) * c20 / (ka - l2)
            + (exp_l3_t - exp_ka_t) * c21 / (ka - l3),
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
    use super::{three_compartments, three_compartments_with_absorption};
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
    fn test_three_compartments() {
        let infusion_dosing = SubjectInfo::InfusionDosing;
        let subject = infusion_dosing.get_subject();

        let ode = equation::ODE::new(
            |x, p, _t, dx, b, rateiv, _cov| {
                fetch_params!(p, k10, k12, k13, k21, k31, _v);

                dx[0] = rateiv[0] - (k10 + k12 + k13) * x[0] + k21 * x[1] + k31 * x[2] + b[0];
                dx[1] = k12 * x[0] - k21 * x[1] + b[1];
                dx[2] = k13 * x[0] - k31 * x[2] + b[2];
            },
            |_p, _t, _cov| lag! {},
            |_p, _t, _cov| fa! {},
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
            |_p, _t, _cov| lag! {},
            |_p, _t, _cov| fa! {},
            |_p, _t, _cov, _x| {},
            |x, p, _t, _cov, y| {
                fetch_params!(p, _k10, _k12, _k13, _k21, _k31, v);
                y[0] = x[0] / v;
            },
            (3, 1),
        );

        let op_ode = ode
            .estimate_predictions(&subject, &vec![0.1, 3.0, 2.0, 1.0, 0.5, 1.0])
            .unwrap();
        let pred_ode = &op_ode.flat_predictions()[..];

        let op_analytical = analytical
            .estimate_predictions(&subject, &vec![0.1, 3.0, 2.0, 1.0, 0.5, 1.0])
            .unwrap();
        let pred_analytical = &op_analytical.flat_predictions()[..];

        println!("ode: {:?}", pred_ode);
        println!("analitycal: {:?}", pred_analytical);

        for (&od, &an) in pred_ode.iter().zip(pred_analytical.iter()) {
            assert_relative_eq!(od, an, max_relative = 1e-3, epsilon = 1e-3);
        }
    }

    #[test]
    fn test_three_compartments_with_absorption() {
        let oral_infusion_dosing = SubjectInfo::OralInfusionDosage;
        let subject = oral_infusion_dosing.get_subject();

        let ode = equation::ODE::new(
            |x, p, _t, dx, b, rateiv, _cov| {
                fetch_params!(p, ka, k10, k12, k13, k21, k31, _v);

                dx[0] = -ka * x[0] + b[0];
                dx[1] = rateiv[0] - (k10 + k12 + k13) * x[1]
                    + ka * x[0]
                    + k21 * x[2]
                    + k31 * x[3]
                    + b[1];
                dx[2] = k12 * x[1] - k21 * x[2] + b[2];
                dx[3] = k13 * x[1] - k31 * x[3] + b[3];
            },
            |_p, _t, _cov| lag! {},
            |_p, _t, _cov| fa! {},
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
            |_p, _t, _cov| lag! {},
            |_p, _t, _cov| fa! {},
            |_p, _t, _cov, _x| {},
            |x, p, _t, _cov, y| {
                fetch_params!(p, _ka, _k10, _k12, _k13, _k21, _k31, v);
                y[0] = x[1] / v;
            },
            (4, 1),
        );

        let op_ode = ode
            .estimate_predictions(&subject, &vec![1.0, 0.1, 3.0, 2.0, 1.0, 0.5, 1.0])
            .unwrap();
        let op_analytical = analytical
            .estimate_predictions(&subject, &vec![1.0, 0.1, 3.0, 2.0, 1.0, 0.5, 1.0])
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
