use crate::{data::Covariates, simulator::*};
use diffsol::VectorCommon;
use nalgebra::{DVector, Matrix2, Vector2};

/// Analytical solution for two compartment model parameterized by clearances.
///
/// Converts CL/Q/V1/V2 to ke/k12/k21 and delegates to [`two_compartments`].
///
/// # Assumptions
/// - `p` is a vector of length 4 with CL, Q, V1 and V2 in that order
/// - `rateiv` is a vector of length 1 with the value of the infusion rate (only one drug)
/// - `x` is a vector of length 2
/// - covariates are not used
pub fn two_compartments_cl(x: &V, p: &V, t: T, rateiv: &V, _cov: &Covariates) -> V {
    let cl = p[0];
    let q = p[1];
    let v1 = p[2];
    let v2 = p[3];

    let ke = cl / v1;
    let k12 = q / v1;
    let k21 = q / v2;

    let sqrt = (ke + k12 + k21).powi(2) - 4.0 * ke * k21;
    if sqrt < 0.0 {
        panic!("Imaginary solutions, program stopped!");
    }
    let sqrt = sqrt.sqrt();
    let l1 = (ke + k12 + k21 + sqrt) / 2.0;
    let l2 = (ke + k12 + k21 - sqrt) / 2.0;
    let exp_l1_t = (-l1 * t).exp();
    let exp_l2_t = (-l2 * t).exp();
    let non_zero_matrix = Matrix2::new(
        (l1 - k21) * exp_l1_t + (k21 - l2) * exp_l2_t,
        -k21 * exp_l1_t + k21 * exp_l2_t,
        -k12 * exp_l1_t + k12 * exp_l2_t,
        (l1 - ke - k12) * exp_l1_t + (ke + k12 - l2) * exp_l2_t,
    );

    let non_zero = (non_zero_matrix * x.inner()) / (l1 - l2);

    let infusion_vector = Vector2::new(
        ((l1 - k21) / l1) * (1.0 - exp_l1_t) + ((k21 - l2) / l2) * (1.0 - exp_l2_t),
        (-k12 / l1) * (1.0 - exp_l1_t) + (k12 / l2) * (1.0 - exp_l2_t),
    );

    let infusion = infusion_vector * (rateiv[0] / (l1 - l2));

    let result_vector = non_zero + infusion;

    DVector::from_vec(vec![result_vector[0], result_vector[1]]).into()
}

/// Analytical solution for two compartment model with first-order absorption,
/// parameterized by clearances.
///
/// Converts CL/Q/Vc/Vp to ke/kcp/kpc and delegates to [`two_compartments_with_absorption`].
///
/// # Assumptions
/// - `p` is a vector of length 5 with ka, CL, Q, V2 and V3 in that order
/// - `rateiv` is a vector of length 1 with the value of the infusion rate (only one drug)
/// - `x` is a vector of length 3
/// - covariates are not used
pub fn two_compartments_cl_with_absorption(x: &V, p: &V, t: T, rateiv: &V, _cov: &Covariates) -> V {
    let ka = p[0];
    let cl = p[1];
    let q = p[2];
    let v2 = p[3];
    let v3 = p[4];
    let mut xout = x.clone();

    let ke = cl / v2;
    let k12 = q / v2;
    let k21 = q / v3;

    let sqrt = (ke + k12 + k21).powi(2) - 4.0 * ke * k21;
    if sqrt < 0.0 {
        panic!("Imaginary solutions, program stopped!");
    }
    let sqrt = sqrt.sqrt();
    let l1 = (ke + k12 + k21 + sqrt) / 2.0;
    let l2 = (ke + k12 + k21 - sqrt) / 2.0;

    let exp_l1_t = (-l1 * t).exp();
    let exp_l2_t = (-l2 * t).exp();

    let non_zero_matrix = Matrix2::new(
        (l1 - k21) * exp_l1_t + (k21 - l2) * exp_l2_t,
        -k21 * exp_l1_t + k21 * exp_l2_t,
        -k12 * exp_l1_t + k12 * exp_l2_t,
        (l1 - ke - k12) * exp_l1_t + (ke + k12 - l2) * exp_l2_t,
    );

    let non_zero = (non_zero_matrix * Vector2::new(x[1], x[2])) / (l1 - l2);

    let infusion_vector = Vector2::new(
        ((l1 - k21) / l1) * (1.0 - exp_l1_t) + ((k21 - l2) / l2) * (1.0 - exp_l2_t),
        (-k12 / l1) * (1.0 - exp_l1_t) + (k12 / l2) * (1.0 - exp_l2_t),
    );

    let infusion = infusion_vector * (rateiv[0] / (l1 - l2));

    let exp_ka_t = (-ka * t).exp();

    let absorption_vector = Vector2::new(
        ((l1 - k21) / (ka - l1)) * (exp_l1_t - exp_ka_t)
            + ((k21 - l2) / (ka - l2)) * (exp_l2_t - exp_ka_t),
        (-k12 / (ka - l1)) * (exp_l1_t - exp_ka_t) + (k12 / (ka - l2)) * (exp_l2_t - exp_ka_t),
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
    use super::super::tests::SubjectInfo;
    use super::{two_compartments_cl, two_compartments_cl_with_absorption};
    use crate::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_two_compartments_cl() {
        let infusion_dosing = SubjectInfo::InfusionDosing;
        let subject = infusion_dosing.get_subject();

        let ode = equation::ODE::new(
            |x, p, _t, dx, b, rateiv, _cov| {
                fetch_params!(p, cl, q, v1, v2);

                let ke = cl / v1;
                let k12 = q / v1;
                let k21 = q / v2;

                dx[0] = rateiv[0] - ke * x[0] - k12 * x[0] + k21 * x[1] + b[0];
                dx[1] = k12 * x[0] - k21 * x[1] + b[1];
            },
            |_p, _t, _cov| lag! {},
            |_p, _t, _cov| fa! {},
            |_p, _t, _cov, _x| {},
            |x, p, _t, _cov, y| {
                fetch_params!(p, _cl, _q, v1, _v2);
                y[0] = x[0] / v1;
            },
        )
        .with_nstates(2)
        .with_ndrugs(2)
        .with_nout(1);

        let analytical = equation::Analytical::new(
            two_compartments_cl,
            |_p, _t, _cov| {},
            |_p, _t, _cov| lag! {},
            |_p, _t, _cov| fa! {},
            |_p, _t, _cov, _x| {},
            |x, p, _t, _cov, y| {
                fetch_params!(p, _cl, _q, v1, _v2);
                y[0] = x[0] / v1;
            },
        )
        .with_nstates(2)
        .with_ndrugs(2)
        .with_nout(1);

        let op_ode = ode
            .estimate_predictions(&subject, &vec![0.1, 3.0, 1.0, 3.0])
            .unwrap();
        let op_analytical = analytical
            .estimate_predictions(&subject, &vec![0.1, 3.0, 1.0, 3.0])
            .unwrap();

        let pred_ode = &op_ode.flat_predictions()[..];
        let pred_analytical = &op_analytical.flat_predictions()[..];

        for (&od, &an) in pred_ode.iter().zip(pred_analytical.iter()) {
            assert_relative_eq!(od, an, max_relative = 1e-4, epsilon = 1.0);
        }
    }

    #[test]
    fn test_two_compartments_cl_with_absorption() {
        let oral_infusion_dosing = SubjectInfo::OralInfusionDosage;
        let subject = oral_infusion_dosing.get_subject();

        let ode = equation::ODE::new(
            |x, p, _t, dx, b, rateiv, _cov| {
                fetch_params!(p, ka, cl, q, v1, v2);

                let ke = cl / v1;
                let k12 = q / v1;
                let k21 = q / v2;

                dx[0] = -ka * x[0] + b[0];
                dx[1] = rateiv[0] - ke * x[1] + ka * x[0] - k12 * x[1] + k21 * x[2] + b[1];
                dx[2] = k12 * x[1] - k21 * x[2] + b[2];
            },
            |_p, _t, _cov| lag! {},
            |_p, _t, _cov| fa! {},
            |_p, _t, _cov, _x| {},
            |x, p, _t, _cov, y| {
                fetch_params!(p, _ka, _cl, _q, v1, _v2);
                y[0] = x[1] / v1;
            },
        )
        .with_nstates(3)
        .with_ndrugs(3)
        .with_nout(1);

        let analytical = equation::Analytical::new(
            two_compartments_cl_with_absorption,
            |_p, _t, _cov| {},
            |_p, _t, _cov| lag! {},
            |_p, _t, _cov| fa! {},
            |_p, _t, _cov, _x| {},
            |x, p, _t, _cov, y| {
                fetch_params!(p, _ka, _cl, _q, v1, _v2);
                y[0] = x[1] / v1;
            },
        )
        .with_nstates(3)
        .with_ndrugs(3)
        .with_nout(1);

        let op_ode = ode
            .estimate_predictions(&subject, &vec![1.0, 0.1, 3.0, 1.0, 3.0])
            .unwrap();
        let op_analytical = analytical
            .estimate_predictions(&subject, &vec![1.0, 0.1, 3.0, 1.0, 3.0])
            .unwrap();

        let pred_ode = &op_ode.flat_predictions()[..];
        let pred_analytical = &op_analytical.flat_predictions()[..];

        for (&od, &an) in pred_ode.iter().zip(pred_analytical.iter()) {
            assert_relative_eq!(od, an, max_relative = 1e-3, epsilon = 1e-3);
        }
    }
}
