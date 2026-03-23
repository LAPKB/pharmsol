use crate::{data::Covariates, simulator::*};
use diffsol::{NalgebraContext, Vector};

use super::two_compartment_models::{two_compartments, two_compartments_with_absorption};

/// Analytical solution for two compartment model parameterized by clearances.
///
/// Converts CL/Q/Vc/Vp to ke/kcp/kpc and delegates to [`two_compartments`].
///
/// # Assumptions
/// - `p` is a vector of length 4 with CL, Q, Vc and Vp in that order
/// - `rateiv` is a vector of length 1 with the value of the infusion rate (only one drug)
/// - `x` is a vector of length 2
/// - covariates are not used
pub fn two_compartments_cl(x: &V, p: &V, t: T, rateiv: V, cov: &Covariates) -> V {
    let cl = p[0];
    let q = p[1];
    let vc = p[2];
    let vp = p[3];
    let ke = cl / vc;
    let kcp = q / vc;
    let kpc = q / vp;
    let p_ke = V::from_vec(vec![ke, kcp, kpc], NalgebraContext);
    two_compartments(x, &p_ke, t, rateiv, cov)
}

/// Analytical solution for two compartment model with first-order absorption,
/// parameterized by clearances.
///
/// Converts CL/Q/Vc/Vp to ke/kcp/kpc and delegates to [`two_compartments_with_absorption`].
///
/// # Assumptions
/// - `p` is a vector of length 5 with ka, CL, Q, Vc and Vp in that order
/// - `rateiv` is a vector of length 1 with the value of the infusion rate (only one drug)
/// - `x` is a vector of length 3
/// - covariates are not used
pub fn two_compartments_cl_with_absorption(x: &V, p: &V, t: T, rateiv: V, cov: &Covariates) -> V {
    let ka = p[0];
    let cl = p[1];
    let q = p[2];
    let vc = p[3];
    let vp = p[4];
    let ke = cl / vc;
    let kcp = q / vc;
    let kpc = q / vp;
    let p_ke = V::from_vec(vec![ke, ka, kcp, kpc], NalgebraContext);
    two_compartments_with_absorption(x, &p_ke, t, rateiv, cov)
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
                fetch_params!(p, cl, q, vc, vp);

                let ke = cl / vc;
                let kcp = q / vc;
                let kpc = q / vp;

                dx[0] = rateiv[0] - ke * x[0] - kcp * x[0] + kpc * x[1] + b[0];
                dx[1] = kcp * x[0] - kpc * x[1] + b[1];
            },
            |_p, _t, _cov| lag! {},
            |_p, _t, _cov| fa! {},
            |_p, _t, _cov, _x| {},
            |x, p, _t, _cov, y| {
                fetch_params!(p, _cl, _q, vc, _vp);
                y[0] = x[0] / vc;
            },
            (2, 1),
        );

        let analytical = equation::Analytical::new(
            two_compartments_cl,
            |_p, _t, _cov| {},
            |_p, _t, _cov| lag! {},
            |_p, _t, _cov| fa! {},
            |_p, _t, _cov, _x| {},
            |x, p, _t, _cov, y| {
                fetch_params!(p, _cl, _q, vc, _vp);
                y[0] = x[0] / vc;
            },
            (2, 1),
        );

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
                fetch_params!(p, ka, cl, q, vc, vp);

                let ke = cl / vc;
                let kcp = q / vc;
                let kpc = q / vp;

                dx[0] = -ka * x[0] + b[0];
                dx[1] = rateiv[0] - ke * x[1] + ka * x[0] - kcp * x[1] + kpc * x[2] + b[1];
                dx[2] = kcp * x[1] - kpc * x[2] + b[2];
            },
            |_p, _t, _cov| lag! {},
            |_p, _t, _cov| fa! {},
            |_p, _t, _cov, _x| {},
            |x, p, _t, _cov, y| {
                fetch_params!(p, _ka, _cl, _q, vc, _vp);
                y[0] = x[1] / vc;
            },
            (3, 1),
        );

        let analytical = equation::Analytical::new(
            two_compartments_cl_with_absorption,
            |_p, _t, _cov| {},
            |_p, _t, _cov| lag! {},
            |_p, _t, _cov| fa! {},
            |_p, _t, _cov, _x| {},
            |x, p, _t, _cov, y| {
                fetch_params!(p, _ka, _cl, _q, vc, _vp);
                y[0] = x[1] / vc;
            },
            (3, 1),
        );

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
