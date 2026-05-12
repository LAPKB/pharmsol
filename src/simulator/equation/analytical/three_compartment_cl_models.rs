use crate::{data::Covariates, simulator::*};
use diffsol::{NalgebraContext, Vector};

use super::three_compartment_models::{three_compartments, three_compartments_with_absorption};
use super::wrap_pmetrics_analytical;

/// Analytical solution for three compartment model parameterized by clearances.
///
/// Converts CL/Q2/Q3/Vc/V2/V3 to k10/k12/k13/k21/k31 and delegates to [`three_compartments`].
///
/// # Assumptions
/// - `p` is a vector of length 6 with CL, Q2, Q3, Vc, V2 and V3 in that order
/// - `rateiv` is a vector of length 1 with the value of the infusion rate (only one drug)
/// - `x` is a vector of length 3
/// - covariates are not used
pub fn three_compartments_cl(x: &V, p: &V, t: T, rateiv: &V, cov: &Covariates) -> V {
    let cl = p[0];
    let q2 = p[1];
    let q3 = p[2];
    let vc = p[3];
    let v2 = p[4];
    let v3 = p[5];
    let k10 = cl / vc;
    let k12 = q2 / vc;
    let k13 = q3 / vc;
    let k21 = q2 / v2;
    let k31 = q3 / v3;
    let p_ke = V::from_vec(vec![k10, k12, k13, k21, k31], NalgebraContext);
    three_compartments(x, &p_ke, t, rateiv, cov)
}

pub fn pm_three_compartments_cl(x: &V, p: &V, t: T, rateiv: &V, cov: &Covariates) -> V {
    wrap_pmetrics_analytical(x, p, t, rateiv, cov, three_compartments_cl)
}

/// Analytical solution for three compartment model with first-order absorption,
/// parameterized by clearances.
///
/// Converts CL/Q2/Q3/Vc/V2/V3 to k10/k12/k13/k21/k31 and delegates to [`three_compartments_with_absorption`].
///
/// # Assumptions
/// - `p` is a vector of length 7 with ka, CL, Q2, Q3, Vc, V2 and V3 in that order
/// - `rateiv` is a vector of length 1 with the value of the infusion rate (only one drug)
/// - `x` is a vector of length 4
/// - covariates are not used
pub fn three_compartments_cl_with_absorption(
    x: &V,
    p: &V,
    t: T,
    rateiv: &V,
    cov: &Covariates,
) -> V {
    let ka = p[0];
    let cl = p[1];
    let q2 = p[2];
    let q3 = p[3];
    let vc = p[4];
    let v2 = p[5];
    let v3 = p[6];
    let k10 = cl / vc;
    let k12 = q2 / vc;
    let k13 = q3 / vc;
    let k21 = q2 / v2;
    let k31 = q3 / v3;
    let p_ke = V::from_vec(vec![ka, k10, k12, k13, k21, k31], NalgebraContext);
    three_compartments_with_absorption(x, &p_ke, t, rateiv, cov)
}

pub fn pm_three_compartments_cl_with_absorption(
    x: &V,
    p: &V,
    t: T,
    rateiv: &V,
    cov: &Covariates,
) -> V {
    wrap_pmetrics_analytical(x, p, t, rateiv, cov, three_compartments_cl_with_absorption)
}

#[cfg(test)]
mod tests {
    use super::super::tests::SubjectInfo;
    use super::{three_compartments_cl, three_compartments_cl_with_absorption};
    use crate::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_three_compartments_cl() {
        let infusion_dosing = SubjectInfo::InfusionDosing;
        let subject = infusion_dosing.get_subject();

        // CL=0.1, Q2=3.0, Q3=2.0, Vc=1.0, V2=3.0, V3=4.0
        // => k10=0.1, k12=3.0, k13=2.0, k21=1.0, k31=0.5

        let ode = equation::ODE::new(
            |x, p, _t, dx, b, rateiv, _cov| {
                fetch_params!(p, cl, q2, q3, vc, v2, v3);
                let k10 = cl / vc;
                let k12 = q2 / vc;
                let k13 = q3 / vc;
                let k21 = q2 / v2;
                let k31 = q3 / v3;

                dx[0] = rateiv[0] - (k10 + k12 + k13) * x[0] + k21 * x[1] + k31 * x[2] + b[0];
                dx[1] = k12 * x[0] - k21 * x[1] + b[1];
                dx[2] = k13 * x[0] - k31 * x[2] + b[2];
            },
            |_p, _t, _cov| lag! {},
            |_p, _t, _cov| fa! {},
            |_p, _t, _cov, _x| {},
            |x, p, _t, _cov, y| {
                fetch_params!(p, _cl, _q2, _q3, vc, _v2, _v3);
                y[0] = x[0] / vc;
            },
        )
        .with_nstates(3)
        .with_nout(1)
        .with_ndrugs(3);

        let analytical = equation::Analytical::new(
            three_compartments_cl,
            |_p, _t, _cov| {},
            |_p, _t, _cov| lag! {},
            |_p, _t, _cov| fa! {},
            |_p, _t, _cov, _x| {},
            |x, p, _t, _cov, y| {
                fetch_params!(p, _cl, _q2, _q3, vc, _v2, _v3);
                y[0] = x[0] / vc;
            },
        )
        .with_nstates(3)
        .with_nout(1)
        .with_ndrugs(3);

        let op_ode = ode
            .estimate_predictions(
                &subject,
                &crate::Parameters::dense([0.1, 3.0, 2.0, 1.0, 3.0, 4.0]),
            )
            .unwrap();
        let op_analytical = analytical
            .estimate_predictions(
                &subject,
                &crate::Parameters::dense([0.1, 3.0, 2.0, 1.0, 3.0, 4.0]),
            )
            .unwrap();

        let pred_ode = &op_ode.flat_predictions()[..];
        let pred_analytical = &op_analytical.flat_predictions()[..];

        for (&od, &an) in pred_ode.iter().zip(pred_analytical.iter()) {
            assert_relative_eq!(od, an, max_relative = 1e-3, epsilon = 1e-3);
        }
    }

    #[test]
    fn test_three_compartments_cl_with_absorption() {
        let oral_infusion_dosing = SubjectInfo::OralInfusionDosage;
        let subject = oral_infusion_dosing.get_subject();

        // ka=1.0, CL=0.1, Q2=3.0, Q3=2.0, Vc=1.0, V2=3.0, V3=4.0
        // => k10=0.1, k12=3.0, k13=2.0, k21=1.0, k31=0.5

        let ode = equation::ODE::new(
            |x, p, _t, dx, b, rateiv, _cov| {
                fetch_params!(p, ka, cl, q2, q3, vc, v2, v3);
                let k10 = cl / vc;
                let k12 = q2 / vc;
                let k13 = q3 / vc;
                let k21 = q2 / v2;
                let k31 = q3 / v3;

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
                fetch_params!(p, _ka, _cl, _q2, _q3, vc, _v2, _v3);
                y[0] = x[1] / vc;
            },
        );

        let analytical = equation::Analytical::new(
            three_compartments_cl_with_absorption,
            |_p, _t, _cov| {},
            |_p, _t, _cov| lag! {},
            |_p, _t, _cov| fa! {},
            |_p, _t, _cov, _x| {},
            |x, p, _t, _cov, y| {
                fetch_params!(p, _ka, _cl, _q2, _q3, vc, _v2, _v3);
                y[0] = x[1] / vc;
            },
        );

        let op_ode = ode
            .estimate_predictions(
                &subject,
                &crate::Parameters::dense([1.0, 0.1, 3.0, 2.0, 1.0, 3.0, 4.0]),
            )
            .unwrap();
        let op_analytical = analytical
            .estimate_predictions(
                &subject,
                &crate::Parameters::dense([1.0, 0.1, 3.0, 2.0, 1.0, 3.0, 4.0]),
            )
            .unwrap();

        let pred_ode = &op_ode.flat_predictions()[..];
        let pred_analytical = &op_analytical.flat_predictions()[..];

        for (&od, &an) in pred_ode.iter().zip(pred_analytical.iter()) {
            assert_relative_eq!(od, an, max_relative = 1e-3, epsilon = 1e-3);
        }
    }
}
