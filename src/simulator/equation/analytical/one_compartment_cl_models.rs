use crate::{data::Covariates, simulator::*};
use diffsol::{NalgebraContext, Vector};

use super::one_compartment_models::{one_compartment, one_compartment_with_absorption};
use super::wrap_pmetrics_analytical;

/// Analytical solution for one compartment model parameterized by clearance.
///
/// Converts CL/V to ke and delegates to [`one_compartment`].
///
/// # Assumptions
/// - `p` is a vector of length 2 with CL and V in that order
/// - `rateiv` is a vector of length 1 with the value of the infusion rate (only one drug)
/// - `x` is a vector of length 1
/// - covariates are not used
pub fn one_compartment_cl(x: &V, p: &V, t: T, rateiv: &V, cov: &Covariates) -> V {
    let cl = p[0];
    let v = p[1];
    let ke = cl / v;
    let p_ke = V::from_vec(vec![ke], NalgebraContext);
    one_compartment(x, &p_ke, t, rateiv, cov)
}

pub fn pm_one_compartment_cl(x: &V, p: &V, t: T, rateiv: &V, cov: &Covariates) -> V {
    wrap_pmetrics_analytical(x, p, t, rateiv, cov, one_compartment_cl)
}

/// Analytical solution for one compartment model with first-order absorption,
/// parameterized by clearance.
///
/// Converts CL/V to ke and delegates to [`one_compartment_with_absorption`].
///
/// # Assumptions
/// - `p` is a vector of length 3 with ka, CL and V in that order
/// - `rateiv` is a vector of length 1 with the value of the infusion rate (only one drug)
/// - `x` is a vector of length 2
/// - covariates are not used
pub fn one_compartment_cl_with_absorption(x: &V, p: &V, t: T, rateiv: &V, cov: &Covariates) -> V {
    let ka = p[0];
    let cl = p[1];
    let v = p[2];
    let ke = cl / v;
    let p_ke = V::from_vec(vec![ka, ke], NalgebraContext);
    one_compartment_with_absorption(x, &p_ke, t, rateiv, cov)
}

pub fn pm_one_compartment_cl_with_absorption(
    x: &V,
    p: &V,
    t: T,
    rateiv: &V,
    cov: &Covariates,
) -> V {
    wrap_pmetrics_analytical(x, p, t, rateiv, cov, one_compartment_cl_with_absorption)
}

#[cfg(test)]
mod tests {
    use super::super::tests::SubjectInfo;
    use super::{one_compartment_cl, one_compartment_cl_with_absorption};
    use crate::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_one_compartment_cl() {
        let infusion_dosing = SubjectInfo::InfusionDosing;
        let subject = infusion_dosing.get_subject();

        let ode = equation::ODE::new(
            |x, p, _t, dx, b, rateiv, _cov| {
                fetch_params!(p, cl, v);
                let ke = cl / v;

                dx[0] = -ke * x[0] + rateiv[0] + b[0];
            },
            |_p, _t, _cov| lag! {},
            |_p, _t, _cov| fa! {},
            |_p, _t, _cov, _x| {},
            |x, p, _t, _cov, y| {
                fetch_params!(p, _cl, v);
                y[0] = x[0] / v;
            },
        )
        .with_nstates(1)
        .with_nout(1);

        let analytical = equation::Analytical::new(
            one_compartment_cl,
            |_p, _t, _cov| {},
            |_p, _t, _cov| lag! {},
            |_p, _t, _cov| fa! {},
            |_p, _t, _cov, _x| {},
            |x, p, _t, _cov, y| {
                fetch_params!(p, _cl, v);
                y[0] = x[0] / v;
            },
        )
        .with_nstates(1)
        .with_nout(1);

        let op_ode = ode
            .estimate_predictions(&subject, &crate::parameters::dense([0.1, 1.0]))
            .unwrap();
        let op_analytical = analytical
            .estimate_predictions(&subject, &crate::parameters::dense([0.1, 1.0]))
            .unwrap();

        let pred_ode = &op_ode.flat_predictions()[..];
        let pred_analytical = &op_analytical.flat_predictions()[..];

        for (&od, &an) in pred_ode.iter().zip(pred_analytical.iter()) {
            assert_relative_eq!(od, an, max_relative = 1e-4, epsilon = 1.0,);
        }
    }

    #[test]
    fn test_one_compartment_cl_with_absorption() {
        let oral_infusion_dosing = SubjectInfo::OralInfusionDosage;
        let subject = oral_infusion_dosing.get_subject();

        let ode = equation::ODE::new(
            |x, p, _t, dx, b, rateiv, _cov| {
                fetch_params!(p, ka, cl, v);
                let ke = cl / v;

                dx[0] = -ka * x[0] + b[0];
                dx[1] = ka * x[0] - ke * x[1] + rateiv[0] + b[1];
            },
            |_p, _t, _cov| lag! {},
            |_p, _t, _cov| fa! {},
            |_p, _t, _cov, _x| {},
            |x, p, _t, _cov, y| {
                fetch_params!(p, _ka, _cl, v);
                y[0] = x[1] / v;
            },
        )
        .with_nstates(2)
        .with_nout(1);

        let analytical = equation::Analytical::new(
            one_compartment_cl_with_absorption,
            |_p, _t, _cov| {},
            |_p, _t, _cov| lag! {},
            |_p, _t, _cov| fa! {},
            |_p, _t, _cov, _x| {},
            |x, p, _t, _cov, y| {
                fetch_params!(p, _ka, _cl, v);
                y[0] = x[1] / v;
            },
        )
        .with_nstates(2)
        .with_nout(1);

        let op_ode = ode
            .estimate_predictions(&subject, &crate::parameters::dense([1.0, 0.1, 1.0]))
            .unwrap();
        let op_analytical = analytical
            .estimate_predictions(&subject, &crate::parameters::dense([1.0, 0.1, 1.0]))
            .unwrap();

        let pred_ode = &op_ode.flat_predictions()[..];
        let pred_analytical = &op_analytical.flat_predictions()[..];

        for (&od, &an) in pred_ode.iter().zip(pred_analytical.iter()) {
            assert_relative_eq!(od, an, max_relative = 1e-4, epsilon = 1.0);
        }
    }
}
