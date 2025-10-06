use crate::{data::Covariates, simulator::*};

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

#[cfg(test)]
mod tests {
    use super::{one_compartment, one_compartment_with_absorption};
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
    fn test_one_compartment() {
        let infusion_dosing = SubjectInfo::InfusionDosing;
        let subject = infusion_dosing.get_subject();

        let ode = equation::ODE::new(
            |x, p, _t, dx, b, rateiv, _cov| {
                fetch_params!(p, ke, _v);

                dx[0] = -ke * x[0] + rateiv[0] + b[0];
            },
            |_p, _t, _cov| lag! {},
            |_p, _t, _cov| fa! {},
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
            |_p, _t, _cov| lag! {},
            |_p, _t, _cov| fa! {},
            |_p, _t, _cov, _x| {},
            |x, p, _t, _cov, y| {
                fetch_params!(p, _ke, v);
                y[0] = x[0] / v;
            },
            (1, 1),
        );

        let op_ode = ode.estimate_predictions(&subject, &vec![0.1, 1.0]).unwrap();
        let op_analytical = analytical
            .estimate_predictions(&subject, &vec![0.1, 1.0])
            .unwrap();

        let pred_ode = &op_ode.flat_predictions()[..];
        let pred_analytical = &op_analytical.flat_predictions()[..];

        println!("ode: {:?}", pred_ode);
        println!("analitycal: {:?}", pred_analytical);

        for (&od, &an) in pred_ode.iter().zip(pred_analytical.iter()) {
            assert_relative_eq!(od, an, max_relative = 1e-4, epsilon = 1.0,);
        }
    }

    #[test]
    fn test_one_compartment_with_absorption() {
        let oral_infusion_dosing = SubjectInfo::OralInfusionDosage;
        let subject = oral_infusion_dosing.get_subject();

        let ode = equation::ODE::new(
            |x, p, _t, dx, b, rateiv, _cov| {
                fetch_params!(p, ka, ke, _v);

                dx[0] = -ka * x[0] + b[0];
                dx[1] = ka * x[0] - ke * x[1] + rateiv[0] + b[1];
            },
            |_p, _t, _cov| lag! {},
            |_p, _t, _cov| fa! {},
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
            |_p, _t, _cov| lag! {},
            |_p, _t, _cov| fa! {},
            |_p, _t, _cov, _x| {},
            |x, p, _t, _cov, y| {
                fetch_params!(p, _ka, _ke, v);
                y[0] = x[1] / v;
            },
            (2, 1),
        );

        let op_ode = ode
            .estimate_predictions(&subject, &vec![1.0, 0.1, 1.0])
            .unwrap();
        let op_analytical = analytical
            .estimate_predictions(&subject, &vec![1.0, 0.1, 1.0])
            .unwrap();

        let pred_ode = &op_ode.flat_predictions()[..];
        let pred_analytical = &op_analytical.flat_predictions()[..];

        println!("ode: {:?}", pred_ode);
        println!("analitycal: {:?}", pred_analytical);

        for (&od, &an) in pred_ode.iter().zip(pred_analytical.iter()) {
            assert_relative_eq!(od, an, max_relative = 1e-4, epsilon = 1.0);
        }
    }
}
