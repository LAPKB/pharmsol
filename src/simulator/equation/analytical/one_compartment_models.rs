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
    use crate::prelude::simulator::Prediction;
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

    /// Helper function to compare two vectors of predictions for testing
    fn compare_predictions(
        ode_predictions: &[Prediction],
        analytical_predictions: &[Prediction],
        tolerance: f64,
    ) {
        assert_eq!(
            ode_predictions.len(),
            analytical_predictions.len(),
            "Prediction vectors must have the same length"
        );

        for (ode_pred, analytical_pred) in ode_predictions.iter().zip(analytical_predictions.iter())
        {
            // Check that times match
            assert_relative_eq!(ode_pred.time(), analytical_pred.time(), epsilon = 1e-8,);

            // Check that observations match
            assert_eq!(
                ode_pred.observation(),
                analytical_pred.observation(),
                "Observations do not match at time {}",
                ode_pred.time()
            );

            // Check that outeq matches
            assert_eq!(
                ode_pred.outeq(),
                analytical_pred.outeq(),
                "Output equations do not match at time {}",
                ode_pred.time()
            );

            // Check that prediction values match within tolerance
            let ode_val = ode_pred.prediction();
            let analytical_val = analytical_pred.prediction();
            assert_relative_eq!(
                ode_val,
                analytical_val,
                max_relative = tolerance,
                epsilon = 1.0
            );
        }
    }

    // Function to pretty-print comparison of predictions
    fn print_comparison(ode_predictions: &[Prediction], analytical_predictions: &[Prediction]) {
        println!("Time\t\tAnalytical\tODE\t\tDelta");
        for (analytical_pred, ode_pred) in analytical_predictions.iter().zip(ode_predictions.iter())
        {
            let analytical_val = analytical_pred.prediction();
            let ode_val = ode_pred.prediction();
            let delta = (analytical_val - ode_val).abs();
            let time = analytical_pred.time();
            println!(
                "{:.2}\t\t{:.6}\t{:.6}\t{:.6}",
                time, analytical_val, ode_val, delta
            );
        }
    }

    #[test]
    fn test_one_compartment() {
        let infusion_dosing = SubjectInfo::InfusionDosing;
        let subject = infusion_dosing.get_subject();

        let ode = equation::ODE::new(
            |x, p, _t, dx, rateiv, _cov, bolus| {
                fetch_params!(p, ke, _v);

                dx[0] = -ke * x[0] + rateiv[0] + bolus[0];
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

        let ode_predictions = op_ode.predictions();
        let analytical_predictions = op_analytical.predictions();

        print_comparison(ode_predictions, analytical_predictions);
        compare_predictions(ode_predictions, analytical_predictions, 1e-4);
    }

    #[test]
    fn test_one_compartment_with_absorption() {
        let oral_infusion_dosing = SubjectInfo::OralInfusionDosage;
        let subject = oral_infusion_dosing.get_subject();

        let ode = equation::ODE::new(
            |x, p, _t, dx, rateiv, _cov, bolus| {
                fetch_params!(p, ka, ke, _v);

                dx[0] = -ka * x[0] + bolus[0];
                dx[1] = ka * x[0] - ke * x[1] + rateiv[0] + bolus[1];
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

        let ode_predictions = op_ode.predictions();
        let analytical_predictions = op_analytical.predictions();

        print_comparison(ode_predictions, analytical_predictions);
        compare_predictions(ode_predictions, analytical_predictions, 1e-4);
    }
}
