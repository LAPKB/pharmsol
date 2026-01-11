fn main() {
    use pharmsol::prelude::*;

    // Create a subject with a bolus dose, observations, and covariates
    let subject = Subject::builder("id1")
        // Administer a bolus dose of 100 units at time 0
        .bolus(0.0, 100.0, 0)
        // Give two additional doses at 2-hour intervals
        .repeat(2, 2.0)
        .observation(0.5, 0.1, 0)
        .observation(1.0, 0.4, 0)
        .observation(2.0, 1.0, 0)
        .observation(2.5, 1.1, 0)
        // Creatinine covariate changes over time, with initial value of 80 at time 0
        .covariate("creatinine", 0.0, 80.0)
        // New obseration of creatinine at time 6 hours
        // The value will be linearly interpolated between time 0 and time 6
        .covariate("creatinine", 1.0, 40.0)
        // For age, the covariate is constant over time, as there are no changes
        .covariate("age", 0.0, 25.0)
        .missing_observation(8.0, 0)
        .build();

    let ode = equation::ODE::new(
        |x, p, t, dx, b, _rateiv, cov| {
            // Macro to get the (possibly interpolated) covariate values at time `t`
            fetch_cov!(cov, t, creatinine, age);
            // Macro to fetch parameter values from `p`
            // Note the order must match the order in which parameters are defined later
            fetch_params!(p, ka, ke, _tlag, _v);

            let ke = ke * (creatinine / 75.0).powf(0.75) * (age / 25.0).powf(0.5);

            //Struct
            dx[0] = -ka * x[0] + b[0];
            dx[1] = ka * x[0] - ke * x[1];
        },
        // This blocks defines the lag-time of the bolus dose
        |p, _t, _cov| {
            fetch_params!(p, _ka, _ke, tlag, _v);
            // Macro used to define the lag-time for the input of the bolus dose
            lag! {0=>tlag}
        },
        |_p, _t, _cov| fa! {},
        |_p, _t, _cov, _x| {},
        |x, p, _t, _cov, y| {
            fetch_params!(p, _ka, _ke, _tlag, v);

            // Define the predicted concentration as the amount in the central compartment divided by volume
            y[0] = x[1] / v;
        },
        (2, 1),
    );

    // Define parameter values
    // Note that the order matters and should correspond to the order in which parameters are fetched in the model
    // This is subject to change in future versions
    let ka = 1.0; // Absorption rate constant
    let ke = 0.2; // Elimination rate constant
    let tlag = 0.0; // Lag time
    let v = 70.0; // Volume of distribution
    let params = vec![ka, ke, tlag, v];

    let result = ode.estimate_predictions(&subject, &params).unwrap();

    for pred in result.predictions() {
        println!(
            "Time: {:.2} h, Predicted Concentration: {:.4} units",
            pred.time(),
            pred.prediction()
        );
    }
}
