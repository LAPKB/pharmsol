fn main() -> Result<(), pharmsol::PharmsolError> {
    use pharmsol::*;

    // Create a subject using the builder pattern
    let subject = Subject::builder("Nikola Tesla")
        // An initial infusion of 500 units over 0.5 time units
        .infusion(0., 500.0, 0, 0.5)
        // Observations at various time points
        .observation(0.5, 1.645, 0)
        .observation(1., 1.216, 0)
        .observation(2., 0.462, 0)
        .observation(3., 0.169, 0)
        .observation(4., 0.063, 0)
        .observation(6., 0.009, 0)
        .observation(8., 0.001, 0)
        // A missing observation, to force the simulator to predict to this time point
        .missing_observation(12.0, 0)
        // Build the subject
        .build();

    // Define the one-compartment analytical solution function
    let an = equation::Analytical::new(
        one_compartment,
        |_p, _t, _cov| {},
        |_p, _t, _cov| lag! {},
        |_p, _t, _cov| fa! {},
        |_p, _t, _cov, _x| {},
        |x, p, _t, _cov, y| {
            fetch_params!(p, _ke, v);
            // Calculate the output concentration, here defined as amount over volume
            y[0] = x[0] / v;
        },
        (1, 1),
    );

    let ode = equation::ODE::new(
        |x, p, _t, dx, _b, rateiv, _cov| {
            // Macro to fetch parameters from the parameter vector
            // This exposes them as local variables
            fetch_params!(p, ke, _v);

            // Define the ODE for the one-compartment model
            // Note that rateiv is used to include infusion rates
            dx[0] = -ke * x[0] + rateiv[0];
        },
        |_p, _t, _cov| lag! {},
        |_p, _t, _cov| fa! {},
        |_p, _t, _cov, _x| {},
        |x, p, _t, _cov, y| {
            fetch_params!(p, _ke, v);
            // Calculate the output concentration, here defined as amount over volume
            y[0] = x[0] / v;
        },
        (1, 1),
    );

    // Define the error models for the observations
    let ems = ErrorModels::new().
    // For this example, we use a simple additive error model with 5% error
    add(
        0,
        ErrorModel::additive(ErrorPoly::new(0.0, 0.05, 0.0, 0.0), 0.0),
    )?;

    // Define the parameter values for the simulations
    let ke = 1.022; // Elimination rate constant
    let v = 194.0; // Volume of distribution

    // Compute likelihoods and predictions for both models
    let analytical_likelihoods = an
        .estimate_likelihood(&subject, &vec![ke, v], &ems, false)
        .unwrap();

    let analytical_predictions = an.estimate_predictions(&subject, &vec![ke, v])?;

    let ode_likelihoods = ode.estimate_likelihood(&subject, &vec![ke, v], &ems, false)?;

    let ode_predictions = ode.estimate_predictions(&subject, &vec![ke, v])?;

    // Print comparison table
    println!("\n┌───────────┬─────────────────┬─────────────────┬─────────────────────┐");
    println!("│           │   Analytical    │       ODE       │     Difference      │");
    println!("├───────────┼─────────────────┼─────────────────┼─────────────────────┤");
    println!(
        "│ Likelihood│ {:>15.6} │ {:>15.6} │ {:>19.2e} │",
        analytical_likelihoods,
        ode_likelihoods,
        analytical_likelihoods - ode_likelihoods
    );
    println!("├───────────┼─────────────────┼─────────────────┼─────────────────────┤");
    println!("│   Time    │   Prediction    │   Prediction    │                     │");
    println!("├───────────┼─────────────────┼─────────────────┼─────────────────────┤");

    let times = analytical_predictions.flat_times();
    let analytical_preds = analytical_predictions.flat_predictions();
    let ode_preds = ode_predictions.flat_predictions();

    for ((t, a), b) in times
        .iter()
        .zip(analytical_preds.iter())
        .zip(ode_preds.iter())
    {
        let diff = a - b;
        println!(
            "│ {:>9.2} │ {:>15.9} │ {:>15.9} │ {:>19.2e} │",
            t, a, b, diff
        );
    }

    println!("└───────────┴─────────────────┴─────────────────┴─────────────────────┘\n");

    Ok(())
}
