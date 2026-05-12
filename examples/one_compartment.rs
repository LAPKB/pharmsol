fn main() -> Result<(), pharmsol::PharmsolError> {
    use pharmsol::{Parameters, prelude::*};

    let analytical = analytical! {
        name: "one_cmt_iv",
        params: [ke, v],
        states: [central],
        outputs: [cp],
        routes: [
            infusion(iv) -> central,
        ],
        structure: one_compartment,
        out: |x, _p, _t, _cov, y| {
            y[cp] = x[central] / v;
        },
    };

    let ode = ode! {
        name: "one_cmt_iv",
        params: [ke, v],
        states: [central],
        outputs: [cp],
        routes: [
            infusion(iv) -> central,
        ],
        diffeq: |x, _p, _t, dx, _cov| {
            dx[central] = -ke * x[central];
        },
        out: |x, _p, _t, _cov, y| {
            y[cp] = x[central] / v;
        },
    };

    // Create a subject using route and output labels directly.
    let subject = Subject::builder("Nikola Tesla")
        .infusion(0., 500.0, "iv", 0.5)
        .observation(0.5, 1.645, "cp")
        .observation(1., 1.216, "cp")
        .observation(2., 0.462, "cp")
        .observation(3., 0.169, "cp")
        .observation(4., 0.063, "cp")
        .observation(6., 0.009, "cp")
        .observation(8., 0.001, "cp")
        .missing_observation(12.0, "cp")
        .build();

    // Define the assay error models once by label and reuse them across both
    // equations.
    let ems = AssayErrorModels::new().add(
        "cp",
        AssayErrorModel::additive(ErrorPoly::new(0.0, 0.05, 0.0, 0.0), 0.0),
    )?;

    // Define the parameter values for the simulations
    let ke = 1.022; // Elimination rate constant
    let v = 194.0; // Volume of distribution
    let analytical_parameters = Parameters::with_model(&analytical, [("ke", ke), ("v", v)])
        .expect("valid named parameters");
    let ode_parameters = Parameters::with_model(&ode, [("ke", ke), ("v", v)])
        .expect("valid named parameters");

    // Compute likelihoods and predictions for both models
    let analytical_likelihoods =
        analytical.estimate_log_likelihood(&subject, &analytical_parameters, &ems)?;

    let analytical_predictions = analytical.estimate_predictions(&subject, &analytical_parameters)?;

    let ode_likelihoods = ode.estimate_log_likelihood(&subject, &ode_parameters, &ems)?;

    let ode_predictions = ode.estimate_predictions(&subject, &ode_parameters)?;

    // Print comparison table
    println!("\n┌───────────┬─────────────────┬─────────────────┬─────────────────────┐");
    println!("│           │   Analytical    │       ODE       │     Difference      │");
    println!("├───────────┼─────────────────┼─────────────────┼─────────────────────┤");
    println!(
        "│ Log-Likeli│ {:>15.6} │ {:>15.6} │ {:>19.2e} │",
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
