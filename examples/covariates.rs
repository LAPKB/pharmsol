fn main() {
    use pharmsol::{prelude::*, Parameters};

    let ode = ode! {
        name: "one_cmt_covariates",
        params: [ka, ke, tlag, v],
        covariates: [creatinine, age],
        states: [gut, central],
        outputs: [cp],
        routes: [
            bolus(oral) -> gut,
        ],
        diffeq: |x, _t, dx| {
            let scaled_ke = ke * (creatinine / 75.0).powf(0.75) * (age / 25.0).powf(0.5);

            dx[gut] = -ka * x[gut];
            dx[central] = ka * x[gut] - scaled_ke * x[central];
        },
        // This blocks defines the lag-time of the bolus dose
        lag: |_t| {
            // Macro used to define the lag-time for the input of the bolus dose
            lag! { oral => tlag }
        },
        out: |x, _t, y| {
            // Define the predicted concentration as the amount in the central compartment divided by volume
            y[cp] = x[central] / v;
        },
    };

    // Create a subject using route and output labels directly.
    let subject = Subject::builder("id1")
        .bolus(0.0, 100.0, "oral")
        .repeat(2, 2.0)
        .observation(0.5, 0.1, "cp")
        .observation(1.0, 0.4, "cp")
        .observation(2.0, 1.0, "cp")
        .observation(2.5, 1.1, "cp")
        .covariate("creatinine", 0.0, 80.0)
        .covariate("creatinine", 1.0, 40.0)
        .covariate("age", 0.0, 25.0)
        .missing_observation(8.0, "cp")
        .build();

    // Define parameter values
    let ka = 1.0; // Absorption rate constant
    let ke = 0.2; // Elimination rate constant
    let tlag = 0.0; // Lag time
    let v = 70.0; // Volume of distribution
    let params = Parameters::with_model(&ode, [("ka", ka), ("ke", ke), ("tlag", tlag), ("v", v)])
        .expect("valid named parameters");

    let result = ode.estimate_predictions(&subject, &params).unwrap();

    for pred in result.predictions() {
        println!(
            "Time: {:.2} h, Predicted Concentration: {:.4} units",
            pred.time(),
            pred.prediction()
        );
    }
}
