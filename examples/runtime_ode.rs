fn main() -> Result<(), pharmsol::PharmsolError> {
    use pharmsol::prelude::*;
    use pharmsol::RuntimeODE;

    let subject = Subject::builder("runtime_subject")
        .infusion(0.0, 500.0, 0, 0.5)
        .covariate("wt", 0.0, 70.0)
        .covariate("wt", 1.0, 84.0)
        .observation(0.5, 1.645, 0)
        .observation(1.0, 1.216, 0)
        .observation(2.0, 0.462, 0)
        .observation(4.0, 0.063, 0)
        .build();

    // Runtime-defined ODE model. No Rust code generation or model compilation required.
    let model_json = r#"
    {
      "states": ["A"],
      "parameters": ["ke", "v"],
      "outputs": ["cp"],
      "covariates": ["wt"],
      "derivatives": ["-(ke * wt / 70.0) * A + rateiv[0]"],
      "output_equations": ["A / v"]
    }
    "#;

    let runtime_ode = RuntimeODE::from_json(model_json)?;

    let params = vec![1.022, 194.0];
    let predictions = runtime_ode.estimate_predictions(&subject, &params)?;

    println!("Runtime ODE predictions:");
    println!("time\tprediction");
    for prediction in predictions.predictions() {
        println!("{:.2}\t{:.6}", prediction.time(), prediction.prediction());
    }

    Ok(())
}
