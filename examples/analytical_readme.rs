fn main() -> Result<(), pharmsol::PharmsolError> {
    use pharmsol::{prelude::*, Parameters};

    let analytical = analytical! {
        name: "one_cmt_oral",
        params: [ka, ke0, v],
        derived: [ke],
        covariates: [wt],
        states: [gut, central],
        outputs: [cp],
        routes: [
            bolus(oral) -> gut,
        ],
        structure: one_compartment_with_absorption,
        derive: |_t| {
            ke = ke0 * (wt / 70.0).powf(0.75);
        },
        out: |x, _t, y| {
            y[cp] = x[central] / v;
        },
    };

    let subject = Subject::builder("analytical_readme")
        .bolus(0.0, 500.0, "oral")
        .missing_observation(0.5, "cp")
        .missing_observation(1.0, "cp")
        .missing_observation(2.0, "cp")
        .missing_observation(4.0, "cp")
        .covariate("wt", 0.0, 75.0)
        .build();

    let parameters =
        Parameters::with_model(&analytical, [("ka", 1.2), ("ke0", 0.08), ("v", 194.0)])?;
    let predictions = analytical.estimate_predictions(&subject, &parameters)?;

    println!("times => {:?}", predictions.flat_times());
    println!("predictions => {:?}", predictions.flat_predictions());

    Ok(())
}
