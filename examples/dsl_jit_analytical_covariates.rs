//! Run with:
//! cargo run --example dsl_jit_analytical_covariates --features dsl-jit

#[cfg(feature = "dsl-jit")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use std::io;

    use pharmsol::{prelude::*, Parameters};

    let model_source = r#"
name = one_cmt_covariates
kind = analytical

params = ka, ke0, v
covariates = wt@linear
derived = ke
states = gut, central
outputs = cp

bolus(oral) -> gut

ke = ke0 * pow(wt / 70.0, 0.75)
structure = one_compartment_with_absorption
out(cp) = central / v ~ continuous()
"#;

    let jit_model = pharmsol::dsl::compile_module_source_to_runtime(
        model_source,
        Some("one_cmt_covariates"),
        pharmsol::dsl::RuntimeCompilationTarget::Jit,
        |_kind, _message| {},
    )?;

    let analytical = analytical! {
        name: "one_cmt_covariates",
        params: [ke0, ka, v],
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

    let subject = Subject::builder("covariates")
        .bolus(0.0, 100.0, "oral")
        .missing_observation(0.5, "cp")
        .missing_observation(1.0, "cp")
        .missing_observation(2.0, "cp")
        .missing_observation(4.0, "cp")
        .covariate("wt", 0.0, 75.0)
        .build();

    let values = [("ka", 1.2), ("ke0", 0.08), ("v", 20.0)];
    let jit_parameters = Parameters::with_model(&jit_model, values)?;
    let analytical_parameters = Parameters::with_model(&analytical, values)?;

    let jit_predictions = jit_model
        .estimate_predictions(&subject, &jit_parameters)?
        .into_subject()
        .ok_or_else(|| io::Error::other("expected subject predictions"))?;
    let analytical_predictions =
        analytical.estimate_predictions(&subject, &analytical_parameters)?;

    println!("t      dsl-jit      analytical!");
    for (jit, analytical) in jit_predictions
        .predictions()
        .iter()
        .zip(analytical_predictions.predictions())
    {
        println!(
            "{:<4.1} {:>12.6} {:>16.6}",
            jit.time(),
            jit.prediction(),
            analytical.prediction()
        );
    }

    Ok(())
}

#[cfg(not(feature = "dsl-jit"))]
fn main() {
    eprintln!("Run with: cargo run --example dsl_jit_analytical_covariates --features dsl-jit");
    std::process::exit(1);
}
