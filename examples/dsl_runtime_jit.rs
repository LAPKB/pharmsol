//! Run with:
//! cargo run --example dsl_runtime_jit --features dsl-jit

#[cfg(feature = "dsl-jit")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use std::io;

    use pharmsol::{prelude::*, Parameters};

    let model_source = r#"
name = bimodal_ke
kind = ode

params = ke, v
states = central
outputs = cp

infusion(iv) -> central

dx(central) = -ke * central

out(cp) = central / v
"#;
    let show_compile_logs = false;
    let on_compile_event = move |kind: String, message: String| {
        if !show_compile_logs || message.is_empty() {
            return;
        }

        if kind == "log" {
            eprint!("{message}");
        } else {
            eprintln!("[compile:{kind}] {message}");
        }
    };

    // 1. Compile the model to the runtime JIT backend.
    let model = pharmsol::dsl::compile_module_source_to_runtime(
        model_source,
        Some("bimodal_ke"),
        pharmsol::dsl::RuntimeCompilationTarget::Jit,
        on_compile_event,
    )?;
    let support_point = Parameters::with_model(&model, [("ke", 1.2), ("v", 50.0)])?;

    // 3. Define the subject data.
    let subject = Subject::builder("bimodal_ke")
        .infusion(0.0, 500.0, "iv", 0.5)
        .missing_observation(0.5, "cp")
        .missing_observation(1.0, "cp")
        .missing_observation(2.0, "cp")
        .missing_observation(3.0, "cp")
        .missing_observation(4.0, "cp")
        .missing_observation(6.0, "cp")
        .missing_observation(8.0, "cp")
        .build();

    // 4. Estimate predictions for one support point.
    let predictions = model
        .estimate_predictions(&subject, &support_point)?
        .into_subject()
        .ok_or_else(|| io::Error::other("expected subject predictions"))?;

    // 5. Report the predictions.
    println!("bimodal_ke compiled with runtime JIT");
    println!("{:<6} {:>14}", "t", "prediction");
    for prediction in predictions.predictions() {
        println!(
            "{:<6.1} {:>14.6}",
            prediction.time(),
            prediction.prediction()
        );
    }

    Ok(())
}

#[cfg(not(feature = "dsl-jit"))]
fn main() {
    eprintln!("Run with: cargo run --example bimodal_ke_dsl_runtime_jit --features dsl-jit");
    std::process::exit(1);
}
