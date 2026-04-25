//! Run with:
//! cargo run --example bimodal_ke_dsl_aot --features "dsl-aot dsl-aot-load"

#[cfg(all(feature = "dsl-aot", feature = "dsl-aot-load"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use std::{fs, io, path::PathBuf};

    use pharmsol::prelude::*;

    let model_source = r#"
model = bimodal_ke
kind = ode

params = ke, v
states = central
outputs = cp

infusion(iv) -> central

dx(central) = -ke * central

out(cp) = central / v ~ continuous()
"#;
    let support_point = [1.2, 50.0];
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

    // 1. Compile the model to a native artifact, then load it back.
    let workspace = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("target")
        .join("example-artifacts")
        .join("bimodal_ke_dsl_aot");
    fs::create_dir_all(&workspace)?;
    let artifact = pharmsol::dsl::compile_module_source_to_aot(
        model_source,
        Some("bimodal_ke"),
        Some(workspace.join("bimodal-ke-direct-aot.pkm")),
        workspace.join("direct-aot-build"),
        on_compile_event,
    )?;
    let model = pharmsol::dsl::load_runtime_artifact(
        &artifact,
        pharmsol::dsl::RuntimeArtifactFormat::NativeAot,
    )?;

    // 2. Resolve the route and output indices declared by the model.
    let iv = model
        .route_index("iv")
        .ok_or_else(|| io::Error::other("missing iv route"))?;
    let cp = model
        .output_index("cp")
        .ok_or_else(|| io::Error::other("missing cp output"))?;

    // 3. Define the subject data.
    let subject = Subject::builder("bimodal_ke")
        .infusion(0.0, 500.0, iv, 0.5)
        .missing_observation(0.5, cp)
        .missing_observation(1.0, cp)
        .missing_observation(2.0, cp)
        .missing_observation(3.0, cp)
        .missing_observation(4.0, cp)
        .missing_observation(6.0, cp)
        .missing_observation(8.0, cp)
        .build();

    // 4. Estimate predictions for one support point.
    let predictions = model
        .estimate_predictions(&subject, &support_point)?
        .into_subject()
        .ok_or_else(|| io::Error::other("expected subject predictions"))?;

    // 5. Report the predictions.
    println!("bimodal_ke compiled with compile_module_source_to_aot");
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

#[cfg(not(all(feature = "dsl-aot", feature = "dsl-aot-load")))]
fn main() {
    eprintln!(
        "Run with: cargo run --example bimodal_ke_dsl_aot --features \"dsl-aot dsl-aot-load\""
    );
    std::process::exit(1);
}
