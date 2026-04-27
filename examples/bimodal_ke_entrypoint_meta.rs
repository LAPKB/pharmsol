//! Run with:
//! cargo run --example bimodal_ke_entrypoint_meta --features "dsl-jit dsl-aot dsl-aot-load dsl-wasm"

#[cfg(all(
    feature = "dsl-jit",
    feature = "dsl-aot",
    feature = "dsl-aot-load",
    feature = "dsl-wasm"
))]
const MODEL_SOURCE: &str = r#"
model = bimodal_ke
kind = ode

params = ke, v
states = central
outputs = cp

infusion(iv) -> central

dx(central) = -ke * central

out(cp) = central / v ~ continuous()
"#;

#[cfg(all(
    feature = "dsl-jit",
    feature = "dsl-aot",
    feature = "dsl-aot-load",
    feature = "dsl-wasm"
))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use std::{fs, io, path::PathBuf};

    use pharmsol::{Subject, SubjectBuilderExt};

    println!("Compare the same bimodal_ke model across the public DSL entrypoints");
    let support_point = [1.2, 50.0];
    let workspace = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("target")
        .join("example-artifacts")
        .join("bimodal_ke_entrypoint_meta");
    fs::create_dir_all(&workspace)?;
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

    // 1. Compile the model with the runtime JIT entrypoint.
    let runtime_jit_model = pharmsol::dsl::compile_module_source_to_runtime(
        MODEL_SOURCE,
        Some("bimodal_ke"),
        pharmsol::dsl::RuntimeCompilationTarget::Jit,
        on_compile_event,
    )?;
    let runtime_jit_iv = runtime_jit_model
        .route_index("iv")
        .ok_or_else(|| io::Error::other("runtime JIT: missing iv route"))?;
    let runtime_jit_cp = runtime_jit_model
        .output_index("cp")
        .ok_or_else(|| io::Error::other("runtime JIT: missing cp output"))?;
    let runtime_jit_subject = Subject::builder("bimodal_ke")
        .infusion(0.0, 500.0, runtime_jit_iv, 0.5)
        .missing_observation(0.5, runtime_jit_cp)
        .missing_observation(1.0, runtime_jit_cp)
        .missing_observation(2.0, runtime_jit_cp)
        .missing_observation(3.0, runtime_jit_cp)
        .missing_observation(4.0, runtime_jit_cp)
        .missing_observation(6.0, runtime_jit_cp)
        .missing_observation(8.0, runtime_jit_cp)
        .build();
    let runtime_jit_predictions = runtime_jit_model
        .estimate_predictions(&runtime_jit_subject, &support_point)?
        .into_subject()
        .ok_or_else(|| io::Error::other("runtime JIT: expected subject predictions"))?;
    print_subject_predictions("runtime JIT", &runtime_jit_predictions);

    // 2. Compile the same model with the runtime Native AoT entrypoint.
    let runtime_native_aot_model = pharmsol::dsl::compile_module_source_to_runtime(
        MODEL_SOURCE,
        Some("bimodal_ke"),
        pharmsol::dsl::RuntimeCompilationTarget::NativeAot(
            pharmsol::dsl::NativeAotCompileOptions::new(
                workspace.join("bimodal-ke-meta-runtime-native-aot-build"),
            )
            .with_output(workspace.join("bimodal-ke-meta-runtime-native-aot.pkm")),
        ),
        on_compile_event,
    )?;
    let runtime_native_aot_iv = runtime_native_aot_model
        .route_index("iv")
        .ok_or_else(|| io::Error::other("runtime Native AoT: missing iv route"))?;
    let runtime_native_aot_cp = runtime_native_aot_model
        .output_index("cp")
        .ok_or_else(|| io::Error::other("runtime Native AoT: missing cp output"))?;
    let runtime_native_aot_subject = Subject::builder("bimodal_ke")
        .infusion(0.0, 500.0, runtime_native_aot_iv, 0.5)
        .missing_observation(0.5, runtime_native_aot_cp)
        .missing_observation(1.0, runtime_native_aot_cp)
        .missing_observation(2.0, runtime_native_aot_cp)
        .missing_observation(3.0, runtime_native_aot_cp)
        .missing_observation(4.0, runtime_native_aot_cp)
        .missing_observation(6.0, runtime_native_aot_cp)
        .missing_observation(8.0, runtime_native_aot_cp)
        .build();
    let runtime_native_aot_predictions = runtime_native_aot_model
        .estimate_predictions(&runtime_native_aot_subject, &support_point)?
        .into_subject()
        .ok_or_else(|| io::Error::other("runtime Native AoT: expected subject predictions"))?;
    print_subject_predictions("runtime Native AoT", &runtime_native_aot_predictions);

    // 3. Compile the same model with the runtime WASM entrypoint.
    let runtime_wasm_model =
        pharmsol::dsl::compile_module_source_to_runtime_wasm(MODEL_SOURCE, Some("bimodal_ke"))?;
    let runtime_wasm_iv = runtime_wasm_model
        .route_index("iv")
        .ok_or_else(|| io::Error::other("runtime WASM: missing iv route"))?;
    let runtime_wasm_cp = runtime_wasm_model
        .output_index("cp")
        .ok_or_else(|| io::Error::other("runtime WASM: missing cp output"))?;
    let runtime_wasm_subject = Subject::builder("bimodal_ke")
        .infusion(0.0, 500.0, runtime_wasm_iv, 0.5)
        .missing_observation(0.5, runtime_wasm_cp)
        .missing_observation(1.0, runtime_wasm_cp)
        .missing_observation(2.0, runtime_wasm_cp)
        .missing_observation(3.0, runtime_wasm_cp)
        .missing_observation(4.0, runtime_wasm_cp)
        .missing_observation(6.0, runtime_wasm_cp)
        .missing_observation(8.0, runtime_wasm_cp)
        .build();
    let runtime_wasm_predictions = runtime_wasm_model
        .estimate_predictions(&runtime_wasm_subject, &support_point)?
        .into_subject()
        .ok_or_else(|| io::Error::other("runtime WASM: expected subject predictions"))?;
    print_subject_predictions("runtime WASM", &runtime_wasm_predictions);

    // 4. Compile the same model to a native artifact, then load it.
    let direct_aot_artifact = pharmsol::dsl::compile_module_source_to_aot(
        MODEL_SOURCE,
        Some("bimodal_ke"),
        pharmsol::dsl::NativeAotCompileOptions::new(
            workspace.join("bimodal-ke-meta-direct-aot-build"),
        )
        .with_output(workspace.join("bimodal-ke-meta-direct-aot.pkm")),
        on_compile_event,
    )?;
    let direct_aot_model = pharmsol::dsl::load_runtime_artifact(
        &direct_aot_artifact,
        pharmsol::dsl::RuntimeArtifactFormat::NativeAot,
    )?;
    let direct_aot_iv = direct_aot_model
        .route_index("iv")
        .ok_or_else(|| io::Error::other("compile_module_source_to_aot: missing iv route"))?;
    let direct_aot_cp = direct_aot_model
        .output_index("cp")
        .ok_or_else(|| io::Error::other("compile_module_source_to_aot: missing cp output"))?;
    let direct_aot_subject = Subject::builder("bimodal_ke")
        .infusion(0.0, 500.0, direct_aot_iv, 0.5)
        .missing_observation(0.5, direct_aot_cp)
        .missing_observation(1.0, direct_aot_cp)
        .missing_observation(2.0, direct_aot_cp)
        .missing_observation(3.0, direct_aot_cp)
        .missing_observation(4.0, direct_aot_cp)
        .missing_observation(6.0, direct_aot_cp)
        .missing_observation(8.0, direct_aot_cp)
        .build();
    let direct_aot_predictions = direct_aot_model
        .estimate_predictions(&direct_aot_subject, &support_point)?
        .into_subject()
        .ok_or_else(|| {
            io::Error::other("compile_module_source_to_aot: expected subject predictions")
        })?;
    print_subject_predictions(
        "compile_module_source_to_aot + load_runtime_artifact",
        &direct_aot_predictions,
    );

    // 5. Compile the same model to in-memory WASM bytes, then load them.
    let direct_wasm_bytes =
        pharmsol::dsl::compile_module_source_to_wasm_bytes(MODEL_SOURCE, Some("bimodal_ke"))?;
    let direct_wasm_model = pharmsol::dsl::load_runtime_wasm_bytes(&direct_wasm_bytes)?;
    let direct_wasm_iv = direct_wasm_model
        .route_index("iv")
        .ok_or_else(|| io::Error::other("compile_module_source_to_wasm: missing iv route"))?;
    let direct_wasm_cp = direct_wasm_model
        .output_index("cp")
        .ok_or_else(|| io::Error::other("compile_module_source_to_wasm: missing cp output"))?;
    let direct_wasm_subject = Subject::builder("bimodal_ke")
        .infusion(0.0, 500.0, direct_wasm_iv, 0.5)
        .missing_observation(0.5, direct_wasm_cp)
        .missing_observation(1.0, direct_wasm_cp)
        .missing_observation(2.0, direct_wasm_cp)
        .missing_observation(3.0, direct_wasm_cp)
        .missing_observation(4.0, direct_wasm_cp)
        .missing_observation(6.0, direct_wasm_cp)
        .missing_observation(8.0, direct_wasm_cp)
        .build();
    let direct_wasm_predictions = direct_wasm_model
        .estimate_predictions(&direct_wasm_subject, &support_point)?
        .into_subject()
        .ok_or_else(|| {
            io::Error::other("compile_module_source_to_wasm: expected subject predictions")
        })?;
    print_subject_predictions(
        "compile_module_source_to_wasm_bytes + load_runtime_wasm_bytes",
        &direct_wasm_predictions,
    );

    Ok(())
}

#[cfg(all(
    feature = "dsl-jit",
    feature = "dsl-aot",
    feature = "dsl-aot-load",
    feature = "dsl-wasm"
))]
fn print_subject_predictions(label: &str, predictions: &pharmsol::prelude::SubjectPredictions) {
    println!("\n{label}");
    println!("{:<6} {:>14}", "t", "prediction");
    for prediction in predictions.predictions() {
        println!(
            "{:<6.1} {:>14.6}",
            prediction.time(),
            prediction.prediction()
        );
    }
}

#[cfg(not(all(
    feature = "dsl-jit",
    feature = "dsl-aot",
    feature = "dsl-aot-load",
    feature = "dsl-wasm"
)))]
fn main() {
    eprintln!(
        "Run with: cargo run --example bimodal_ke_entrypoint_meta --features \"dsl-jit dsl-aot dsl-aot-load dsl-wasm\""
    );
    std::process::exit(1);
}
