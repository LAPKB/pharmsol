//! Run with:
//! cargo run --example proposal_dsl_runtime_meta --features "dsl-jit dsl-aot dsl-aot-load dsl-wasm"

const ODE_SOURCE: &str = r#"
model = example_ode
kind = ode

params = ka, cl, v, tlag, f_oral
covariates = wt@linear
states = depot, central
derived = cl_i, ke
outputs = cp

bolus(oral) -> depot
infusion(iv) -> central

lag(oral) = tlag
fa(oral) = f_oral

cl_i = cl * pow(wt / 70.0, 0.75)
ke = cl_i / v

dx(depot) = -ka * depot
dx(central) = ka * depot - ke * central

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

    use pharmsol::dsl::{self, RuntimeCompilationTarget};
    use pharmsol::{Subject, SubjectBuilderExt};

    println!("Compare one DSL model across runtime JIT, Native AoT, and WASM");
    let support_point = [1.2, 5.0, 40.0, 0.25, 0.8];
    let workspace = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("target")
        .join("example-artifacts")
        .join("proposal_dsl_runtime_meta");
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

    // 1. Compile the model with the JIT backend.
    let jit_model = dsl::compile_module_source_to_runtime(
        ODE_SOURCE,
        Some("example_ode"),
        RuntimeCompilationTarget::Jit,
        on_compile_event,
    )?;
    let jit_oral = jit_model
        .route_index("oral")
        .ok_or_else(|| io::Error::other("runtime JIT: missing oral route"))?;
    let jit_iv = jit_model
        .route_index("iv")
        .ok_or_else(|| io::Error::other("runtime JIT: missing iv route"))?;
    let jit_cp = jit_model
        .output_index("cp")
        .ok_or_else(|| io::Error::other("runtime JIT: missing cp output"))?;
    let jit_subject = Subject::builder("example_ode")
        .covariate("wt", 0.0, 70.0)
        .bolus(0.0, 120.0, jit_oral)
        .infusion(6.0, 60.0, jit_iv, 2.0)
        .missing_observation(0.5, jit_cp)
        .missing_observation(1.0, jit_cp)
        .missing_observation(2.0, jit_cp)
        .missing_observation(6.0, jit_cp)
        .missing_observation(7.0, jit_cp)
        .missing_observation(9.0, jit_cp)
        .build();
    let jit_predictions = jit_model.estimate_predictions(&jit_subject, &support_point)?;
    print_predictions("ODE model via runtime JIT", jit_predictions);

    // 2. Compile the same model with the Native AoT backend.
    let native_aot_model = dsl::compile_module_source_to_runtime(
        ODE_SOURCE,
        Some("example_ode"),
        RuntimeCompilationTarget::NativeAot {
            output: Some(workspace.join("example-ode-runtime-meta-native-aot.pkm")),
            template_root: workspace.join("example-ode-runtime-meta-native-aot-build"),
        },
        on_compile_event,
    )?;
    let native_aot_oral = native_aot_model
        .route_index("oral")
        .ok_or_else(|| io::Error::other("runtime Native AoT: missing oral route"))?;
    let native_aot_iv = native_aot_model
        .route_index("iv")
        .ok_or_else(|| io::Error::other("runtime Native AoT: missing iv route"))?;
    let native_aot_cp = native_aot_model
        .output_index("cp")
        .ok_or_else(|| io::Error::other("runtime Native AoT: missing cp output"))?;
    let native_aot_subject = Subject::builder("example_ode")
        .covariate("wt", 0.0, 70.0)
        .bolus(0.0, 120.0, native_aot_oral)
        .infusion(6.0, 60.0, native_aot_iv, 2.0)
        .missing_observation(0.5, native_aot_cp)
        .missing_observation(1.0, native_aot_cp)
        .missing_observation(2.0, native_aot_cp)
        .missing_observation(6.0, native_aot_cp)
        .missing_observation(7.0, native_aot_cp)
        .missing_observation(9.0, native_aot_cp)
        .build();
    let native_aot_predictions =
        native_aot_model.estimate_predictions(&native_aot_subject, &support_point)?;
    print_predictions("ODE model via runtime Native AoT", native_aot_predictions);

    // 3. Compile the same model with the WASM backend.
    let wasm_model = dsl::compile_module_source_to_runtime(
        ODE_SOURCE,
        Some("example_ode"),
        RuntimeCompilationTarget::Wasm {
            output: Some(workspace.join("example-ode-runtime-meta-wasm.wasm")),
            template_root: workspace.join("example-ode-runtime-meta-wasm-build"),
        },
        on_compile_event,
    )?;
    let wasm_oral = wasm_model
        .route_index("oral")
        .ok_or_else(|| io::Error::other("runtime WASM: missing oral route"))?;
    let wasm_iv = wasm_model
        .route_index("iv")
        .ok_or_else(|| io::Error::other("runtime WASM: missing iv route"))?;
    let wasm_cp = wasm_model
        .output_index("cp")
        .ok_or_else(|| io::Error::other("runtime WASM: missing cp output"))?;
    let wasm_subject = Subject::builder("example_ode")
        .covariate("wt", 0.0, 70.0)
        .bolus(0.0, 120.0, wasm_oral)
        .infusion(6.0, 60.0, wasm_iv, 2.0)
        .missing_observation(0.5, wasm_cp)
        .missing_observation(1.0, wasm_cp)
        .missing_observation(2.0, wasm_cp)
        .missing_observation(6.0, wasm_cp)
        .missing_observation(7.0, wasm_cp)
        .missing_observation(9.0, wasm_cp)
        .build();
    let wasm_predictions = wasm_model.estimate_predictions(&wasm_subject, &support_point)?;
    print_predictions("ODE model via runtime WASM", wasm_predictions);

    Ok(())
}

#[cfg(all(
    feature = "dsl-jit",
    feature = "dsl-aot",
    feature = "dsl-aot-load",
    feature = "dsl-wasm"
))]
fn print_predictions(title: &str, predictions: pharmsol::dsl::RuntimePredictions) {
    println!("\n{title}");
    match predictions {
        pharmsol::dsl::RuntimePredictions::Subject(predictions) => {
            println!("{:<6} {:>14}", "t", "prediction");
            for prediction in predictions.predictions() {
                println!(
                    "{:<6.1} {:>14.6}",
                    prediction.time(),
                    prediction.prediction()
                );
            }
        }
        pharmsol::dsl::RuntimePredictions::Particles(predictions) => {
            println!(
                "showing mean prediction across {} particles",
                predictions.nrows()
            );
            println!("{:<6} {:>14}", "t", "prediction");
            for col in 0..predictions.ncols() {
                let time = predictions[(0, col)].time();
                let mean = (0..predictions.nrows())
                    .map(|row| predictions[(row, col)].prediction())
                    .sum::<f64>()
                    / predictions.nrows() as f64;
                println!("{:<6.1} {:>14.6}", time, mean);
            }
        }
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
        "Run with: cargo run --example proposal_dsl_runtime_meta --features \"dsl-jit dsl-aot dsl-aot-load dsl-wasm\""
    );
    std::process::exit(1);
}
