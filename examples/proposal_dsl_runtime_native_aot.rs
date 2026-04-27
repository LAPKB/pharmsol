//! Run with:
//! cargo run --example proposal_dsl_runtime_native_aot --features "dsl-aot dsl-aot-load"

#[cfg(all(feature = "dsl-aot", feature = "dsl-aot-load"))]
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

#[cfg(all(feature = "dsl-aot", feature = "dsl-aot-load"))]
const ANALYTICAL_SOURCE: &str = r#"
model = example_analytical
kind = analytical

params = ka, ke, v, tlag, f_oral
states = depot, central
outputs = cp

bolus(oral) -> depot

lag(oral) = tlag
fa(oral) = f_oral

kernel = one_compartment_with_absorption

out(cp) = central / v ~ continuous()
"#;

#[cfg(all(feature = "dsl-aot", feature = "dsl-aot-load"))]
const SDE_SOURCE: &str = r#"
model = example_sde
kind = sde

params = ka, ke0, kcp, kpc, vol, ske
covariates = wt@locf
states = depot, central, peripheral, ke_latent
particles = 16
outputs = cp

bolus(oral) -> depot

init(ke_latent) = ke0

dx(depot) = -ka * depot
dx(central) = ka * depot - (ke_latent + kcp) * central + kpc * peripheral
dx(peripheral) = kcp * central - kpc * peripheral
dx(ke_latent) = -ke_latent + ke0

noise(ke_latent) = ske

out(cp) = central / (vol * wt) ~ continuous()
"#;

#[cfg(all(feature = "dsl-aot", feature = "dsl-aot-load"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use std::{fs, io, path::PathBuf};

    use pharmsol::dsl::{self, RuntimeCompilationTarget};
    use pharmsol::{Subject, SubjectBuilderExt};

    println!("Sugared DSL models compiled with runtime Native AoT");
    let workspace = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("target")
        .join("example-artifacts")
        .join("proposal_dsl_runtime_native_aot");
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

    // 1. Define an ODE model, compile it, simulate one subject, and print predictions.
    let ode_support_point = [1.2, 5.0, 40.0, 0.25, 0.8];
    let ode_model = dsl::compile_module_source_to_runtime(
        ODE_SOURCE,
        Some("example_ode"),
        RuntimeCompilationTarget::NativeAot(
            dsl::NativeAotCompileOptions::new(
                workspace.join("example-ode-runtime-native-aot-build"),
            )
            .with_output(workspace.join("example-ode-runtime-native-aot.pkm")),
        ),
        on_compile_event,
    )?;
    let ode_oral = ode_model
        .route_index("oral")
        .ok_or_else(|| io::Error::other("ODE model: missing oral route"))?;
    let ode_iv = ode_model
        .route_index("iv")
        .ok_or_else(|| io::Error::other("ODE model: missing iv route"))?;
    let ode_cp = ode_model
        .output_index("cp")
        .ok_or_else(|| io::Error::other("ODE model: missing cp output"))?;
    let ode_subject = Subject::builder("example_ode")
        .covariate("wt", 0.0, 70.0)
        .bolus(0.0, 120.0, ode_oral)
        .infusion(6.0, 60.0, ode_iv, 2.0)
        .missing_observation(0.5, ode_cp)
        .missing_observation(1.0, ode_cp)
        .missing_observation(2.0, ode_cp)
        .missing_observation(6.0, ode_cp)
        .missing_observation(7.0, ode_cp)
        .missing_observation(9.0, ode_cp)
        .build();
    let ode_predictions = ode_model.estimate_predictions(&ode_subject, &ode_support_point)?;
    print_predictions("ODE model via runtime Native AoT", ode_predictions);

    // 2. Define an analytical model, compile it, simulate one subject, and print predictions.
    let analytical_support_point = [1.0, 0.15, 25.0, 0.5, 0.8];
    let analytical_model = dsl::compile_module_source_to_runtime(
        ANALYTICAL_SOURCE,
        Some("example_analytical"),
        RuntimeCompilationTarget::NativeAot(
            dsl::NativeAotCompileOptions::new(
                workspace.join("example-analytical-runtime-native-aot-build"),
            )
            .with_output(workspace.join("example-analytical-runtime-native-aot.pkm")),
        ),
        on_compile_event,
    )?;
    let analytical_oral = analytical_model
        .route_index("oral")
        .ok_or_else(|| io::Error::other("Analytical model: missing oral route"))?;
    let analytical_cp = analytical_model
        .output_index("cp")
        .ok_or_else(|| io::Error::other("Analytical model: missing cp output"))?;
    let analytical_subject = Subject::builder("example_analytical")
        .bolus(0.0, 100.0, analytical_oral)
        .missing_observation(0.5, analytical_cp)
        .missing_observation(1.0, analytical_cp)
        .missing_observation(2.0, analytical_cp)
        .missing_observation(4.0, analytical_cp)
        .build();
    let analytical_predictions =
        analytical_model.estimate_predictions(&analytical_subject, &analytical_support_point)?;
    print_predictions(
        "Analytical model via runtime Native AoT",
        analytical_predictions,
    );

    // 3. Define an SDE model, compile it, simulate one subject, and print predictions.
    let sde_support_point = [1.1, 0.2, 0.12, 0.08, 15.0, 0.05];
    let sde_model = dsl::compile_module_source_to_runtime(
        SDE_SOURCE,
        Some("example_sde"),
        RuntimeCompilationTarget::NativeAot(
            dsl::NativeAotCompileOptions::new(
                workspace.join("example-sde-runtime-native-aot-build"),
            )
            .with_output(workspace.join("example-sde-runtime-native-aot.pkm")),
        ),
        on_compile_event,
    )?;
    let sde_oral = sde_model
        .route_index("oral")
        .ok_or_else(|| io::Error::other("SDE model: missing oral route"))?;
    let sde_cp = sde_model
        .output_index("cp")
        .ok_or_else(|| io::Error::other("SDE model: missing cp output"))?;
    let sde_subject = Subject::builder("example_sde")
        .covariate("wt", 0.0, 70.0)
        .bolus(0.0, 80.0, sde_oral)
        .missing_observation(0.5, sde_cp)
        .missing_observation(1.0, sde_cp)
        .missing_observation(2.0, sde_cp)
        .missing_observation(4.0, sde_cp)
        .build();
    let sde_predictions = sde_model.estimate_predictions(&sde_subject, &sde_support_point)?;
    print_predictions("SDE model via runtime Native AoT", sde_predictions);

    Ok(())
}

#[cfg(all(feature = "dsl-aot", feature = "dsl-aot-load"))]
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
            print_particle_mean_predictions(&predictions);
        }
    }
}

#[cfg(all(feature = "dsl-aot", feature = "dsl-aot-load"))]
fn print_particle_mean_predictions(predictions: &ndarray::Array2<pharmsol::prelude::Prediction>) {
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

#[cfg(not(all(feature = "dsl-aot", feature = "dsl-aot-load")))]
fn main() {
    eprintln!(
        "Run with: cargo run --example proposal_dsl_runtime_native_aot --features \"dsl-aot dsl-aot-load\""
    );
    std::process::exit(1);
}
