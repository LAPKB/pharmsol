// Run with: cargo run --example json_exa --features exa
//
// This example demonstrates JSON model compilation using the `exa` feature.
// It compares predictions from:
// 1. A statically defined ODE model (Rust code)
// 2. A dynamically compiled ODE model (via exa, raw Rust string)
// 3. A JSON-defined ODE model (via compile_json)
// 4. A JSON-defined Analytical model (via compile_json)

#[cfg(feature = "exa")]
fn main() {
    use pharmsol::prelude::*;
    use pharmsol::{exa, json, Analytical, ODE};
    use std::path::PathBuf;

    // Create test subject with infusion and observations
    let subject = Subject::builder("1")
        .infusion(0.0, 500.0, 0, 0.5)
        .observation(0.5, 1.645776, 0)
        .observation(1.0, 1.216442, 0)
        .observation(2.0, 0.4622729, 0)
        .observation(3.0, 0.1697458, 0)
        .observation(4.0, 0.06382178, 0)
        .observation(6.0, 0.009099384, 0)
        .observation(8.0, 0.001017932, 0)
        .build();

    // Parameters: ke (elimination rate constant), V (volume of distribution)
    let params = vec![1.2, 50.0];

    let test_dir = std::env::current_dir().expect("Failed to get current directory");

    // Shared template path for all compilations (they run sequentially)
    let template_path = std::env::temp_dir().join("exa_json_example");

    // =========================================================================
    // 1. Create ODE model directly (static Rust code)
    // =========================================================================
    println!("1. Creating static ODE model...");
    let static_ode = equation::ODE::new(
        |x, p, _t, dx, _bolus, rateiv, _cov| {
            fetch_params!(p, ke, _v);
            dx[0] = -ke * x[0] + rateiv[0];
        },
        |_p, _t, _cov| lag! {},
        |_p, _t, _cov| fa! {},
        |_p, _t, _cov, _x| {},
        |x, p, _t, _cov, y| {
            fetch_params!(p, _ke, v);
            y[0] = x[0] / v;
        },
        (1, 1),
    );
    println!("   ✓ Static ODE model created\n");

    // =========================================================================
    // 2. Compile ODE model dynamically using exa (raw Rust string)
    // =========================================================================
    println!("2. Compiling ODE model via exa (raw Rust)...");
    let exa_ode_path = test_dir.join("exa_ode_model.pkm");

    let exa_ode_compiled = exa::build::compile::<ODE>(
        r#"
            equation::ODE::new(
                |x, p, _t, dx, _bolus, rateiv, _cov| {
                    fetch_params!(p, ke, _V);
                    dx[0] = -ke * x[0] + rateiv[0];
                },
                |_p, _t, _cov| lag! {},
                |_p, _t, _cov| fa! {},
                |_p, _t, _cov, _x| {},
                |x, p, _t, _cov, y| {
                    fetch_params!(p, _ke, V);
                    y[0] = x[0] / V;
                },
                (1, 1),
            )
        "#
        .to_string(),
        Some(exa_ode_path.clone()),
        vec!["ke".to_string(), "V".to_string()],
        template_path.clone(),
        |_, _| {},
    )
    .expect("Failed to compile ODE model via exa");

    let exa_ode_path = PathBuf::from(&exa_ode_compiled);
    let (_lib_exa_ode, (dynamic_exa_ode, _)) =
        unsafe { exa::load::load::<ODE>(exa_ode_path.clone()) };
    println!("   ✓ Compiled to: {}\n", exa_ode_compiled);

    // =========================================================================
    // 3. Compile ODE model from JSON using compile_json
    // =========================================================================
    println!("3. Compiling ODE model from JSON...");

    let json_ode = r#"{
        "schema": "1.0",
        "id": "pk_1cmt_iv_ode",
        "type": "ode",
        "parameters": ["ke", "V"],
        "compartments": ["central"],
        "diffeq": {
            "central": "-ke * central + rateiv[0]"
        },
        "output": "central / V",
        "display": {
            "name": "One-Compartment IV ODE",
            "category": "pk"
        }
    }"#;

    // First, show the generated code
    let generated = json::generate_code(json_ode).expect("Failed to generate code from JSON");
    println!("   Generated Rust code:");
    println!("   ─────────────────────────────────────");
    for line in generated.equation_code.lines().take(15) {
        println!("   {}", line);
    }
    println!("   ...\n");

    let json_ode_path = test_dir.join("json_ode_model.pkm");

    let json_ode_compiled = json::compile_json::<ODE>(
        json_ode,
        Some(json_ode_path.clone()),
        template_path.clone(),
        |_, _| {},
    )
    .expect("Failed to compile JSON ODE model");

    let json_ode_path = PathBuf::from(&json_ode_compiled);
    let (_lib_json_ode, (dynamic_json_ode, meta_ode)) =
        unsafe { exa::load::load::<ODE>(json_ode_path.clone()) };
    println!(
        "   ✓ Compiled to: {} (params: {:?})\n",
        json_ode_compiled,
        meta_ode.get_params()
    );

    // =========================================================================
    // 4. Compile Analytical model from JSON using compile_json
    // =========================================================================
    println!("4. Compiling Analytical model from JSON...");

    let json_analytical = r#"{
        "schema": "1.0",
        "id": "pk_1cmt_iv_analytical",
        "type": "analytical",
        "analytical": "one_compartment",
        "parameters": ["ke", "V"],
        "output": "x[0] / V",
        "display": {
            "name": "One-Compartment IV Analytical",
            "category": "pk"
        }
    }"#;

    let json_analytical_path = test_dir.join("json_analytical_model.pkm");

    let json_analytical_compiled = json::compile_json::<Analytical>(
        json_analytical,
        Some(json_analytical_path.clone()),
        template_path.clone(),
        |_, _| {},
    )
    .expect("Failed to compile JSON Analytical model");

    let json_analytical_path = PathBuf::from(&json_analytical_compiled);
    let (_lib_json_analytical, (dynamic_json_analytical, meta_analytical)) =
        unsafe { exa::load::load::<Analytical>(json_analytical_path.clone()) };
    println!(
        "   ✓ Compiled to: {} (params: {:?})\n",
        json_analytical_compiled,
        meta_analytical.get_params()
    );

    // =========================================================================
    // 5. Compare predictions from all four models
    // =========================================================================
    println!("{}", "═".repeat(80));
    println!("Comparing predictions (ke={}, V={})", params[0], params[1]);
    println!("{}", "═".repeat(80));

    let static_preds = static_ode
        .estimate_predictions(&subject, &params)
        .expect("Static ODE prediction failed");
    let exa_ode_preds = dynamic_exa_ode
        .estimate_predictions(&subject, &params)
        .expect("Exa ODE prediction failed");
    let json_ode_preds = dynamic_json_ode
        .estimate_predictions(&subject, &params)
        .expect("JSON ODE prediction failed");
    let json_analytical_preds = dynamic_json_analytical
        .estimate_predictions(&subject, &params)
        .expect("JSON Analytical prediction failed");

    let static_flat = static_preds.flat_predictions();
    let exa_ode_flat = exa_ode_preds.flat_predictions();
    let json_ode_flat = json_ode_preds.flat_predictions();
    let json_analytical_flat = json_analytical_preds.flat_predictions();

    println!(
        "\n{:<8} {:>14} {:>14} {:>14} {:>14}",
        "Time", "Static ODE", "Exa ODE", "JSON ODE", "JSON Analyt."
    );
    println!("{}", "─".repeat(80));

    let times = [0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0];
    for (i, &time) in times.iter().enumerate() {
        println!(
            "{:<8.1} {:>14.6} {:>14.6} {:>14.6} {:>14.6}",
            time, static_flat[i], exa_ode_flat[i], json_ode_flat[i], json_analytical_flat[i]
        );
    }

    // =========================================================================
    // 6. Verification
    // =========================================================================
    println!("\n{}", "═".repeat(80));
    println!("Verification:");
    println!("{}", "─".repeat(80));

    // Static ODE vs Exa ODE
    let static_vs_exa = static_flat
        .iter()
        .zip(exa_ode_flat.iter())
        .all(|(a, b)| (a - b).abs() < 1e-10);
    println!(
        "  Static ODE vs Exa ODE:        {} (tolerance: 1e-10)",
        if static_vs_exa {
            "✓ MATCH"
        } else {
            "✗ MISMATCH"
        }
    );

    // Static ODE vs JSON ODE
    let static_vs_json_ode = static_flat
        .iter()
        .zip(json_ode_flat.iter())
        .all(|(a, b)| (a - b).abs() < 1e-10);
    println!(
        "  Static ODE vs JSON ODE:       {} (tolerance: 1e-10)",
        if static_vs_json_ode {
            "✓ MATCH"
        } else {
            "✗ MISMATCH"
        }
    );

    // Static ODE vs JSON Analytical
    let static_vs_json_analytical = static_flat
        .iter()
        .zip(json_analytical_flat.iter())
        .all(|(a, b)| (a - b).abs() < 1e-3);
    println!(
        "  Static ODE vs JSON Analytical: {} (tolerance: 1e-3)",
        if static_vs_json_analytical {
            "✓ CLOSE"
        } else {
            "✗ DIFFERS"
        }
    );

    // =========================================================================
    // 7. Demonstrate JSON Model Library
    // =========================================================================
    println!("\n{}", "═".repeat(80));
    println!("JSON Model Library:");
    println!("{}", "─".repeat(80));

    let library = json::ModelLibrary::builtin();
    println!("  Available builtin models ({}):", library.list().len());
    for id in library.list() {
        let model = library.get(id).unwrap();
        let model_type = match &model.model_type {
            json::ModelType::Analytical => "Analytical",
            json::ModelType::Ode => "ODE",
            json::ModelType::Sde => "SDE",
        };
        let name = model
            .display
            .as_ref()
            .and_then(|d| d.name.as_ref())
            .map(|s| s.as_str())
            .unwrap_or("(unnamed)");
        println!("    • {} [{}]: {}", id, model_type, name);
    }

    // =========================================================================
    // 8. Clean up
    // =========================================================================
    println!("\n{}", "═".repeat(80));
    println!("Cleaning up...");

    std::fs::remove_file(&exa_ode_path).ok();
    std::fs::remove_file(&json_ode_path).ok();
    std::fs::remove_file(&json_analytical_path).ok();
    std::fs::remove_dir_all(&template_path).ok();

    println!("  ✓ Removed compiled model files");
    println!("  ✓ Removed temporary build directory");
    println!("\nDone!");
}

#[cfg(not(feature = "exa"))]
fn main() {
    eprintln!("This example requires the 'exa' feature.");
    eprintln!("Run with: cargo run --example json_exa --features exa");
    std::process::exit(1);
}
