// Run with: cargo run --example exa --features exa
//
// This example demonstrates the `exa` feature for dynamically compiling and loading
// pharmacometric models at runtime. It compares predictions from:
// 1. A statically defined ODE model
// 2. A dynamically compiled ODE model (via exa)
// 3. A dynamically compiled Analytical model (via exa)

#[cfg(feature = "exa")]
fn main() {
    use pharmsol::prelude::*;
    use pharmsol::{build::temp_path, exa, Analytical, ODE};
    use std::path::PathBuf;

    // Create test subject with infusion and observations
    // Including missing observations to verify predictions work without observed values
    let subject = Subject::builder("1")
        .infusion(0.0, 500.0, 0, 0.5)
        .observation(0.5, 1.645776, 0)
        .missing_observation(0.75, 0) // Missing observation
        .observation(1.0, 1.216442, 0)
        .missing_observation(1.5, 0) // Missing observation
        .observation(2.0, 0.4622729, 0)
        .missing_observation(2.5, 0) // Missing observation
        .observation(3.0, 0.1697458, 0)
        .observation(4.0, 0.06382178, 0)
        .missing_observation(5.0, 0) // Missing observation
        .observation(6.0, 0.009099384, 0)
        .missing_observation(7.0, 0) // Missing observation
        .observation(8.0, 0.001017932, 0)
        .build();

    // Parameters: ke (elimination rate constant), v (volume of distribution)
    let params = vec![1.2, 50.0];

    // =========================================================================
    // 1. Create ODE model directly (static compilation)
    // =========================================================================
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

    // =========================================================================
    // 2. Compile and load ODE model dynamically using exa
    // =========================================================================
    let test_dir = std::env::current_dir().expect("Failed to get current directory");
    let ode_output_path = test_dir.join("dynamic_ode_model.pkm");

    println!("Compiling ODE model...");
    let ode_compiled_path = exa::build::compile::<ODE>(
        r#"
            equation::ODE::new(
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
            )
        "#
        .to_string(),
        Some(ode_output_path),
        vec!["ke".to_string(), "v".to_string()],
        temp_path(),
        |key, msg| println!("  [{key}] {msg}"),
    )
    .expect("Failed to compile ODE model");

    println!("ODE model compiled to: {ode_compiled_path}");

    let ode_path = PathBuf::from(&ode_compiled_path);
    let (_lib_ode, (dynamic_ode, _meta)) = unsafe { exa::load::load::<ODE>(ode_path.clone()) };

    // =========================================================================
    // 3. Compile and load Analytical model dynamically using exa
    // =========================================================================
    let analytical_output_path = test_dir.join("dynamic_analytical_model.pkm");

    println!("\nCompiling Analytical model...");
    let analytical_compiled_path = exa::build::compile::<Analytical>(
        r#"
            equation::Analytical::new(
                one_compartment,
                |_p, _t, _cov| {},
                |_p, _t, _cov| lag! {},
                |_p, _t, _cov| fa! {},
                |_p, _t, _cov, _x| {},
                |x, p, _t, _cov, y| {
                    fetch_params!(p, _ke, v);
                    y[0] = x[0] / v;
                },
                (1, 1),
            )
        "#
        .to_string(),
        Some(analytical_output_path),
        vec!["ke".to_string(), "v".to_string()],
        temp_path(),
        |key, msg| println!("  [{key}] {msg}"),
    )
    .expect("Failed to compile Analytical model");

    println!("Analytical model compiled to: {analytical_compiled_path}");

    let analytical_path = PathBuf::from(&analytical_compiled_path);
    let (_lib_analytical, (dynamic_analytical, _meta)) =
        unsafe { exa::load::load::<Analytical>(analytical_path.clone()) };

    // =========================================================================
    // 4. Compare predictions from all three models
    // =========================================================================
    println!("\n{}", "=".repeat(60));
    println!("Comparing predictions (ke={}, v={})", params[0], params[1]);
    println!("{}", "=".repeat(60));

    let static_ode_preds = static_ode
        .estimate_predictions(&subject, &params)
        .expect("Static ODE prediction failed");
    let dynamic_ode_preds = dynamic_ode
        .estimate_predictions(&subject, &params)
        .expect("Dynamic ODE prediction failed");
    let dynamic_analytical_preds = dynamic_analytical
        .estimate_predictions(&subject, &params)
        .expect("Dynamic Analytical prediction failed");

    let static_flat = static_ode_preds.flat_predictions();
    let dynamic_ode_flat = dynamic_ode_preds.flat_predictions();
    let dynamic_analytical_flat = dynamic_analytical_preds.flat_predictions();

    let static_times = static_ode_preds.flat_times();
    let static_obs = static_ode_preds.flat_observations();

    println!(
        "\n{:<12} {:>12} {:>15} {:>15} {:>15}",
        "Time", "Obs", "Static ODE", "Dynamic ODE", "Analytical"
    );
    println!("{}", "-".repeat(75));

    for i in 0..static_times.len() {
        let obs_str = match static_obs[i] {
            Some(v) => format!("{:.4}", v),
            None => "MISSING".to_string(),
        };
        println!(
            "{:<12.2} {:>12} {:>15.6} {:>15.6} {:>15.6}",
            static_times[i],
            obs_str,
            static_flat[i],
            dynamic_ode_flat[i],
            dynamic_analytical_flat[i]
        );
    }

    // Verify predictions match
    println!("\n{}", "=".repeat(75));
    println!("Verification:");

    let ode_match = static_flat
        .iter()
        .zip(dynamic_ode_flat.iter())
        .all(|(a, b)| (a - b).abs() < 1e-10);
    println!(
        "  Static ODE vs Dynamic ODE: {}",
        if ode_match {
            "✓ MATCH"
        } else {
            "✗ MISMATCH"
        }
    );

    let analytical_close = static_flat
        .iter()
        .zip(dynamic_analytical_flat.iter())
        .all(|(a, b)| (a - b).abs() < 1e-3);
    println!(
        "  Static ODE vs Analytical:  {}",
        if analytical_close {
            "✓ CLOSE (within 1e-3)"
        } else {
            "✗ DIFFERS"
        }
    );

    // Count zero predictions for missing observations
    let zero_count = static_flat.iter().filter(|&&v| v == 0.0).count();
    println!("  Zero predictions count: {} (should be 0)", zero_count);

    // =========================================================================
    // 5. Clean up compiled model files
    // =========================================================================
    std::fs::remove_file(&ode_path).ok();
    std::fs::remove_file(&analytical_path).ok();
    println!("\nCleaned up temporary model files.");
}

#[cfg(not(feature = "exa"))]
fn main() {
    eprintln!("This example requires the 'exa' feature.");
    eprintln!("Run with: cargo run --example exa --features exa");
    std::process::exit(1);
}
