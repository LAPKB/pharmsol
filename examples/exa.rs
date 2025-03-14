//cargo run --example exa --features exa

#[cfg(feature = "exa")]
fn main() {
    use pharmsol::*;
    use std::path::PathBuf;
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

    // Create ODE model directly
    let ode = equation::ODE::new(
        |x, p, _t, dx, rateiv, _cov| {
            fetch_params!(p, ke, _v);
            dx[0] = -ke * x[0] + rateiv[0];
        },
        |_p| lag! {},
        |_p| fa! {},
        |_p, _t, _cov, _x| {},
        |x, p, _t, _cov, y| {
            fetch_params!(p, _ke, v);
            y[0] = x[0] / v;
        },
        (1, 1),
    );

    // Compile the same model using exa
    let model_path = exa::build::compile(
        format!(
            r#"
                equation::ODE::new(
            |x, p, _t, dx, rateiv, _cov| {{
                fetch_params!(p, ke, _v);
                dx[0] = -ke * x[0] + rateiv[0];
            }},
            |_p| lag! {{}},
            |_p| fa! {{}},
            |_p, _t, _cov, _x| {{}},
            |x, p, _t, _cov, y| {{
                fetch_params!(p, _ke, v);
                y[0] = x[0] / v;
            }},
            (1, 1),
        )
        "#
        ),
        Some(PathBuf::from("model.pkm")),
        vec!["ke".to_string(), "v".to_string()],
        |_, _| {}, // Empty callback for tests
    )
    .unwrap();

    // Load the compiled model
    let model_path = PathBuf::from(model_path);
    let (_lib, (dyn_ode, _meta)) = unsafe { exa::load::load_ode(model_path.clone()) };

    // Parameters for model evaluation
    let params = vec![1.02282724609375, 194.51904296875];

    // Get predictions from both models
    let dyn_predictions = dyn_ode.estimate_predictions(&subject, &params);
    let ode_predictions = ode.estimate_predictions(&subject, &params);

    // Check that predictions are the same
    let dyn_flat = dyn_predictions.flat_predictions();
    let ode_flat = ode_predictions.flat_predictions();
    println!("Dyn_model: {:#?}", dyn_flat);

    let op = ode.estimate_predictions(&subject, &vec![1.02282724609375, 194.51904296875]);
    println!("ODE: {:#?}", ode_flat);
    // Clean up
    std::fs::remove_file(model_path).ok();
}

#[cfg(not(feature = "exa"))]
fn main() {
    panic!("This example requires the 'exa' feature. Please run with `cargo run --example exa --features exa`");
}
