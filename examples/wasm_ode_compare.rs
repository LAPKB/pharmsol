//cargo run --example wasm_ode_compare --features exa

fn main() {
    use pharmsol::{equation, exa_wasm, *};
    // use std::path::PathBuf; // not needed

    let subject = Subject::builder("1")
        .infusion(0.0, 500.0, 0, 0.5)
        .observation(0.5, 1.645776, 0)
        .observation(1.0, 1.216442, 0)
        .observation(2.0, 0.4622729, 0)
        .observation(3.0, 0.1697458, 0)
        .observation(4.0, 0.06382178, 0)
        .observation(6.0, 0.009099384, 0)
        .observation(8.0, 0.001017932, 0)
        .missing_observation(12.0, 0)
        .build();

    // Regular ODE model
    let ode = equation::ODE::new(
        |x, p, _t, dx, _b, rateiv, _cov| {
            fetch_params!(p, ke, _v);
            if true {
                dx[0] = -ke * x[0] + rateiv[0];
            }
            // dx[0] = -ke * x[0] + rateiv[0];
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

    // Compile WASM IR model using exa (interpreter, not native dynlib)
    let test_dir = std::env::current_dir().expect("Failed to get current directory");
    let ir_path = test_dir.join("test_model_ir.pkm");
    // This emits a JSON IR file for the same ODE model
    let _ir_file = exa_wasm::build::emit_ir::<equation::ODE>(
        "|x, p, _t, dx, rateiv, _cov| {
            fetch_params!(p, ke, _v);
            if true {
                if true {
                    dx[0] = -ke * x[0] + rateiv[0];
                }
            }
        }"
        .to_string(),
        None,
        None,
        Some("|p, _t, _cov, x| { }".to_string()),
        Some(
            "|x, p, _t, _cov, y| {
            fetch_params!(p, _ke, v);
            y[0] = x[0] / v;
        }"
            .to_string(),
        ),
        Some(ir_path.clone()),
        vec!["ke".to_string(), "v".to_string()],
    )
    .expect("emit_ir failed");

    //debug the contents of the ir file
    let ir_contents = std::fs::read_to_string(&ir_path).expect("Failed to read IR file");
    println!("Generated IR file contents:\n{}", ir_contents);

    // Load the IR model using the WASM-capable interpreter
    let (wasm_ode, _meta, _id) =
        exa_wasm::interpreter::load_ir_ode(ir_path.clone()).expect("load_ir_ode failed");

    let params = vec![1.02282724609375, 194.51904296875];

    // Get predictions from both models
    let ode_predictions = ode.estimate_predictions(&subject, &params).unwrap();
    let wasm_predictions = wasm_ode.estimate_predictions(&subject, &params).unwrap();

    // Display predictions side by side
    println!("Predictions:");
    println!("ODE\tWASM ODE\tDifference");
    ode_predictions
        .flat_predictions()
        .iter()
        .zip(wasm_predictions.flat_predictions())
        .for_each(|(a, b)| println!("{:.9}\t{:.9}\t{:.9}", a, b, a - b));

    // Optionally, display likelihoods
    let mut ems = ErrorModels::new()
        .add(
            0,
            ErrorModel::additive(ErrorPoly::new(0.0, 0.05, 0.0, 0.0), 0.0),
        )
        .unwrap();
    ems = ems
        .add(
            1,
            ErrorModel::proportional(ErrorPoly::new(0.0, 0.05, 0.0, 0.0), 0.0),
        )
        .unwrap();
    let ll_ode = ode
        .estimate_likelihood(&subject, &params, &ems, false)
        .unwrap();
    let ll_wasm = wasm_ode
        .estimate_likelihood(&subject, &params, &ems, false)
        .unwrap();
    println!("\nLikelihoods:");
    println!("ODE\tWASM ODE");
    println!("{:.6}\t{:.6}", -2.0 * ll_ode, -2.0 * ll_wasm);

    // Clean up
    std::fs::remove_file(ir_path).ok();
}
