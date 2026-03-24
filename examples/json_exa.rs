// cargo run --example json_exa --features exa
//
// Compiles a 1-compartment IV model four different ways and checks that they
// all agree: static Rust, exa (raw Rust string), JSON ODE, JSON analytical.

#[cfg(feature = "exa")]
fn main() {
    use pharmsol::prelude::*;
    use pharmsol::{exa, json, Analytical, ODE};
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

    let params = vec![1.2, 50.0]; // ke, V

    let cwd = std::env::current_dir().unwrap();
    let template_path = std::env::temp_dir().join("exa_json_example");

    // -- Static ODE (plain Rust) ----------------------------------------------

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

    // -- Exa ODE (raw Rust string, compiled at runtime) -----------------------

    let exa_ode_path = cwd.join("exa_ode_model.pkm");
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
    .unwrap();

    let exa_ode_path = PathBuf::from(&exa_ode_compiled);
    let (_lib_exa, (dyn_exa_ode, _)) = unsafe { exa::load::load::<ODE>(exa_ode_path.clone()) };

    // -- JSON ODE -------------------------------------------------------------

    let json_ode_str = r#"{
        "schema": "1.0",
        "id": "pk_1cmt_iv_ode",
        "type": "ode",
        "parameters": ["ke", "V"],
        "compartments": ["central"],
        "diffeq": { "central": "-ke * central + rateiv[0]" },
        "output": "central / V"
    }"#;

    // Show the generated Rust before compiling
    let generated = json::generate_code(json_ode_str).unwrap();
    println!(
        "Generated code from JSON ODE:\n{}\n",
        generated.equation_code
    );

    let json_ode_path = cwd.join("json_ode_model.pkm");
    let json_ode_compiled = json::compile_json::<ODE>(
        json_ode_str,
        Some(json_ode_path.clone()),
        template_path.clone(),
        |_, _| {},
    )
    .unwrap();

    let json_ode_path = PathBuf::from(&json_ode_compiled);
    let (_lib_json_ode, (dyn_json_ode, _)) =
        unsafe { exa::load::load::<ODE>(json_ode_path.clone()) };

    // -- JSON Analytical ------------------------------------------------------

    let json_analytical_str = r#"{
        "schema": "1.0",
        "id": "pk_1cmt_iv_analytical",
        "type": "analytical",
        "analytical": "one_compartment",
        "parameters": ["ke", "V"],
        "output": "x[0] / V"
    }"#;

    let json_an_path = cwd.join("json_analytical_model.pkm");
    let json_an_compiled = json::compile_json::<Analytical>(
        json_analytical_str,
        Some(json_an_path.clone()),
        template_path.clone(),
        |_, _| {},
    )
    .unwrap();

    let json_an_path = PathBuf::from(&json_an_compiled);
    let (_lib_json_an, (dyn_json_an, _)) =
        unsafe { exa::load::load::<Analytical>(json_an_path.clone()) };

    // -- Compare predictions --------------------------------------------------

    let flat = |preds: pharmsol::prelude::SubjectPredictions| preds.flat_predictions();

    let preds_static = flat(static_ode.estimate_predictions(&subject, &params).unwrap());
    let preds_exa = flat(dyn_exa_ode.estimate_predictions(&subject, &params).unwrap());
    let preds_json_ode = flat(
        dyn_json_ode
            .estimate_predictions(&subject, &params)
            .unwrap(),
    );
    let preds_json_an = flat(dyn_json_an.estimate_predictions(&subject, &params).unwrap());

    let times = [0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0];
    println!(
        "{:<6} {:>12} {:>12} {:>12} {:>12}",
        "t", "Static", "Exa", "JSON ODE", "JSON An."
    );
    for (i, t) in times.iter().enumerate() {
        println!(
            "{:<6.1} {:>12.6} {:>12.6} {:>12.6} {:>12.6}",
            t, preds_static[i], preds_exa[i], preds_json_ode[i], preds_json_an[i]
        );
    }

    let close = |a: &[f64], b: &[f64], tol: f64| -> bool {
        a.iter().zip(b).all(|(x, y)| (x - y).abs() < tol)
    };

    println!();
    assert!(close(&preds_static, &preds_exa, 1e-10), "static != exa");
    println!("static vs exa:          OK  (< 1e-10)");
    assert!(
        close(&preds_static, &preds_json_ode, 1e-10),
        "static != json ode"
    );
    println!("static vs json ODE:     OK  (< 1e-10)");
    assert!(
        close(&preds_static, &preds_json_an, 1e-3),
        "static != json analytical"
    );
    println!("static vs json analyt.: OK  (< 1e-3)");

    // -- Builtin library models -----------------------------------------------

    let library = json::ModelLibrary::builtin();
    println!("\nBuiltin models:");
    for id in library.list() {
        println!("  {}", id);
    }

    // -- Cleanup --------------------------------------------------------------

    std::fs::remove_file(&exa_ode_path).ok();
    std::fs::remove_file(&json_ode_path).ok();
    std::fs::remove_file(&json_an_path).ok();
    std::fs::remove_dir_all(&template_path).ok();
}

#[cfg(not(feature = "exa"))]
fn main() {
    eprintln!("Run with: cargo run --example json_exa --features exa");
    std::process::exit(1);
}
