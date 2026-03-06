// Run with: cargo run --example json_comprehensive_test --features exa
//
// Comprehensive test of every JSON model feature against hand-written Rust equivalents.
// Tests:
//   1. Simple ODE (1-compartment IV, named compartment)
//   2. ODE with absorption + rateiv (2-compartment)
//   3. Analytical model (1-compartment IV)
//   4. Analytical model with absorption
//   5. ODE with lag time
//   6. ODE with bioavailability (fa)
//   7. ODE with initial conditions
//   8. ODE with derived parameters (CL/V → ke)
//   9. ODE with multiple output equations
//  10. ODE with covariates (allometric scaling)
//  11. Built-in library model compilation

#[cfg(feature = "exa")]
fn main() {
    use pharmsol::prelude::*;
    use pharmsol::{exa, json, Analytical, ODE};
    use std::path::PathBuf;

    let template_path = std::env::temp_dir().join("json_comprehensive_test");
    let test_dir = std::env::temp_dir().join("json_comprehensive_output");
    std::fs::create_dir_all(&test_dir).ok();

    let mut all_passed = true;
    let mut test_count = 0;
    let mut pass_count = 0;

    // Helper macro to run a test
    macro_rules! run_test {
        ($name:expr, $body:expr) => {{
            test_count += 1;
            print!("  Test {}: {} ... ", test_count, $name);
            match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| $body)) {
                Ok(true) => {
                    println!("PASS");
                    pass_count += 1;
                }
                Ok(false) => {
                    println!("FAIL");
                    all_passed = false;
                }
                Err(e) => {
                    println!("PANIC: {:?}", e.downcast_ref::<String>().unwrap_or(&"unknown".to_string()));
                    all_passed = false;
                }
            }
        }};
    }

    println!("═══════════════════════════════════════════════════════════════════════");
    println!("  JSON Model Comprehensive Test Suite");
    println!("═══════════════════════════════════════════════════════════════════════");

    // ═══════════════════════════════════════════════════════════════════════════
    // PART A: Code generation tests (no compilation needed)
    // ═══════════════════════════════════════════════════════════════════════════
    println!("\n── Part A: Code Generation (parse → validate → codegen) ──────────────\n");

    // Test 1: Simple ODE with named compartment
    run_test!("Simple ODE codegen (named compartment)", {
        let json_str = r#"{
            "schema": "1.0",
            "id": "test_1cmt_iv_ode",
            "type": "ode",
            "parameters": ["ke", "V"],
            "compartments": ["central"],
            "diffeq": {
                "central": "-ke * central + rateiv[0]"
            },
            "output": "central / V"
        }"#;

        let generated = json::generate_code(json_str).expect("codegen failed");
        let code = &generated.equation_code;

        // Must contain compartment binding
        assert!(code.contains("let central = x[0];"), "Missing compartment binding: {}", code);
        // Must contain the diffeq expression
        assert!(code.contains("dx[0] = -ke * central + rateiv[0];"), "Wrong diffeq: {}", code);
        // Must contain the output expression
        assert!(code.contains("y[0] = central / V;"), "Wrong output: {}", code);
        // Must contain fetch_params
        assert!(code.contains("fetch_params!(p, ke, V);"), "Missing fetch_params: {}", code);
        true
    });

    // Test 2: ODE with 2 compartments + absorption
    run_test!("Two-compartment ODE codegen", {
        let json_str = r#"{
            "schema": "1.0",
            "id": "test_2cmt_oral_ode",
            "type": "ode",
            "parameters": ["ka", "ke", "V"],
            "compartments": ["depot", "central"],
            "diffeq": {
                "depot": "-ka * depot",
                "central": "ka * depot - ke * central + rateiv[1]"
            },
            "output": "central / V"
        }"#;

        let generated = json::generate_code(json_str).expect("codegen failed");
        let code = &generated.equation_code;

        assert!(code.contains("let central = x[1];"), "Missing central binding: {}", code);
        assert!(code.contains("let depot = x[0];"), "Missing depot binding: {}", code);
        // Check deterministic ordering (depot=0 before central=1)
        let depot_pos = code.find("dx[0]").unwrap();
        let central_pos = code.find("dx[1]").unwrap();
        assert!(depot_pos < central_pos, "diffeq order wrong: depot should come before central");
        true
    });

    // Test 3: Analytical model
    run_test!("Analytical model codegen", {
        let json_str = r#"{
            "schema": "1.0",
            "id": "test_analytical",
            "type": "analytical",
            "analytical": "one_compartment_with_absorption",
            "parameters": ["ka", "ke", "V"],
            "output": "x[1] / V"
        }"#;

        let generated = json::generate_code(json_str).expect("codegen failed");
        let code = &generated.equation_code;

        assert!(code.contains("equation::Analytical::new"), "Not analytical: {}", code);
        assert!(code.contains("one_compartment_with_absorption"), "Wrong function: {}", code);
        assert!(code.contains("y[0] = x[1] / V;"), "Wrong output: {}", code);
        assert_eq!(generated.parameters, vec!["ka", "ke", "V"]);
        true
    });

    // Test 4: ODE with lag time
    run_test!("ODE with lag time codegen", {
        let json_str = r#"{
            "schema": "1.0",
            "id": "test_lag",
            "type": "ode",
            "parameters": ["ka", "ke", "V", "tlag"],
            "compartments": ["depot", "central"],
            "lag": { "depot": "tlag" },
            "diffeq": {
                "depot": "-ka * depot",
                "central": "ka * depot - ke * central"
            },
            "output": "central / V"
        }"#;

        let generated = json::generate_code(json_str).expect("codegen failed");
        let code = &generated.equation_code;

        assert!(code.contains("lag!"), "Missing lag macro: {}", code);
        assert!(code.contains("0 => tlag"), "Wrong lag entry: {}", code);
        true
    });

    // Test 5: ODE with bioavailability (fa)
    run_test!("ODE with bioavailability codegen", {
        let json_str = r#"{
            "schema": "1.0",
            "id": "test_fa",
            "type": "ode",
            "parameters": ["ka", "ke", "V", "fabs"],
            "compartments": ["depot", "central"],
            "fa": { "depot": "fabs" },
            "diffeq": {
                "depot": "-ka * depot",
                "central": "ka * depot - ke * central"
            },
            "output": "central / V"
        }"#;

        let generated = json::generate_code(json_str).expect("codegen failed");
        let code = &generated.equation_code;

        assert!(code.contains("fa!"), "Missing fa macro: {}", code);
        assert!(code.contains("0 => fabs"), "Wrong fa entry: {}", code);
        true
    });

    // Test 6: ODE with initial conditions
    run_test!("ODE with initial conditions codegen", {
        let json_str = r#"{
            "schema": "1.0",
            "id": "test_init",
            "type": "ode",
            "parameters": ["ke", "V", "A0"],
            "compartments": ["central"],
            "init": { "central": "A0" },
            "diffeq": {
                "central": "-ke * central"
            },
            "output": "central / V"
        }"#;

        let generated = json::generate_code(json_str).expect("codegen failed");
        let code = &generated.equation_code;

        assert!(code.contains("x[0] = A0;"), "Missing init: {}", code);
        true
    });

    // Test 7: ODE with derived parameters
    run_test!("ODE with derived parameters codegen", {
        let json_str = r#"{
            "schema": "1.0",
            "id": "test_derived",
            "type": "ode",
            "parameters": ["CL", "V"],
            "compartments": ["central"],
            "derived": [
                { "symbol": "ke", "expression": "CL / V" }
            ],
            "diffeq": {
                "central": "-ke * central + rateiv[0]"
            },
            "output": "central / V"
        }"#;

        let generated = json::generate_code(json_str).expect("codegen failed");
        let code = &generated.equation_code;

        assert!(code.contains("let ke = CL / V;"), "Missing derived param: {}", code);
        true
    });

    // Test 8: Model with multiple outputs
    run_test!("Multiple output equations codegen", {
        let json_str = r#"{
            "schema": "1.0",
            "id": "test_multi_out",
            "type": "ode",
            "parameters": ["ke", "V"],
            "compartments": ["central"],
            "diffeq": {
                "central": "-ke * central + rateiv[0]"
            },
            "outputs": [
                { "equation": "central / V", "id": "concentration" },
                { "equation": "central", "id": "amount" }
            ],
            "neqs": [1, 2]
        }"#;

        let generated = json::generate_code(json_str).expect("codegen failed");
        let code = &generated.equation_code;

        assert!(code.contains("y[0] = central / V;"), "Missing output 0: {}", code);
        assert!(code.contains("y[1] = central;"), "Missing output 1: {}", code);
        assert!(code.contains("(1, 2)"), "Wrong neqs: {}", code);
        true
    });

    // Test 9: SDE model codegen
    run_test!("SDE model codegen", {
        let json_str = r#"{
            "schema": "1.0",
            "id": "test_sde",
            "type": "sde",
            "parameters": ["ke0", "sigma_ke", "V"],
            "states": ["amount", "ke"],
            "drift": {
                "amount": "-ke * amount",
                "ke": "-0.5 * (ke - ke0)"
            },
            "diffusion": { "ke": "sigma_ke" },
            "init": { "ke": "ke0" },
            "output": "amount / V",
            "neqs": [2, 1],
            "particles": 1000
        }"#;

        let generated = json::generate_code(json_str).expect("codegen failed");
        let code = &generated.equation_code;

        assert!(code.contains("equation::SDE::new"), "Not SDE: {}", code);
        // Drift should have state bindings
        assert!(code.contains("let amount = x[0];"), "Missing amount binding: {}", code);
        assert!(code.contains("let ke = x[1];"), "Missing ke binding: {}", code);
        // Diffusion closure should have correct signature (p, d) - only 2 params
        assert!(code.contains("|p, d|"), "Wrong diffusion signature: {}", code);
        // Should NOT have |x, p, d| (old bug)
        assert!(!code.contains("|x, p, d|"), "Diffusion has old buggy signature: {}", code);
        assert!(code.contains("1000"), "Missing particles: {}", code);
        true
    });

    // Test 10: Analytical with lag (lag uses numeric index)
    run_test!("Analytical with lag codegen", {
        let json_str = r#"{
            "schema": "1.0",
            "id": "test_analytical_lag",
            "type": "analytical",
            "analytical": "one_compartment_with_absorption",
            "parameters": ["ka", "ke", "V", "tlag"],
            "lag": { "0": "tlag" },
            "output": "x[1] / V",
            "neqs": [2, 1]
        }"#;

        let generated = json::generate_code(json_str).expect("codegen failed");
        let code = &generated.equation_code;

        assert!(code.contains("lag!"), "Missing lag: {}", code);
        assert!(code.contains("0 => tlag"), "Wrong lag entry: {}", code);
        true
    });

    // ═══════════════════════════════════════════════════════════════════════════
    // PART B: Validation tests (no compilation needed)
    // ═══════════════════════════════════════════════════════════════════════════
    println!("\n── Part B: Validation Rules ─────────────────────────────────────────\n");

    // Test 11: Named compartments reject x[i]
    run_test!("Named compartments reject x[i] in diffeq", {
        let result = json::validate_json(r#"{
            "schema": "1.0",
            "id": "test",
            "type": "ode",
            "compartments": ["central"],
            "parameters": ["ke", "V"],
            "diffeq": { "central": "-ke * x[0]" },
            "output": "central / V"
        }"#);
        result.is_err()
    });

    // Test 12: Named compartments reject x[i] in output
    run_test!("Named compartments reject x[i] in output", {
        let result = json::validate_json(r#"{
            "schema": "1.0",
            "id": "test",
            "type": "ode",
            "compartments": ["central"],
            "parameters": ["ke", "V"],
            "diffeq": { "central": "-ke * central" },
            "output": "x[0] / V"
        }"#);
        result.is_err()
    });

    // Test 13: No compartments allows x[i]
    run_test!("No compartment names allows x[i]", {
        let result = json::validate_json(r#"{
            "schema": "1.0",
            "id": "test",
            "type": "ode",
            "parameters": ["ke", "V"],
            "diffeq": "dx[0] = -ke * x[0];",
            "output": "x[0] / V"
        }"#);
        result.is_ok()
    });

    // Test 14: Analytical always allows x[i]
    run_test!("Analytical always allows x[i]", {
        let result = json::validate_json(r#"{
            "schema": "1.0",
            "id": "test",
            "type": "analytical",
            "analytical": "one_compartment",
            "parameters": ["ke", "V"],
            "output": "x[0] / V"
        }"#);
        result.is_ok()
    });

    // Test 15: SDE with named states rejects x[i]
    run_test!("SDE named states reject x[i]", {
        let result = json::validate_json(r#"{
            "schema": "1.0",
            "id": "test",
            "type": "sde",
            "parameters": ["ke0", "sigma_ke", "V"],
            "states": ["amount", "ke"],
            "drift": { "amount": "-ke * x[0]", "ke": "-0.5 * (ke - ke0)" },
            "diffusion": { "ke": "sigma_ke" },
            "output": "amount / V",
            "neqs": [2, 1]
        }"#);
        result.is_err()
    });

    // Test 16: Duplicate parameters rejected
    run_test!("Duplicate parameters rejected", {
        let result = json::validate_json(r#"{
            "schema": "1.0",
            "id": "test",
            "type": "ode",
            "parameters": ["ke", "V", "ke"],
            "diffeq": "dx[0] = -ke * x[0];",
            "output": "x[0] / V"
        }"#);
        result.is_err()
    });

    // Test 17: Wrong model type field rejected
    run_test!("Analytical rejects diffeq field", {
        let result = json::parse_json(r#"{
            "schema": "1.0",
            "id": "test",
            "type": "analytical",
            "analytical": "one_compartment",
            "diffeq": "dx[0] = -ke * x[0];",
            "parameters": ["ke"],
            "output": "x[0]"
        }"#);
        // Note: serde's deny_unknown_fields won't reject this since diffeq IS a known field
        // But validation should catch it
        if let Ok(model) = result {
            json::validate_json(&model.to_json().unwrap()).is_err()
        } else {
            true
        }
    });

    // Test 18: Model library lists models
    run_test!("Model library has builtin models", {
        let library = json::ModelLibrary::builtin();
        let models = library.list();
        models.len() >= 10  // Should have at least 10 models
    });

    // Test 19: All library models validate
    run_test!("All library models validate", {
        let library = json::ModelLibrary::builtin();
        let mut all_valid = true;
        for id in library.list() {
            let model = library.get(id).unwrap();
            let json_str = model.to_json().unwrap();
            if let Err(e) = json::validate_json(&json_str) {
                println!("\n    FAIL: {} - {}", id, e);
                all_valid = false;
            }
        }
        all_valid
    });

    // Test 20: All library models generate code
    run_test!("All library models generate code", {
        let library = json::ModelLibrary::builtin();
        let mut all_ok = true;
        for id in library.list() {
            let model = library.get(id).unwrap();
            let json_str = model.to_json().unwrap();
            if let Err(e) = json::generate_code(&json_str) {
                println!("\n    FAIL: {} - {}", id, e);
                all_ok = false;
            }
        }
        all_ok
    });

    // ═══════════════════════════════════════════════════════════════════════════
    // PART C: Compilation + Execution Tests (requires exa feature)
    // ═══════════════════════════════════════════════════════════════════════════
    println!("\n── Part C: Compilation & Prediction Comparison ─────────────────────\n");

    // Create common test subjects
    let iv_subject = Subject::builder("iv_test")
        .infusion(0.0, 500.0, 0, 0.5)
        .observation(0.5, 0.0, 0)
        .observation(1.0, 0.0, 0)
        .observation(2.0, 0.0, 0)
        .observation(4.0, 0.0, 0)
        .observation(8.0, 0.0, 0)
        .build();

    let oral_subject = Subject::builder("oral_test")
        .bolus(0.0, 100.0, 0)
        .observation(0.5, 0.0, 0)
        .observation(1.0, 0.0, 0)
        .observation(2.0, 0.0, 0)
        .observation(4.0, 0.0, 0)
        .observation(8.0, 0.0, 0)
        .build();

    // ─────────────────────────────────────────────────────────────────────────
    // Test 21: ODE 1-compartment IV — JSON vs static Rust
    // ─────────────────────────────────────────────────────────────────────────
    run_test!("ODE 1cmt IV: JSON compiled vs static Rust", {
        let static_ode = equation::ODE::new(
            |x, p, _t, dx, _b, rateiv, _cov| {
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

        let json_ode = r#"{
            "schema": "1.0",
            "id": "test_compile_1cmt_iv",
            "type": "ode",
            "parameters": ["ke", "V"],
            "compartments": ["central"],
            "diffeq": { "central": "-ke * central + rateiv[0]" },
            "output": "central / V"
        }"#;

        let out_path = test_dir.join("test_ode_1cmt_iv.pkm");
        let compiled = json::compile_json::<ODE>(
            json_ode,
            Some(out_path.clone()),
            template_path.clone(),
            |_, _| {},
        ).expect("Compilation failed");

        let compiled_path = PathBuf::from(&compiled);
        let (_lib, (dynamic_ode, meta)) = unsafe { exa::load::load::<ODE>(compiled_path.clone()) };

        assert_eq!(meta.get_params(), &vec!["ke".to_string(), "V".to_string()]);

        let params = vec![1.2, 50.0];
        let static_preds = static_ode.estimate_predictions(&iv_subject, &params).unwrap();
        let dynamic_preds = dynamic_ode.estimate_predictions(&iv_subject, &params).unwrap();

        let sp = static_preds.flat_predictions();
        let dp = dynamic_preds.flat_predictions();

        let max_diff = sp.iter().zip(dp.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f64, f64::max);
        println!("(max diff: {:.2e})", max_diff);
        max_diff < 1e-10
    });

    // ─────────────────────────────────────────────────────────────────────────
    // Test 22: ODE 2-compartment oral — JSON vs static Rust
    // ─────────────────────────────────────────────────────────────────────────
    run_test!("ODE 2cmt oral: JSON compiled vs static Rust", {
        let static_ode = equation::ODE::new(
            |x, p, _t, dx, _b, _rateiv, _cov| {
                fetch_params!(p, ka, ke, _v);
                dx[0] = -ka * x[0];
                dx[1] = ka * x[0] - ke * x[1];
            },
            |_p, _t, _cov| lag! {},
            |_p, _t, _cov| fa! {},
            |_p, _t, _cov, _x| {},
            |x, p, _t, _cov, y| {
                fetch_params!(p, _ka, _ke, v);
                y[0] = x[1] / v;
            },
            (2, 1),
        );

        let json_ode = r#"{
            "schema": "1.0",
            "id": "test_compile_2cmt_oral",
            "type": "ode",
            "parameters": ["ka", "ke", "V"],
            "compartments": ["depot", "central"],
            "diffeq": {
                "depot": "-ka * depot",
                "central": "ka * depot - ke * central"
            },
            "output": "central / V"
        }"#;

        let out_path = test_dir.join("test_ode_2cmt_oral.pkm");
        let compiled = json::compile_json::<ODE>(
            json_ode,
            Some(out_path.clone()),
            template_path.clone(),
            |_, _| {},
        ).expect("Compilation failed");

        let compiled_path = PathBuf::from(&compiled);
        let (_lib, (dynamic_ode, _)) = unsafe { exa::load::load::<ODE>(compiled_path.clone()) };

        let params = vec![1.5, 0.5, 30.0]; // ka, ke, V
        let static_preds = static_ode.estimate_predictions(&oral_subject, &params).unwrap();
        let dynamic_preds = dynamic_ode.estimate_predictions(&oral_subject, &params).unwrap();

        let sp = static_preds.flat_predictions();
        let dp = dynamic_preds.flat_predictions();

        let max_diff = sp.iter().zip(dp.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f64, f64::max);
        println!("(max diff: {:.2e})", max_diff);
        max_diff < 1e-10
    });

    // ─────────────────────────────────────────────────────────────────────────
    // Test 23: Analytical 1-compartment — JSON vs static Rust
    // ─────────────────────────────────────────────────────────────────────────
    run_test!("Analytical 1cmt IV: JSON compiled vs static Rust", {
        let static_an = equation::Analytical::new(
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
        );

        let json_an = r#"{
            "schema": "1.0",
            "id": "test_compile_analytical_1cmt",
            "type": "analytical",
            "analytical": "one_compartment",
            "parameters": ["ke", "V"],
            "output": "x[0] / V"
        }"#;

        let out_path = test_dir.join("test_analytical_1cmt.pkm");
        let compiled = json::compile_json::<Analytical>(
            json_an,
            Some(out_path.clone()),
            template_path.clone(),
            |_, _| {},
        ).expect("Compilation failed");

        let compiled_path = PathBuf::from(&compiled);
        let (_lib, (dynamic_an, _)) = unsafe { exa::load::load::<Analytical>(compiled_path.clone()) };

        let params = vec![1.2, 50.0];
        let static_preds = static_an.estimate_predictions(&iv_subject, &params).unwrap();
        let dynamic_preds = dynamic_an.estimate_predictions(&iv_subject, &params).unwrap();

        let sp = static_preds.flat_predictions();
        let dp = dynamic_preds.flat_predictions();

        let max_diff = sp.iter().zip(dp.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f64, f64::max);
        println!("(max diff: {:.2e})", max_diff);
        max_diff < 1e-10
    });

    // ─────────────────────────────────────────────────────────────────────────
    // Test 24: Analytical with absorption — JSON vs static Rust
    // ─────────────────────────────────────────────────────────────────────────
    run_test!("Analytical 1cmt oral: JSON compiled vs static Rust", {
        let static_an = equation::Analytical::new(
            one_compartment_with_absorption,
            |_p, _t, _cov| {},
            |_p, _t, _cov| lag! {},
            |_p, _t, _cov| fa! {},
            |_p, _t, _cov, _x| {},
            |x, p, _t, _cov, y| {
                fetch_params!(p, _ka, _ke, v);
                y[0] = x[1] / v;
            },
            (2, 1),
        );

        let json_an = r#"{
            "schema": "1.0",
            "id": "test_compile_analytical_oral",
            "type": "analytical",
            "analytical": "one_compartment_with_absorption",
            "parameters": ["ka", "ke", "V"],
            "output": "x[1] / V"
        }"#;

        let out_path = test_dir.join("test_analytical_oral.pkm");
        let compiled = json::compile_json::<Analytical>(
            json_an,
            Some(out_path.clone()),
            template_path.clone(),
            |_, _| {},
        ).expect("Compilation failed");

        let compiled_path = PathBuf::from(&compiled);
        let (_lib, (dynamic_an, _)) = unsafe { exa::load::load::<Analytical>(compiled_path.clone()) };

        let params = vec![1.5, 0.5, 30.0];
        let static_preds = static_an.estimate_predictions(&oral_subject, &params).unwrap();
        let dynamic_preds = dynamic_an.estimate_predictions(&oral_subject, &params).unwrap();

        let sp = static_preds.flat_predictions();
        let dp = dynamic_preds.flat_predictions();

        let max_diff = sp.iter().zip(dp.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f64, f64::max);
        println!("(max diff: {:.2e})", max_diff);
        max_diff < 1e-10
    });

    // ─────────────────────────────────────────────────────────────────────────
    // Test 25: ODE with lag time — JSON vs static Rust
    // ─────────────────────────────────────────────────────────────────────────
    run_test!("ODE with lag time: JSON compiled vs static Rust", {
        let static_ode = equation::ODE::new(
            |x, p, _t, dx, _b, _rateiv, _cov| {
                fetch_params!(p, ka, ke, _v, _tlag);
                dx[0] = -ka * x[0];
                dx[1] = ka * x[0] - ke * x[1];
            },
            |p, _t, _cov| {
                fetch_params!(p, _ka, _ke, _v, tlag);
                lag! { 0 => tlag }
            },
            |_p, _t, _cov| fa! {},
            |_p, _t, _cov, _x| {},
            |x, p, _t, _cov, y| {
                fetch_params!(p, _ka, _ke, v, _tlag);
                y[0] = x[1] / v;
            },
            (2, 1),
        );

        let json_ode = r#"{
            "schema": "1.0",
            "id": "test_compile_lag",
            "type": "ode",
            "parameters": ["ka", "ke", "V", "tlag"],
            "compartments": ["depot", "central"],
            "lag": { "depot": "tlag" },
            "diffeq": {
                "depot": "-ka * depot",
                "central": "ka * depot - ke * central"
            },
            "output": "central / V"
        }"#;

        let out_path = test_dir.join("test_ode_lag.pkm");
        let compiled = json::compile_json::<ODE>(
            json_ode,
            Some(out_path.clone()),
            template_path.clone(),
            |_, _| {},
        ).expect("Compilation failed");

        let compiled_path = PathBuf::from(&compiled);
        let (_lib, (dynamic_ode, _)) = unsafe { exa::load::load::<ODE>(compiled_path.clone()) };

        let params = vec![1.5, 0.5, 30.0, 0.5]; // ka, ke, V, tlag=0.5h
        let static_preds = static_ode.estimate_predictions(&oral_subject, &params).unwrap();
        let dynamic_preds = dynamic_ode.estimate_predictions(&oral_subject, &params).unwrap();

        let sp = static_preds.flat_predictions();
        let dp = dynamic_preds.flat_predictions();

        let max_diff = sp.iter().zip(dp.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f64, f64::max);
        println!("(max diff: {:.2e})", max_diff);
        max_diff < 1e-10
    });

    // ─────────────────────────────────────────────────────────────────────────
    // Test 26: ODE with bioavailability (fa) — JSON vs static Rust
    // ─────────────────────────────────────────────────────────────────────────
    run_test!("ODE with bioavailability: JSON compiled vs static Rust", {
        let static_ode = equation::ODE::new(
            |x, p, _t, dx, _b, _rateiv, _cov| {
                fetch_params!(p, ka, ke, _v, _fabs);
                dx[0] = -ka * x[0];
                dx[1] = ka * x[0] - ke * x[1];
            },
            |_p, _t, _cov| lag! {},
            |p, _t, _cov| {
                fetch_params!(p, _ka, _ke, _v, fabs);
                fa! { 0 => fabs }
            },
            |_p, _t, _cov, _x| {},
            |x, p, _t, _cov, y| {
                fetch_params!(p, _ka, _ke, v, _fabs);
                y[0] = x[1] / v;
            },
            (2, 1),
        );

        let json_ode = r#"{
            "schema": "1.0",
            "id": "test_compile_fa",
            "type": "ode",
            "parameters": ["ka", "ke", "V", "fabs"],
            "compartments": ["depot", "central"],
            "fa": { "depot": "fabs" },
            "diffeq": {
                "depot": "-ka * depot",
                "central": "ka * depot - ke * central"
            },
            "output": "central / V"
        }"#;

        let out_path = test_dir.join("test_ode_fa.pkm");
        let compiled = json::compile_json::<ODE>(
            json_ode,
            Some(out_path.clone()),
            template_path.clone(),
            |_, _| {},
        ).expect("Compilation failed");

        let compiled_path = PathBuf::from(&compiled);
        let (_lib, (dynamic_ode, _)) = unsafe { exa::load::load::<ODE>(compiled_path.clone()) };

        let params = vec![1.5, 0.5, 30.0, 0.7]; // ka, ke, V, fabs=70%
        let static_preds = static_ode.estimate_predictions(&oral_subject, &params).unwrap();
        let dynamic_preds = dynamic_ode.estimate_predictions(&oral_subject, &params).unwrap();

        let sp = static_preds.flat_predictions();
        let dp = dynamic_preds.flat_predictions();

        let max_diff = sp.iter().zip(dp.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f64, f64::max);
        println!("(max diff: {:.2e})", max_diff);
        max_diff < 1e-10
    });

    // ─────────────────────────────────────────────────────────────────────────
    // Test 27: ODE with initial conditions — JSON vs static Rust
    // ─────────────────────────────────────────────────────────────────────────
    run_test!("ODE with init conditions: JSON compiled vs static Rust", {
        // Subject with no dose, just observations — initial condition provides drug
        let init_subject = Subject::builder("init_test")
            .observation(0.5, 0.0, 0)
            .observation(1.0, 0.0, 0)
            .observation(2.0, 0.0, 0)
            .observation(4.0, 0.0, 0)
            .build();

        let static_ode = equation::ODE::new(
            |x, p, _t, dx, _b, _rateiv, _cov| {
                fetch_params!(p, ke, _v, _a0);
                dx[0] = -ke * x[0];
            },
            |_p, _t, _cov| lag! {},
            |_p, _t, _cov| fa! {},
            |p, _t, _cov, x| {
                fetch_params!(p, _ke, _v, a0);
                x[0] = a0;
            },
            |x, p, _t, _cov, y| {
                fetch_params!(p, _ke, v, _a0);
                y[0] = x[0] / v;
            },
            (1, 1),
        );

        let json_ode = r#"{
            "schema": "1.0",
            "id": "test_compile_init",
            "type": "ode",
            "parameters": ["ke", "V", "A0"],
            "compartments": ["central"],
            "init": { "central": "A0" },
            "diffeq": { "central": "-ke * central" },
            "output": "central / V"
        }"#;

        let out_path = test_dir.join("test_ode_init.pkm");
        let compiled = json::compile_json::<ODE>(
            json_ode,
            Some(out_path.clone()),
            template_path.clone(),
            |_, _| {},
        ).expect("Compilation failed");

        let compiled_path = PathBuf::from(&compiled);
        let (_lib, (dynamic_ode, _)) = unsafe { exa::load::load::<ODE>(compiled_path.clone()) };

        let params = vec![0.5, 20.0, 100.0]; // ke, V, A0
        let static_preds = static_ode.estimate_predictions(&init_subject, &params).unwrap();
        let dynamic_preds = dynamic_ode.estimate_predictions(&init_subject, &params).unwrap();

        let sp = static_preds.flat_predictions();
        let dp = dynamic_preds.flat_predictions();

        let max_diff = sp.iter().zip(dp.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f64, f64::max);
        println!("(max diff: {:.2e})", max_diff);
        max_diff < 1e-10
    });

    // ─────────────────────────────────────────────────────────────────────────
    // Test 28: ODE with derived parameters — JSON vs static Rust
    // ─────────────────────────────────────────────────────────────────────────
    run_test!("ODE with derived params: JSON compiled vs static Rust", {
        let static_ode = equation::ODE::new(
            |x, p, _t, dx, _b, rateiv, _cov| {
                fetch_params!(p, cl, v);
                let ke = cl / v;
                dx[0] = -ke * x[0] + rateiv[0];
            },
            |_p, _t, _cov| lag! {},
            |_p, _t, _cov| fa! {},
            |_p, _t, _cov, _x| {},
            |x, p, _t, _cov, y| {
                fetch_params!(p, _cl, v);
                y[0] = x[0] / v;
            },
            (1, 1),
        );

        let json_ode = r#"{
            "schema": "1.0",
            "id": "test_compile_derived",
            "type": "ode",
            "parameters": ["CL", "V"],
            "compartments": ["central"],
            "derived": [{ "symbol": "ke", "expression": "CL / V" }],
            "diffeq": { "central": "-ke * central + rateiv[0]" },
            "output": "central / V"
        }"#;

        let out_path = test_dir.join("test_ode_derived.pkm");
        let compiled = json::compile_json::<ODE>(
            json_ode,
            Some(out_path.clone()),
            template_path.clone(),
            |_, _| {},
        ).expect("Compilation failed");

        let compiled_path = PathBuf::from(&compiled);
        let (_lib, (dynamic_ode, _)) = unsafe { exa::load::load::<ODE>(compiled_path.clone()) };

        // CL=60 L/h, V=50 L → ke=1.2
        let params = vec![60.0, 50.0];
        let static_preds = static_ode.estimate_predictions(&iv_subject, &params).unwrap();
        let dynamic_preds = dynamic_ode.estimate_predictions(&iv_subject, &params).unwrap();

        let sp = static_preds.flat_predictions();
        let dp = dynamic_preds.flat_predictions();

        let max_diff = sp.iter().zip(dp.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f64, f64::max);
        println!("(max diff: {:.2e})", max_diff);
        max_diff < 1e-10
    });

    // ─────────────────────────────────────────────────────────────────────────
    // Test 29: ODE with multiple outputs — JSON vs static Rust
    // ─────────────────────────────────────────────────────────────────────────
    run_test!("ODE multiple outputs: JSON compiled vs static Rust", {
        // Subject with observations on two output equations
        let multi_out_subject = Subject::builder("multi_out_test")
            .infusion(0.0, 500.0, 0, 0.5)
            .observation(1.0, 0.0, 0)  // concentration (outeq 0)
            .observation(1.0, 0.0, 1)  // amount (outeq 1)
            .observation(4.0, 0.0, 0)
            .observation(4.0, 0.0, 1)
            .build();

        let static_ode = equation::ODE::new(
            |x, p, _t, dx, _b, rateiv, _cov| {
                fetch_params!(p, ke, _v);
                dx[0] = -ke * x[0] + rateiv[0];
            },
            |_p, _t, _cov| lag! {},
            |_p, _t, _cov| fa! {},
            |_p, _t, _cov, _x| {},
            |x, p, _t, _cov, y| {
                fetch_params!(p, _ke, v);
                y[0] = x[0] / v; // concentration
                y[1] = x[0];     // amount
            },
            (1, 2),
        );

        let json_ode = r#"{
            "schema": "1.0",
            "id": "test_compile_multi_out",
            "type": "ode",
            "parameters": ["ke", "V"],
            "compartments": ["central"],
            "diffeq": { "central": "-ke * central + rateiv[0]" },
            "outputs": [
                { "equation": "central / V", "id": "concentration" },
                { "equation": "central", "id": "amount" }
            ],
            "neqs": [1, 2]
        }"#;

        let out_path = test_dir.join("test_ode_multi_out.pkm");
        let compiled = json::compile_json::<ODE>(
            json_ode,
            Some(out_path.clone()),
            template_path.clone(),
            |_, _| {},
        ).expect("Compilation failed");

        let compiled_path = PathBuf::from(&compiled);
        let (_lib, (dynamic_ode, _)) = unsafe { exa::load::load::<ODE>(compiled_path.clone()) };

        let params = vec![1.2, 50.0];
        let static_preds = static_ode.estimate_predictions(&multi_out_subject, &params).unwrap();
        let dynamic_preds = dynamic_ode.estimate_predictions(&multi_out_subject, &params).unwrap();

        let sp = static_preds.flat_predictions();
        let dp = dynamic_preds.flat_predictions();

        let max_diff = sp.iter().zip(dp.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f64, f64::max);
        println!("(max diff: {:.2e})", max_diff);
        max_diff < 1e-10
    });

    // ─────────────────────────────────────────────────────────────────────────
    // Test 30: ODE with covariates — JSON vs static Rust
    // ─────────────────────────────────────────────────────────────────────────
    run_test!("ODE with covariates: JSON compiled vs static Rust", {
        // Subject with weight covariate
        let cov_subject = Subject::builder("cov_test")
            .infusion(0.0, 500.0, 0, 0.5)
            .covariate("WT", 0.0, 85.0)
            .observation(0.5, 0.0, 0)
            .observation(1.0, 0.0, 0)
            .observation(2.0, 0.0, 0)
            .observation(4.0, 0.0, 0)
            .build();

        let static_ode = equation::ODE::new(
            |x, p, t, dx, _b, rateiv, cov| {
                fetch_params!(p, ke, _v);
                fetch_cov!(cov, t, WT);
                let ke = ke * (WT / 70.0).powf(0.7500);
                dx[0] = -ke * x[0] + rateiv[0];
            },
            |_p, _t, _cov| lag! {},
            |_p, _t, _cov| fa! {},
            |_p, _t, _cov, _x| {},
            |x, p, t, cov, y| {
                fetch_params!(p, _ke, v);
                fetch_cov!(cov, t, WT);
                let v = v * (WT / 70.0).powf(1.0000);
                y[0] = x[0] / v;
            },
            (1, 1),
        );

        let json_ode = r#"{
            "schema": "1.0",
            "id": "test_compile_covariates",
            "type": "ode",
            "parameters": ["ke", "V"],
            "compartments": ["central"],
            "covariates": [{ "id": "WT", "reference": 70.0 }],
            "covariateEffects": [
                {
                    "on": "ke",
                    "covariate": "WT",
                    "type": "allometric",
                    "exponent": 0.75,
                    "reference": 70.0
                },
                {
                    "on": "V",
                    "covariate": "WT",
                    "type": "allometric",
                    "exponent": 1.0,
                    "reference": 70.0
                }
            ],
            "diffeq": { "central": "-ke * central + rateiv[0]" },
            "output": "central / V"
        }"#;

        let out_path = test_dir.join("test_ode_covariates.pkm");
        let compiled = json::compile_json::<ODE>(
            json_ode,
            Some(out_path.clone()),
            template_path.clone(),
            |_, _| {},
        ).expect("Compilation failed");

        let compiled_path = PathBuf::from(&compiled);
        let (_lib, (dynamic_ode, _)) = unsafe { exa::load::load::<ODE>(compiled_path.clone()) };

        let params = vec![1.2, 50.0]; // ke, V
        let static_preds = static_ode.estimate_predictions(&cov_subject, &params).unwrap();
        let dynamic_preds = dynamic_ode.estimate_predictions(&cov_subject, &params).unwrap();

        let sp = static_preds.flat_predictions();
        let dp = dynamic_preds.flat_predictions();

        let max_diff = sp.iter().zip(dp.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f64, f64::max);
        println!("(max diff: {:.2e})", max_diff);
        max_diff < 1e-10
    });

    // ─────────────────────────────────────────────────────────────────────────
    // Test 31: Analytical with two compartments — JSON vs static Rust
    // ─────────────────────────────────────────────────────────────────────────
    run_test!("Analytical 2cmt IV: JSON compiled vs static Rust", {
        let static_an = equation::Analytical::new(
            two_compartments,
            |_p, _t, _cov| {},
            |_p, _t, _cov| lag! {},
            |_p, _t, _cov| fa! {},
            |_p, _t, _cov, _x| {},
            |x, p, _t, _cov, y| {
                fetch_params!(p, _ke, _kcp, _kpc, v);
                y[0] = x[0] / v;
            },
            (2, 1),
        );

        let json_an = r#"{
            "schema": "1.0",
            "id": "test_compile_analytical_2cmt",
            "type": "analytical",
            "analytical": "two_compartments",
            "parameters": ["ke", "kcp", "kpc", "V"],
            "output": "x[0] / V",
            "neqs": [2, 1]
        }"#;

        let out_path = test_dir.join("test_analytical_2cmt.pkm");
        let compiled = json::compile_json::<Analytical>(
            json_an,
            Some(out_path.clone()),
            template_path.clone(),
            |_, _| {},
        ).expect("Compilation failed");

        let compiled_path = PathBuf::from(&compiled);
        let (_lib, (dynamic_an, _)) = unsafe { exa::load::load::<Analytical>(compiled_path.clone()) };

        let params = vec![0.5, 0.3, 0.2, 50.0]; // ke, kcp, kpc, V
        let static_preds = static_an.estimate_predictions(&iv_subject, &params).unwrap();
        let dynamic_preds = dynamic_an.estimate_predictions(&iv_subject, &params).unwrap();

        let sp = static_preds.flat_predictions();
        let dp = dynamic_preds.flat_predictions();

        let max_diff = sp.iter().zip(dp.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f64, f64::max);
        println!("(max diff: {:.2e})", max_diff);
        max_diff < 1e-10
    });

    // ─────────────────────────────────────────────────────────────────────────
    // Test 32: ODE with lag + fa combined
    // ─────────────────────────────────────────────────────────────────────────
    run_test!("ODE with lag + fa combined: JSON compiled vs static Rust", {
        let static_ode = equation::ODE::new(
            |x, p, _t, dx, _b, _rateiv, _cov| {
                fetch_params!(p, ka, ke, _v, _tlag, _fabs);
                dx[0] = -ka * x[0];
                dx[1] = ka * x[0] - ke * x[1];
            },
            |p, _t, _cov| {
                fetch_params!(p, _ka, _ke, _v, tlag, _fabs);
                lag! { 0 => tlag }
            },
            |p, _t, _cov| {
                fetch_params!(p, _ka, _ke, _v, _tlag, fabs);
                fa! { 0 => fabs }
            },
            |_p, _t, _cov, _x| {},
            |x, p, _t, _cov, y| {
                fetch_params!(p, _ka, _ke, v, _tlag, _fabs);
                y[0] = x[1] / v;
            },
            (2, 1),
        );

        let json_ode = r#"{
            "schema": "1.0",
            "id": "test_compile_lag_fa",
            "type": "ode",
            "parameters": ["ka", "ke", "V", "tlag", "fabs"],
            "compartments": ["depot", "central"],
            "lag": { "depot": "tlag" },
            "fa": { "depot": "fabs" },
            "diffeq": {
                "depot": "-ka * depot",
                "central": "ka * depot - ke * central"
            },
            "output": "central / V"
        }"#;

        let out_path = test_dir.join("test_ode_lag_fa.pkm");
        let compiled = json::compile_json::<ODE>(
            json_ode,
            Some(out_path.clone()),
            template_path.clone(),
            |_, _| {},
        ).expect("Compilation failed");

        let compiled_path = PathBuf::from(&compiled);
        let (_lib, (dynamic_ode, _)) = unsafe { exa::load::load::<ODE>(compiled_path.clone()) };

        let params = vec![1.5, 0.5, 30.0, 0.3, 0.8]; // ka, ke, V, tlag, fabs
        let static_preds = static_ode.estimate_predictions(&oral_subject, &params).unwrap();
        let dynamic_preds = dynamic_ode.estimate_predictions(&oral_subject, &params).unwrap();

        let sp = static_preds.flat_predictions();
        let dp = dynamic_preds.flat_predictions();

        let max_diff = sp.iter().zip(dp.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f64, f64::max);
        println!("(max diff: {:.2e})", max_diff);
        max_diff < 1e-10
    });

    // ═══════════════════════════════════════════════════════════════════════════
    // Summary
    // ═══════════════════════════════════════════════════════════════════════════
    println!("\n═══════════════════════════════════════════════════════════════════════");
    println!("  Results: {}/{} tests passed", pass_count, test_count);
    println!("═══════════════════════════════════════════════════════════════════════");

    // Clean up
    println!("\nCleaning up...");
    std::fs::remove_dir_all(&test_dir).ok();
    std::fs::remove_dir_all(&template_path).ok();
    println!("Done!");

    if !all_passed {
        std::process::exit(1);
    }
}

#[cfg(not(feature = "exa"))]
fn main() {
    eprintln!("This example requires the 'exa' feature.");
    eprintln!("Run with: cargo run --example json_comprehensive_test --features exa");
    std::process::exit(1);
}
