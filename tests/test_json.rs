//! Integration tests for the JSON model system
//!
//! These tests validate the complete pipeline from JSON parsing to code generation.

use pharmsol::json::{
    generate_code, parse_json, validate_json, CodeGenerator, JsonModel, ModelLibrary, ModelType,
    Validator,
};

// ═══════════════════════════════════════════════════════════════════════════════
// Parsing Tests
// ═══════════════════════════════════════════════════════════════════════════════

mod parsing {
    use super::*;

    #[test]
    fn test_parse_complete_analytical_model() {
        let json = r#"{
            "schema": "1.0",
            "id": "pk_2cmt_oral",
            "type": "analytical",
            "version": "1.0.0",
            "analytical": "two_compartments_with_absorption",
            "parameters": ["ke", "ka", "kcp", "kpc", "V"],
            "output": "x[1] / V",
            "neqs": [3, 1],
            "display": {
                "name": "Two-Compartment Oral",
                "category": "pk",
                "tags": ["2-compartment", "oral"]
            },
            "documentation": {
                "summary": "Standard two-compartment oral PK model"
            }
        }"#;

        let model = parse_json(json).expect("Should parse successfully");
        assert_eq!(model.id, "pk_2cmt_oral");
        assert_eq!(model.model_type, ModelType::Analytical);
        assert_eq!(model.parameters.as_ref().unwrap().len(), 5);
    }

    #[test]
    fn test_parse_complete_ode_model() {
        let json = r#"{
            "schema": "1.0",
            "id": "pk_mm_1cmt",
            "type": "ode",
            "parameters": ["Vmax", "Km", "V"],
            "compartments": ["central"],
            "diffeq": {
                "central": "-Vmax * (central/V) / (Km + central/V)"
            },
            "output": "central / V",
            "neqs": [1, 1]
        }"#;

        let model = parse_json(json).expect("Should parse successfully");
        assert_eq!(model.model_type, ModelType::Ode);
        assert!(model.diffeq.is_some());
    }

    #[test]
    fn test_parse_with_covariates() {
        let json = r#"{
            "schema": "1.0",
            "id": "pk_1cmt_wt",
            "type": "analytical",
            "analytical": "one_compartment",
            "parameters": ["ke", "V"],
            "output": "x[0] / V",
            "covariates": [
                { "id": "WT", "reference": 70.0, "units": "kg" }
            ],
            "covariateEffects": [
                {
                    "covariate": "WT",
                    "on": "V",
                    "type": "allometric",
                    "exponent": 0.75,
                    "reference": 70.0
                }
            ]
        }"#;

        let model = parse_json(json).expect("Should parse successfully");
        assert!(model.covariates.is_some());
        assert!(model.covariate_effects.is_some());
        assert_eq!(model.covariate_effects.as_ref().unwrap().len(), 1);
    }

    #[test]
    fn test_parse_with_lag_and_fa() {
        let json = r#"{
            "schema": "1.0",
            "id": "pk_1cmt_lag",
            "type": "ode",
            "parameters": ["ka", "CL", "V", "APTS", "FFA"],
            "compartments": ["depot", "central"],
            "diffeq": {
                "depot": "-ka * depot",
                "central": "ka * depot - CL/V * central"
            },
            "output": "central / V",
            "lag": {
                "depot": "APTS"
            },
            "fa": {
                "depot": "FFA"
            }
        }"#;

        let model = parse_json(json).expect("Should parse successfully");
        assert!(model.lag.is_some());
        assert!(model.fa.is_some());
    }

    #[test]
    fn test_reject_unknown_fields() {
        let json = r#"{
            "schema": "1.0",
            "id": "bad_model",
            "type": "ode",
            "unknownField": "should fail"
        }"#;

        let result = parse_json(json);
        assert!(result.is_err());
    }

    #[test]
    fn test_reject_unsupported_schema() {
        let json = r#"{
            "schema": "99.0",
            "id": "future_model",
            "type": "ode"
        }"#;

        let result = parse_json(json);
        assert!(result.is_err());
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Validation Tests
// ═══════════════════════════════════════════════════════════════════════════════

mod validation {
    use super::*;

    #[test]
    fn test_validate_complete_model() {
        let json = r#"{
            "schema": "1.0",
            "id": "pk_1cmt",
            "type": "analytical",
            "analytical": "one_compartment",
            "parameters": ["ke", "V"],
            "output": "x[0] / V"
        }"#;

        let validated = validate_json(json).expect("Should validate successfully");
        assert_eq!(validated.inner().id, "pk_1cmt");
    }

    #[test]
    fn test_validate_rejects_missing_analytical() {
        let json = r#"{
            "schema": "1.0",
            "id": "bad_analytical",
            "type": "analytical",
            "parameters": ["ke", "V"],
            "output": "x[0] / V"
        }"#;

        let result = validate_json(json);
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_rejects_missing_diffeq() {
        let json = r#"{
            "schema": "1.0",
            "id": "bad_ode",
            "type": "ode",
            "parameters": ["ke", "V"],
            "output": "x[0] / V"
        }"#;

        let result = validate_json(json);
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_rejects_duplicate_parameters() {
        let json = r#"{
            "schema": "1.0",
            "id": "dup_params",
            "type": "analytical",
            "analytical": "one_compartment",
            "parameters": ["ke", "V", "ke"],
            "output": "x[0] / V"
        }"#;

        let result = validate_json(json);
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_ode_with_compartments() {
        let json = r#"{
            "schema": "1.0",
            "id": "ode_with_cmt",
            "type": "ode",
            "parameters": ["ka", "CL", "V"],
            "compartments": ["depot", "central"],
            "diffeq": {
                "depot": "-ka * depot",
                "central": "ka * depot - CL/V * central"
            },
            "output": "central / V"
        }"#;

        let validated = validate_json(json).expect("Should validate successfully");
        assert_eq!(validated.inner().compartments.as_ref().unwrap().len(), 2);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Code Generation Tests
// ═══════════════════════════════════════════════════════════════════════════════

mod codegen {
    use super::*;

    #[test]
    fn test_generate_analytical_code() {
        let json = r#"{
            "schema": "1.0",
            "id": "pk_1cmt",
            "type": "analytical",
            "analytical": "one_compartment_with_absorption",
            "parameters": ["ka", "ke", "V"],
            "output": "x[1] / V"
        }"#;

        let code = generate_code(json).expect("Should generate code");

        // Check generated code contains expected elements
        assert!(code.equation_code.contains("Analytical::new"));
        assert!(code
            .equation_code
            .contains("one_compartment_with_absorption"));
        assert!(code.equation_code.contains("fetch_params!"));
        assert!(code.equation_code.contains("y[0] = x[1] / V"));

        assert_eq!(code.parameters, vec!["ka", "ke", "V"]);
    }

    #[test]
    fn test_generate_ode_code() {
        let json = r#"{
            "schema": "1.0",
            "id": "pk_1cmt_ode",
            "type": "ode",
            "parameters": ["CL", "V"],
            "compartments": ["central"],
            "diffeq": {
                "central": "-CL/V * central"
            },
            "output": "central / V"
        }"#;

        let code = generate_code(json).expect("Should generate code");

        assert!(code.equation_code.contains("ODE::new"));
        assert!(code.equation_code.contains("fetch_params!"));
        // ODE uses dx[idx] = expression format
        assert!(code.equation_code.contains("dx[0]"));
    }

    #[test]
    fn test_generate_with_lag() {
        let json = r#"{
            "schema": "1.0",
            "id": "pk_with_lag",
            "type": "ode",
            "parameters": ["ka", "CL", "V", "APTS"],
            "compartments": ["depot", "central"],
            "diffeq": {
                "depot": "-ka * depot",
                "central": "ka * depot - CL/V * central"
            },
            "output": "central / V",
            "lag": {
                "depot": "APTS"
            }
        }"#;

        let code = generate_code(json).expect("Should generate code");

        assert!(code.equation_code.contains("lag!"));
        // depot is compartment 0, so should be "0 => APTS"
        assert!(code.equation_code.contains("=> APTS"));
    }

    #[test]
    fn test_generate_with_init() {
        let json = r#"{
            "schema": "1.0",
            "id": "pk_with_init",
            "type": "ode",
            "parameters": ["CL", "V", "A0"],
            "compartments": ["central"],
            "diffeq": {
                "central": "-CL/V * central"
            },
            "init": {
                "central": "A0"
            },
            "output": "central / V"
        }"#;

        let code = generate_code(json).expect("Should generate code");

        assert!(code.equation_code.contains("x[0] = A0"));
    }

    #[test]
    fn test_generate_with_covariates() {
        let json = r#"{
            "schema": "1.0",
            "id": "pk_cov",
            "type": "analytical",
            "analytical": "one_compartment",
            "parameters": ["ke", "V"],
            "output": "x[0] / V",
            "covariates": [
                { "id": "WT", "reference": 70.0 }
            ],
            "covariateEffects": [
                {
                    "covariate": "WT",
                    "on": "V",
                    "type": "allometric",
                    "exponent": 0.75,
                    "reference": 70.0
                }
            ]
        }"#;

        let code = generate_code(json).expect("Should generate code");

        // Should include covariate access and effect
        assert!(code.equation_code.contains("cov.get_covariate"));
        // Allometric: V * (WT / ref)^exp
        assert!(code.equation_code.contains("powf"));
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Library Tests
// ═══════════════════════════════════════════════════════════════════════════════

mod library {
    use super::*;

    #[test]
    fn test_builtin_library_contains_standard_models() {
        let library = ModelLibrary::builtin();

        // Should have all expected models
        assert!(library.contains("pk/1cmt-iv"));
        assert!(library.contains("pk/1cmt-oral"));
        assert!(library.contains("pk/2cmt-iv"));
        assert!(library.contains("pk/2cmt-oral"));
        assert!(library.contains("pk/1cmt-iv-ode"));
        assert!(library.contains("pk/1cmt-oral-ode"));
    }

    #[test]
    fn test_library_search() {
        let library = ModelLibrary::builtin();

        // Search by ID substring
        let oral_models = library.search("oral");
        assert!(!oral_models.is_empty());
        assert!(oral_models.iter().all(|m| m.id.contains("oral")));
    }

    #[test]
    fn test_library_filter_by_type() {
        let library = ModelLibrary::builtin();

        let analytical = library.filter_by_type(ModelType::Analytical);
        let ode = library.filter_by_type(ModelType::Ode);

        assert!(!analytical.is_empty());
        assert!(!ode.is_empty());

        // All filtered models should have correct type
        assert!(analytical
            .iter()
            .all(|m| m.model_type == ModelType::Analytical));
        assert!(ode.iter().all(|m| m.model_type == ModelType::Ode));
    }

    #[test]
    fn test_library_filter_by_tag() {
        let library = ModelLibrary::builtin();

        let oral_models = library.filter_by_tag("oral");
        assert!(!oral_models.is_empty());
    }

    #[test]
    fn test_library_inheritance() {
        let mut library = ModelLibrary::new();

        // Add base model
        let base = JsonModel::from_str(
            r#"{
            "schema": "1.0",
            "id": "base/pk-1cmt",
            "type": "analytical",
            "analytical": "one_compartment",
            "parameters": ["ke", "V"],
            "output": "x[0] / V",
            "display": {
                "name": "Base One-Compartment",
                "category": "pk"
            }
        }"#,
        )
        .unwrap();
        library.add(base);

        // Create derived model with weight covariate
        let derived = JsonModel::from_str(
            r#"{
            "schema": "1.0",
            "id": "derived/pk-1cmt-wt",
            "extends": "base/pk-1cmt",
            "type": "analytical",
            "analytical": "one_compartment",
            "parameters": ["ke", "V"],
            "covariates": [
                { "id": "WT", "reference": 70.0 }
            ],
            "covariateEffects": [
                {
                    "covariate": "WT",
                    "on": "V",
                    "type": "allometric",
                    "exponent": 0.75,
                    "reference": 70.0
                }
            ]
        }"#,
        )
        .unwrap();

        let resolved = library.resolve(&derived).unwrap();

        // Should inherit output from base
        assert!(resolved.output.is_some());
        assert_eq!(resolved.output.as_ref().unwrap(), "x[0] / V");

        // Should have covariates from derived
        assert!(resolved.covariates.is_some());
        assert!(resolved.covariate_effects.is_some());
    }

    #[test]
    fn test_library_generates_code_from_model() {
        let library = ModelLibrary::builtin();

        let model = library.get("pk/1cmt-oral").unwrap();
        let generator = CodeGenerator::new(model);
        let code = generator.generate().expect("Should generate code");

        assert!(code
            .equation_code
            .contains("one_compartment_with_absorption"));
        assert_eq!(code.parameters, vec!["ka", "ke", "V"]);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// End-to-End Tests
// ═══════════════════════════════════════════════════════════════════════════════

mod end_to_end {
    use super::*;

    #[test]
    fn test_full_pipeline_analytical() {
        // 1. Define model in JSON
        let json = r#"{
            "schema": "1.0",
            "id": "e2e_1cmt",
            "type": "analytical",
            "analytical": "one_compartment_with_absorption",
            "parameters": ["ka", "ke", "V"],
            "output": "x[1] / V",
            "display": {
                "name": "E2E Test Model",
                "category": "pk"
            }
        }"#;

        // 2. Parse
        let model = parse_json(json).unwrap();
        assert_eq!(model.id, "e2e_1cmt");

        // 3. Validate
        let validator = Validator::new();
        let validated = validator.validate(&model).unwrap();

        // 4. Generate code
        let generator = CodeGenerator::new(validated.inner());
        let code = generator.generate().unwrap();

        // 5. Verify code is valid Rust syntax (basic check)
        assert!(code.equation_code.contains("Analytical::new"));
        assert!(!code.equation_code.is_empty());
        assert_eq!(code.parameters.len(), 3);
    }

    #[test]
    fn test_full_pipeline_ode() {
        let json = r#"{
            "schema": "1.0",
            "id": "e2e_mm",
            "type": "ode",
            "parameters": ["Vmax", "Km", "V"],
            "compartments": ["central"],
            "diffeq": {
                "central": "-Vmax * (central/V) / (Km + central/V)"
            },
            "output": "central / V"
        }"#;

        // Full pipeline
        let code = generate_code(json).unwrap();

        assert!(code.equation_code.contains("ODE::new"));
        assert!(code.equation_code.contains("Vmax"));
        assert!(code.equation_code.contains("Km"));
    }

    #[test]
    fn test_library_to_code_pipeline() {
        let library = ModelLibrary::builtin();

        // Get all models and verify they all generate valid code
        for id in library.list() {
            let model = library.get(id).unwrap();
            let generator = CodeGenerator::new(model);
            let result = generator.generate();

            assert!(result.is_ok(), "Failed to generate code for model: {}", id);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// EXA Compilation Tests (requires `exa` feature)
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(feature = "exa")]
mod exa_integration {
    use approx::assert_relative_eq;
    use pharmsol::json::compile_json;
    use pharmsol::{equation, exa, Equation, Subject, SubjectBuilderExt, ODE};
    use pharmsol::{fa, fetch_params, lag};
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicUsize, Ordering};

    // Unique counter for test file names
    static TEST_COUNTER: AtomicUsize = AtomicUsize::new(0);

    fn unique_model_path(prefix: &str) -> PathBuf {
        let count = TEST_COUNTER.fetch_add(1, Ordering::SeqCst);
        let pid = std::process::id();
        std::env::current_dir()
            .expect("Failed to get current directory")
            .join(format!(
                "{}_{}_{}_{}.pkm",
                prefix,
                pid,
                count,
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            ))
    }

    /// Create a unique temp path for each test to avoid race conditions
    fn unique_temp_path() -> PathBuf {
        let count = TEST_COUNTER.fetch_add(1, Ordering::SeqCst);
        let pid = std::process::id();
        std::env::temp_dir().join(format!("exa_test_{}_{}", pid, count))
    }

    #[test]
    fn test_compile_json_ode_model() {
        // Define a simple ODE model in JSON
        let json = r#"{
            "schema": "1.0",
            "id": "test_compiled_ode",
            "type": "ode",
            "parameters": ["ke", "V"],
            "compartments": ["central"],
            "diffeq": {
                "central": "-ke * central + rateiv[0]"
            },
            "output": "central / V"
        }"#;

        let model_output_path = unique_model_path("test_json_compiled");
        let template_path = unique_temp_path();

        // Compile using compile_json
        let model_path = compile_json::<ODE>(
            json,
            Some(model_output_path.clone()),
            template_path.clone(),
            |_, _| {}, // Empty callback for tests
        )
        .expect("compile_json should succeed");

        // Load the compiled model
        let model_path = PathBuf::from(&model_path);
        let (_lib, (dyn_ode, _meta)) = unsafe { exa::load::load::<ODE>(model_path.clone()) };

        // Create a test subject
        let subject = Subject::builder("1")
            .infusion(0.0, 500.0, 0, 0.5)
            .observation(0.5, 1.5, 0)
            .observation(1.0, 1.2, 0)
            .observation(2.0, 0.5, 0)
            .build();

        // Test that the model produces predictions
        let params = vec![1.0, 100.0]; // ke=1.0, V=100
        let predictions = dyn_ode.estimate_predictions(&subject, &params);
        assert!(predictions.is_ok(), "Should produce predictions");

        let preds = predictions.unwrap().flat_predictions();
        assert_eq!(preds.len(), 3, "Should have 3 predictions");

        // Predictions should be positive (concentrations)
        for p in &preds {
            assert!(*p > 0.0, "Concentration should be positive");
        }

        // Clean up
        std::fs::remove_file(model_path).ok();
        std::fs::remove_dir_all(template_path).ok();
    }

    #[test]
    fn test_compile_json_matches_handwritten_ode() {
        // Define model in JSON
        let json = r#"{
            "schema": "1.0",
            "id": "compare_ode",
            "type": "ode",
            "parameters": ["ke", "V"],
            "compartments": ["central"],
            "diffeq": {
                "central": "-ke * central + rateiv[0]"
            },
            "output": "central / V"
        }"#;

        // Compile JSON model
        let model_output_path = unique_model_path("test_json_vs_handwritten");
        let template_path = unique_temp_path();

        let model_path = compile_json::<ODE>(
            json,
            Some(model_output_path.clone()),
            template_path.clone(),
            |_, _| {},
        )
        .expect("compile_json should succeed");

        let model_path = PathBuf::from(&model_path);
        let (_lib, (dyn_ode, _meta)) = unsafe { exa::load::load::<ODE>(model_path.clone()) };

        // Create equivalent handwritten ODE
        let handwritten_ode = equation::ODE::new(
            |x, p, _t, dx, _b, rateiv, _cov| {
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
        );

        // Test subject
        let subject = Subject::builder("1")
            .infusion(0.0, 500.0, 0, 0.5)
            .observation(0.5, 1.645776, 0)
            .observation(1.0, 1.216442, 0)
            .observation(2.0, 0.4622729, 0)
            .build();

        let params = vec![1.02282724609375, 194.51904296875];

        // Compare predictions
        let json_preds = dyn_ode.estimate_predictions(&subject, &params).unwrap();
        let hand_preds = handwritten_ode
            .estimate_predictions(&subject, &params)
            .unwrap();

        let json_flat = json_preds.flat_predictions();
        let hand_flat = hand_preds.flat_predictions();

        assert_eq!(json_flat.len(), hand_flat.len());

        for (json_val, hand_val) in json_flat.iter().zip(hand_flat.iter()) {
            assert_relative_eq!(json_val, hand_val, max_relative = 1e-10, epsilon = 1e-10);
        }

        // Clean up
        std::fs::remove_file(model_path).ok();
        std::fs::remove_dir_all(template_path).ok();
    }

    #[test]
    fn test_compile_json_library_model() {
        use pharmsol::json::ModelLibrary;

        let library = ModelLibrary::builtin();

        // Get an ODE model from the library
        let model = library
            .get("pk/1cmt-iv-ode")
            .expect("Should have pk/1cmt-iv-ode");

        // Convert back to JSON and compile
        let json = serde_json::to_string(model).expect("Should serialize");

        let model_output_path = unique_model_path("test_library_compiled");
        let template_path = unique_temp_path();

        let model_path = compile_json::<ODE>(
            &json,
            Some(model_output_path.clone()),
            template_path.clone(),
            |_, _| {},
        )
        .expect("compile_json should succeed for library model");

        let model_path = PathBuf::from(&model_path);

        // Verify it loads
        let (_lib, (dyn_ode, meta)) = unsafe { exa::load::load::<ODE>(model_path.clone()) };

        // Verify metadata
        assert_eq!(meta.get_params(), &vec!["CL".to_string(), "V".to_string()]);

        // Test it produces valid predictions
        let subject = Subject::builder("1")
            .bolus(0.0, 100.0, 0)
            .observation(1.0, 50.0, 0)
            .build();

        let params = vec![5.0, 10.0]; // CL=5, V=10 (ke = CL/V = 0.5)
        let predictions = dyn_ode.estimate_predictions(&subject, &params);
        assert!(predictions.is_ok());

        // Clean up
        std::fs::remove_file(model_path).ok();
        std::fs::remove_dir_all(template_path).ok();
    }
}
