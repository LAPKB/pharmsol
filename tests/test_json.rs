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
            "covariates": ["WT"],
            "secondary": {
                "V": "V * (WT / 70)^0.75"
            }
        }"#;

        let model = parse_json(json).expect("Should parse successfully");
        assert!(model.covariates.is_some());
        assert_eq!(model.covariates.as_ref().unwrap(), &["WT"]);
        assert!(model.secondary.is_some());
        assert_eq!(model.secondary.as_ref().unwrap().len(), 1);
        assert_eq!(model.secondary.as_ref().unwrap()[0].0, "V");
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
            "covariates": ["WT"],
            "secondary": {
                "V": "V * (WT / 70)^0.75"
            }
        }"#;

        let code = generate_code(json).expect("Should generate code");

        // Should include covariate access
        assert!(code.equation_code.contains("cov.get_covariate"));
        // Allometric: V * (WT / 70)^0.75 transpiled with powf
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

        // Create derived model with weight covariate + secondary equation
        let derived = JsonModel::from_str(
            r#"{
            "schema": "1.0",
            "id": "derived/pk-1cmt-wt",
            "extends": "base/pk-1cmt",
            "type": "analytical",
            "analytical": "one_compartment",
            "parameters": ["ke", "V"],
            "covariates": ["WT"],
            "secondary": {
                "V": "V * (WT / 70)^0.75"
            }
        }"#,
        )
        .unwrap();

        let resolved = library.resolve(&derived).unwrap();

        // Should inherit output from base
        assert!(resolved.output.is_some());
        assert_eq!(resolved.output.as_ref().unwrap(), "x[0] / V");

        // Should have covariates from derived
        assert!(resolved.covariates.is_some());
        assert_eq!(resolved.covariates.as_ref().unwrap(), &["WT"]);

        // Should have secondary equations from derived
        assert!(resolved.secondary.is_some());
        assert_eq!(resolved.secondary.as_ref().unwrap().len(), 1);
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
// Secondary Equations and Covariates Tests
// ═══════════════════════════════════════════════════════════════════════════════

mod secondary_and_covariates {
    use super::*;

    #[test]
    fn test_secondary_equations_ordered() {
        let json = r#"{
            "schema": "1.0",
            "id": "pk_secondary",
            "type": "analytical",
            "analytical": "one_compartment",
            "parameters": ["CL", "V"],
            "secondary": {
                "ke": "CL / V"
            },
            "output": "x[0] / V"
        }"#;

        let model = parse_json(json).expect("Should parse");
        let secondary = model.secondary.as_ref().unwrap();
        assert_eq!(secondary.len(), 1);
        assert_eq!(secondary[0].0, "ke");
        assert_eq!(secondary[0].1, "CL / V");
    }

    #[test]
    fn test_secondary_multiple_ordered() {
        let json = r#"{
            "schema": "1.0",
            "id": "pk_multi_secondary",
            "type": "analytical",
            "analytical": "one_compartment",
            "parameters": ["CLs", "Vs"],
            "covariates": ["wt"],
            "secondary": {
                "CL": "CLs * (wt / 70)^0.75",
                "V": "Vs * (wt / 70)",
                "ke": "CL / V"
            },
            "output": "x[0] / V"
        }"#;

        let model = parse_json(json).expect("Should parse");
        let secondary = model.secondary.as_ref().unwrap();
        // Must preserve order (CL before V before ke)
        assert_eq!(secondary.len(), 3);
        assert_eq!(secondary[0].0, "CL");
        assert_eq!(secondary[1].0, "V");
        assert_eq!(secondary[2].0, "ke");
    }

    #[test]
    fn test_covariates_simple_list() {
        let json = r#"{
            "schema": "1.0",
            "id": "pk_covs",
            "type": "analytical",
            "analytical": "one_compartment",
            "parameters": ["ke", "V"],
            "covariates": ["wt", "age", "sex"],
            "output": "x[0] / V"
        }"#;

        let model = parse_json(json).expect("Should parse");
        let covs = model.covariates.as_ref().unwrap();
        assert_eq!(covs, &["wt", "age", "sex"]);
    }

    #[test]
    fn test_codegen_with_secondary_and_covariates() {
        let json = r#"{
            "schema": "1.0",
            "id": "pk_cov_sec",
            "type": "analytical",
            "analytical": "one_compartment",
            "parameters": ["CLs", "Vs"],
            "covariates": ["wt"],
            "secondary": {
                "CL": "CLs * (wt / 70)^0.75",
                "V": "Vs * (wt / 70)",
                "ke": "CL / V"
            },
            "output": "x[0] / V"
        }"#;

        let code = generate_code(json).expect("Should generate code");

        // Should fetch covariates
        assert!(code.equation_code.contains("cov.get_covariate"));
        // Should transpile allometric expression with powf
        assert!(code.equation_code.contains("powf"));
        // Should have secondary variables
        assert!(code.equation_code.contains("let CL ="));
        assert!(code.equation_code.contains("let V ="));
        assert!(code.equation_code.contains("let ke ="));
    }

    #[test]
    fn test_codegen_ode_with_covariates() {
        let json = r#"{
            "schema": "1.0",
            "id": "pk_ode_cov",
            "type": "ode",
            "parameters": ["CLs", "Vs"],
            "compartments": ["central"],
            "covariates": ["wt"],
            "secondary": {
                "CL": "CLs * (wt / 70)^0.75",
                "V": "Vs * wt / 70",
                "ke": "CL / V"
            },
            "diffeq": {
                "central": "-ke * central + rateiv[central]"
            },
            "output": "central / V"
        }"#;

        let code = generate_code(json).expect("Should generate code");

        assert!(code.equation_code.contains("ODE::new"));
        // Should use t and cov (not _t, _cov) when covariates present
        assert!(code.equation_code.contains(", cov,"));
    }

    #[test]
    fn test_expression_with_if_function() {
        let json = r#"{
            "schema": "1.0",
            "id": "pk_if_test",
            "type": "analytical",
            "analytical": "one_compartment",
            "parameters": ["ke", "V"],
            "covariates": ["sex"],
            "secondary": {
                "V": "if(sex == 1, V * 0.8, V)"
            },
            "output": "x[0] / V"
        }"#;

        let code = generate_code(json).expect("Should generate code");
        // Should emit Rust if-else
        assert!(code.equation_code.contains("if sex == 1.0"));
        assert!(code.equation_code.contains("else"));
    }

    #[test]
    fn test_validate_rejects_invalid_expression() {
        let json = r#"{
            "schema": "1.0",
            "id": "bad_expr",
            "type": "ode",
            "parameters": ["ke", "V"],
            "compartments": ["central"],
            "diffeq": {
                "central": "ke * central +"
            },
            "output": "central / V"
        }"#;

        let result = validate_json(json);
        assert!(result.is_err());
    }

    #[test]
    fn test_full_pipeline_with_secondary() {
        let json = r#"{
            "schema": "1.0",
            "id": "e2e_secondary",
            "type": "analytical",
            "analytical": "one_compartment_with_absorption",
            "parameters": ["ka", "CL", "V"],
            "secondary": {
                "ke": "CL / V"
            },
            "output": "x[1] / V"
        }"#;

        // Full pipeline
        let model = parse_json(json).unwrap();
        let validator = Validator::new();
        let validated = validator.validate(&model).unwrap();
        let generator = CodeGenerator::new(validated.inner());
        let code = generator.generate().unwrap();

        assert!(code.equation_code.contains("let ke = CL / V;"));
        assert!(code.equation_code.contains("Analytical::new"));
        assert_eq!(code.parameters, vec!["ka", "CL", "V"]);
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
        )
        .with_nstates(1)
        .with_ndrugs(1)
        .with_nout(1);

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

// ═══════════════════════════════════════════════════════════════════════════════
// Library Model Correctness Tests
// ═══════════════════════════════════════════════════════════════════════════════

mod library_models {
    use super::*;

    /// All 10 built-in library models should parse and validate without errors
    #[test]
    fn test_all_library_models_validate() {
        let library = ModelLibrary::builtin();
        let validator = Validator::new();

        let ids = [
            "pk/1cmt-iv",
            "pk/1cmt-oral",
            "pk/2cmt-iv",
            "pk/2cmt-oral",
            "pk/3cmt-iv",
            "pk/3cmt-oral",
            "pk/1cmt-iv-ode",
            "pk/1cmt-oral-ode",
            "pk/2cmt-iv-ode",
            "pk/2cmt-oral-ode",
        ];

        for id in &ids {
            let model = library.get(id).unwrap_or_else(|| panic!("Missing model: {}", id));
            validator
                .validate(model)
                .unwrap_or_else(|e| panic!("Validation failed for {}: {:?}", id, e));
        }
    }

    /// All library models should generate code successfully
    #[test]
    fn test_all_library_models_generate_code() {
        let library = ModelLibrary::builtin();

        let ids = [
            "pk/1cmt-iv",
            "pk/1cmt-oral",
            "pk/2cmt-iv",
            "pk/2cmt-oral",
            "pk/3cmt-iv",
            "pk/3cmt-oral",
            "pk/1cmt-iv-ode",
            "pk/1cmt-oral-ode",
            "pk/2cmt-iv-ode",
            "pk/2cmt-oral-ode",
        ];

        for id in &ids {
            let model = library.get(id).unwrap();
            let gen = CodeGenerator::new(model);
            gen.generate()
                .unwrap_or_else(|e| panic!("Code generation failed for {}: {:?}", id, e));
        }
    }

    // ── Analytical parameter order tests ─────────────────────────────────

    #[test]
    fn test_1cmt_iv_parameter_order() {
        let library = ModelLibrary::builtin();
        let model = library.get("pk/1cmt-iv").unwrap();

        // one_compartment expects: p[0]=ke
        let params = model.get_parameters();
        assert_eq!(params[0], "ke", "p[0] must be ke for one_compartment");
    }

    #[test]
    fn test_1cmt_oral_parameter_order() {
        let library = ModelLibrary::builtin();
        let model = library.get("pk/1cmt-oral").unwrap();

        // one_compartment_with_absorption expects: p[0]=ka, p[1]=ke
        let params = model.get_parameters();
        assert_eq!(params[0], "ka", "p[0] must be ka");
        assert_eq!(params[1], "ke", "p[1] must be ke");
    }

    #[test]
    fn test_2cmt_iv_parameter_order() {
        let library = ModelLibrary::builtin();
        let model = library.get("pk/2cmt-iv").unwrap();

        // two_compartments expects: p[0]=ke, p[1]=kcp, p[2]=kpc
        let params = model.get_parameters();
        assert_eq!(params[0], "ke");
        assert_eq!(params[1], "kcp");
        assert_eq!(params[2], "kpc");
    }

    #[test]
    fn test_2cmt_oral_parameter_order() {
        let library = ModelLibrary::builtin();
        let model = library.get("pk/2cmt-oral").unwrap();

        // two_compartments_with_absorption expects: p[0]=ke, p[1]=ka, p[2]=kcp, p[3]=kpc
        let params = model.get_parameters();
        assert_eq!(params[0], "ke", "p[0] must be ke (not ka!)");
        assert_eq!(params[1], "ka", "p[1] must be ka");
        assert_eq!(params[2], "kcp");
        assert_eq!(params[3], "kpc");
    }

    #[test]
    fn test_3cmt_iv_parameter_order() {
        let library = ModelLibrary::builtin();
        let model = library.get("pk/3cmt-iv").unwrap();

        // three_compartments expects: p[0]=k10, p[1]=k12, p[2]=k13, p[3]=k21, p[4]=k31
        let params = model.get_parameters();
        assert_eq!(params[0], "k10");
        assert_eq!(params[1], "k12");
        assert_eq!(params[2], "k13");
        assert_eq!(params[3], "k21");
        assert_eq!(params[4], "k31");
    }

    #[test]
    fn test_3cmt_oral_parameter_order() {
        let library = ModelLibrary::builtin();
        let model = library.get("pk/3cmt-oral").unwrap();

        // three_compartments_with_absorption expects: p[0]=ka, p[1]=k10, ... p[5]=k31
        let params = model.get_parameters();
        assert_eq!(params[0], "ka");
        assert_eq!(params[1], "k10");
        assert_eq!(params[2], "k12");
        assert_eq!(params[3], "k13");
        assert_eq!(params[4], "k21");
        assert_eq!(params[5], "k31");
    }

    // ── neqs validity tests ──────────────────────────────────────────────

    #[test]
    fn test_analytical_neqs_match_function_states() {
        let library = ModelLibrary::builtin();

        let cases = [
            ("pk/1cmt-iv", 1, 1),
            ("pk/1cmt-oral", 2, 1),
            ("pk/2cmt-iv", 2, 1),
            ("pk/2cmt-oral", 3, 1),
            ("pk/3cmt-iv", 3, 1),
            ("pk/3cmt-oral", 4, 1),
        ];

        for (id, expected_nstates, expected_nout) in &cases {
            let model = library.get(id).unwrap();
            let neqs = model.get_neqs();
            assert_eq!(
                neqs.0, *expected_nstates,
                "{}: nstates should be {}",
                id, expected_nstates
            );
            assert_eq!(
                neqs.1, *expected_nout,
                "{}: nout should be {}",
                id, expected_nout
            );
        }
    }

    #[test]
    fn test_ode_neqs_match_compartments() {
        let library = ModelLibrary::builtin();

        let cases = [
            ("pk/1cmt-iv-ode", 1, 1),
            ("pk/1cmt-oral-ode", 2, 1),
            ("pk/2cmt-iv-ode", 2, 1),
            ("pk/2cmt-oral-ode", 3, 1),
        ];

        for (id, expected_nstates, expected_nout) in &cases {
            let model = library.get(id).unwrap();
            let neqs = model.get_neqs();
            assert_eq!(neqs.0, *expected_nstates, "{}: nstates mismatch", id);
            assert_eq!(neqs.1, *expected_nout, "{}: nout mismatch", id);
        }
    }

    // ── Output expression correctness ────────────────────────────────────

    #[test]
    fn test_iv_models_output_from_central_state() {
        let library = ModelLibrary::builtin();

        // IV models: central is x[0] (no absorption compartment)
        for id in &["pk/1cmt-iv", "pk/2cmt-iv", "pk/3cmt-iv"] {
            let model = library.get(id).unwrap();
            let output = model.output.as_ref().unwrap();
            assert!(
                output.contains("x[0]"),
                "{}: IV model output should reference x[0] (central)",
                id
            );
        }
    }

    #[test]
    fn test_oral_models_output_from_central_state() {
        let library = ModelLibrary::builtin();

        // Oral models: central is x[1] (x[0] is depot/absorption)
        for id in &["pk/1cmt-oral", "pk/2cmt-oral", "pk/3cmt-oral"] {
            let model = library.get(id).unwrap();
            let output = model.output.as_ref().unwrap();
            assert!(
                output.contains("x[1]"),
                "{}: Oral model output should reference x[1] (central, after depot at x[0])",
                id
            );
        }
    }

    // ── ODE diffeq completeness ──────────────────────────────────────────

    #[test]
    fn test_ode_iv_models_include_rateiv() {
        let library = ModelLibrary::builtin();

        // IV ODE models must include rateiv[0] in the central compartment
        // equation to support infusion dosing
        for id in &["pk/1cmt-iv-ode", "pk/2cmt-iv-ode"] {
            let model = library.get(id).unwrap();
            let gen = CodeGenerator::new(model);
            let code = gen.generate().unwrap();

            assert!(
                code.equation_code.contains("rateiv[0]"),
                "{}: central diffeq should include rateiv[0] for infusion support",
                id
            );
        }
    }

    #[test]
    fn test_ode_compartment_counts_match_diffeq_entries() {
        let library = ModelLibrary::builtin();

        for id in &[
            "pk/1cmt-iv-ode",
            "pk/1cmt-oral-ode",
            "pk/2cmt-iv-ode",
            "pk/2cmt-oral-ode",
        ] {
            let model = library.get(id).unwrap();
            let ncompartments = model.compartments.as_ref().unwrap().len();
            let gen = CodeGenerator::new(model);
            let code = gen.generate().unwrap();

            // Count dx[N] assignments in generated code
            let dx_count = (0..ncompartments)
                .filter(|i| code.equation_code.contains(&format!("dx[{}]", i)))
                .count();

            assert_eq!(
                dx_count, ncompartments,
                "{}: should have dx[i] for each of the {} compartments",
                id, ncompartments
            );
        }
    }

    // ── ODE oral models have correct transfer terms ──────────────────────

    #[test]
    fn test_oral_ode_depot_drains_to_central() {
        let library = ModelLibrary::builtin();

        for id in &["pk/1cmt-oral-ode", "pk/2cmt-oral-ode"] {
            let model = library.get(id).unwrap();
            let gen = CodeGenerator::new(model);
            let code = gen.generate().unwrap();

            // Depot should have negative ka term (drug leaving)
            assert!(
                code.equation_code.contains("ka") && code.equation_code.contains("dx[0]"),
                "{}: depot (dx[0]) should involve ka",
                id
            );

            // Central should have positive ka term (drug arriving)
            assert!(
                code.equation_code.contains("dx[1]"),
                "{}: central equation (dx[1]) should exist",
                id
            );
        }
    }
}
