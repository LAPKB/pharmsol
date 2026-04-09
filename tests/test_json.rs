#![cfg(feature = "json")]
//! Integration tests for the JSON model system.

use pharmsol::json::{
    generate_code, normalize_json, parse_json, validate_json, CodeGenerator, JsonModel,
    ModelLibrary, ModelType, Validator,
};

mod parsing {
    use super::*;

    #[test]
    fn test_parse_complete_analytical_model() {
        let json = r#"{
            "schema": "2.0",
            "id": "pk/2cmt-oral",
            "type": "analytical",
            "version": "1.0.0",
            "analytical": "two_compartments_with_absorption",
            "parameters": ["ke", "ka", "kcp", "kpc", "V"],
            "compartments": ["depot", "central", "peripheral"],
            "outputs": [
                { "id": "cp", "equation": "central / V" }
            ],
            "neqs": [3, 1],
            "editor": {
                "display": {
                    "name": "Two-Compartment Oral",
                    "category": "pk",
                    "tags": ["2-compartment", "oral"]
                },
                "documentation": {
                    "summary": "Standard two-compartment oral PK model"
                }
            }
        }"#;

        let model = parse_json(json).expect("Should parse successfully");
        assert_eq!(model.id, "pk/2cmt-oral");
        assert_eq!(model.model_type, ModelType::Analytical);
        assert_eq!(model.parameters.as_ref().unwrap().len(), 5);
        assert_eq!(model.outputs.as_ref().unwrap()[0].id, "cp");
    }

    #[test]
    fn test_parse_with_covariates_and_secondary() {
        let json = r#"{
            "schema": "2.0",
            "id": "pk/1cmt-wt",
            "type": "analytical",
            "analytical": "one_compartment",
            "parameters": ["CLs", "Vs"],
            "compartments": ["central"],
            "covariates": [{ "id": "WT", "column": "WT" }],
            "secondary": [
                { "id": "CL", "equation": "CLs * (WT / 70)^0.75" },
                { "id": "V", "equation": "Vs * (WT / 70)" },
                { "id": "ke", "equation": "CL / V" }
            ],
            "outputs": [{ "id": "cp", "equation": "central / V" }]
        }"#;

        let model = parse_json(json).expect("Should parse successfully");
        assert_eq!(model.covariates.as_ref().unwrap()[0].id, "WT");
        assert_eq!(model.secondary.as_ref().unwrap().len(), 3);
        assert_eq!(model.secondary.as_ref().unwrap()[2].id, "ke");
    }

    #[test]
    fn test_reject_unknown_fields() {
        let json = r#"{
            "schema": "2.0",
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
            "schema": "1.0",
            "id": "legacy_model",
            "type": "ode"
        }"#;

        let result = parse_json(json);
        assert!(result.is_err());
    }
}

mod validation {
    use super::*;

    #[test]
    fn test_validate_complete_model() {
        let json = r#"{
            "schema": "2.0",
            "id": "pk/1cmt",
            "type": "analytical",
            "analytical": "one_compartment",
            "parameters": ["ke", "V"],
            "compartments": ["central"],
            "outputs": [{ "id": "cp", "equation": "central / V" }]
        }"#;

        let validated = validate_json(json).expect("Should validate successfully");
        assert_eq!(validated.inner().id, "pk/1cmt");
    }

    #[test]
    fn test_validate_rejects_missing_outputs() {
        let json = r#"{
            "schema": "2.0",
            "id": "bad_ode",
            "type": "ode",
            "parameters": ["ke", "V"],
            "compartments": ["central"],
            "diffeq": { "central": "-ke * central" }
        }"#;

        let result = validate_json(json);
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_rejects_duplicate_parameters() {
        let json = r#"{
            "schema": "2.0",
            "id": "dup_params",
            "type": "analytical",
            "analytical": "one_compartment",
            "parameters": ["ke", "V", "ke"],
            "compartments": ["central"],
            "outputs": [{ "id": "cp", "equation": "central / V" }]
        }"#;

        let result = validate_json(json);
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_rejects_numeric_lag_keys() {
        let json = r#"{
            "schema": "2.0",
            "id": "bad_lag",
            "type": "analytical",
            "analytical": "one_compartment_with_absorption",
            "parameters": ["ka", "ke", "V", "tlag"],
            "compartments": ["depot", "central"],
            "lag": { "0": "tlag" },
            "outputs": [{ "id": "cp", "equation": "central / V" }]
        }"#;

        let result = validate_json(json);
        assert!(result.is_err());
    }
}

mod codegen {
    use super::*;

    #[test]
    fn test_generate_analytical_code() {
        let json = r#"{
            "schema": "2.0",
            "id": "pk/1cmt",
            "type": "analytical",
            "analytical": "one_compartment_with_absorption",
            "parameters": ["ka", "ke", "V"],
            "compartments": ["depot", "central"],
            "outputs": [{ "id": "cp", "equation": "central / V" }]
        }"#;

        let code = generate_code(json).expect("Should generate code");
        assert!(code.equation_code.contains("Analytical::new"));
        assert!(code.equation_code.contains("one_compartment_with_absorption"));
        assert!(code.equation_code.contains("fetch_params!"));
        assert_eq!(code.parameters, vec!["ka", "ke", "V"]);
    }

    #[test]
    fn test_generate_ode_code_with_init_and_lag() {
        let json = r#"{
            "schema": "2.0",
            "id": "pk/1cmt-ode",
            "type": "ode",
            "parameters": ["ka", "CL", "V", "A0", "tlag"],
            "compartments": ["depot", "central"],
            "diffeq": {
                "depot": "-ka * depot",
                "central": "ka * depot - CL/V * central"
            },
            "init": {
                "central": "A0"
            },
            "lag": {
                "depot": "tlag"
            },
            "outputs": [{ "id": "cp", "equation": "central / V" }]
        }"#;

        let code = generate_code(json).expect("Should generate code");
        assert!(code.equation_code.contains("ODE::new"));
        assert!(code.equation_code.contains("x[1] = A0") || code.equation_code.contains("x[0] = A0"));
        assert!(code.equation_code.contains("lag!"));
    }

    #[test]
    fn test_generate_with_covariates_and_secondary() {
        let json = r#"{
            "schema": "2.0",
            "id": "pk/cov",
            "type": "analytical",
            "analytical": "one_compartment",
            "parameters": ["CLs", "Vs"],
            "compartments": ["central"],
            "covariates": [{ "id": "wt" }],
            "secondary": [
                { "id": "CL", "equation": "CLs * (wt / 70)^0.75" },
                { "id": "V", "equation": "Vs * wt / 70" },
                { "id": "ke", "equation": "CL / V" }
            ],
            "outputs": [{ "id": "cp", "equation": "central / V" }]
        }"#;

        let code = generate_code(json).expect("Should generate code");
        assert!(code.equation_code.contains("cov.get_covariate"));
        assert!(code.equation_code.contains("powf"));
        assert!(code.equation_code.contains("let CL ="));
        assert!(code.equation_code.contains("let ke ="));
    }
}

mod library {
    use super::*;

    #[test]
    fn test_builtin_library_contains_standard_models() {
        let library = ModelLibrary::builtin();
        assert!(library.contains("pk/1cmt-iv"));
        assert!(library.contains("pk/1cmt-oral"));
        assert!(library.contains("pk/2cmt-iv"));
        assert!(library.contains("pk/2cmt-oral"));
        assert!(library.contains("pk/1cmt-iv-ode"));
        assert!(library.contains("pk/1cmt-oral-ode"));
    }

    #[test]
    fn test_library_search_and_filter() {
        let library = ModelLibrary::builtin();
        let oral_models = library.search("oral");
        let analytical = library.filter_by_type(ModelType::Analytical);
        let oral_tagged = library.filter_by_tag("oral");

        assert!(!oral_models.is_empty());
        assert!(!analytical.is_empty());
        assert!(!oral_tagged.is_empty());
    }

    #[test]
    fn test_library_inheritance() {
        let mut library = ModelLibrary::new();

        let base = JsonModel::from_str(
            r#"{
            "schema": "2.0",
            "id": "base/pk-1cmt",
            "type": "analytical",
            "analytical": "one_compartment",
            "parameters": ["ke", "V"],
            "compartments": ["central"],
            "outputs": [{ "id": "cp", "equation": "central / V" }],
            "editor": {
                "display": {
                    "name": "Base One-Compartment",
                    "category": "pk"
                }
            }
        }"#,
        )
        .unwrap();
        library.add(base);

        let derived = JsonModel::from_str(
            r#"{
            "schema": "2.0",
            "id": "derived/pk-1cmt-wt",
            "extends": "base/pk-1cmt",
            "type": "analytical",
            "analytical": "one_compartment",
            "parameters": ["ke", "V"],
            "compartments": ["central"],
            "covariates": [{ "id": "WT", "column": "WT" }],
            "secondary": [{ "id": "Vadj", "equation": "V * (WT / 70)^0.75" }]
        }"#,
        )
        .unwrap();

        let resolved = library.resolve(&derived).unwrap();
        assert_eq!(resolved.outputs.as_ref().unwrap()[0].id, "cp");
        assert_eq!(resolved.covariates.as_ref().unwrap()[0].id, "WT");
        assert_eq!(resolved.secondary.as_ref().unwrap()[0].id, "Vadj");
    }

    #[test]
    fn test_library_generates_code_from_model() {
        let library = ModelLibrary::builtin();
        let model = library.get("pk/1cmt-oral").unwrap();
        let generator = CodeGenerator::new(model);
        let code = generator.generate().expect("Should generate code");

        assert!(code.equation_code.contains("one_compartment_with_absorption"));
        assert_eq!(code.parameters, vec!["ka", "ke", "V"]);
    }
}

mod end_to_end {
    use super::*;

    #[test]
    fn test_full_pipeline_analytical() {
        let json = r#"{
            "schema": "2.0",
            "id": "e2e/1cmt",
            "type": "analytical",
            "analytical": "one_compartment_with_absorption",
            "parameters": ["ka", "ke", "V"],
            "compartments": ["depot", "central"],
            "outputs": [{ "id": "cp", "equation": "central / V" }],
            "editor": {
                "display": {
                    "name": "E2E Test Model",
                    "category": "pk"
                }
            }
        }"#;

        let model = parse_json(json).unwrap();
        let validated = Validator::new().validate(&model).unwrap();
        let generator = CodeGenerator::new(validated.inner());
        let code = generator.generate().unwrap();

        assert!(code.equation_code.contains("Analytical::new"));
        assert!(!code.equation_code.is_empty());
        assert_eq!(code.parameters.len(), 3);
    }

    #[test]
    fn test_library_to_code_pipeline() {
        let library = ModelLibrary::builtin();
        for id in library.list() {
            let model = library.get(id).unwrap();
            let generator = CodeGenerator::new(model);
            let result = generator.generate();
            assert!(result.is_ok(), "Failed to generate code for model: {}", id);
        }
    }

    #[test]
    fn test_normalize_secondary_symbols_for_codegen() {
        let json = r#"{
            "schema": "2.0",
            "id": "pk/1cmt-secondary",
            "type": "ode",
            "parameters": ["CL", "V"],
            "compartments": ["central"],
            "secondary": [
                { "id": "ke", "equation": "CL / V" }
            ],
            "diffeq": {
                "central": "-ke * central"
            },
            "outputs": [{ "id": "cp", "equation": "central / V" }]
        }"#;

        let executable = normalize_json(json).unwrap();
        assert_eq!(executable.outputs[0].id, "cp");
        assert_eq!(executable.calculations[0].id, "ke");

        let code = generate_code(json).unwrap();
        assert!(code.equation_code.contains("let ke = CL / V;"));
        assert!(code.equation_code.contains("dx[0]"));
        assert!(code.equation_code.contains("ke * central"));
    }
}

#[cfg(feature = "exa")]
mod exa_integration {
    use approx::assert_relative_eq;
    use pharmsol::json::compile_json;
    use pharmsol::{equation, exa, Equation, Subject, SubjectBuilderExt, ODE};
    use pharmsol::{fa, fetch_params, lag};
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicUsize, Ordering};

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

    fn unique_temp_path() -> PathBuf {
        let count = TEST_COUNTER.fetch_add(1, Ordering::SeqCst);
        let pid = std::process::id();
        std::env::temp_dir().join(format!("exa_test_{}_{}", pid, count))
    }

    #[test]
    fn test_compile_json_ode_model() {
        let json = r#"{
            "schema": "2.0",
            "id": "test/compiled-ode",
            "type": "ode",
            "parameters": ["ke", "V"],
            "compartments": ["central"],
            "diffeq": {
                "central": "-ke * central + rateiv[0]"
            },
            "outputs": [{ "id": "cp", "equation": "central / V" }]
        }"#;

        let model_output_path = unique_model_path("test_json_compiled");
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

        let subject = Subject::builder("1")
            .infusion(0.0, 500.0, 0, 0.5)
            .observation(0.5, 1.5, 0)
            .observation(1.0, 1.2, 0)
            .observation(2.0, 0.5, 0)
            .build();

        let params = vec![1.0, 100.0];
        let predictions = dyn_ode.estimate_predictions(&subject, &params);
        assert!(predictions.is_ok());

        let preds = predictions.unwrap().flat_predictions();
        assert_eq!(preds.len(), 3);
        for prediction in &preds {
            assert!(*prediction > 0.0);
        }

        std::fs::remove_file(model_path).ok();
        std::fs::remove_dir_all(template_path).ok();
    }

    #[test]
    fn test_compile_json_matches_handwritten_ode() {
        let json = r#"{
            "schema": "2.0",
            "id": "compare/ode",
            "type": "ode",
            "parameters": ["ke", "V"],
            "compartments": ["central"],
            "diffeq": {
                "central": "-ke * central + rateiv[0]"
            },
            "outputs": [{ "id": "cp", "equation": "central / V" }]
        }"#;

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

        let handwritten_ode = equation::ODE::new(
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
        )
        .with_nstates(1)
        .with_ndrugs(1)
        .with_nout(1);

        let subject = Subject::builder("1")
            .infusion(0.0, 500.0, 0, 0.5)
            .observation(0.5, 1.645776, 0)
            .observation(1.0, 1.216442, 0)
            .observation(2.0, 0.4622729, 0)
            .build();

        let params = vec![1.02282724609375, 194.51904296875];

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

        std::fs::remove_file(model_path).ok();
        std::fs::remove_dir_all(template_path).ok();
    }
}
