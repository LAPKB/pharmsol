#[cfg(feature = "exa")]
mod exa_tests {
    use approx::assert_relative_eq;
    use pharmsol::{build::temp_path, *};
    use std::path::PathBuf;

    #[test]
    fn test_exa_predictions_match_ode_predictions() {
        // Create subject with observations
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
            |x, p, _t, dx, b, rateiv, _cov| {
                fetch_params!(p, ke, _v);
                dx[0] = -ke * x[0] + rateiv[0] + b[0];
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

        // Compile the same model using exa
        // Use absolute path in the current test directory to avoid path issues
        let test_dir = std::env::current_dir().expect("Failed to get current directory");
        let model_output_path = test_dir.join("test_model.pkm");

        let model_path = exa::build::compile::<ODE>(
            format!(
                r#"
                equation::ODE::new(
            |x, p, _t, dx, b, rateiv, _cov| {{
                fetch_params!(p, ke, _v);
                dx[0] = -ke * x[0] + rateiv[0] + b[0];
            }},
            |_p, _t, _cov| lag! {{}},
            |_p, _t, _cov| fa! {{}},
            |_p, _t, _cov, _x| {{}},
            |x, p, _t, _cov, y| {{
                fetch_params!(p, _ke, v);
                y[0] = x[0] / v;
            }},
            (1, 1),
        )
        "#
            ),
            Some(model_output_path),
            vec!["ke".to_string(), "v".to_string()],
            temp_path(),
            |_, _| {}, // Empty callback for tests
        )
        .unwrap();

        // Load the compiled model
        let model_path = PathBuf::from(model_path);
        let (_lib, (dyn_ode, _meta)) = unsafe { exa::load::load::<ODE>(model_path.clone()) };

        // Parameters for model evaluation
        let params = vec![1.02282724609375, 194.51904296875];

        // Get predictions from both models
        let dyn_predictions = dyn_ode.estimate_predictions(&subject, &params).unwrap();
        let ode_predictions = ode.estimate_predictions(&subject, &params).unwrap();

        // Check that predictions are the same
        let dyn_flat = dyn_predictions.flat_predictions();
        let ode_flat = ode_predictions.flat_predictions();

        assert_eq!(
            dyn_flat.len(),
            ode_flat.len(),
            "Prediction arrays have different lengths"
        );

        // Compare each prediction with relative tolerance
        for (_i, (dyn_val, ode_val)) in dyn_flat.iter().zip(ode_flat.iter()).enumerate() {
            assert_relative_eq!(dyn_val, ode_val, max_relative = 1e-10, epsilon = 1e-10,);
        }

        // Clean up
        std::fs::remove_file(model_path).ok();
    }

    #[test]
    fn test_cache_invalidation_on_version_change() {
        // Use an isolated temp directory for this test
        let test_dir = tempfile::tempdir().expect("Failed to create temp dir");
        let template_path = test_dir.path().to_path_buf();

        // Step 1: Initial compilation creates the template
        pharmsol::build::dummy_compile(template_path.clone(), |_, _| {}).unwrap();

        let cargo_toml_path = template_path.join("template").join("Cargo.toml");
        let target_dir = template_path.join("template").join("target");

        assert!(cargo_toml_path.exists(), "Cargo.toml should be created");
        assert!(
            target_dir.exists(),
            "target/ should exist after compilation"
        );

        // Save the original Cargo.toml content for later comparison
        let original_content = std::fs::read_to_string(&cargo_toml_path).unwrap();

        // Step 2: Simulate a pharmsol version change by modifying the Cargo.toml
        let fake_old_content =
            original_content.replace(env!("CARGO_PKG_VERSION"), "0.0.0-fake-old-version");
        std::fs::write(&cargo_toml_path, &fake_old_content).unwrap();

        // Also create a marker file inside target/ to verify it gets removed
        let marker = target_dir.join("release").join("cache_marker.txt");
        std::fs::create_dir_all(marker.parent().unwrap()).ok();
        std::fs::write(&marker, "should be deleted").unwrap();
        assert!(marker.exists());

        // Step 3: Call dummy_compile again — should detect the version mismatch,
        // rewrite Cargo.toml, and remove target/
        pharmsol::build::dummy_compile(template_path.clone(), |_, _| {}).unwrap();

        // Verify Cargo.toml was restored to the current version
        let updated_content = std::fs::read_to_string(&cargo_toml_path).unwrap();
        assert!(
            updated_content.contains(env!("CARGO_PKG_VERSION")),
            "Cargo.toml should contain the current pharmsol version after cache invalidation"
        );
        assert!(
            !updated_content.contains("0.0.0-fake-old-version"),
            "Old version should no longer be in Cargo.toml"
        );

        // Verify the old target/ was removed (the marker file should be gone)
        // Note: dummy_compile rebuilds, so target/ exists again, but our marker should be gone
        assert!(
            !marker.exists(),
            "Cache marker should be gone after invalidation — target/ was cleaned"
        );
    }

    #[test]
    fn test_incomplete_template_dir_is_recreated() {
        // If the template directory exists but is not a valid cargo project
        // (e.g. missing src/), create_template should re-scaffold it.
        let test_dir = tempfile::tempdir().expect("Failed to create temp dir");
        let template_path = test_dir.path().to_path_buf();

        // Create a malformed template directory (no src/, no Cargo.toml)
        let template_dir = template_path.join("template");
        std::fs::create_dir_all(&template_dir).unwrap();
        assert!(template_dir.exists());
        assert!(!template_dir.join("src").exists());

        // dummy_compile should recover from this and succeed
        pharmsol::build::dummy_compile(template_path.clone(), |_, _| {}).unwrap();

        // Verify the template was properly created
        assert!(
            template_dir.join("Cargo.toml").exists(),
            "Cargo.toml should exist"
        );
        assert!(template_dir.join("src").exists(), "src/ should exist");
        assert!(
            template_dir.join("target").exists(),
            "target/ should exist after successful build"
        );
    }
}

// When exa feature is not enabled, provide an empty test module
#[cfg(not(feature = "exa"))]
mod exa_tests {}
