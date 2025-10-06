#[cfg(feature = "exa")]
mod exa_tests {
    use approx::assert_relative_eq;
    use pharmsol::*;
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
}

// When exa feature is not enabled, provide an empty test module
#[cfg(not(feature = "exa"))]
mod exa_tests {}
