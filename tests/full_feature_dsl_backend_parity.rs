#[path = "support/runtime_corpus.rs"]
mod runtime_corpus;

#[cfg(all(feature = "dsl-jit", feature = "dsl-wasm"))]
mod tests {
    use super::runtime_corpus::{self as corpus, CorpusCase};
    use pharmsol::dsl::{CompiledRuntimeModel, RuntimeBackend};

    fn owned_names(names: &[&str]) -> Vec<String> {
        names.iter().map(|name| (*name).to_owned()).collect()
    }

    fn assert_info_matches(
        left_label: &str,
        left: &CompiledRuntimeModel,
        right_label: &str,
        right: &CompiledRuntimeModel,
    ) {
        assert_eq!(
            left.info(),
            right.info(),
            "{left_label} model info diverged from {right_label}"
        );
    }

    fn assert_ode_full_public_shape(model: &CompiledRuntimeModel) {
        let info = model.info();

        assert_eq!(info.name, "ode_full_feature_parity");
        assert_eq!(
            info.parameters,
            owned_names(&[
                "ka",
                "ke",
                "kcp",
                "kpc",
                "v",
                "tlag",
                "f_oral",
                "base_depot",
                "base_central",
                "base_peripheral",
            ])
        );
        assert_eq!(
            info.covariates
                .iter()
                .map(|covariate| covariate.name.as_str())
                .collect::<Vec<_>>(),
            vec!["wt", "renal"]
        );
        assert_eq!(
            info.routes
                .iter()
                .map(|route| route.name.as_str())
                .collect::<Vec<_>>(),
            vec!["oral", "load", "iv"]
        );
        assert_eq!(
            info.routes
                .iter()
                .map(|route| route.declaration_index)
                .collect::<Vec<_>>(),
            vec![0, 1, 2]
        );
        assert_eq!(
            info.routes
                .iter()
                .map(|route| route.index)
                .collect::<Vec<_>>(),
            vec![0, 1, 0]
        );
        assert_eq!(
            info.outputs
                .iter()
                .map(|output| output.name.as_str())
                .collect::<Vec<_>>(),
            vec!["cp"]
        );
    }

    fn assert_analytical_full_public_shape(model: &CompiledRuntimeModel) {
        let info = model.info();

        assert_eq!(info.name, "analytical_full_feature_parity");
        assert_eq!(
            info.parameters,
            owned_names(&[
                "ka",
                "ke",
                "v",
                "tlag",
                "f_oral",
                "base_gut",
                "base_central",
                "tvke",
            ])
        );
        assert_eq!(
            info.covariates
                .iter()
                .map(|covariate| covariate.name.as_str())
                .collect::<Vec<_>>(),
            vec!["wt", "renal"]
        );
        assert_eq!(
            info.routes
                .iter()
                .map(|route| route.name.as_str())
                .collect::<Vec<_>>(),
            vec!["oral", "load", "iv"]
        );
        assert_eq!(
            info.routes
                .iter()
                .map(|route| route.declaration_index)
                .collect::<Vec<_>>(),
            vec![0, 1, 2]
        );
        assert_eq!(
            info.routes
                .iter()
                .map(|route| route.index)
                .collect::<Vec<_>>(),
            vec![0, 1, 0]
        );
        assert_eq!(
            info.outputs
                .iter()
                .map(|output| output.name.as_str())
                .collect::<Vec<_>>(),
            vec!["cp"]
        );
    }

    fn assert_full_backend_parity(
        case: CorpusCase,
        assert_public_shape: fn(&CompiledRuntimeModel),
    ) -> Result<(), Box<dyn std::error::Error>> {
        #[cfg(all(feature = "dsl-aot", feature = "dsl-aot-load"))]
        let workspace = super::runtime_corpus::ArtifactWorkspace::new()?;

        let jit = corpus::compile_runtime_jit_model(case)?;
        assert_eq!(jit.backend(), RuntimeBackend::Jit);
        assert_public_shape(&jit);
        corpus::assert_runtime_model_matches_reference(case, "runtime-jit", &jit)?;

        #[cfg(all(feature = "dsl-aot", feature = "dsl-aot-load"))]
        let aot = corpus::compile_runtime_native_aot_model(case, &workspace)?;
        #[cfg(all(feature = "dsl-aot", feature = "dsl-aot-load"))]
        {
            assert_eq!(aot.backend(), RuntimeBackend::NativeAot);
            assert_public_shape(&aot);
            corpus::assert_runtime_model_matches_reference(case, "runtime-native-aot", &aot)?;
        }

        let wasm = corpus::compile_runtime_wasm_model(case)?;
        assert_eq!(wasm.backend(), RuntimeBackend::Wasm);
        assert_public_shape(&wasm);
        corpus::assert_runtime_model_matches_reference(case, "runtime-wasm", &wasm)?;

        assert_info_matches("runtime-jit", &jit, "runtime-wasm", &wasm);
        corpus::assert_runtime_models_match_each_other(
            case,
            "runtime-jit",
            &jit,
            "runtime-wasm",
            &wasm,
        )?;

        #[cfg(all(feature = "dsl-aot", feature = "dsl-aot-load"))]
        {
            assert_info_matches("runtime-jit", &jit, "runtime-native-aot", &aot);
            assert_info_matches("runtime-native-aot", &aot, "runtime-wasm", &wasm);
            corpus::assert_runtime_models_match_each_other(
                case,
                "runtime-jit",
                &jit,
                "runtime-native-aot",
                &aot,
            )?;
            corpus::assert_runtime_models_match_each_other(
                case,
                "runtime-native-aot",
                &aot,
                "runtime-wasm",
                &wasm,
            )?;
        }

        Ok(())
    }

    #[test]
    fn ode_full_feature_dsl_matches_handwritten_across_backends(
    ) -> Result<(), Box<dyn std::error::Error>> {
        assert_full_backend_parity(CorpusCase::OdeFull, assert_ode_full_public_shape)
    }

    #[test]
    fn analytical_full_feature_dsl_matches_handwritten_across_backends(
    ) -> Result<(), Box<dyn std::error::Error>> {
        assert_full_backend_parity(
            CorpusCase::AnalyticalFull,
            assert_analytical_full_public_shape,
        )
    }
}
