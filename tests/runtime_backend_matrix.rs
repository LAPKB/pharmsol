#[path = "support/runtime_corpus.rs"]
mod runtime_corpus;

#[cfg(all(feature = "dsl-jit", feature = "dsl-wasm"))]
mod tests {
    use super::runtime_corpus::{self as corpus, ArtifactWorkspace, CorpusCase};
    use pharmsol::dsl::RuntimeBackend;

    #[test]
    fn ode_runtime_backend_matrix_matches_reference_predictions(
    ) -> Result<(), Box<dyn std::error::Error>> {
        let workspace = ArtifactWorkspace::new()?;

        let jit = corpus::compile_runtime_jit_model(CorpusCase::Ode)?;
        assert_eq!(jit.backend(), RuntimeBackend::Jit);
        corpus::assert_runtime_model_matches_reference(CorpusCase::Ode, "runtime-jit", &jit)?;

        #[cfg(all(feature = "dsl-aot", feature = "dsl-aot-load"))]
        let aot = corpus::compile_runtime_native_aot_model(CorpusCase::Ode, &workspace)?;
        #[cfg(all(feature = "dsl-aot", feature = "dsl-aot-load"))]
        assert_eq!(aot.backend(), RuntimeBackend::NativeAot);
        #[cfg(all(feature = "dsl-aot", feature = "dsl-aot-load"))]
        corpus::assert_runtime_model_matches_reference(
            CorpusCase::Ode,
            "runtime-native-aot",
            &aot,
        )?;

        let wasm = corpus::compile_runtime_wasm_model(CorpusCase::Ode)?;
        assert_eq!(wasm.backend(), RuntimeBackend::Wasm);
        corpus::assert_runtime_model_matches_reference(CorpusCase::Ode, "runtime-wasm", &wasm)?;
        corpus::assert_runtime_models_match_each_other(
            CorpusCase::Ode,
            "runtime-jit",
            &jit,
            "runtime-wasm",
            &wasm,
        )?;

        Ok(())
    }

    #[test]
    fn analytical_runtime_backend_matrix_matches_reference_predictions(
    ) -> Result<(), Box<dyn std::error::Error>> {
        let workspace = ArtifactWorkspace::new()?;

        let jit = corpus::compile_runtime_jit_model(CorpusCase::Analytical)?;
        assert_eq!(jit.backend(), RuntimeBackend::Jit);
        corpus::assert_runtime_model_matches_reference(
            CorpusCase::Analytical,
            "runtime-jit",
            &jit,
        )?;

        #[cfg(all(feature = "dsl-aot", feature = "dsl-aot-load"))]
        let aot = corpus::compile_runtime_native_aot_model(CorpusCase::Analytical, &workspace)?;
        #[cfg(all(feature = "dsl-aot", feature = "dsl-aot-load"))]
        assert_eq!(aot.backend(), RuntimeBackend::NativeAot);
        #[cfg(all(feature = "dsl-aot", feature = "dsl-aot-load"))]
        corpus::assert_runtime_model_matches_reference(
            CorpusCase::Analytical,
            "runtime-native-aot",
            &aot,
        )?;

        let wasm = corpus::compile_runtime_wasm_model(CorpusCase::Analytical)?;
        assert_eq!(wasm.backend(), RuntimeBackend::Wasm);
        corpus::assert_runtime_model_matches_reference(
            CorpusCase::Analytical,
            "runtime-wasm",
            &wasm,
        )?;
        corpus::assert_runtime_models_match_each_other(
            CorpusCase::Analytical,
            "runtime-jit",
            &jit,
            "runtime-wasm",
            &wasm,
        )?;

        Ok(())
    }

    #[test]
    fn sde_runtime_backend_matrix_matches_reference_predictions(
    ) -> Result<(), Box<dyn std::error::Error>> {
        let workspace = ArtifactWorkspace::new()?;

        let jit = corpus::compile_runtime_jit_model(CorpusCase::Sde)?;
        assert_eq!(jit.backend(), RuntimeBackend::Jit);
        corpus::assert_runtime_model_matches_reference(CorpusCase::Sde, "runtime-jit", &jit)?;

        #[cfg(all(feature = "dsl-aot", feature = "dsl-aot-load"))]
        let aot = corpus::compile_runtime_native_aot_model(CorpusCase::Sde, &workspace)?;
        #[cfg(all(feature = "dsl-aot", feature = "dsl-aot-load"))]
        assert_eq!(aot.backend(), RuntimeBackend::NativeAot);
        #[cfg(all(feature = "dsl-aot", feature = "dsl-aot-load"))]
        corpus::assert_runtime_model_matches_reference(
            CorpusCase::Sde,
            "runtime-native-aot",
            &aot,
        )?;

        let wasm = corpus::compile_runtime_wasm_model(CorpusCase::Sde)?;
        assert_eq!(wasm.backend(), RuntimeBackend::Wasm);
        corpus::assert_runtime_model_matches_reference(CorpusCase::Sde, "runtime-wasm", &wasm)?;
        corpus::assert_runtime_models_match_each_other(
            CorpusCase::Sde,
            "runtime-jit",
            &jit,
            "runtime-wasm",
            &wasm,
        )?;

        Ok(())
    }
}
