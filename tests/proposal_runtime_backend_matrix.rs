#[path = "support/proposal_runtime_corpus.rs"]
mod proposal_runtime_corpus;

#[cfg(all(
    feature = "dsl-jit",
    feature = "dsl-aot",
    feature = "dsl-aot-load",
    feature = "dsl-wasm"
))]
mod tests {
    use super::proposal_runtime_corpus::{self as corpus, ArtifactWorkspace, CorpusCase};
    use pharmsol::dsl::RuntimeBackend;

    #[test]
    fn proposal_ode_runtime_backend_matrix_matches_reference_predictions(
    ) -> Result<(), Box<dyn std::error::Error>> {
        let workspace = ArtifactWorkspace::new()?;

        let jit = corpus::compile_runtime_jit_model(CorpusCase::Ode)?;
        assert_eq!(jit.backend(), RuntimeBackend::Jit);
        corpus::assert_runtime_model_matches_reference(CorpusCase::Ode, "runtime-jit", &jit)?;

        let aot = corpus::compile_runtime_native_aot_model(CorpusCase::Ode, &workspace)?;
        assert_eq!(aot.backend(), RuntimeBackend::NativeAot);
        corpus::assert_runtime_model_matches_reference(
            CorpusCase::Ode,
            "runtime-native-aot",
            &aot,
        )?;

        let wasm = corpus::compile_runtime_wasm_model(CorpusCase::Ode, &workspace)?;
        assert_eq!(wasm.backend(), RuntimeBackend::Wasm);
        corpus::assert_runtime_model_matches_reference(CorpusCase::Ode, "runtime-wasm", &wasm)?;

        Ok(())
    }

    #[test]
    fn proposal_analytical_runtime_backend_matrix_matches_reference_predictions(
    ) -> Result<(), Box<dyn std::error::Error>> {
        let workspace = ArtifactWorkspace::new()?;

        let jit = corpus::compile_runtime_jit_model(CorpusCase::Analytical)?;
        assert_eq!(jit.backend(), RuntimeBackend::Jit);
        corpus::assert_runtime_model_matches_reference(
            CorpusCase::Analytical,
            "runtime-jit",
            &jit,
        )?;

        let aot = corpus::compile_runtime_native_aot_model(CorpusCase::Analytical, &workspace)?;
        assert_eq!(aot.backend(), RuntimeBackend::NativeAot);
        corpus::assert_runtime_model_matches_reference(
            CorpusCase::Analytical,
            "runtime-native-aot",
            &aot,
        )?;

        let wasm = corpus::compile_runtime_wasm_model(CorpusCase::Analytical, &workspace)?;
        assert_eq!(wasm.backend(), RuntimeBackend::Wasm);
        corpus::assert_runtime_model_matches_reference(
            CorpusCase::Analytical,
            "runtime-wasm",
            &wasm,
        )?;

        Ok(())
    }

    #[test]
    fn proposal_sde_runtime_backend_matrix_matches_reference_predictions(
    ) -> Result<(), Box<dyn std::error::Error>> {
        let workspace = ArtifactWorkspace::new()?;

        let jit = corpus::compile_runtime_jit_model(CorpusCase::Sde)?;
        assert_eq!(jit.backend(), RuntimeBackend::Jit);
        corpus::assert_runtime_model_matches_reference(CorpusCase::Sde, "runtime-jit", &jit)?;

        let aot = corpus::compile_runtime_native_aot_model(CorpusCase::Sde, &workspace)?;
        assert_eq!(aot.backend(), RuntimeBackend::NativeAot);
        corpus::assert_runtime_model_matches_reference(
            CorpusCase::Sde,
            "runtime-native-aot",
            &aot,
        )?;

        let wasm = corpus::compile_runtime_wasm_model(CorpusCase::Sde, &workspace)?;
        assert_eq!(wasm.backend(), RuntimeBackend::Wasm);
        corpus::assert_runtime_model_matches_reference(CorpusCase::Sde, "runtime-wasm", &wasm)?;

        Ok(())
    }
}
