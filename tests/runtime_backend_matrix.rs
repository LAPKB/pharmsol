#[path = "support/runtime_corpus.rs"]
mod runtime_corpus;

#[cfg(feature = "dsl-jit")]
mod tests {
    use super::runtime_corpus::{self as corpus, CorpusCase};
    use pharmsol::dsl::RuntimeBackend;

    #[test]
    fn ode_runtime_backend_matrix_matches_reference_predictions(
    ) -> Result<(), Box<dyn std::error::Error>> {
        #[cfg(all(feature = "dsl-aot", feature = "dsl-aot-load"))]
        let workspace = super::runtime_corpus::ArtifactWorkspace::new()?;

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

        Ok(())
    }

    #[test]
    fn analytical_runtime_backend_matrix_matches_reference_predictions(
    ) -> Result<(), Box<dyn std::error::Error>> {
        #[cfg(all(feature = "dsl-aot", feature = "dsl-aot-load"))]
        let workspace = super::runtime_corpus::ArtifactWorkspace::new()?;

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

        Ok(())
    }

    #[test]
    fn analytical_full_runtime_backend_matrix_matches_reference_predictions(
    ) -> Result<(), Box<dyn std::error::Error>> {
        #[cfg(all(feature = "dsl-aot", feature = "dsl-aot-load"))]
        let workspace = super::runtime_corpus::ArtifactWorkspace::new()?;

        let jit = corpus::compile_runtime_jit_model(CorpusCase::AnalyticalFull)?;
        assert_eq!(jit.backend(), RuntimeBackend::Jit);
        corpus::assert_runtime_model_matches_reference(
            CorpusCase::AnalyticalFull,
            "runtime-jit",
            &jit,
        )?;

        #[cfg(all(feature = "dsl-aot", feature = "dsl-aot-load"))]
        let aot = corpus::compile_runtime_native_aot_model(CorpusCase::AnalyticalFull, &workspace)?;
        #[cfg(all(feature = "dsl-aot", feature = "dsl-aot-load"))]
        assert_eq!(aot.backend(), RuntimeBackend::NativeAot);
        #[cfg(all(feature = "dsl-aot", feature = "dsl-aot-load"))]
        corpus::assert_runtime_model_matches_reference(
            CorpusCase::AnalyticalFull,
            "runtime-native-aot",
            &aot,
        )?;

        Ok(())
    }

    #[test]
    fn ode_full_runtime_backend_matrix_matches_reference_predictions(
    ) -> Result<(), Box<dyn std::error::Error>> {
        #[cfg(all(feature = "dsl-aot", feature = "dsl-aot-load"))]
        let workspace = super::runtime_corpus::ArtifactWorkspace::new()?;

        let jit = corpus::compile_runtime_jit_model(CorpusCase::OdeFull)?;
        assert_eq!(jit.backend(), RuntimeBackend::Jit);
        corpus::assert_runtime_model_matches_reference(CorpusCase::OdeFull, "runtime-jit", &jit)?;

        #[cfg(all(feature = "dsl-aot", feature = "dsl-aot-load"))]
        let aot = corpus::compile_runtime_native_aot_model(CorpusCase::OdeFull, &workspace)?;
        #[cfg(all(feature = "dsl-aot", feature = "dsl-aot-load"))]
        assert_eq!(aot.backend(), RuntimeBackend::NativeAot);
        #[cfg(all(feature = "dsl-aot", feature = "dsl-aot-load"))]
        corpus::assert_runtime_model_matches_reference(
            CorpusCase::OdeFull,
            "runtime-native-aot",
            &aot,
        )?;

        Ok(())
    }

    #[test]
    fn sde_runtime_backend_matrix_matches_reference_predictions(
    ) -> Result<(), Box<dyn std::error::Error>> {
        #[cfg(all(feature = "dsl-aot", feature = "dsl-aot-load"))]
        let workspace = super::runtime_corpus::ArtifactWorkspace::new()?;

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

        Ok(())
    }
}
