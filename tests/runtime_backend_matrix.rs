#[path = "support/runtime_corpus.rs"]
mod runtime_corpus;

#[cfg(feature = "dsl")]
mod tests {
    use super::runtime_corpus::{self as corpus, CorpusCase};

    #[test]
    fn ode_runtime_backend_matrix_matches_reference_predictions(
    ) -> Result<(), Box<dyn std::error::Error>> {
        let jit = corpus::compile_runtime_jit_model(CorpusCase::Ode)?;
        corpus::assert_runtime_model_matches_reference(CorpusCase::Ode, "runtime-jit", &jit)?;

        Ok(())
    }

    #[test]
    fn analytical_runtime_backend_matrix_matches_reference_predictions(
    ) -> Result<(), Box<dyn std::error::Error>> {
        let jit = corpus::compile_runtime_jit_model(CorpusCase::Analytical)?;
        corpus::assert_runtime_model_matches_reference(
            CorpusCase::Analytical,
            "runtime-jit",
            &jit,
        )?;

        Ok(())
    }

    #[test]
    fn analytical_full_runtime_backend_matrix_matches_reference_predictions(
    ) -> Result<(), Box<dyn std::error::Error>> {
        let jit = corpus::compile_runtime_jit_model(CorpusCase::AnalyticalFull)?;
        corpus::assert_runtime_model_matches_reference(
            CorpusCase::AnalyticalFull,
            "runtime-jit",
            &jit,
        )?;

        Ok(())
    }

    #[test]
    fn ode_full_runtime_backend_matrix_matches_reference_predictions(
    ) -> Result<(), Box<dyn std::error::Error>> {
        let jit = corpus::compile_runtime_jit_model(CorpusCase::OdeFull)?;
        corpus::assert_runtime_model_matches_reference(CorpusCase::OdeFull, "runtime-jit", &jit)?;

        Ok(())
    }

    #[test]
    fn sde_runtime_backend_matrix_matches_reference_predictions(
    ) -> Result<(), Box<dyn std::error::Error>> {
        let jit = corpus::compile_runtime_jit_model(CorpusCase::Sde)?;
        corpus::assert_runtime_model_matches_reference(CorpusCase::Sde, "runtime-jit", &jit)?;

        Ok(())
    }
}
