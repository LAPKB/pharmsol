#[path = "support/bimodal_ke.rs"]
mod bimodal_ke;

#[cfg(feature = "dsl-jit")]
mod tests {
    use super::bimodal_ke;
    use pharmsol::dsl::RuntimeBackend;

    #[test]
    fn bimodal_ke_entrypoint_matrix_matches_reference_predictions(
    ) -> Result<(), Box<dyn std::error::Error>> {
        let runtime_jit = bimodal_ke::compile_runtime_jit_model()?;
        assert_eq!(runtime_jit.backend(), RuntimeBackend::Jit);
        bimodal_ke::report_runtime_model(
            "dsl::compile_module_source_to_runtime(Jit)",
            &runtime_jit,
            1e-10,
        )?;

        Ok(())
    }
}
