#[path = "support/bimodal_ke.rs"]
mod bimodal_ke;

#[cfg(all(feature = "dsl-jit", feature = "dsl-aot", feature = "dsl-aot-load", feature = "dsl-wasm"))]
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

        let runtime_aot_workspace = bimodal_ke::ArtifactWorkspace::new()?;
        let runtime_aot = bimodal_ke::compile_runtime_native_aot_model(&runtime_aot_workspace)?;
        assert_eq!(runtime_aot.backend(), RuntimeBackend::NativeAot);
        bimodal_ke::report_runtime_model(
            "dsl::compile_module_source_to_runtime(NativeAot)",
            &runtime_aot,
            1e-10,
        )?;

        let runtime_wasm_workspace = bimodal_ke::ArtifactWorkspace::new()?;
        let runtime_wasm = bimodal_ke::compile_runtime_wasm_model(&runtime_wasm_workspace)?;
        assert_eq!(runtime_wasm.backend(), RuntimeBackend::Wasm);
        bimodal_ke::report_runtime_model(
            "dsl::compile_module_source_to_runtime(Wasm)",
            &runtime_wasm,
            1e-4,
        )?;

        let direct_aot_workspace = bimodal_ke::ArtifactWorkspace::new()?;
        let direct_aot = bimodal_ke::compile_direct_aot_model(&direct_aot_workspace)?;
        assert_eq!(direct_aot.backend(), RuntimeBackend::NativeAot);
        bimodal_ke::report_runtime_model(
            "dsl::compile_module_source_to_aot + load_runtime_artifact",
            &direct_aot,
            1e-10,
        )?;

        let direct_wasm_workspace = bimodal_ke::ArtifactWorkspace::new()?;
        let direct_wasm = bimodal_ke::compile_direct_wasm_model(&direct_wasm_workspace)?;
        assert_eq!(direct_wasm.backend(), RuntimeBackend::Wasm);
        bimodal_ke::report_runtime_model(
            "dsl::compile_module_source_to_wasm + load_runtime_artifact",
            &direct_wasm,
            1e-4,
        )?;

        Ok(())
    }
}
