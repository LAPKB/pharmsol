use std::fmt;
use std::path::Path;

use ndarray::Array2;
use thiserror::Error;

#[cfg(all(feature = "dsl-aot", feature = "dsl-aot-load"))]
use super::aot::{
    export_execution_model_to_aot, load_aot_model, AotError, NativeAotCompileOptions,
};
#[cfg(feature = "dsl-jit")]
use super::jit::{compile_execution_model_to_jit, JitCompileError};
#[cfg(feature = "dsl-wasm")]
use super::native::RuntimeArtifact;
use super::native::{
    CompiledNativeModel, NativeAnalyticalModel, NativeCovariateInfo, NativeModelInfo,
    NativeOdeModel, NativeOutputInfo, NativeRouteInfo, NativeSdeModel, RuntimeBackend,
};
#[cfg(feature = "dsl-wasm")]
use super::wasm::{load_wasm_artifact, load_wasm_artifact_bytes};
#[cfg(feature = "dsl-wasm")]
use super::wasm_compile::{
    compile_execution_model_to_wasm_bytes, compile_module_source_to_wasm_bytes, WasmError,
};
use crate::{
    simulator::likelihood::{Prediction, SubjectPredictions},
    PharmsolError, Subject,
};
use pharmsol_dsl::{
    analyze_module, lower_typed_model, parse_module, Diagnostic, DiagnosticReport, ExecutionModel,
    LoweringError, ModelKind, ParseError, SemanticError,
};

pub type RuntimeModelInfo = NativeModelInfo;
pub type RuntimeCovariateInfo = NativeCovariateInfo;
pub type RuntimeRouteInfo = NativeRouteInfo;
pub type RuntimeOutputInfo = NativeOutputInfo;
pub type RuntimeOdeModel = NativeOdeModel;
pub type RuntimeAnalyticalModel = NativeAnalyticalModel;
pub type RuntimeSdeModel = NativeSdeModel;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RuntimeCompilationTarget {
    #[cfg(feature = "dsl-jit")]
    Jit,
    #[cfg(all(feature = "dsl-aot", feature = "dsl-aot-load"))]
    NativeAot(NativeAotCompileOptions),
    #[cfg(feature = "dsl-wasm")]
    Wasm,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeArtifactFormat {
    #[cfg(all(feature = "dsl-aot", feature = "dsl-aot-load"))]
    NativeAot,
    #[cfg(feature = "dsl-wasm")]
    Wasm,
}

#[derive(Clone, Debug)]
pub enum RuntimePredictions {
    Subject(SubjectPredictions),
    Particles(Array2<Prediction>),
}

impl RuntimePredictions {
    pub fn as_subject(&self) -> Option<&SubjectPredictions> {
        match self {
            Self::Subject(predictions) => Some(predictions),
            Self::Particles(_) => None,
        }
    }

    pub fn as_particles(&self) -> Option<&Array2<Prediction>> {
        match self {
            Self::Particles(predictions) => Some(predictions),
            Self::Subject(_) => None,
        }
    }

    pub fn into_subject(self) -> Option<SubjectPredictions> {
        match self {
            Self::Subject(predictions) => Some(predictions),
            Self::Particles(_) => None,
        }
    }

    pub fn into_particles(self) -> Option<Array2<Prediction>> {
        match self {
            Self::Particles(predictions) => Some(predictions),
            Self::Subject(_) => None,
        }
    }
}

#[derive(Clone, Debug)]
pub enum CompiledRuntimeModel {
    Ode(RuntimeOdeModel),
    Analytical(RuntimeAnalyticalModel),
    Sde(RuntimeSdeModel),
}

impl From<CompiledNativeModel> for CompiledRuntimeModel {
    fn from(value: CompiledNativeModel) -> Self {
        match value {
            CompiledNativeModel::Ode(model) => Self::Ode(model),
            CompiledNativeModel::Analytical(model) => Self::Analytical(model),
            CompiledNativeModel::Sde(model) => Self::Sde(model),
        }
    }
}

impl CompiledRuntimeModel {
    pub fn backend(&self) -> RuntimeBackend {
        match self {
            Self::Ode(model) => model.backend(),
            Self::Analytical(model) => model.backend(),
            Self::Sde(model) => model.backend(),
        }
    }

    pub fn info(&self) -> &RuntimeModelInfo {
        match self {
            Self::Ode(model) => model.info(),
            Self::Analytical(model) => model.info(),
            Self::Sde(model) => model.info(),
        }
    }

    pub fn kind(&self) -> ModelKind {
        self.info().kind
    }

    pub fn route_index(&self, name: &str) -> Option<usize> {
        match self {
            Self::Ode(model) => model.route_index(name),
            Self::Analytical(model) => model.route_index(name),
            Self::Sde(model) => model.route_index(name),
        }
    }

    pub fn output_index(&self, name: &str) -> Option<usize> {
        match self {
            Self::Ode(model) => model.output_index(name),
            Self::Analytical(model) => model.output_index(name),
            Self::Sde(model) => model.output_index(name),
        }
    }

    pub fn estimate_predictions(
        &self,
        subject: &Subject,
        support_point: &[f64],
    ) -> Result<RuntimePredictions, RuntimeError> {
        Ok(match self {
            Self::Ode(model) => {
                RuntimePredictions::Subject(model.estimate_predictions(subject, support_point)?)
            }
            Self::Analytical(model) => {
                RuntimePredictions::Subject(model.estimate_predictions(subject, support_point)?)
            }
            Self::Sde(model) => {
                RuntimePredictions::Particles(model.estimate_predictions(subject, support_point)?)
            }
        })
    }
}

#[derive(Error)]
pub enum RuntimeError {
    #[error("failed to parse DSL source: {0}")]
    Parse(#[source] ParseError),
    #[error("failed to analyze DSL source: {0}")]
    Semantic(#[source] SemanticError),
    #[error("failed to lower DSL model: {0}")]
    Lowering(#[source] LoweringError),
    #[error("{0}")]
    ModelSelection(String),
    #[cfg(feature = "dsl-jit")]
    #[error(transparent)]
    Jit(#[from] JitCompileError),
    #[cfg(all(feature = "dsl-aot", feature = "dsl-aot-load"))]
    #[error(transparent)]
    Aot(#[from] AotError),
    #[cfg(feature = "dsl-wasm")]
    #[error(transparent)]
    Wasm(#[from] WasmError),
    #[error(transparent)]
    Runtime(#[from] PharmsolError),
}

impl RuntimeError {
    pub fn diagnostic(&self) -> Option<&Diagnostic> {
        match self {
            Self::Parse(error) => Some(error.diagnostic()),
            Self::Semantic(error) => Some(error.diagnostic()),
            Self::Lowering(error) => Some(error.diagnostic()),
            #[cfg(feature = "dsl-jit")]
            Self::Jit(error) => Some(error.diagnostic()),
            _ => None,
        }
    }

    pub fn render_diagnostic(&self, src: &str) -> Option<String> {
        self.diagnostic().map(|diagnostic| diagnostic.render(src))
    }

    pub fn diagnostic_report(&self, source_name: impl Into<String>) -> Option<DiagnosticReport> {
        let source_name = source_name.into();
        match self {
            Self::Parse(error) => Some(error.diagnostic_report(source_name)),
            Self::Semantic(error) => Some(error.diagnostic_report(source_name)),
            Self::Lowering(error) => Some(error.diagnostic_report(source_name)),
            #[cfg(feature = "dsl-jit")]
            Self::Jit(error) => Some(error.diagnostic_report(source_name)),
            _ => None,
        }
    }
}

impl fmt::Debug for RuntimeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Parse(error) => fmt::Display::fmt(error, f),
            Self::Semantic(error) => fmt::Display::fmt(error, f),
            Self::Lowering(error) => fmt::Display::fmt(error, f),
            #[cfg(feature = "dsl-jit")]
            Self::Jit(error) => fmt::Display::fmt(error, f),
            _ => fmt::Display::fmt(self, f),
        }
    }
}

pub fn compile_module_source_to_runtime(
    source: &str,
    model_name: Option<&str>,
    target: RuntimeCompilationTarget,
    event_callback: impl Fn(String, String) + Send + Sync + 'static,
) -> Result<CompiledRuntimeModel, RuntimeError> {
    let parsed =
        parse_module(source).map_err(|error| RuntimeError::Parse(error.with_source(source)))?;
    let typed = analyze_module(&parsed)
        .map_err(|error| RuntimeError::Semantic(error.with_source(source)))?;

    let model = match model_name {
        Some(name) => typed
            .models
            .iter()
            .find(|model| model.name == name)
            .ok_or_else(|| {
                RuntimeError::ModelSelection(format!("model `{name}` not found in module"))
            })?,
        None if typed.models.len() == 1 => &typed.models[0],
        None => {
            return Err(RuntimeError::ModelSelection(
                "module contains multiple models; pass an explicit model name".to_string(),
            ))
        }
    };

    let execution = lower_typed_model(model)
        .map_err(|error| RuntimeError::Lowering(error.with_source(source)))?;
    compile_execution_model_to_runtime(&execution, target, event_callback).map_err(|error| {
        #[cfg(feature = "dsl-jit")]
        if let RuntimeError::Jit(error) = error {
            return RuntimeError::Jit(error.with_source(source));
        }
        error
    })
}

pub fn compile_execution_model_to_runtime(
    model: &ExecutionModel,
    target: RuntimeCompilationTarget,
    event_callback: impl Fn(String, String) + Send + Sync + 'static,
) -> Result<CompiledRuntimeModel, RuntimeError> {
    match target {
        #[cfg(feature = "dsl-jit")]
        RuntimeCompilationTarget::Jit => {
            event_callback(
                "started".into(),
                format!("Compiling jit model `{}`", model.name),
            );
            let compiled = compile_execution_model_to_jit(model)?;
            event_callback(
                "finished".into(),
                format!("Compiled jit model `{}`", model.name),
            );
            Ok(compiled.into())
        }
        #[cfg(all(feature = "dsl-aot", feature = "dsl-aot-load"))]
        RuntimeCompilationTarget::NativeAot(options) => {
            let artifact = export_execution_model_to_aot(model, options, event_callback)?;
            load_runtime_artifact(&artifact, RuntimeArtifactFormat::NativeAot)
        }
        #[cfg(feature = "dsl-wasm")]
        RuntimeCompilationTarget::Wasm => {
            event_callback(
                "started".into(),
                format!("Compiling runtime wasm model `{}`", model.name),
            );
            let compiled = compile_execution_model_to_runtime_wasm(model)?;
            event_callback(
                "finished".into(),
                format!("Compiled runtime wasm model `{}`", model.name),
            );
            Ok(compiled)
        }
    }
}

pub fn load_runtime_artifact(
    path: impl AsRef<Path>,
    format: RuntimeArtifactFormat,
) -> Result<CompiledRuntimeModel, RuntimeError> {
    #[cfg(not(any(
        all(feature = "dsl-aot", feature = "dsl-aot-load"),
        feature = "dsl-wasm"
    )))]
    let _ = path.as_ref();
    match format {
        #[cfg(all(feature = "dsl-aot", feature = "dsl-aot-load"))]
        RuntimeArtifactFormat::NativeAot => Ok(load_aot_model(path)?.into()),
        #[cfg(feature = "dsl-wasm")]
        RuntimeArtifactFormat::Wasm => {
            let (info, artifact) = load_wasm_artifact(path)?;
            Ok(runtime_model_from_parts(info, artifact))
        }
    }
}

#[cfg(feature = "dsl-wasm")]
pub fn compile_module_source_to_runtime_wasm(
    source: &str,
    model_name: Option<&str>,
) -> Result<CompiledRuntimeModel, RuntimeError> {
    let bytes = compile_module_source_to_wasm_bytes(source, model_name)?;
    load_runtime_wasm_bytes(&bytes)
}

#[cfg(feature = "dsl-wasm")]
pub fn compile_execution_model_to_runtime_wasm(
    model: &ExecutionModel,
) -> Result<CompiledRuntimeModel, RuntimeError> {
    let bytes = compile_execution_model_to_wasm_bytes(model)?;
    load_runtime_wasm_bytes(&bytes)
}

#[cfg(feature = "dsl-wasm")]
pub fn load_runtime_wasm_bytes(bytes: &[u8]) -> Result<CompiledRuntimeModel, RuntimeError> {
    let (info, artifact) = load_wasm_artifact_bytes(bytes)?;
    Ok(runtime_model_from_parts(info, artifact))
}

#[cfg(feature = "dsl-wasm")]
fn runtime_model_from_parts(
    info: NativeModelInfo,
    artifact: impl RuntimeArtifact + 'static,
) -> CompiledRuntimeModel {
    match info.kind {
        ModelKind::Ode => CompiledRuntimeModel::Ode(NativeOdeModel::new(info, artifact)),
        ModelKind::Analytical => {
            CompiledRuntimeModel::Analytical(NativeAnalyticalModel::new(info, artifact))
        }
        ModelKind::Sde => CompiledRuntimeModel::Sde(NativeSdeModel::new(info, artifact)),
    }
}

#[cfg(all(
    test,
    feature = "dsl-jit",
    feature = "dsl-aot",
    feature = "dsl-aot-load",
    feature = "dsl-wasm"
))]
mod tests {
    use super::*;
    use crate::dsl::compile_sde_model_to_jit;
    use crate::SubjectBuilderExt;
    use approx::assert_relative_eq;
    use pharmsol_dsl::{DiagnosticPhase, DSL_BACKEND_GENERIC, DSL_PARSE_GENERIC};
    use tempfile::tempdir;

    fn corpus_source() -> &'static str {
        include_str!("../../tests/fixtures/dsl/02-structured-block-imperative.dsl")
    }

    fn corpus_model(name: &str) -> ExecutionModel {
        let parsed = pharmsol_dsl::parse_module(corpus_source()).expect("parse corpus module");
        let typed = pharmsol_dsl::analyze_module(&parsed).expect("analyze corpus module");
        let model = typed
            .models
            .iter()
            .find(|model| model.name == name)
            .expect("model present in corpus module");
        pharmsol_dsl::lower_typed_model(model).expect("lower corpus model")
    }

    fn ode_subject(output: usize, oral: usize, iv: usize) -> Subject {
        Subject::builder("ode")
            .covariate("wt", 0.0, 70.0)
            .bolus(0.0, 120.0, oral)
            .infusion(6.0, 60.0, iv, 2.0)
            .missing_observation(0.5, output)
            .missing_observation(1.0, output)
            .missing_observation(2.0, output)
            .missing_observation(6.0, output)
            .missing_observation(7.0, output)
            .missing_observation(9.0, output)
            .build()
    }

    fn subject_values(predictions: &RuntimePredictions) -> Vec<f64> {
        predictions
            .as_subject()
            .expect("subject predictions")
            .predictions()
            .iter()
            .map(|prediction| prediction.prediction())
            .collect()
    }

    #[test]
    fn runtime_backend_matrix_matches_ode_predictions() {
        let work_dir = tempdir().expect("tempdir");
        let support = vec![1.2, 5.0, 40.0, 0.5, 0.8];

        let jit = compile_module_source_to_runtime(
            corpus_source(),
            Some("one_cmt_oral_iv"),
            RuntimeCompilationTarget::Jit,
            |_, _| {},
        )
        .expect("compile jit runtime model");
        let aot = compile_module_source_to_runtime(
            corpus_source(),
            Some("one_cmt_oral_iv"),
            RuntimeCompilationTarget::NativeAot(
                NativeAotCompileOptions::new(work_dir.path().join("aot-build"))
                    .with_output(work_dir.path().join("one_cmt_oral_iv.pkm")),
            ),
            |_, _| {},
        )
        .expect("compile aot runtime model");
        let wasm = compile_module_source_to_runtime(
            corpus_source(),
            Some("one_cmt_oral_iv"),
            RuntimeCompilationTarget::Wasm,
            |_, _| {},
        )
        .expect("compile wasm runtime model");

        assert_eq!(jit.backend(), RuntimeBackend::Jit);
        assert_eq!(aot.backend(), RuntimeBackend::NativeAot);
        assert_eq!(wasm.backend(), RuntimeBackend::Wasm);
        assert_eq!(jit.info().name, "one_cmt_oral_iv");
        assert_eq!(
            wasm.info().parameters,
            vec!["ka", "cl", "v", "tlag", "f_oral"]
        );

        let oral = jit.route_index("oral").expect("oral route");
        let iv = jit.route_index("iv").expect("iv route");
        let cp = jit.output_index("cp").expect("cp output");
        let subject = ode_subject(cp, oral, iv);

        let jit_values = subject_values(
            &jit.estimate_predictions(&subject, &support)
                .expect("jit predictions"),
        );
        let aot_values = subject_values(
            &aot.estimate_predictions(&subject, &support)
                .expect("aot predictions"),
        );
        let wasm_values = subject_values(
            &wasm
                .estimate_predictions(&subject, &support)
                .expect("wasm predictions"),
        );

        for ((jit_value, aot_value), wasm_value) in jit_values
            .iter()
            .zip(aot_values.iter())
            .zip(wasm_values.iter())
        {
            assert_relative_eq!(jit_value, aot_value, max_relative = 1e-4);
            assert_relative_eq!(jit_value, wasm_value, max_relative = 1e-4);
        }
    }

    #[test]
    fn runtime_compile_preserves_parse_diagnostic_structure() {
        let source = "model broken { kind ode outputs { cp = 1 + } }";
        let error = compile_module_source_to_runtime(
            source,
            None,
            RuntimeCompilationTarget::Jit,
            |_, _| {},
        )
        .expect_err("invalid DSL should fail before runtime compilation");

        let diagnostic = error
            .diagnostic()
            .expect("runtime should expose diagnostic");
        assert_eq!(diagnostic.phase, DiagnosticPhase::Parse);
        assert_eq!(diagnostic.code, DSL_PARSE_GENERIC);
        assert!(diagnostic.message.contains("expected expression"));
        let rendered = error
            .render_diagnostic(source)
            .expect("rendered diagnostic");
        assert!(rendered.contains("error[DSL1000]"), "{}", rendered);
        assert!(rendered.contains("expected expression"), "{}", rendered);
        let debugged = format!("{error:?}");
        assert!(debugged.contains("error[DSL1000]"), "{}", debugged);
        assert!(debugged.contains("expected expression"), "{}", debugged);
        let report = error
            .diagnostic_report("inline.dsl")
            .expect("diagnostic report");
        assert_eq!(report.source.name, "inline.dsl");
        assert_eq!(report.diagnostics[0].code, "DSL1000");
        assert_eq!(report.diagnostics[0].labels[0].span.start_line, Some(1));
        assert!(report
            .to_json()
            .expect("serialize report")
            .contains("\"name\":\"inline.dsl\""),);
    }

    #[test]
    fn runtime_exposes_jit_backend_diagnostic_structure() {
        let source = corpus_source();
        let model = corpus_model("one_cmt_oral_iv");
        let error = RuntimeError::from(
            compile_sde_model_to_jit(&model)
                .expect_err("ODE model should not compile through the SDE JIT entrypoint")
                .with_source(source),
        );

        let diagnostic = error
            .diagnostic()
            .expect("runtime should expose jit diagnostic");
        assert_eq!(diagnostic.phase, DiagnosticPhase::Backend);
        assert_eq!(diagnostic.code, DSL_BACKEND_GENERIC);
        assert!(diagnostic.message.contains("not an SDE model"));

        let rendered = error
            .render_diagnostic(source)
            .expect("rendered backend diagnostic");
        assert!(rendered.contains("error[DSL4000]"), "{}", rendered);
        assert!(rendered.contains("not an SDE model"), "{}", rendered);

        let report = error
            .diagnostic_report("model.dsl")
            .expect("diagnostic report");
        assert_eq!(report.source.name, "model.dsl");
        assert_eq!(report.diagnostics[0].code, "DSL4000");
        assert_eq!(report.diagnostics[0].phase, "backend");
        assert!(report.diagnostics[0].labels[0].span.start_line.is_some());

        let debugged = format!("{error:?}");
        assert!(debugged.contains("error[DSL4000]"), "{}", debugged);
    }
}
