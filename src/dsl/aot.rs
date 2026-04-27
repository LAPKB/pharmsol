use std::fmt;
#[cfg(feature = "dsl-aot")]
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
#[cfg(feature = "dsl-aot")]
use std::sync::Arc;

#[cfg(feature = "dsl-aot-load")]
use libloading::{Library, Symbol};
#[cfg(feature = "dsl-aot")]
use rand::RngExt;
#[cfg(feature = "dsl-aot")]
use rand_distr::Alphanumeric;
use serde_json;
use thiserror::Error;

use super::compiled_backend_abi::{
    decode_compiled_model_info, API_VERSION_SYMBOL, DERIVE_SYMBOL, DIFFUSION_SYMBOL, DRIFT_SYMBOL,
    DYNAMICS_SYMBOL, INIT_SYMBOL, MODEL_INFO_JSON_LEN_SYMBOL, MODEL_INFO_JSON_PTR_SYMBOL,
    OUTPUTS_SYMBOL, ROUTE_BIOAVAILABILITY_SYMBOL, ROUTE_LAG_SYMBOL,
};
#[cfg(feature = "dsl-aot")]
use super::execution::ExecutionModel;
#[cfg(feature = "dsl-aot-load")]
use super::native::{CompiledNativeModel, DenseKernelFn, NativeExecutionArtifact, NativeModelInfo};
#[cfg(feature = "dsl-aot")]
use super::rust_backend::{emit_rust_backend_source, RustBackendFlavor};
#[cfg(feature = "dsl-aot-load")]
use super::ModelKind;
#[cfg(feature = "dsl-aot")]
use super::{analyze_module, lower_typed_model, parse_module};
use super::{Diagnostic, DiagnosticReport, LoweringError, ParseError, SemanticError};
#[cfg(feature = "dsl-aot")]
use crate::build_support::{
    build_cargo_template, create_cargo_template, native_cdylib_filename_for_target,
    write_template_source,
};
#[cfg(all(test, feature = "dsl-aot"))]
use crate::build_support::{rustc_host_target, rustup_installed_targets};

pub const AOT_API_VERSION: u32 = 1;

#[cfg(feature = "dsl-aot")]
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum NativeAotTarget {
    #[default]
    Host,
    Triple(String),
}

#[cfg(feature = "dsl-aot")]
impl NativeAotTarget {
    pub fn triple(target: impl Into<String>) -> Self {
        Self::Triple(target.into())
    }

    fn cargo_target(&self) -> Option<&str> {
        match self {
            Self::Host => None,
            Self::Triple(target) => Some(target.as_str()),
        }
    }
}

#[cfg(feature = "dsl-aot")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeAotCompileOptions {
    pub target: NativeAotTarget,
    pub output: Option<PathBuf>,
    pub template_root: PathBuf,
}

#[cfg(feature = "dsl-aot")]
impl NativeAotCompileOptions {
    pub fn new(template_root: PathBuf) -> Self {
        Self {
            target: NativeAotTarget::Host,
            output: None,
            template_root,
        }
    }

    pub fn with_output(mut self, output: PathBuf) -> Self {
        self.output = Some(output);
        self
    }

    pub fn with_target(mut self, target: NativeAotTarget) -> Self {
        self.target = target;
        self
    }
}

#[derive(Error)]
pub enum AotError {
    #[error(transparent)]
    Io(#[from] io::Error),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    #[error("failed to parse DSL source: {0}")]
    Parse(#[source] ParseError),
    #[error("failed to analyze DSL source: {0}")]
    Semantic(#[source] SemanticError),
    #[error("failed to lower DSL model: {0}")]
    Lowering(#[source] LoweringError),
    #[error("{0}")]
    ModelSelection(String),
    #[error("AoT artifact API version mismatch: expected {expected}, found {found}")]
    ApiVersionMismatch { expected: u32, found: u32 },
    #[error("missing required AoT symbol `{0}`")]
    MissingSymbol(&'static str),
    #[error("failed to emit AoT library source: {0}")]
    Emit(String),
    #[error("failed to load AoT artifact: {0}")]
    Load(String),
}

impl AotError {
    pub fn diagnostic(&self) -> Option<&Diagnostic> {
        match self {
            Self::Parse(error) => Some(error.diagnostic()),
            Self::Semantic(error) => Some(error.diagnostic()),
            Self::Lowering(error) => Some(error.diagnostic()),
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
            _ => None,
        }
    }
}

impl fmt::Debug for AotError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Parse(error) => fmt::Display::fmt(error, f),
            Self::Semantic(error) => fmt::Display::fmt(error, f),
            Self::Lowering(error) => fmt::Display::fmt(error, f),
            _ => fmt::Display::fmt(self, f),
        }
    }
}

#[cfg(feature = "dsl-aot")]
pub fn compile_module_source_to_aot(
    source: &str,
    model_name: Option<&str>,
    options: NativeAotCompileOptions,
    event_callback: impl Fn(String, String) + Send + Sync + 'static,
) -> Result<PathBuf, AotError> {
    let parsed =
        parse_module(source).map_err(|error| AotError::Parse(error.with_source(source)))?;
    let typed =
        analyze_module(&parsed).map_err(|error| AotError::Semantic(error.with_source(source)))?;

    let model = match model_name {
        Some(name) => typed
            .models
            .iter()
            .find(|model| model.name == name)
            .ok_or_else(|| {
                AotError::ModelSelection(format!("model `{name}` not found in module"))
            })?,
        None if typed.models.len() == 1 => &typed.models[0],
        None => {
            return Err(AotError::ModelSelection(
                "module contains multiple models; pass an explicit model name".to_string(),
            ))
        }
    };

    let execution =
        lower_typed_model(model).map_err(|error| AotError::Lowering(error.with_source(source)))?;
    export_execution_model_to_aot(&execution, options, event_callback)
}

#[cfg(feature = "dsl-aot")]
pub fn export_execution_model_to_aot(
    model: &ExecutionModel,
    options: NativeAotCompileOptions,
    event_callback: impl Fn(String, String) + Send + Sync + 'static,
) -> Result<PathBuf, AotError> {
    let event_callback = Arc::new(event_callback);
    let NativeAotCompileOptions {
        target,
        output,
        template_root,
    } = options;
    let cargo_target = target.cargo_target();
    let template_dir = create_cargo_template(template_root.clone(), &aot_template_manifest())?;
    let source = emit_rust_backend_source(
        model,
        RustBackendFlavor::NativeAot {
            api_version: AOT_API_VERSION,
        },
    )
    .map_err(AotError::Emit)?;
    write_template_source(&template_dir, &source)?;

    let dylib_name = native_cdylib_filename_for_target("model_lib", cargo_target);
    let dylib_path = match cargo_target {
        Some(target) => build_cargo_template(
            template_dir,
            event_callback.clone(),
            "native-aot",
            model.name.clone(),
            Some(target),
            &[target, "release", dylib_name.as_str()],
        )?,
        None => build_cargo_template(
            template_dir,
            event_callback.clone(),
            "native-aot",
            model.name.clone(),
            None,
            &["release", dylib_name.as_str()],
        )?,
    };

    let output_path = output.unwrap_or_else(|| default_output_path(&template_root, &target));
    fs::copy(&dylib_path, &output_path)?;
    event_callback(
        "finished".into(),
        format!(
            "Compiled native-aot model `{}` -> {}",
            model.name,
            output_path.display()
        ),
    );
    Ok(output_path)
}

#[cfg(feature = "dsl-aot-load")]
pub fn read_aot_model_info(path: impl AsRef<Path>) -> Result<NativeModelInfo, AotError> {
    let library = unsafe { Library::new(path.as_ref()) }
        .map_err(|error| AotError::Load(error.to_string()))?;
    let info = unsafe { read_model_info_from_library(&library)? };
    Ok(info)
}

#[cfg(feature = "dsl-aot-load")]
pub fn load_aot_model(path: impl AsRef<Path>) -> Result<CompiledNativeModel, AotError> {
    let path = path.as_ref();
    let library =
        unsafe { Library::new(path) }.map_err(|error| AotError::Load(error.to_string()))?;

    unsafe { ensure_api_version(&library)? };
    let info = unsafe { read_model_info_from_library(&library)? };
    let model_name = info.name.clone();
    let artifact = unsafe {
        NativeExecutionArtifact::from_library(
            model_name,
            load_optional_kernel(&library, DERIVE_SYMBOL),
            load_optional_kernel(&library, DYNAMICS_SYMBOL),
            load_required_kernel(&library, OUTPUTS_SYMBOL)?,
            load_optional_kernel(&library, INIT_SYMBOL),
            load_optional_kernel(&library, DRIFT_SYMBOL),
            load_optional_kernel(&library, DIFFUSION_SYMBOL),
            load_optional_kernel(&library, ROUTE_LAG_SYMBOL),
            load_optional_kernel(&library, ROUTE_BIOAVAILABILITY_SYMBOL),
            library,
        )
    };

    Ok(match info.kind {
        ModelKind::Ode => CompiledNativeModel::Ode(super::NativeOdeModel::new(info, artifact)),
        ModelKind::Analytical => {
            CompiledNativeModel::Analytical(super::NativeAnalyticalModel::new(info, artifact))
        }
        ModelKind::Sde => CompiledNativeModel::Sde(super::NativeSdeModel::new(info, artifact)),
    })
}

#[cfg(feature = "dsl-aot")]
fn default_output_path(template_root: &Path, target: &NativeAotTarget) -> PathBuf {
    let random_suffix: String = rand::rng()
        .sample_iter(&Alphanumeric)
        .take(5)
        .map(char::from)
        .collect();
    let target_label = match target {
        NativeAotTarget::Host => default_target_label(),
        NativeAotTarget::Triple(target) => sanitize_target_label(target),
    };
    template_root.join(format!("model_{}_{}.pkm", target_label, random_suffix))
}

#[cfg(feature = "dsl-aot")]
fn default_target_label() -> String {
    sanitize_target_label(&format!(
        "{}-{}",
        std::env::consts::ARCH,
        std::env::consts::OS
    ))
}

#[cfg(feature = "dsl-aot")]
fn sanitize_target_label(target: &str) -> String {
    target
        .chars()
        .map(|ch| if ch.is_ascii_alphanumeric() { ch } else { '_' })
        .collect()
}

#[cfg(feature = "dsl-aot")]
fn aot_template_manifest() -> String {
    r#"
        [package]
        name = "model_lib"
        version = "0.1.0"
        edition = "2021"

        [lib]
        crate-type = ["cdylib"]

        [workspace]
        "#
    .to_string()
}

#[cfg(feature = "dsl-aot-load")]
unsafe fn ensure_api_version(library: &Library) -> Result<(), AotError> {
    let symbol: Symbol<unsafe extern "C" fn() -> u32> = library
        .get(API_VERSION_SYMBOL.as_bytes())
        .map_err(|_| AotError::MissingSymbol(API_VERSION_SYMBOL))?;
    let found = symbol();
    if found != AOT_API_VERSION {
        return Err(AotError::ApiVersionMismatch {
            expected: AOT_API_VERSION,
            found,
        });
    }
    Ok(())
}

#[cfg(feature = "dsl-aot-load")]
unsafe fn read_model_info_from_library(library: &Library) -> Result<NativeModelInfo, AotError> {
    ensure_api_version(library)?;
    let ptr_symbol: Symbol<unsafe extern "C" fn() -> *const u8> = library
        .get(MODEL_INFO_JSON_PTR_SYMBOL.as_bytes())
        .map_err(|_| AotError::MissingSymbol(MODEL_INFO_JSON_PTR_SYMBOL))?;
    let len_symbol: Symbol<unsafe extern "C" fn() -> usize> = library
        .get(MODEL_INFO_JSON_LEN_SYMBOL.as_bytes())
        .map_err(|_| AotError::MissingSymbol(MODEL_INFO_JSON_LEN_SYMBOL))?;

    let ptr = ptr_symbol();
    let len = len_symbol();
    let bytes = std::slice::from_raw_parts(ptr, len);
    let envelope = decode_compiled_model_info(bytes)?;
    if envelope.abi_version != AOT_API_VERSION {
        return Err(AotError::ApiVersionMismatch {
            expected: AOT_API_VERSION,
            found: envelope.abi_version,
        });
    }
    Ok(envelope.model)
}

#[cfg(feature = "dsl-aot-load")]
unsafe fn load_required_kernel(
    library: &Library,
    name: &'static str,
) -> Result<DenseKernelFn, AotError> {
    let symbol: Symbol<DenseKernelFn> = library
        .get(name.as_bytes())
        .map_err(|_| AotError::MissingSymbol(name))?;
    Ok(*symbol)
}

#[cfg(feature = "dsl-aot-load")]
unsafe fn load_optional_kernel(library: &Library, name: &'static str) -> Option<DenseKernelFn> {
    library
        .get::<DenseKernelFn>(name.as_bytes())
        .ok()
        .map(|symbol| *symbol)
}

#[cfg(all(
    test,
    feature = "dsl-aot",
    feature = "dsl-aot-load",
    feature = "dsl-jit"
))]
mod tests {
    use super::*;
    use crate::dsl::{compile_ode_model_to_jit, lower_typed_model, parse_module};
    use crate::dsl::{DiagnosticPhase, DSL_SEMANTIC_GENERIC};
    use crate::SubjectBuilderExt;
    use approx::assert_relative_eq;
    use std::sync::{Arc, Mutex};
    use tempfile::tempdir;

    const CROSS_TARGET_SMOKE_ENV: &str = "PHARMSOL_NATIVE_AOT_SMOKE_TARGET";

    enum CrossTargetSmokeDecision {
        Run(String),
        Skip(String),
    }

    fn load_proposal_model(name: &str) -> ExecutionModel {
        let source = std::fs::read_to_string("dsl-proposals/02-structured-block-imperative.dsl")
            .expect("proposal source");
        let parsed = parse_module(&source).expect("parse proposal module");
        let typed = analyze_module(&parsed).expect("analyze proposal module");
        let model = typed
            .models
            .iter()
            .find(|model| model.name == name)
            .expect("model in proposal module");
        lower_typed_model(model).expect("lower proposal model")
    }

    fn resolve_cross_target_smoke_target() -> Result<CrossTargetSmokeDecision, String> {
        let requested_target = std::env::var(CROSS_TARGET_SMOKE_ENV)
            .ok()
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty());
        let host_target = rustc_host_target()
            .map_err(|error| format!("failed to detect the Rust host target: {error}"))?;

        let installed_targets = match rustup_installed_targets() {
            Ok(targets) => targets,
            Err(error) if requested_target.is_none() => {
                return Ok(CrossTargetSmokeDecision::Skip(format!(
                    "rustup target discovery is unavailable: {error}"
                )))
            }
            Err(error) => {
                return Err(format!(
                    "{CROSS_TARGET_SMOKE_ENV} is set, but installed targets could not be queried: {error}"
                ))
            }
        };

        if let Some(target) = requested_target {
            if target == host_target {
                return Err(format!(
                    "{CROSS_TARGET_SMOKE_ENV} must name a non-host native target, but `{target}` matches the host"
                ));
            }
            if !is_native_target_triple(&target) {
                return Err(format!(
                    "{CROSS_TARGET_SMOKE_ENV} must name a native target triple, but `{target}` is not supported for native AoT"
                ));
            }
            if !installed_targets
                .iter()
                .any(|installed| installed == &target)
            {
                return Err(format!(
                    "{CROSS_TARGET_SMOKE_ENV} requested `{target}`, but it is not installed. Run `rustup target add {target}` first."
                ));
            }
            return Ok(CrossTargetSmokeDecision::Run(target));
        }

        if let Some(target) =
            auto_detect_cross_target_smoke_target(&host_target, &installed_targets)
        {
            return Ok(CrossTargetSmokeDecision::Run(target));
        }

        Ok(CrossTargetSmokeDecision::Skip(format!(
            "no supported non-host native target is installed; set {CROSS_TARGET_SMOKE_ENV} after installing a target and linker"
        )))
    }

    fn auto_detect_cross_target_smoke_target(
        host_target: &str,
        installed_targets: &[String],
    ) -> Option<String> {
        let preferred = match host_target {
            "aarch64-apple-darwin" => &["x86_64-apple-darwin"][..],
            "x86_64-apple-darwin" => &["aarch64-apple-darwin"][..],
            _ => &[][..],
        };

        preferred
            .iter()
            .find(|candidate| {
                installed_targets
                    .iter()
                    .any(|installed| installed == *candidate)
            })
            .map(|candidate| (*candidate).to_string())
    }

    fn is_native_target_triple(target: &str) -> bool {
        !target.starts_with("wasm32-") && !target.starts_with("wasm64-")
    }

    fn render_captured_events(events: &Arc<Mutex<Vec<(String, String)>>>) -> String {
        let events = events
            .lock()
            .expect("cross-target smoke event log mutex poisoned");
        if events.is_empty() {
            return "<no compile events captured>".to_string();
        }

        events
            .iter()
            .map(|(kind, message)| format!("[{kind}] {}", message.trim_end()))
            .collect::<Vec<_>>()
            .join("\n")
    }

    #[test]
    fn aot_ode_artifact_matches_jit_predictions() {
        let model = load_proposal_model("one_cmt_oral_iv");
        let work_dir = tempdir().expect("tempdir");
        let output_path = work_dir.path().join("one_cmt_oral_iv.pkm");

        let jit = compile_ode_model_to_jit(&model).expect("compile jit model");
        export_execution_model_to_aot(
            &model,
            NativeAotCompileOptions::new(work_dir.path().join("build"))
                .with_output(output_path.clone()),
            |_, _| {},
        )
        .expect("export aot model");

        let loaded = load_aot_model(&output_path).expect("load aot model");
        let aot = match loaded {
            CompiledNativeModel::Ode(model) => model,
            other => panic!("expected ode model, got {other:?}"),
        };

        let oral = jit.route_index("oral").expect("jit oral route");
        let iv = jit.route_index("iv").expect("jit iv route");
        let cp = jit.output_index("cp").expect("jit cp output");
        assert_eq!(aot.route_index("oral"), Some(oral));
        assert_eq!(aot.route_index("iv"), Some(iv));
        assert_eq!(aot.output_index("cp"), Some(cp));

        let subject = crate::Subject::builder("ode")
            .covariate("wt", 0.0, 70.0)
            .bolus(0.0, 120.0, oral)
            .infusion(6.0, 60.0, iv, 2.0)
            .missing_observation(0.5, cp)
            .missing_observation(1.0, cp)
            .missing_observation(2.0, cp)
            .missing_observation(6.0, cp)
            .missing_observation(7.0, cp)
            .missing_observation(9.0, cp)
            .build();

        let support = vec![1.2, 5.0, 40.0, 0.5, 0.8];
        let jit_predictions = jit
            .estimate_predictions(&subject, &support)
            .expect("jit predictions");
        let aot_predictions = aot
            .estimate_predictions(&subject, &support)
            .expect("aot predictions");

        for (jit_pred, aot_pred) in jit_predictions
            .predictions()
            .iter()
            .zip(aot_predictions.predictions())
        {
            assert_relative_eq!(
                jit_pred.prediction(),
                aot_pred.prediction(),
                max_relative = 1e-4
            );
        }

        let info = read_aot_model_info(&output_path).expect("aot model info");
        assert_eq!(info.name, "one_cmt_oral_iv");
        assert_eq!(info.kind, ModelKind::Ode);
        assert_eq!(info.parameters, vec!["ka", "cl", "v", "tlag", "f_oral"]);
    }

    #[test]
    fn native_cdylib_filename_tracks_requested_target() {
        assert_eq!(
            native_cdylib_filename_for_target("model_lib", Some("x86_64-pc-windows-msvc")),
            "model_lib.dll"
        );
        assert_eq!(
            native_cdylib_filename_for_target("model_lib", Some("aarch64-apple-darwin")),
            "libmodel_lib.dylib"
        );
        assert_eq!(
            native_cdylib_filename_for_target("model_lib", Some("x86_64-unknown-linux-gnu")),
            "libmodel_lib.so"
        );
    }

    #[test]
    fn default_output_path_uses_requested_target_label() {
        let work_dir = tempdir().expect("tempdir");
        let output = default_output_path(
            work_dir.path(),
            &NativeAotTarget::triple("x86_64-pc-windows-msvc"),
        );
        let file_name = output
            .file_name()
            .expect("output file name")
            .to_string_lossy();
        assert!(file_name.starts_with("model_x86_64_pc_windows_msvc_"));
        assert!(file_name.ends_with(".pkm"));
    }

    #[test]
    fn native_aot_compile_options_default_to_host_target() {
        let work_dir = tempdir().expect("tempdir");
        let options = NativeAotCompileOptions::new(work_dir.path().join("build"));
        assert_eq!(options.target, NativeAotTarget::Host);
        assert_eq!(options.output, None);
    }

    #[test]
    fn native_aot_cross_target_smoke_builds_when_supported() {
        let target = match resolve_cross_target_smoke_target() {
            Ok(CrossTargetSmokeDecision::Run(target)) => target,
            Ok(CrossTargetSmokeDecision::Skip(reason)) => {
                eprintln!("skipping Native AoT cross-target smoke test: {reason}");
                return;
            }
            Err(error) => panic!("invalid cross-target smoke configuration: {error}"),
        };

        let model = load_proposal_model("one_cmt_oral_iv");
        let work_dir = tempdir().expect("tempdir");
        let output_path = work_dir.path().join(format!(
            "one_cmt_oral_iv_{}.pkm",
            sanitize_target_label(&target)
        ));
        let events = Arc::new(Mutex::new(Vec::<(String, String)>::new()));
        let captured_events = Arc::clone(&events);

        let result = export_execution_model_to_aot(
            &model,
            NativeAotCompileOptions::new(work_dir.path().join("cross-target-build"))
                .with_target(NativeAotTarget::triple(target.clone()))
                .with_output(output_path.clone()),
            move |kind, message| {
                captured_events
                    .lock()
                    .expect("cross-target smoke event log mutex poisoned")
                    .push((kind, message));
            },
        );

        match result {
            Ok(path) => {
                assert_eq!(path, output_path);
                assert!(path.exists());
            }
            Err(error) => panic!(
                "Native AoT cross-target smoke build failed for `{target}`: {error}\n{}",
                render_captured_events(&events)
            ),
        }
    }

    #[test]
    fn aot_compile_preserves_semantic_diagnostic_structure() {
        let source = r#"
model broken {
  kind ode
  states { central }
  dynamics {
    ddt(central) = rate(oral)
  }
  outputs {
    cp = central
  }
}
"#;
        let work_dir = tempdir().expect("tempdir");
        let error = compile_module_source_to_aot(
            source,
            None,
            NativeAotCompileOptions::new(work_dir.path().join("build")),
            |_, _| {},
        )
        .expect_err("invalid DSL should fail before AoT compilation");

        let diagnostic = error.diagnostic().expect("AoT should expose diagnostic");
        assert_eq!(diagnostic.phase, DiagnosticPhase::Semantic);
        assert_eq!(diagnostic.code, DSL_SEMANTIC_GENERIC);
        assert!(diagnostic.message.contains("unknown route `oral`"));
        let rendered = error
            .render_diagnostic(source)
            .expect("rendered diagnostic");
        assert!(rendered.contains("error[DSL2000]"), "{}", rendered);
        assert!(rendered.contains("unknown route `oral`"), "{}", rendered);
        let debugged = format!("{error:?}");
        assert!(debugged.contains("error[DSL2000]"), "{}", debugged);
        assert!(debugged.contains("unknown route `oral`"), "{}", debugged);
        let report = error
            .diagnostic_report("inline.dsl")
            .expect("diagnostic report");
        assert_eq!(report.source.name, "inline.dsl");
        assert_eq!(report.diagnostics[0].code, "DSL2000");
        assert!(report.diagnostics[0].labels.len() >= 1);
    }

    #[test]
    fn aot_compile_preserves_semantic_suggestions() {
        let source = r#"
model broken {
    kind ode
    states { central }
    routes { oral -> central }
    dynamics {
        ddt(central) = rate(orla)
    }
    outputs {
        cp = central
    }
}
"#;
        let work_dir = tempdir().expect("tempdir");
        let error = compile_module_source_to_aot(
            source,
            None,
            NativeAotCompileOptions::new(work_dir.path().join("build-suggestions")),
            |_, _| {},
        )
        .expect_err("invalid DSL should fail before AoT compilation");

        let diagnostic = error.diagnostic().expect("AoT should expose diagnostic");
        assert!(diagnostic
            .suggestions
            .iter()
            .any(|suggestion| suggestion.message.contains("did you mean `oral`?")));

        let rendered = error
            .render_diagnostic(source)
            .expect("rendered diagnostic");
        assert!(
            rendered.contains("suggestion: did you mean `oral`?"),
            "{}",
            rendered
        );
    }
}
