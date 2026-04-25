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

#[cfg(feature = "dsl-aot")]
use super::execution::ExecutionModel;
#[cfg(feature = "dsl-aot-load")]
use super::native::{CompiledNativeModel, DenseKernelFn, NativeExecutionArtifact, NativeModelInfo};
#[cfg(feature = "dsl-aot")]
use super::rust_backend::{emit_rust_backend_source, RustBackendFlavor};
#[cfg(any(feature = "dsl-aot", feature = "dsl-aot-load"))]
use super::rust_backend::{
    API_VERSION_SYMBOL, DERIVE_SYMBOL, DIFFUSION_SYMBOL, DRIFT_SYMBOL, DYNAMICS_SYMBOL,
    INIT_SYMBOL, MODEL_INFO_JSON_LEN_SYMBOL, MODEL_INFO_JSON_PTR_SYMBOL, OUTPUTS_SYMBOL,
    ROUTE_BIOAVAILABILITY_SYMBOL, ROUTE_LAG_SYMBOL,
};
#[cfg(feature = "dsl-aot-load")]
use super::ModelKind;
#[cfg(feature = "dsl-aot")]
use super::{analyze_module, lower_typed_model, parse_module};
#[cfg(feature = "dsl-aot")]
use crate::build_support::{
    build_cargo_template, create_cargo_template, native_cdylib_filename, write_template_source,
};

pub const AOT_API_VERSION: u32 = 1;

#[derive(Debug, Error)]
pub enum AotError {
    #[error(transparent)]
    Io(#[from] io::Error),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    #[error("failed to parse DSL source: {0}")]
    Parse(String),
    #[error("failed to analyze DSL source: {0}")]
    Semantic(String),
    #[error("failed to lower DSL model: {0}")]
    Lowering(String),
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

#[cfg(feature = "dsl-aot")]
pub fn compile_module_source_to_aot(
    source: &str,
    model_name: Option<&str>,
    output: Option<PathBuf>,
    template_root: PathBuf,
    event_callback: impl Fn(String, String) + Send + Sync + 'static,
) -> Result<PathBuf, AotError> {
    let parsed = parse_module(source).map_err(|error| AotError::Parse(error.render(source)))?;
    let typed =
        analyze_module(&parsed).map_err(|error| AotError::Semantic(error.render(source)))?;

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
        lower_typed_model(model).map_err(|error| AotError::Lowering(error.render(source)))?;
    export_execution_model_to_aot(&execution, output, template_root, event_callback)
}

#[cfg(feature = "dsl-aot")]
pub fn export_execution_model_to_aot(
    model: &ExecutionModel,
    output: Option<PathBuf>,
    template_root: PathBuf,
    event_callback: impl Fn(String, String) + Send + Sync + 'static,
) -> Result<PathBuf, AotError> {
    let event_callback = Arc::new(event_callback);
    let template_dir = create_cargo_template(template_root.clone(), &aot_template_manifest())?;
    let source = emit_rust_backend_source(
        model,
        RustBackendFlavor::NativeAot {
            api_version: AOT_API_VERSION,
        },
    )
    .map_err(AotError::Emit)?;
    write_template_source(&template_dir, &source)?;

    let dylib_name = native_cdylib_filename("model_lib");
    let dylib_path = build_cargo_template(
        template_dir,
        event_callback.clone(),
        "native-aot",
        model.name.clone(),
        None,
        &["release", dylib_name.as_str()],
    )?;

    let output_path = output.unwrap_or_else(|| default_output_path(&template_root));
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
fn default_output_path(template_root: &Path) -> PathBuf {
    let random_suffix: String = rand::rng()
        .sample_iter(&Alphanumeric)
        .take(5)
        .map(char::from)
        .collect();
    template_root.join(format!(
        "model_{}_{}_{}.pkm",
        std::env::consts::OS,
        std::env::consts::ARCH,
        random_suffix
    ))
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
    Ok(serde_json::from_slice(bytes)?)
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
    use crate::SubjectBuilderExt;
    use approx::assert_relative_eq;
    use tempfile::tempdir;

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

    #[test]
    fn aot_ode_artifact_matches_jit_predictions() {
        let model = load_proposal_model("one_cmt_oral_iv");
        let work_dir = tempdir().expect("tempdir");
        let output_path = work_dir.path().join("one_cmt_oral_iv.pkm");

        let jit = compile_ode_model_to_jit(&model).expect("compile jit model");
        export_execution_model_to_aot(
            &model,
            Some(output_path.clone()),
            work_dir.path().join("build"),
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
}
