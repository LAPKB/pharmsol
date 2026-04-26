use std::fs;
use std::io;
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use rand::RngExt;
use rand_distr::Alphanumeric;
use serde_json;
use thiserror::Error;
use wasmtime::{Engine, Instance, Linker, Memory, Module, Store, TypedFunc};

use super::execution::ExecutionModel;
use super::execution::KernelRole;
use super::native::{KernelSession, NativeModelInfo, RuntimeArtifact, RuntimeBackend};
use super::rust_backend::{
    emit_rust_backend_source, RustBackendFlavor, ALLOC_F64_BUFFER_SYMBOL, API_VERSION_SYMBOL,
    DERIVE_SYMBOL, DIFFUSION_SYMBOL, DRIFT_SYMBOL, DYNAMICS_SYMBOL, FREE_F64_BUFFER_SYMBOL,
    INIT_SYMBOL, MODEL_INFO_JSON_LEN_SYMBOL, MODEL_INFO_JSON_PTR_SYMBOL, OUTPUTS_SYMBOL,
    ROUTE_BIOAVAILABILITY_SYMBOL, ROUTE_LAG_SYMBOL,
};
use super::{analyze_module, lower_typed_model, parse_module};
use crate::build_support::{build_cargo_template, create_cargo_template, write_template_source};
use crate::PharmsolError;

pub const WASM_API_VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WasmArtifactBundle {
    pub wasm_path: PathBuf,
    pub browser_loader_path: PathBuf,
}

#[derive(Debug, Error)]
pub enum WasmError {
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
    #[error("failed to emit WASM module source: {0}")]
    Emit(String),
    #[error("WASM artifact API version mismatch: expected {expected}, found {found}")]
    ApiVersionMismatch { expected: u32, found: u32 },
    #[error("missing required WASM export `{0}`")]
    MissingExport(&'static str),
    #[error(
        "WASM memory access out of bounds for {region}: ptr={ptr}, len={len}, memory_len={memory_len}"
    )]
    MemoryOutOfBounds {
        region: &'static str,
        ptr: i64,
        len: i64,
        memory_len: usize,
    },
    #[error("failed to load WASM artifact: {0}")]
    Load(String),
}

#[derive(Clone, Copy, Debug, Default)]
struct WasmKernelAvailability {
    derive: bool,
    dynamics: bool,
    outputs: bool,
    init: bool,
    drift: bool,
    diffusion: bool,
    route_lag: bool,
    route_bioavailability: bool,
}

impl WasmKernelAvailability {
    fn has(self, role: KernelRole) -> bool {
        match role {
            KernelRole::Derive => self.derive,
            KernelRole::Dynamics => self.dynamics,
            KernelRole::Outputs => self.outputs,
            KernelRole::Init => self.init,
            KernelRole::Drift => self.drift,
            KernelRole::Diffusion => self.diffusion,
            KernelRole::RouteLag => self.route_lag,
            KernelRole::RouteBioavailability => self.route_bioavailability,
            KernelRole::Analytical => false,
        }
    }
}

pub(crate) struct WasmExecutionArtifact {
    info: NativeModelInfo,
    engine: Engine,
    module: Module,
    kernels: WasmKernelAvailability,
    session_pool: Mutex<Vec<WasmKernelSession>>,
}

impl std::fmt::Debug for WasmExecutionArtifact {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WasmExecutionArtifact")
            .field("model", &self.info.name)
            .field("kind", &self.info.kind)
            .field("kernels", &self.kernels)
            .finish()
    }
}

#[derive(Debug)]
struct WasmBuffer {
    ptr: i32,
    len: usize,
}

struct WasmKernelSession {
    info: NativeModelInfo,
    store: Store<()>,
    memory: Memory,
    free: TypedFunc<(i32, i32), ()>,
    derive: Option<TypedFunc<(f64, i32, i32, i32, i32, i32, i32), ()>>,
    dynamics: Option<TypedFunc<(f64, i32, i32, i32, i32, i32, i32), ()>>,
    outputs: TypedFunc<(f64, i32, i32, i32, i32, i32, i32), ()>,
    init: Option<TypedFunc<(f64, i32, i32, i32, i32, i32, i32), ()>>,
    drift: Option<TypedFunc<(f64, i32, i32, i32, i32, i32, i32), ()>>,
    diffusion: Option<TypedFunc<(f64, i32, i32, i32, i32, i32, i32), ()>>,
    route_lag: Option<TypedFunc<(f64, i32, i32, i32, i32, i32, i32), ()>>,
    route_bioavailability: Option<TypedFunc<(f64, i32, i32, i32, i32, i32, i32), ()>>,
    states: WasmBuffer,
    params: WasmBuffer,
    covariates: WasmBuffer,
    routes: WasmBuffer,
    derived: WasmBuffer,
    out: WasmBuffer,
}

struct PooledWasmKernelSession<'a> {
    session: Option<WasmKernelSession>,
    pool: &'a Mutex<Vec<WasmKernelSession>>,
}

impl WasmKernelSession {
    fn new(
        info: &NativeModelInfo,
        engine: &Engine,
        module: &Module,
        kernels: WasmKernelAvailability,
    ) -> Result<Self, WasmError> {
        let mut store = Store::new(engine, ());
        let linker = Linker::new(engine);
        let instance = linker
            .instantiate(&mut store, module)
            .map_err(|error| WasmError::Load(error.to_string()))?;
        let memory = instance
            .get_memory(&mut store, "memory")
            .ok_or(WasmError::MissingExport("memory"))?;
        let alloc = typed_func::<i32, i32>(&instance, &mut store, ALLOC_F64_BUFFER_SYMBOL)?;
        let free = typed_func::<(i32, i32), ()>(&instance, &mut store, FREE_F64_BUFFER_SYMBOL)?;

        let states = alloc_buffer(&alloc, &mut store, info.state_len)?;
        let params = alloc_buffer(&alloc, &mut store, info.parameters.len())?;
        let covariates = alloc_buffer(&alloc, &mut store, info.covariates.len())?;
        let routes = alloc_buffer(&alloc, &mut store, info.route_len)?;
        let derived = alloc_buffer(&alloc, &mut store, info.derived_len)?;
        let out = alloc_buffer(
            &alloc,
            &mut store,
            info.state_len
                .max(info.derived_len)
                .max(info.output_len)
                .max(info.route_len),
        )?;

        let derive = if kernels.derive {
            optional_typed_func(&instance, &mut store, DERIVE_SYMBOL)?
        } else {
            None
        };
        let dynamics = if kernels.dynamics {
            optional_typed_func(&instance, &mut store, DYNAMICS_SYMBOL)?
        } else {
            None
        };
        let outputs = typed_func(&instance, &mut store, OUTPUTS_SYMBOL)?;
        let init = if kernels.init {
            optional_typed_func(&instance, &mut store, INIT_SYMBOL)?
        } else {
            None
        };
        let drift = if kernels.drift {
            optional_typed_func(&instance, &mut store, DRIFT_SYMBOL)?
        } else {
            None
        };
        let diffusion = if kernels.diffusion {
            optional_typed_func(&instance, &mut store, DIFFUSION_SYMBOL)?
        } else {
            None
        };
        let route_lag = if kernels.route_lag {
            optional_typed_func(&instance, &mut store, ROUTE_LAG_SYMBOL)?
        } else {
            None
        };
        let route_bioavailability = if kernels.route_bioavailability {
            optional_typed_func(&instance, &mut store, ROUTE_BIOAVAILABILITY_SYMBOL)?
        } else {
            None
        };

        Ok(Self {
            info: info.clone(),
            store,
            memory,
            free,
            derive,
            dynamics,
            outputs,
            init,
            drift,
            diffusion,
            route_lag,
            route_bioavailability,
            states,
            params,
            covariates,
            routes,
            derived,
            out,
        })
    }

    fn kernel(
        &self,
        role: KernelRole,
    ) -> Result<TypedFunc<(f64, i32, i32, i32, i32, i32, i32), ()>, PharmsolError> {
        match role {
            KernelRole::Derive => self.derive.clone(),
            KernelRole::Dynamics => self.dynamics.clone(),
            KernelRole::Outputs => Some(self.outputs.clone()),
            KernelRole::Init => self.init.clone(),
            KernelRole::Drift => self.drift.clone(),
            KernelRole::Diffusion => self.diffusion.clone(),
            KernelRole::RouteLag => self.route_lag.clone(),
            KernelRole::RouteBioavailability => self.route_bioavailability.clone(),
            KernelRole::Analytical => None,
        }
        .ok_or_else(|| {
            PharmsolError::OtherError(format!(
                "model `{}` does not provide a {:?} kernel",
                self.info.name, role
            ))
        })
    }
}

impl Drop for WasmKernelSession {
    fn drop(&mut self) {
        for buffer in [
            &self.states,
            &self.params,
            &self.covariates,
            &self.routes,
            &self.derived,
            &self.out,
        ] {
            if buffer.ptr != 0 && buffer.len != 0 {
                let _ = self
                    .free
                    .call(&mut self.store, (buffer.ptr, buffer.len as i32));
            }
        }
    }
}

impl KernelSession for WasmKernelSession {
    unsafe fn invoke_raw(
        &mut self,
        role: KernelRole,
        time: f64,
        states: *const f64,
        params: *const f64,
        covariates: *const f64,
        routes: *const f64,
        derived: *const f64,
        out: *mut f64,
    ) -> Result<(), PharmsolError> {
        let map_memory_error = |error: WasmError| {
            PharmsolError::OtherError(format!(
                "WASM memory access failed for model `{}`: {error}",
                self.info.name
            ))
        };
        write_f64s(
            &self.memory,
            &mut self.store,
            self.states.ptr,
            raw_slice(states, self.info.state_len),
        )
        .map_err(map_memory_error)?;
        write_f64s(
            &self.memory,
            &mut self.store,
            self.params.ptr,
            raw_slice(params, self.info.parameters.len()),
        )
        .map_err(map_memory_error)?;
        write_f64s(
            &self.memory,
            &mut self.store,
            self.covariates.ptr,
            raw_slice(covariates, self.info.covariates.len()),
        )
        .map_err(map_memory_error)?;
        write_f64s(
            &self.memory,
            &mut self.store,
            self.routes.ptr,
            raw_slice(routes, self.info.route_len),
        )
        .map_err(map_memory_error)?;
        write_f64s(
            &self.memory,
            &mut self.store,
            self.derived.ptr,
            raw_slice(derived, self.info.derived_len),
        )
        .map_err(map_memory_error)?;

        let out_ptr = if std::ptr::eq(out as *const f64, states) {
            self.states.ptr
        } else if std::ptr::eq(out as *const f64, derived) {
            self.derived.ptr
        } else {
            self.out.ptr
        };
        let out_len = kernel_output_len(&self.info, role);
        if out_ptr == self.out.ptr {
            zero_f64s(&self.memory, &mut self.store, out_ptr, out_len).map_err(map_memory_error)?;
        }

        self.kernel(role)?
            .call(
                &mut self.store,
                (
                    time,
                    self.states.ptr,
                    self.params.ptr,
                    self.covariates.ptr,
                    self.routes.ptr,
                    self.derived.ptr,
                    out_ptr,
                ),
            )
            .map_err(|error| {
                PharmsolError::OtherError(format!(
                    "WASM kernel {:?} trap for model `{}`: {error}",
                    role, self.info.name
                ))
            })?;

        read_f64s_into(
            &self.memory,
            &mut self.store,
            out_ptr,
            raw_slice_mut(out, out_len),
        )
        .map_err(map_memory_error)?;
        Ok(())
    }
}

impl KernelSession for PooledWasmKernelSession<'_> {
    unsafe fn invoke_raw(
        &mut self,
        role: KernelRole,
        time: f64,
        states: *const f64,
        params: *const f64,
        covariates: *const f64,
        routes: *const f64,
        derived: *const f64,
        out: *mut f64,
    ) -> Result<(), PharmsolError> {
        self.session
            .as_mut()
            .expect("pooled wasm session should be present")
            .invoke_raw(role, time, states, params, covariates, routes, derived, out)
    }
}

impl Drop for PooledWasmKernelSession<'_> {
    fn drop(&mut self) {
        if let Some(session) = self.session.take() {
            self.pool
                .lock()
                .expect("pooled wasm session mutex poisoned")
                .push(session);
        }
    }
}

impl RuntimeArtifact for WasmExecutionArtifact {
    fn backend(&self) -> RuntimeBackend {
        RuntimeBackend::Wasm
    }

    fn has_kernel(&self, role: KernelRole) -> bool {
        self.kernels.has(role)
    }

    fn start_session(&self) -> Result<Box<dyn KernelSession + '_>, PharmsolError> {
        let session = self
            .session_pool
            .lock()
            .expect("pooled wasm session mutex poisoned")
            .pop();

        let session = match session {
            Some(session) => session,
            None => WasmKernelSession::new(&self.info, &self.engine, &self.module, self.kernels)
                .map_err(|error| {
                    PharmsolError::OtherError(format!(
                        "failed to instantiate WASM runtime for model `{}`: {error}",
                        self.info.name
                    ))
                })?,
        };

        Ok(Box::new(PooledWasmKernelSession {
            session: Some(session),
            pool: &self.session_pool,
        }) as Box<dyn KernelSession>)
    }
}

pub fn read_wasm_model_info(path: impl AsRef<Path>) -> Result<NativeModelInfo, WasmError> {
    let (info, _) = load_wasm_artifact(path)?;
    Ok(info)
}

pub(crate) fn load_wasm_artifact(
    path: impl AsRef<Path>,
) -> Result<(NativeModelInfo, WasmExecutionArtifact), WasmError> {
    let engine = Engine::default();
    let module =
        Module::from_file(&engine, path).map_err(|error| WasmError::Load(error.to_string()))?;
    let mut store = Store::new(&engine, ());
    let linker = Linker::new(&engine);
    let instance = linker
        .instantiate(&mut store, &module)
        .map_err(|error| WasmError::Load(error.to_string()))?;
    let api_version = typed_func::<(), u32>(&instance, &mut store, API_VERSION_SYMBOL)?
        .call(&mut store, ())
        .map_err(|error| WasmError::Load(error.to_string()))?;
    if api_version != WASM_API_VERSION {
        return Err(WasmError::ApiVersionMismatch {
            expected: WASM_API_VERSION,
            found: api_version,
        });
    }

    let memory = instance
        .get_memory(&mut store, "memory")
        .ok_or(WasmError::MissingExport("memory"))?;
    let info = read_model_info(&instance, &mut store, &memory)?;
    let kernels = WasmKernelAvailability {
        derive: instance.get_func(&mut store, DERIVE_SYMBOL).is_some(),
        dynamics: instance.get_func(&mut store, DYNAMICS_SYMBOL).is_some(),
        outputs: instance.get_func(&mut store, OUTPUTS_SYMBOL).is_some(),
        init: instance.get_func(&mut store, INIT_SYMBOL).is_some(),
        drift: instance.get_func(&mut store, DRIFT_SYMBOL).is_some(),
        diffusion: instance.get_func(&mut store, DIFFUSION_SYMBOL).is_some(),
        route_lag: instance.get_func(&mut store, ROUTE_LAG_SYMBOL).is_some(),
        route_bioavailability: instance
            .get_func(&mut store, ROUTE_BIOAVAILABILITY_SYMBOL)
            .is_some(),
    };

    Ok((
        info.clone(),
        WasmExecutionArtifact {
            info,
            engine,
            module,
            kernels,
            session_pool: Mutex::new(Vec::new()),
        },
    ))
}

fn alloc_buffer(
    alloc: &TypedFunc<i32, i32>,
    store: &mut Store<()>,
    len: usize,
) -> Result<WasmBuffer, WasmError> {
    let ptr = if len == 0 {
        0
    } else {
        alloc
            .call(store, len as i32)
            .map_err(|error| WasmError::Load(error.to_string()))?
    };
    Ok(WasmBuffer { ptr, len })
}

fn kernel_output_len(info: &NativeModelInfo, role: KernelRole) -> usize {
    match role {
        KernelRole::Derive => info.derived_len,
        KernelRole::Dynamics | KernelRole::Init | KernelRole::Drift | KernelRole::Diffusion => {
            info.state_len
        }
        KernelRole::Outputs => info.output_len,
        KernelRole::RouteLag | KernelRole::RouteBioavailability => info.route_len,
        KernelRole::Analytical => 0,
    }
}

fn typed_func<Params, Results>(
    instance: &Instance,
    store: &mut Store<()>,
    name: &'static str,
) -> Result<TypedFunc<Params, Results>, WasmError>
where
    Params: wasmtime::WasmParams,
    Results: wasmtime::WasmResults,
{
    instance
        .get_typed_func(store, name)
        .map_err(|_| WasmError::MissingExport(name))
}

fn optional_typed_func<Params, Results>(
    instance: &Instance,
    store: &mut Store<()>,
    name: &'static str,
) -> Result<Option<TypedFunc<Params, Results>>, WasmError>
where
    Params: wasmtime::WasmParams,
    Results: wasmtime::WasmResults,
{
    match instance.get_typed_func(store, name) {
        Ok(func) => Ok(Some(func)),
        Err(_) => Ok(None),
    }
}

unsafe fn raw_slice<'a>(ptr: *const f64, len: usize) -> &'a [f64] {
    if len == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(ptr, len)
    }
}

unsafe fn raw_slice_mut<'a>(ptr: *mut f64, len: usize) -> &'a mut [f64] {
    if len == 0 {
        &mut []
    } else {
        std::slice::from_raw_parts_mut(ptr, len)
    }
}

fn read_model_info(
    instance: &Instance,
    store: &mut Store<()>,
    memory: &Memory,
) -> Result<NativeModelInfo, WasmError> {
    let ptr = typed_func::<(), i32>(instance, store, MODEL_INFO_JSON_PTR_SYMBOL)?
        .call(&mut *store, ())
        .map_err(|error| WasmError::Load(error.to_string()))?;
    let len = typed_func::<(), i32>(instance, store, MODEL_INFO_JSON_LEN_SYMBOL)?
        .call(&mut *store, ())
        .map_err(|error| WasmError::Load(error.to_string()))?;
    let data = memory.data(&mut *store);
    let len = usize::try_from(len).map_err(|_| WasmError::MemoryOutOfBounds {
        region: "model info",
        ptr: ptr as i64,
        len: len as i64,
        memory_len: data.len(),
    })?;
    let range = byte_range(ptr, len, data.len(), "model info")?;
    Ok(serde_json::from_slice(&data[range])?)
}

fn write_f64s(
    memory: &Memory,
    store: &mut Store<()>,
    ptr: i32,
    values: &[f64],
) -> Result<(), WasmError> {
    if values.is_empty() || ptr == 0 {
        return Ok(());
    }
    let data = memory.data_mut(store);
    let range = f64_buffer_range(ptr, values.len(), data.len(), "host write")?;
    for (chunk, value) in data[range]
        .chunks_exact_mut(std::mem::size_of::<f64>())
        .zip(values.iter())
    {
        chunk.copy_from_slice(&value.to_le_bytes());
    }
    Ok(())
}

fn zero_f64s(
    memory: &Memory,
    store: &mut Store<()>,
    ptr: i32,
    len: usize,
) -> Result<(), WasmError> {
    if len == 0 || ptr == 0 {
        return Ok(());
    }
    let data = memory.data_mut(store);
    let range = f64_buffer_range(ptr, len, data.len(), "host zero")?;
    data[range].fill(0);
    Ok(())
}

fn read_f64s_into(
    memory: &Memory,
    store: &mut Store<()>,
    ptr: i32,
    out: &mut [f64],
) -> Result<(), WasmError> {
    if out.is_empty() || ptr == 0 {
        return Ok(());
    }
    let data = memory.data(store);
    let range = f64_buffer_range(ptr, out.len(), data.len(), "host read")?;
    for (slot, chunk) in out
        .iter_mut()
        .zip(data[range].chunks_exact(std::mem::size_of::<f64>()))
    {
        let mut bytes = [0u8; std::mem::size_of::<f64>()];
        bytes.copy_from_slice(chunk);
        *slot = f64::from_le_bytes(bytes);
    }
    Ok(())
}

fn f64_buffer_range(
    ptr: i32,
    len: usize,
    memory_len: usize,
    region: &'static str,
) -> Result<Range<usize>, WasmError> {
    let byte_len =
        len.checked_mul(std::mem::size_of::<f64>())
            .ok_or(WasmError::MemoryOutOfBounds {
                region,
                ptr: ptr as i64,
                len: len as i64,
                memory_len,
            })?;
    byte_range(ptr, byte_len, memory_len, region)
}

fn byte_range(
    ptr: i32,
    byte_len: usize,
    memory_len: usize,
    region: &'static str,
) -> Result<Range<usize>, WasmError> {
    let start = usize::try_from(ptr).map_err(|_| WasmError::MemoryOutOfBounds {
        region,
        ptr: ptr as i64,
        len: byte_len as i64,
        memory_len,
    })?;
    let end = start
        .checked_add(byte_len)
        .ok_or(WasmError::MemoryOutOfBounds {
            region,
            ptr: ptr as i64,
            len: byte_len as i64,
            memory_len,
        })?;
    if end > memory_len {
        return Err(WasmError::MemoryOutOfBounds {
            region,
            ptr: ptr as i64,
            len: byte_len as i64,
            memory_len,
        });
    }
    Ok(start..end)
}

pub fn compile_module_source_to_wasm(
    source: &str,
    model_name: Option<&str>,
    output: Option<PathBuf>,
    template_root: PathBuf,
    event_callback: impl Fn(String, String) + Send + Sync + 'static,
) -> Result<WasmArtifactBundle, WasmError> {
    let parsed = parse_module(source).map_err(|error| WasmError::Parse(error.render(source)))?;
    let typed =
        analyze_module(&parsed).map_err(|error| WasmError::Semantic(error.render(source)))?;

    let model = match model_name {
        Some(name) => typed
            .models
            .iter()
            .find(|model| model.name == name)
            .ok_or_else(|| {
                WasmError::ModelSelection(format!("model `{name}` not found in module"))
            })?,
        None if typed.models.len() == 1 => &typed.models[0],
        None => {
            return Err(WasmError::ModelSelection(
                "module contains multiple models; pass an explicit model name".to_string(),
            ))
        }
    };

    let execution =
        lower_typed_model(model).map_err(|error| WasmError::Lowering(error.render(source)))?;
    export_execution_model_to_wasm(&execution, output, template_root, event_callback)
}

pub fn export_execution_model_to_wasm(
    model: &ExecutionModel,
    output: Option<PathBuf>,
    template_root: PathBuf,
    event_callback: impl Fn(String, String) + Send + Sync + 'static,
) -> Result<WasmArtifactBundle, WasmError> {
    let event_callback = Arc::new(event_callback);
    let template_dir = create_cargo_template(template_root.clone(), &wasm_template_manifest())?;
    let source = emit_rust_backend_source(
        model,
        RustBackendFlavor::Wasm {
            api_version: WASM_API_VERSION,
        },
    )
    .map_err(WasmError::Emit)?;
    write_template_source(&template_dir, &source)?;

    let built_wasm_path = build_cargo_template(
        template_dir,
        event_callback.clone(),
        "wasm",
        model.name.clone(),
        Some("wasm32-unknown-unknown"),
        &["wasm32-unknown-unknown", "release", "model_lib.wasm"],
    )?;

    let wasm_path = output.unwrap_or_else(|| default_wasm_output_path(&template_root));
    fs::copy(&built_wasm_path, &wasm_path)?;

    let browser_loader_path = wasm_path.with_extension("mjs");
    fs::write(&browser_loader_path, browser_loader_source())?;

    event_callback(
        "finished".into(),
        format!(
            "Compiled wasm model `{}` -> {} (loader -> {})",
            model.name,
            wasm_path.display(),
            browser_loader_path.display()
        ),
    );

    Ok(WasmArtifactBundle {
        wasm_path,
        browser_loader_path,
    })
}

pub fn browser_loader_source() -> String {
    format!(
        r#"const API_VERSION = {api_version};
const API_VERSION_SYMBOL = "{api_version_symbol}";
const MODEL_INFO_JSON_PTR_SYMBOL = "{model_info_ptr_symbol}";
const MODEL_INFO_JSON_LEN_SYMBOL = "{model_info_len_symbol}";
const ALLOC_F64_BUFFER_SYMBOL = "{alloc_f64_buffer_symbol}";
const FREE_F64_BUFFER_SYMBOL = "{free_f64_buffer_symbol}";

const KERNEL_SYMBOLS = Object.freeze({{
  derive: "{derive_symbol}",
  dynamics: "{dynamics_symbol}",
  outputs: "{outputs_symbol}",
    init: "{init_symbol}",
    drift: "{drift_symbol}",
    diffusion: "{diffusion_symbol}",
    route_lag: "{route_lag_symbol}",
    route_bioavailability: "{route_bioavailability_symbol}",
}});

function readUtf8(memory, ptr, len) {{
  const bytes = new Uint8Array(memory.buffer, ptr, len);
  return new TextDecoder().decode(bytes);
}}

function createBufferHandle(exports, memory, length) {{
  const ptr = Number(exports[ALLOC_F64_BUFFER_SYMBOL](length));
  return {{
    ptr,
    length,
    view() {{
      return length === 0 ? new Float64Array() : new Float64Array(memory.buffer, ptr, length);
    }},
    free() {{
      exports[FREE_F64_BUFFER_SYMBOL](ptr, length);
    }},
  }};
}}

export async function loadPharmsolDslWasmModel(source) {{
  const response = source instanceof Response ? source : await fetch(source);
  const {{ instance }} = await WebAssembly.instantiateStreaming(response, {{}});
  return createPharmsolDslWasmModel(instance);
}}

export function createPharmsolDslWasmModel(instance) {{
  const exports = instance.exports;
  const version = Number(exports[API_VERSION_SYMBOL]());
  if (version !== API_VERSION) {{
    throw new Error(`Expected pharmsol DSL WASM API version ${{API_VERSION}}, got ${{version}}`);
  }}

  const memory = exports.memory;
  const infoPtr = Number(exports[MODEL_INFO_JSON_PTR_SYMBOL]());
  const infoLen = Number(exports[MODEL_INFO_JSON_LEN_SYMBOL]());
  const info = JSON.parse(readUtf8(memory, infoPtr, infoLen));

  const kernels = Object.fromEntries(
    Object.entries(KERNEL_SYMBOLS)
      .filter(([, symbol]) => typeof exports[symbol] === "function")
      .map(([name, symbol]) => [name, exports[symbol].bind(exports)])
  );

  return {{
    info,
    instance,
    memory,
    kernels,
    createF64Buffer(length) {{
      return createBufferHandle(exports, memory, length);
    }},
  }};
}}
"#,
        api_version = WASM_API_VERSION,
        api_version_symbol = API_VERSION_SYMBOL,
        model_info_ptr_symbol = MODEL_INFO_JSON_PTR_SYMBOL,
        model_info_len_symbol = MODEL_INFO_JSON_LEN_SYMBOL,
        alloc_f64_buffer_symbol = ALLOC_F64_BUFFER_SYMBOL,
        free_f64_buffer_symbol = FREE_F64_BUFFER_SYMBOL,
        derive_symbol = DERIVE_SYMBOL,
        dynamics_symbol = DYNAMICS_SYMBOL,
        outputs_symbol = OUTPUTS_SYMBOL,
        init_symbol = INIT_SYMBOL,
        drift_symbol = DRIFT_SYMBOL,
        diffusion_symbol = DIFFUSION_SYMBOL,
        route_lag_symbol = ROUTE_LAG_SYMBOL,
        route_bioavailability_symbol = ROUTE_BIOAVAILABILITY_SYMBOL,
    )
}

fn default_wasm_output_path(template_root: &Path) -> PathBuf {
    let random_suffix: String = rand::rng()
        .sample_iter(&Alphanumeric)
        .take(5)
        .map(char::from)
        .collect();
    template_root.join(format!(
        "model_{}_{}_{}.wasm",
        std::env::consts::OS,
        std::env::consts::ARCH,
        random_suffix
    ))
}

fn wasm_template_manifest() -> String {
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

#[cfg(all(test, feature = "dsl-jit"))]
mod tests {
    use super::*;
    use crate::dsl::{
        compile_execution_artifact, lower_typed_model, parse_module, NativeModelInfo,
    };
    use approx::assert_relative_eq;
    use tempfile::tempdir;
    use wasmtime::{Engine, Instance, Linker, Memory, Module, Store, TypedFunc};

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
    fn wasm_ode_artifact_exports_browser_bundle_and_matches_jit_kernels() {
        let model = load_proposal_model("one_cmt_oral_iv");
        let work_dir = tempdir().expect("tempdir");
        let output_path = work_dir.path().join("one_cmt_oral_iv.wasm");
        let bundle = export_execution_model_to_wasm(
            &model,
            Some(output_path.clone()),
            work_dir.path().join("build"),
            |_, _| {},
        )
        .expect("export wasm model");

        assert_eq!(bundle.wasm_path, output_path);
        assert!(bundle.wasm_path.exists());
        assert!(bundle.browser_loader_path.exists());

        let loader = std::fs::read_to_string(&bundle.browser_loader_path).expect("loader source");
        assert!(loader.contains("loadPharmsolDslWasmModel"));
        assert!(loader.contains(API_VERSION_SYMBOL));
        assert!(loader.contains(ALLOC_F64_BUFFER_SYMBOL));
        assert!(loader.contains(INIT_SYMBOL));
        assert!(loader.contains(DRIFT_SYMBOL));
        assert!(loader.contains(DIFFUSION_SYMBOL));
        assert!(loader.contains(ROUTE_LAG_SYMBOL));
        assert!(loader.contains(ROUTE_BIOAVAILABILITY_SYMBOL));

        let jit = compile_execution_artifact(&model).expect("compile jit kernels");
        let (mut store, instance, memory) = instantiate_module(&bundle.wasm_path);

        let api_version = typed_func::<(), u32>(&instance, &mut store, API_VERSION_SYMBOL);
        assert_eq!(
            api_version.call(&mut store, ()).expect("api version"),
            WASM_API_VERSION
        );

        let info = read_model_info(&instance, &mut store, &memory);
        assert_eq!(info.name, "one_cmt_oral_iv");
        assert_eq!(info.kind, crate::dsl::ModelKind::Ode);

        let alloc = typed_func::<i32, i32>(&instance, &mut store, ALLOC_F64_BUFFER_SYMBOL);
        let free = typed_func::<(i32, i32), ()>(&instance, &mut store, FREE_F64_BUFFER_SYMBOL);
        let derive = typed_func::<(f64, i32, i32, i32, i32, i32, i32), ()>(
            &instance,
            &mut store,
            DERIVE_SYMBOL,
        );
        let outputs = typed_func::<(f64, i32, i32, i32, i32, i32, i32), ()>(
            &instance,
            &mut store,
            OUTPUTS_SYMBOL,
        );
        let dynamics = typed_func::<(f64, i32, i32, i32, i32, i32, i32), ()>(
            &instance,
            &mut store,
            DYNAMICS_SYMBOL,
        );

        let states = vec![100.0, 0.0];
        let params = vec![1.2, 5.0, 40.0, 0.5, 0.8];
        let covariates = vec![70.0];
        let routes = vec![0.0, 0.0];
        let mut jit_derived = vec![0.0; info.derived_len];
        let mut jit_outputs = vec![0.0; info.output_len];
        let mut jit_dynamics = vec![0.0; info.state_len];

        unsafe {
            jit.derive.expect("jit derive")(
                0.0,
                states.as_ptr(),
                params.as_ptr(),
                covariates.as_ptr(),
                routes.as_ptr(),
                jit_derived.as_ptr(),
                jit_derived.as_mut_ptr(),
            );
            (jit.outputs)(
                0.0,
                states.as_ptr(),
                params.as_ptr(),
                covariates.as_ptr(),
                routes.as_ptr(),
                jit_derived.as_ptr(),
                jit_outputs.as_mut_ptr(),
            );
            jit.dynamics.expect("jit dynamics")(
                0.0,
                states.as_ptr(),
                params.as_ptr(),
                covariates.as_ptr(),
                routes.as_ptr(),
                jit_derived.as_ptr(),
                jit_dynamics.as_mut_ptr(),
            );
        }

        let states_ptr = alloc
            .call(&mut store, states.len() as i32)
            .expect("alloc states");
        let params_ptr = alloc
            .call(&mut store, params.len() as i32)
            .expect("alloc params");
        let covariates_ptr = alloc
            .call(&mut store, covariates.len() as i32)
            .expect("alloc covariates");
        let routes_ptr = alloc
            .call(&mut store, routes.len() as i32)
            .expect("alloc routes");
        let derived_ptr = alloc
            .call(&mut store, info.derived_len as i32)
            .expect("alloc derived");
        let outputs_ptr = alloc
            .call(&mut store, info.output_len as i32)
            .expect("alloc outputs");
        let dynamics_ptr = alloc
            .call(&mut store, info.state_len as i32)
            .expect("alloc dynamics");

        write_f64s(&memory, &mut store, states_ptr, &states);
        write_f64s(&memory, &mut store, params_ptr, &params);
        write_f64s(&memory, &mut store, covariates_ptr, &covariates);
        write_f64s(&memory, &mut store, routes_ptr, &routes);

        derive
            .call(
                &mut store,
                (
                    0.0,
                    states_ptr,
                    params_ptr,
                    covariates_ptr,
                    routes_ptr,
                    derived_ptr,
                    derived_ptr,
                ),
            )
            .expect("wasm derive");
        outputs
            .call(
                &mut store,
                (
                    0.0,
                    states_ptr,
                    params_ptr,
                    covariates_ptr,
                    routes_ptr,
                    derived_ptr,
                    outputs_ptr,
                ),
            )
            .expect("wasm outputs");
        dynamics
            .call(
                &mut store,
                (
                    0.0,
                    states_ptr,
                    params_ptr,
                    covariates_ptr,
                    routes_ptr,
                    derived_ptr,
                    dynamics_ptr,
                ),
            )
            .expect("wasm dynamics");

        let wasm_derived = read_f64s(&memory, &mut store, derived_ptr, info.derived_len);
        let wasm_outputs = read_f64s(&memory, &mut store, outputs_ptr, info.output_len);
        let wasm_dynamics = read_f64s(&memory, &mut store, dynamics_ptr, info.state_len);

        for (wasm_value, jit_value) in wasm_derived.iter().zip(jit_derived.iter()) {
            assert_relative_eq!(wasm_value, jit_value, max_relative = 1e-10);
        }
        for (wasm_value, jit_value) in wasm_outputs.iter().zip(jit_outputs.iter()) {
            assert_relative_eq!(wasm_value, jit_value, max_relative = 1e-10);
        }
        for (wasm_value, jit_value) in wasm_dynamics.iter().zip(jit_dynamics.iter()) {
            assert_relative_eq!(wasm_value, jit_value, max_relative = 1e-10);
        }

        for (ptr, len) in [
            (states_ptr, states.len()),
            (params_ptr, params.len()),
            (covariates_ptr, covariates.len()),
            (routes_ptr, routes.len()),
            (derived_ptr, info.derived_len),
            (outputs_ptr, info.output_len),
            (dynamics_ptr, info.state_len),
        ] {
            free.call(&mut store, (ptr, len as i32))
                .expect("free buffer");
        }
    }

    #[test]
    fn reuses_wasm_kernel_sessions_across_start_session_calls() {
        let model = load_proposal_model("one_cmt_oral_iv");
        let work_dir = tempdir().expect("tempdir");
        let output_path = work_dir.path().join("one_cmt_oral_iv_reuse.wasm");
        let bundle = export_execution_model_to_wasm(
            &model,
            Some(output_path),
            work_dir.path().join("build-reuse"),
            |_, _| {},
        )
        .expect("export wasm model");

        let (_, artifact) = load_wasm_artifact(&bundle.wasm_path).expect("load wasm artifact");
        assert_eq!(artifact.session_pool.lock().expect("session pool").len(), 0);

        {
            let _session = artifact.start_session().expect("first pooled session");
        }
        assert_eq!(artifact.session_pool.lock().expect("session pool").len(), 1);

        {
            let _session = artifact.start_session().expect("second pooled session");
        }
        assert_eq!(artifact.session_pool.lock().expect("session pool").len(), 1);
    }

    #[test]
    fn rejects_out_of_bounds_wasm_memory_ranges() {
        let error = byte_range(8, 16, 16, "test range").expect_err("range should fail");
        assert!(matches!(
            error,
            WasmError::MemoryOutOfBounds {
                region: "test range",
                ptr: 8,
                len: 16,
                memory_len: 16,
            }
        ));
    }

    fn instantiate_module(path: &Path) -> (Store<()>, Instance, Memory) {
        let engine = Engine::default();
        let module = Module::from_file(&engine, path).expect("compile wasm module");
        let mut store = Store::new(&engine, ());
        let linker = Linker::new(&engine);
        let instance = linker
            .instantiate(&mut store, &module)
            .expect("instantiate wasm module");
        let memory = instance
            .get_memory(&mut store, "memory")
            .expect("wasm memory export");
        (store, instance, memory)
    }

    fn typed_func<Params, Results>(
        instance: &Instance,
        store: &mut Store<()>,
        name: &str,
    ) -> TypedFunc<Params, Results>
    where
        Params: wasmtime::WasmParams,
        Results: wasmtime::WasmResults,
    {
        instance
            .get_typed_func(store, name)
            .unwrap_or_else(|_| panic!("missing wasm export `{name}`"))
    }

    fn read_model_info(
        instance: &Instance,
        store: &mut Store<()>,
        memory: &Memory,
    ) -> NativeModelInfo {
        let ptr = typed_func::<(), i32>(instance, store, MODEL_INFO_JSON_PTR_SYMBOL)
            .call(&mut *store, ())
            .expect("model info ptr");
        let len = typed_func::<(), i32>(instance, store, MODEL_INFO_JSON_LEN_SYMBOL)
            .call(&mut *store, ())
            .expect("model info len");
        let data = memory.data(&mut *store);
        let start = ptr as usize;
        let end = start + len as usize;
        serde_json::from_slice(&data[start..end]).expect("parse model info")
    }

    fn write_f64s(memory: &Memory, store: &mut Store<()>, ptr: i32, values: &[f64]) {
        let data = memory.data_mut(store);
        let start = ptr as usize;
        for (index, value) in values.iter().enumerate() {
            let offset = start + index * std::mem::size_of::<f64>();
            data[offset..offset + std::mem::size_of::<f64>()].copy_from_slice(&value.to_le_bytes());
        }
    }

    fn read_f64s(memory: &Memory, store: &mut Store<()>, ptr: i32, len: usize) -> Vec<f64> {
        let data = memory.data(store);
        let start = ptr as usize;
        (0..len)
            .map(|index| {
                let offset = start + index * std::mem::size_of::<f64>();
                let mut bytes = [0u8; std::mem::size_of::<f64>()];
                bytes.copy_from_slice(&data[offset..offset + std::mem::size_of::<f64>()]);
                f64::from_le_bytes(bytes)
            })
            .collect()
    }
}
