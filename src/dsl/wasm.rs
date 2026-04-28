use std::ops::Range;
use std::path::Path;
use std::sync::Mutex;

use serde_json;
use wasmtime::{Engine, Instance, Linker, Memory, Module, Store, TypedFunc};

use super::compiled_backend_abi::{
    decode_compiled_model_info, CompiledKernelAvailability, ALLOC_F64_BUFFER_SYMBOL,
    API_VERSION_SYMBOL, DERIVE_SYMBOL, DIFFUSION_SYMBOL, DRIFT_SYMBOL, DYNAMICS_SYMBOL,
    FREE_F64_BUFFER_SYMBOL, INIT_SYMBOL, MODEL_INFO_JSON_LEN_SYMBOL, MODEL_INFO_JSON_PTR_SYMBOL,
    OUTPUTS_SYMBOL, ROUTE_BIOAVAILABILITY_SYMBOL, ROUTE_LAG_SYMBOL,
};
use super::native::{KernelSession, NativeModelInfo, RuntimeArtifact, RuntimeBackend};
use super::wasm_compile::{WasmError, WASM_API_VERSION};
use super::wasm_direct_emitter::{
    DIRECT_WASM_BINARY_MATH_IMPORTS, DIRECT_WASM_IMPORT_MODULE, DIRECT_WASM_UNARY_MATH_IMPORTS,
};
use crate::PharmsolError;
use pharmsol_dsl::execution::KernelRole;

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

    fn compiled(self) -> CompiledKernelAvailability {
        CompiledKernelAvailability {
            derive: self.derive,
            dynamics: self.dynamics,
            outputs: self.outputs,
            init: self.init,
            drift: self.drift,
            diffusion: self.diffusion,
            route_lag: self.route_lag,
            route_bioavailability: self.route_bioavailability,
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

type WasmKernelParams = (f64, i32, i32, i32, i32, i32, i32);
type WasmKernelFunc = TypedFunc<WasmKernelParams, ()>;
type WasmSessionPool = Mutex<Vec<WasmKernelSession>>;

struct WasmKernelSession {
    info: NativeModelInfo,
    store: Store<()>,
    memory: Memory,
    free: TypedFunc<(i32, i32), ()>,
    derive: Option<WasmKernelFunc>,
    dynamics: Option<WasmKernelFunc>,
    outputs: WasmKernelFunc,
    init: Option<WasmKernelFunc>,
    drift: Option<WasmKernelFunc>,
    diffusion: Option<WasmKernelFunc>,
    route_lag: Option<WasmKernelFunc>,
    route_bioavailability: Option<WasmKernelFunc>,
    states: WasmBuffer,
    params: WasmBuffer,
    covariates: WasmBuffer,
    routes: WasmBuffer,
    derived: WasmBuffer,
    out: WasmBuffer,
}

struct PooledWasmKernelSession<'a> {
    session: Option<WasmKernelSession>,
    pool: &'a WasmSessionPool,
}

impl WasmKernelSession {
    fn new(
        info: &NativeModelInfo,
        engine: &Engine,
        module: &Module,
        kernels: WasmKernelAvailability,
    ) -> Result<Self, WasmError> {
        let mut store = Store::new(engine, ());
        let linker = configured_wasm_linker(engine)?;
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

    fn kernel(&self, role: KernelRole) -> Result<WasmKernelFunc, PharmsolError> {
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

pub fn read_wasm_model_info_bytes(bytes: &[u8]) -> Result<NativeModelInfo, WasmError> {
    let (info, _) = load_wasm_artifact_bytes(bytes)?;
    Ok(info)
}

pub(crate) fn load_wasm_artifact(
    path: impl AsRef<Path>,
) -> Result<(NativeModelInfo, WasmExecutionArtifact), WasmError> {
    let engine = Engine::default();
    let module =
        Module::from_file(&engine, path).map_err(|error| WasmError::Load(error.to_string()))?;
    load_wasm_artifact_from_module(engine, module)
}

pub(crate) fn load_wasm_artifact_bytes(
    bytes: &[u8],
) -> Result<(NativeModelInfo, WasmExecutionArtifact), WasmError> {
    let engine = Engine::default();
    let module = Module::new(&engine, bytes).map_err(|error| WasmError::Load(error.to_string()))?;
    load_wasm_artifact_from_module(engine, module)
}

fn load_wasm_artifact_from_module(
    engine: Engine,
    module: Module,
) -> Result<(NativeModelInfo, WasmExecutionArtifact), WasmError> {
    let mut store = Store::new(&engine, ());
    let linker = configured_wasm_linker(&engine)?;
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
    let (info, expected_kernels) = read_model_info_envelope(&instance, &mut store, &memory)?;
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
    if let Some(expected_kernels) = expected_kernels {
        let found_kernels = kernels.compiled();
        if found_kernels != expected_kernels {
            return Err(WasmError::KernelMetadataMismatch {
                model: info.name.clone(),
                expected: expected_kernels,
                found: found_kernels,
            });
        }
    }

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

pub(crate) fn configured_wasm_linker(engine: &Engine) -> Result<Linker<()>, WasmError> {
    let mut linker = Linker::new(engine);
    for import in DIRECT_WASM_UNARY_MATH_IMPORTS {
        let name = import.name;
        linker
            .func_wrap(
                DIRECT_WASM_IMPORT_MODULE,
                name,
                move |value: f64| match name {
                    "exp" => value.exp(),
                    "ln" => value.ln(),
                    "log10" => value.log10(),
                    "log2" => value.log2(),
                    "round" => value.round(),
                    "sin" => value.sin(),
                    "cos" => value.cos(),
                    "tan" => value.tan(),
                    _ => unreachable!("unsupported direct unary math import {name}"),
                },
            )
            .map_err(|error| WasmError::Load(error.to_string()))?;
    }
    for import in DIRECT_WASM_BINARY_MATH_IMPORTS {
        let name = import.name;
        linker
            .func_wrap(
                DIRECT_WASM_IMPORT_MODULE,
                name,
                move |lhs: f64, rhs: f64| match name {
                    "pow" => lhs.powf(rhs),
                    _ => unreachable!("unsupported direct binary math import {name}"),
                },
            )
            .map_err(|error| WasmError::Load(error.to_string()))?;
    }
    Ok(linker)
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

fn read_model_info_envelope(
    instance: &Instance,
    store: &mut Store<()>,
    memory: &Memory,
) -> Result<(NativeModelInfo, Option<CompiledKernelAvailability>), WasmError> {
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
    let bytes = &data[range];
    if let Ok(envelope) = decode_compiled_model_info(bytes) {
        if envelope.abi_version != WASM_API_VERSION {
            return Err(WasmError::ApiVersionMismatch {
                expected: WASM_API_VERSION,
                found: envelope.abi_version,
            });
        }
        return Ok((envelope.model, Some(envelope.kernels)));
    }
    Ok((serde_json::from_slice(bytes)?, None))
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

#[cfg(all(test, feature = "dsl-jit"))]
mod tests {
    use super::*;
    use crate::dsl::{
        compile_execution_artifact, CompiledKernelAvailability, CompiledModelInfoEnvelope,
        NativeModelInfo, NativeOutputInfo, NativeRouteInfo,
    };
    use crate::test_fixtures::STRUCTURED_BLOCK_CORPUS;
    use approx::assert_relative_eq;
    use pharmsol_dsl::{
        analyze_module, lower_typed_model, parse_module, ExecutionModel, ModelKind,
    };
    use std::path::{Path, PathBuf};
    use tempfile::tempdir;
    use wasm_encoder::{
        CodeSection, ConstExpr, ExportKind, ExportSection, Function, FunctionSection,
        MemorySection, MemoryType, Module as EncoderModule, TypeSection, ValType,
    };
    use wasmtime::{Engine, Instance, Memory, Module, Store, TypedFunc};

    fn load_corpus_model(name: &str) -> ExecutionModel {
        let source = STRUCTURED_BLOCK_CORPUS;
        let parsed = parse_module(source).expect("parse corpus module");
        let typed = analyze_module(&parsed).expect("analyze corpus module");
        let model = typed
            .models
            .iter()
            .find(|model| model.name == name)
            .expect("model in corpus module");
        lower_typed_model(model).expect("lower corpus model")
    }

    fn loader_test_model_info(name: &str) -> NativeModelInfo {
        NativeModelInfo {
            name: name.to_string(),
            kind: ModelKind::Ode,
            parameters: vec!["ka".to_string()],
            covariates: Vec::new(),
            routes: vec![NativeRouteInfo {
                name: "oral".to_string(),
                index: 0,
                destination_offset: 0,
                inject_input_to_destination: true,
            }],
            outputs: vec![NativeOutputInfo {
                name: "cp".to_string(),
                index: 0,
            }],
            state_len: 1,
            derived_len: 0,
            output_len: 1,
            route_len: 1,
            analytical: None,
            particles: None,
        }
    }

    fn loader_test_module_bytes(
        api_version_export: u32,
        model_info_bytes: &[u8],
        export_outputs: bool,
    ) -> Vec<u8> {
        let mut module = EncoderModule::new();

        let mut types = TypeSection::new();
        types.ty().function([], [ValType::I32]);
        types.ty().function([], []);
        module.section(&types);

        let mut functions = FunctionSection::new();
        functions.function(0);
        functions.function(0);
        functions.function(0);
        if export_outputs {
            functions.function(1);
        }
        module.section(&functions);

        let mut memories = MemorySection::new();
        memories.memory(MemoryType {
            minimum: 1,
            maximum: None,
            memory64: false,
            shared: false,
            page_size_log2: None,
        });
        module.section(&memories);

        let mut exports = ExportSection::new();
        exports.export("memory", ExportKind::Memory, 0);
        exports.export(API_VERSION_SYMBOL, ExportKind::Func, 0);
        exports.export(MODEL_INFO_JSON_PTR_SYMBOL, ExportKind::Func, 1);
        exports.export(MODEL_INFO_JSON_LEN_SYMBOL, ExportKind::Func, 2);
        if export_outputs {
            exports.export(OUTPUTS_SYMBOL, ExportKind::Func, 3);
        }
        module.section(&exports);

        let mut codes = CodeSection::new();

        let mut api_version = Function::new([]);
        api_version.instruction(&wasm_encoder::Instruction::I32Const(
            api_version_export as i32,
        ));
        api_version.instruction(&wasm_encoder::Instruction::End);
        codes.function(&api_version);

        let mut model_info_ptr = Function::new([]);
        model_info_ptr.instruction(&wasm_encoder::Instruction::I32Const(0));
        model_info_ptr.instruction(&wasm_encoder::Instruction::End);
        codes.function(&model_info_ptr);

        let mut model_info_len = Function::new([]);
        model_info_len.instruction(&wasm_encoder::Instruction::I32Const(
            model_info_bytes.len() as i32
        ));
        model_info_len.instruction(&wasm_encoder::Instruction::End);
        codes.function(&model_info_len);

        if export_outputs {
            let mut outputs = Function::new([]);
            outputs.instruction(&wasm_encoder::Instruction::End);
            codes.function(&outputs);
        }
        module.section(&codes);

        let mut data = wasm_encoder::DataSection::new();
        data.active(
            0,
            &ConstExpr::i32_const(0),
            model_info_bytes.iter().copied(),
        );
        module.section(&data);

        module.finish()
    }

    fn write_wasm_bundle_files(model: &ExecutionModel, output_path: &Path) -> PathBuf {
        let bytes = super::super::wasm_compile::compile_execution_model_to_wasm_bytes(model)
            .expect("emit direct wasm bytes");
        let loader_path = output_path.with_extension("mjs");
        std::fs::write(output_path, &bytes).expect("write direct wasm artifact");
        std::fs::write(
            &loader_path,
            super::super::wasm_compile::browser_loader_source(),
        )
        .expect("write browser loader");
        loader_path
    }

    #[test]
    fn rejects_wasm_export_api_version_mismatch() {
        let model_info = loader_test_model_info("api_version_export_mismatch");
        let metadata = serde_json::to_vec(&CompiledModelInfoEnvelope {
            abi_version: WASM_API_VERSION,
            model: model_info,
            kernels: CompiledKernelAvailability {
                outputs: true,
                ..CompiledKernelAvailability::default()
            },
        })
        .expect("metadata json");

        let error = load_wasm_artifact_bytes(&loader_test_module_bytes(
            WASM_API_VERSION + 1,
            &metadata,
            true,
        ))
        .expect_err("mismatched export api version should fail");

        assert!(matches!(
            error,
            WasmError::ApiVersionMismatch {
                expected,
                found,
            } if expected == WASM_API_VERSION && found == WASM_API_VERSION + 1
        ));
    }

    #[test]
    fn rejects_compiled_metadata_abi_version_mismatch() {
        let model_info = loader_test_model_info("metadata_api_version_mismatch");
        let metadata = serde_json::to_vec(&CompiledModelInfoEnvelope {
            abi_version: WASM_API_VERSION + 1,
            model: model_info,
            kernels: CompiledKernelAvailability {
                outputs: true,
                ..CompiledKernelAvailability::default()
            },
        })
        .expect("metadata json");

        let error =
            load_wasm_artifact_bytes(&loader_test_module_bytes(WASM_API_VERSION, &metadata, true))
                .expect_err("mismatched compiled metadata abi version should fail");

        assert!(matches!(
            error,
            WasmError::ApiVersionMismatch {
                expected,
                found,
            } if expected == WASM_API_VERSION && found == WASM_API_VERSION + 1
        ));
    }

    #[test]
    fn rejects_kernel_metadata_mismatch_from_compiled_envelope() {
        let model_info = loader_test_model_info("kernel_metadata_mismatch");
        let metadata = serde_json::to_vec(&CompiledModelInfoEnvelope {
            abi_version: WASM_API_VERSION,
            model: model_info,
            kernels: CompiledKernelAvailability {
                outputs: true,
                ..CompiledKernelAvailability::default()
            },
        })
        .expect("metadata json");

        let error = load_wasm_artifact_bytes(&loader_test_module_bytes(
            WASM_API_VERSION,
            &metadata,
            false,
        ))
        .expect_err("missing outputs export should fail against compiled metadata");

        assert!(matches!(
            error,
            WasmError::KernelMetadataMismatch {
                ref model,
                expected,
                found,
            } if model == "kernel_metadata_mismatch"
                && expected.outputs
                && !found.outputs
        ));
    }

    #[test]
    fn accepts_legacy_plain_model_info_metadata() {
        let model_info = loader_test_model_info("legacy_plain_metadata");
        let metadata = serde_json::to_vec(&model_info).expect("legacy metadata json");

        let (loaded, artifact) =
            load_wasm_artifact_bytes(&loader_test_module_bytes(WASM_API_VERSION, &metadata, true))
                .expect("legacy metadata should still load");

        assert_eq!(loaded, model_info);
        assert!(artifact.has_kernel(KernelRole::Outputs));
    }

    #[test]
    fn direct_browser_smoke_bundle_is_emitted_when_requested() {
        let output_dir = std::env::var_os("PHARMSOL_DSL_BROWSER_SMOKE_DIR")
            .or_else(|| std::env::var_os("PHARMSOL_DSL_W03_BROWSER_SMOKE_DIR"));
        let Some(output_dir) = output_dir else {
            return;
        };

        let output_dir = PathBuf::from(output_dir);
        std::fs::create_dir_all(&output_dir).expect("create browser smoke directory");

        let model = super::super::wasm_direct_emitter::w03_minimal_outputs_execution_model();
        let bytes = super::super::wasm_compile::compile_execution_model_to_wasm_bytes(&model)
            .expect("emit direct browser smoke wasm bytes");
        let info = read_wasm_model_info_bytes(&bytes).expect("read direct wasm model info");
        assert_eq!(info.name, "direct_w03_minimal");

        let loader = super::super::wasm_compile::browser_loader_source();
        assert!(loader.contains("createPharmsolDslWasmSession"));
        assert!(loader.contains("evaluateOutput(name, inputs = {}, options = {})"));

        std::fs::write(output_dir.join("direct.wasm"), &bytes).expect("write direct wasm");
        std::fs::write(output_dir.join("direct.mjs"), loader).expect("write direct loader");
    }

    #[test]
    fn wasm_ode_artifact_exports_browser_bundle_and_matches_jit_kernels() {
        let model = load_corpus_model("one_cmt_oral_iv");
        let work_dir = tempdir().expect("tempdir");
        let output_path = work_dir.path().join("one_cmt_oral_iv.wasm");
        let loader_path = write_wasm_bundle_files(&model, &output_path);

        assert!(output_path.exists());
        assert!(loader_path.exists());

        let loader = std::fs::read_to_string(&loader_path).expect("loader source");
        assert!(loader.contains("loadPharmsolDslWasmModel"));
        assert!(loader.contains(API_VERSION_SYMBOL));
        assert!(loader.contains(ALLOC_F64_BUFFER_SYMBOL));
        assert!(loader.contains(INIT_SYMBOL));
        assert!(loader.contains(DRIFT_SYMBOL));
        assert!(loader.contains(DIFFUSION_SYMBOL));
        assert!(loader.contains(ROUTE_LAG_SYMBOL));
        assert!(loader.contains(ROUTE_BIOAVAILABILITY_SYMBOL));

        let jit = compile_execution_artifact(&model).expect("compile jit kernels");
        let (mut store, instance, memory) = instantiate_module(&output_path);

        let api_version = typed_func::<(), u32>(&instance, &mut store, API_VERSION_SYMBOL);
        assert_eq!(
            api_version.call(&mut store, ()).expect("api version"),
            WASM_API_VERSION
        );

        let info = read_model_info(&instance, &mut store, &memory);
        assert_eq!(info.name, "one_cmt_oral_iv");
        assert_eq!(info.kind, ModelKind::Ode);

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

        let states = [100.0, 0.0];
        let params = [1.2, 5.0, 40.0, 0.5, 0.8];
        let covariates = [70.0];
        let routes = [0.0, 0.0];
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
        let model = load_corpus_model("one_cmt_oral_iv");
        let work_dir = tempdir().expect("tempdir");
        let output_path = work_dir.path().join("one_cmt_oral_iv_reuse.wasm");
        write_wasm_bundle_files(&model, &output_path);

        let (_, artifact) = load_wasm_artifact(&output_path).expect("load wasm artifact");
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
    fn wasm_runtime_preserves_state_aliasing_for_dynamics_kernel() {
        let model = load_corpus_model("one_cmt_oral_iv");
        let jit = compile_execution_artifact(&model).expect("compile jit kernels");
        let bytes = super::super::wasm_compile::compile_execution_model_to_wasm_bytes(&model)
            .expect("emit direct wasm bytes");
        let (info, artifact) = load_wasm_artifact_bytes(&bytes).expect("load direct wasm");
        let mut session = artifact.start_session().expect("start wasm session");

        let mut actual = vec![100.0, 0.0];
        let mut expected = actual.clone();
        let params = [1.2, 5.0, 40.0, 0.5, 0.8];
        let covariates = [70.0];
        let routes = [0.0, 0.0];
        let mut derived = vec![0.0; info.derived_len];

        unsafe {
            jit.derive.expect("jit derive")(
                0.0,
                expected.as_ptr(),
                params.as_ptr(),
                covariates.as_ptr(),
                routes.as_ptr(),
                derived.as_ptr(),
                derived.as_mut_ptr(),
            );
            jit.dynamics.expect("jit dynamics")(
                0.0,
                expected.as_ptr(),
                params.as_ptr(),
                covariates.as_ptr(),
                routes.as_ptr(),
                derived.as_ptr(),
                expected.as_mut_ptr(),
            );
            session
                .invoke_raw(
                    KernelRole::Dynamics,
                    0.0,
                    actual.as_ptr(),
                    params.as_ptr(),
                    covariates.as_ptr(),
                    routes.as_ptr(),
                    derived.as_ptr(),
                    actual.as_mut_ptr(),
                )
                .expect("invoke aliased dynamics kernel");
        }

        for (actual, expected) in actual.iter().zip(expected.iter()) {
            assert_relative_eq!(actual, expected, max_relative = 1e-10);
        }
    }

    #[test]
    fn wasm_runtime_zeroes_non_aliased_diffusion_outputs() {
        let model = load_corpus_model("vanco_sde");
        let jit = compile_execution_artifact(&model).expect("compile jit kernels");
        let bytes = super::super::wasm_compile::compile_execution_model_to_wasm_bytes(&model)
            .expect("emit direct wasm bytes");
        let (info, artifact) = load_wasm_artifact_bytes(&bytes).expect("load direct wasm");
        let mut session = artifact.start_session().expect("start wasm session");

        let states = [0.0, 0.0, 0.0, 0.2];
        let params = [1.1, 0.2, 0.12, 0.08, 15.0, 0.7];
        let covariates = [70.0];
        let routes = [0.0];
        let derived = vec![0.0; info.derived_len];
        let mut expected = vec![0.0; info.state_len];
        let mut actual = vec![42.0; info.state_len];

        unsafe {
            jit.diffusion.expect("jit diffusion")(
                0.0,
                states.as_ptr(),
                params.as_ptr(),
                covariates.as_ptr(),
                routes.as_ptr(),
                derived.as_ptr(),
                expected.as_mut_ptr(),
            );
            session
                .invoke_raw(
                    KernelRole::Diffusion,
                    0.0,
                    states.as_ptr(),
                    params.as_ptr(),
                    covariates.as_ptr(),
                    routes.as_ptr(),
                    derived.as_ptr(),
                    actual.as_mut_ptr(),
                )
                .expect("invoke diffusion kernel");
        }

        for (actual, expected) in actual.iter().zip(expected.iter()) {
            assert_relative_eq!(actual, expected, max_relative = 1e-10);
        }
        assert!(actual.iter().take(3).all(|value| value.abs() <= 1e-12));
    }

    #[test]
    fn wasm_runtime_matches_jit_route_property_kernels() {
        let model = load_corpus_model("one_cmt_oral_iv");
        let jit = compile_execution_artifact(&model).expect("compile jit kernels");
        let bytes = super::super::wasm_compile::compile_execution_model_to_wasm_bytes(&model)
            .expect("emit direct wasm bytes");
        let (info, artifact) = load_wasm_artifact_bytes(&bytes).expect("load direct wasm");
        let mut session = artifact.start_session().expect("start wasm session");

        let states = [100.0, 0.0];
        let params = [1.2, 5.0, 40.0, 0.5, 0.8];
        let covariates = [70.0];
        let routes = [0.0, 0.0];
        let derived = vec![0.0; info.derived_len];
        let mut expected_lag = vec![0.0; info.route_len];
        let mut expected_bioavailability = vec![0.0; info.route_len];
        let mut actual_lag = vec![f64::NAN; info.route_len];
        let mut actual_bioavailability = vec![f64::NAN; info.route_len];

        unsafe {
            jit.route_lag.expect("jit route lag")(
                0.0,
                states.as_ptr(),
                params.as_ptr(),
                covariates.as_ptr(),
                routes.as_ptr(),
                derived.as_ptr(),
                expected_lag.as_mut_ptr(),
            );
            jit.route_bioavailability
                .expect("jit route bioavailability")(
                0.0,
                states.as_ptr(),
                params.as_ptr(),
                covariates.as_ptr(),
                routes.as_ptr(),
                derived.as_ptr(),
                expected_bioavailability.as_mut_ptr(),
            );
            session
                .invoke_raw(
                    KernelRole::RouteLag,
                    0.0,
                    states.as_ptr(),
                    params.as_ptr(),
                    covariates.as_ptr(),
                    routes.as_ptr(),
                    derived.as_ptr(),
                    actual_lag.as_mut_ptr(),
                )
                .expect("invoke route lag kernel");
            session
                .invoke_raw(
                    KernelRole::RouteBioavailability,
                    0.0,
                    states.as_ptr(),
                    params.as_ptr(),
                    covariates.as_ptr(),
                    routes.as_ptr(),
                    derived.as_ptr(),
                    actual_bioavailability.as_mut_ptr(),
                )
                .expect("invoke route bioavailability kernel");
        }

        for (actual, expected) in actual_lag.iter().zip(expected_lag.iter()) {
            assert_relative_eq!(actual, expected, max_relative = 1e-10);
        }
        for (actual, expected) in actual_bioavailability
            .iter()
            .zip(expected_bioavailability.iter())
        {
            assert_relative_eq!(actual, expected, max_relative = 1e-10);
        }
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
        let linker = configured_wasm_linker(&engine).expect("configured wasm linker");
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
        read_model_info_envelope(instance, store, memory)
            .expect("read model info envelope")
            .0
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
