use std::collections::{HashMap, VecDeque};
use std::fmt;
use std::sync::{Mutex, OnceLock};

use serde::{Deserialize, Serialize};
use thiserror::Error;

use super::compiled_backend_abi::{
    compiled_model_info_envelope, CompiledKernelAvailability, CompiledModelInfoEnvelope,
    ALLOC_F64_BUFFER_SYMBOL, API_VERSION_SYMBOL, FREE_F64_BUFFER_SYMBOL, JS_KERNEL_EXPORTS,
    MODEL_INFO_JSON_LEN_SYMBOL, MODEL_INFO_JSON_PTR_SYMBOL,
};
use super::execution::ExecutionModel;
use super::wasm_direct_emitter::{
    compile_execution_model_to_wasm_bytes as emit_execution_model_to_wasm_bytes,
    DIRECT_WASM_BINARY_MATH_IMPORTS, DIRECT_WASM_IMPORT_MODULE, DIRECT_WASM_UNARY_MATH_IMPORTS,
};
use super::{
    analyze_module, lower_typed_model, parse_module, Diagnostic, DiagnosticReport, LoweringError,
    ParseError, SemanticError,
};

pub const WASM_API_VERSION: u32 = 1;
pub const DEFAULT_WASM_COMPILE_CACHE_CAPACITY: usize = 32;

static BROWSER_LOADER_SOURCE: OnceLock<String> = OnceLock::new();

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompiledWasmModule {
    pub wasm_bytes: Vec<u8>,
    pub metadata: CompiledModelInfoEnvelope,
    pub browser_loader_source: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct WasmCompileCacheKey {
    source: String,
    model_name: Option<String>,
}

impl WasmCompileCacheKey {
    fn new(source: &str, model_name: Option<&str>) -> Self {
        Self {
            source: source.to_string(),
            model_name: model_name.map(str::to_string),
        }
    }
}

#[derive(Debug, Default)]
struct WasmCompileCacheState {
    entries: HashMap<WasmCompileCacheKey, CompiledWasmModule>,
    lru: VecDeque<WasmCompileCacheKey>,
}

#[derive(Debug)]
pub struct WasmCompileCache {
    capacity: usize,
    state: Mutex<WasmCompileCacheState>,
}

impl Default for WasmCompileCache {
    fn default() -> Self {
        Self::new(DEFAULT_WASM_COMPILE_CACHE_CAPACITY)
    }
}

impl WasmCompileCache {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity: capacity.max(1),
            state: Mutex::new(WasmCompileCacheState::default()),
        }
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn entry_count(&self) -> usize {
        self.state
            .lock()
            .expect("wasm compile cache mutex poisoned")
            .entries
            .len()
    }

    pub fn clear(&self) {
        let mut state = self
            .state
            .lock()
            .expect("wasm compile cache mutex poisoned");
        state.entries.clear();
        state.lru.clear();
    }

    pub fn compile_module_source_to_wasm_module(
        &self,
        source: &str,
        model_name: Option<&str>,
    ) -> Result<CompiledWasmModule, WasmError> {
        let key = WasmCompileCacheKey::new(source, model_name);
        if let Some(compiled) = self.get(&key) {
            return Ok(compiled);
        }

        let compiled = compile_module_source_to_wasm_module_uncached(source, model_name)?;
        self.insert(key, compiled.clone());
        Ok(compiled)
    }

    pub fn compile_module_source_to_wasm_bytes(
        &self,
        source: &str,
        model_name: Option<&str>,
    ) -> Result<Vec<u8>, WasmError> {
        Ok(self
            .compile_module_source_to_wasm_module(source, model_name)?
            .wasm_bytes)
    }

    fn get(&self, key: &WasmCompileCacheKey) -> Option<CompiledWasmModule> {
        let mut state = self
            .state
            .lock()
            .expect("wasm compile cache mutex poisoned");
        let compiled = state.entries.get(key)?.clone();
        touch_cache_key(&mut state.lru, key);
        Some(compiled)
    }

    fn insert(&self, key: WasmCompileCacheKey, compiled: CompiledWasmModule) {
        let mut state = self
            .state
            .lock()
            .expect("wasm compile cache mutex poisoned");
        touch_cache_key(&mut state.lru, &key);
        state.entries.insert(key, compiled);

        while state.entries.len() > self.capacity {
            let Some(evicted) = state.lru.pop_front() else {
                break;
            };
            state.entries.remove(&evicted);
        }
    }
}

#[derive(Error)]
pub enum WasmError {
    #[error(transparent)]
    Io(#[from] std::io::Error),
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
    #[error("failed to emit WASM module source: {0}")]
    Emit(String),
    #[error("direct WASM backend does not support model `{model}` yet: {reason}")]
    DirectBackendUnsupported { model: String, reason: String },
    #[error("WASM artifact API version mismatch: expected {expected}, found {found}")]
    ApiVersionMismatch { expected: u32, found: u32 },
    #[error(
        "WASM kernel metadata mismatch for model `{model}`: expected {expected:?}, found {found:?}"
    )]
    KernelMetadataMismatch {
        model: String,
        expected: CompiledKernelAvailability,
        found: CompiledKernelAvailability,
    },
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

impl WasmError {
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

impl fmt::Debug for WasmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Parse(error) => fmt::Display::fmt(error, f),
            Self::Semantic(error) => fmt::Display::fmt(error, f),
            Self::Lowering(error) => fmt::Display::fmt(error, f),
            _ => fmt::Display::fmt(self, f),
        }
    }
}

pub fn compile_execution_model_to_wasm_bytes(model: &ExecutionModel) -> Result<Vec<u8>, WasmError> {
    emit_execution_model_to_wasm_bytes(model, WASM_API_VERSION)
}

pub fn compile_execution_model_to_wasm_module(
    model: &ExecutionModel,
) -> Result<CompiledWasmModule, WasmError> {
    Ok(CompiledWasmModule {
        wasm_bytes: compile_execution_model_to_wasm_bytes(model)?,
        metadata: compiled_model_info_envelope(model, WASM_API_VERSION),
        browser_loader_source: browser_loader_source(),
    })
}

pub fn compile_module_source_to_wasm_bytes(
    source: &str,
    model_name: Option<&str>,
) -> Result<Vec<u8>, WasmError> {
    Ok(compile_module_source_to_wasm_module(source, model_name)?.wasm_bytes)
}

pub fn compile_module_source_to_wasm_module(
    source: &str,
    model_name: Option<&str>,
) -> Result<CompiledWasmModule, WasmError> {
    compile_module_source_to_wasm_module_uncached(source, model_name)
}

fn compile_module_source_to_wasm_module_uncached(
    source: &str,
    model_name: Option<&str>,
) -> Result<CompiledWasmModule, WasmError> {
    let parsed =
        parse_module(source).map_err(|error| WasmError::Parse(error.with_source(source)))?;
    let typed =
        analyze_module(&parsed).map_err(|error| WasmError::Semantic(error.with_source(source)))?;

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
        lower_typed_model(model).map_err(|error| WasmError::Lowering(error.with_source(source)))?;
    compile_execution_model_to_wasm_module(&execution)
}

pub fn browser_loader_source() -> String {
    BROWSER_LOADER_SOURCE
        .get_or_init(build_browser_loader_source)
        .clone()
}

fn build_browser_loader_source() -> String {
    let kernel_symbol_entries = JS_KERNEL_EXPORTS
        .iter()
        .map(|(name, symbol)| format!("  {name}: \"{symbol}\""))
        .collect::<Vec<_>>()
        .join(",\n");
    let direct_wasm_import_object = direct_wasm_browser_import_object_source();
    let runtime_wrapper_source = browser_runtime_wrapper_source();
    format!(
        r#"const API_VERSION = {api_version};
const API_VERSION_SYMBOL = "{api_version_symbol}";
const MODEL_INFO_JSON_PTR_SYMBOL = "{model_info_ptr_symbol}";
const MODEL_INFO_JSON_LEN_SYMBOL = "{model_info_len_symbol}";
const ALLOC_F64_BUFFER_SYMBOL = "{alloc_f64_buffer_symbol}";
const FREE_F64_BUFFER_SYMBOL = "{free_f64_buffer_symbol}";

{direct_wasm_import_object}

const KERNEL_SYMBOLS = Object.freeze({{
{kernel_symbol_entries}
}});

{runtime_wrapper_source}

function readUtf8(memory, ptr, len) {{
  const bytes = new Uint8Array(memory.buffer, ptr, len);
  return new TextDecoder().decode(bytes);
}}

function normalizeWasmBytes(bytes) {{
  if (bytes instanceof Uint8Array) {{
    return bytes;
  }}
  if (bytes instanceof ArrayBuffer) {{
    return new Uint8Array(bytes);
  }}
  throw new Error("Expected Uint8Array or ArrayBuffer WASM bytes");
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

export async function instantiatePharmsolDslWasmBytes(bytes) {{
  const {{ instance }} = await WebAssembly.instantiate(normalizeWasmBytes(bytes), DIRECT_WASM_IMPORTS);
  return createPharmsolDslWasmModel(instance);
}}

export async function loadPharmsolDslWasmModel(source) {{
  const response = source instanceof Response ? source : await fetch(source);
  if (typeof WebAssembly.instantiateStreaming === "function") {{
    const {{ instance }} = await WebAssembly.instantiateStreaming(response, DIRECT_WASM_IMPORTS);
    return createPharmsolDslWasmModel(instance);
  }}
  return instantiatePharmsolDslWasmBytes(await response.arrayBuffer());
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
  const infoEnvelope = JSON.parse(readUtf8(memory, infoPtr, infoLen));
  if (Number(infoEnvelope.abi_version) !== API_VERSION) {{
    throw new Error(`Expected pharmsol DSL WASM metadata version ${{API_VERSION}}, got ${{infoEnvelope.abi_version}}`);
  }}

  const kernels = Object.fromEntries(
    Object.entries(KERNEL_SYMBOLS)
      .filter(([, symbol]) => typeof exports[symbol] === "function")
      .map(([name, symbol]) => [name, exports[symbol].bind(exports)])
  );

  for (const [name, available] of Object.entries(infoEnvelope.kernels)) {{
    if (Boolean(available) !== Object.prototype.hasOwnProperty.call(kernels, name)) {{
      throw new Error(`Kernel metadata mismatch for ${{name}}`);
    }}
  }}

    const lowLevelModel = {{
    info: infoEnvelope.model,
    instance,
    memory,
    kernels,
    createF64Buffer(length) {{
      return createBufferHandle(exports, memory, length);
    }},
  }};

    return createHighLevelPharmsolDslWasmModelApi(lowLevelModel);
}}
"#,
        api_version = WASM_API_VERSION,
        api_version_symbol = API_VERSION_SYMBOL,
        model_info_ptr_symbol = MODEL_INFO_JSON_PTR_SYMBOL,
        model_info_len_symbol = MODEL_INFO_JSON_LEN_SYMBOL,
        alloc_f64_buffer_symbol = ALLOC_F64_BUFFER_SYMBOL,
        free_f64_buffer_symbol = FREE_F64_BUFFER_SYMBOL,
        direct_wasm_import_object = direct_wasm_import_object,
        kernel_symbol_entries = kernel_symbol_entries,
        runtime_wrapper_source = runtime_wrapper_source,
    )
}

fn touch_cache_key(queue: &mut VecDeque<WasmCompileCacheKey>, key: &WasmCompileCacheKey) {
    if let Some(position) = queue.iter().position(|entry| entry == key) {
        queue.remove(position);
    }
    queue.push_back(key.clone());
}

fn browser_runtime_wrapper_source() -> String {
    r#"const KNOWN_KERNELS = Object.freeze(Object.keys(KERNEL_SYMBOLS));

function createNamedIndexResolver(entries, label) {
    const byName = new Map(entries.map((entry) => [entry.name, entry.index]));
    return Object.freeze({
        optional(name) {
            return byName.get(name);
        },
        required(name) {
            const index = byName.get(name);
            if (index === undefined) {
                throw new Error(`Unknown ${label} \`${name}\``);
            }
            return index;
        },
    });
}

function createParameterIndexResolver(parameters) {
    const byName = new Map(parameters.map((name, index) => [name, index]));
    return Object.freeze({
        optional(name) {
            return byName.get(name);
        },
        required(name) {
            const index = byName.get(name);
            if (index === undefined) {
                throw new Error(`Unknown parameter \`${name}\``);
            }
            return index;
        },
    });
}

function normalizeKernelName(name) {
    if (!KNOWN_KERNELS.includes(name)) {
        throw new Error(`Unknown kernel \`${name}\``);
    }
    return name;
}

function hasKernel(model, name) {
    return KNOWN_KERNELS.includes(name) && typeof model.kernels[name] === "function";
}

function normalizeTime(time) {
    const value = Number(time ?? 0);
    if (!Number.isFinite(value)) {
        throw new Error(`Expected a finite time value, got ${time}`);
    }
    return value;
}

function writeArrayLikeToView(values, view, label) {
    if (values == null) {
        return false;
    }
    if (values.length !== view.length) {
        throw new Error(`Expected ${label} length ${view.length}, got ${values.length}`);
    }
    for (let index = 0; index < view.length; index += 1) {
        const value = Number(values[index]);
        if (!Number.isFinite(value)) {
            throw new Error(`Expected ${label}[${index}] to be finite, got ${values[index]}`);
        }
        view[index] = value;
    }
    return true;
}

function zeroViews(views) {
    for (const view of Object.values(views)) {
        view.fill(0);
    }
}

function kernelUsesStateInput(kernelName) {
    return (
        kernelName === "derive" ||
        kernelName === "dynamics" ||
        kernelName === "outputs" ||
        kernelName === "drift" ||
        kernelName === "diffusion"
    );
}

function kernelUsesDerivedInput(kernelName) {
    return kernelName !== "derive" && kernelName !== "init";
}

function kernelOutputTarget(handles, views, info, kernelName) {
    switch (kernelName) {
        case "derive":
            return {
                ptr: handles.derived.ptr,
                view: views.derived,
                length: info.derived_len,
                zeroBeforeCall: false,
            };
        case "outputs":
            return {
                ptr: handles.outputs.ptr,
                view: views.outputs,
                length: info.output_len,
                zeroBeforeCall: true,
            };
        case "route_lag":
        case "route_bioavailability":
            return {
                ptr: handles.routeScratch.ptr,
                view: views.routeScratch,
                length: info.route_len,
                zeroBeforeCall: true,
            };
        case "dynamics":
        case "init":
        case "drift":
        case "diffusion":
            return {
                ptr: handles.stateScratch.ptr,
                view: views.stateScratch,
                length: info.state_len,
                zeroBeforeCall: true,
            };
        default:
            throw new Error(`Kernel \`${kernelName}\` is not supported by the high-level runtime wrapper`);
    }
}

/**
 * High-level reusable browser session layered on top of the raw kernel exports.
 * Use this for repeated evaluations. It keeps stable typed-array views alive and
 * exposes a zero-copy fast path via `invokeKernelView(...)` and `evaluateOutputsView(...)`.
 *
 * If you need custom output aliasing, hand-managed buffer lifetimes, or direct
 * access to the raw kernel ABI, use `model.kernels` plus `model.createF64Buffer(...)` directly.
 */
export function createPharmsolDslWasmSession(model) {
    const info = model.info;
    const handles = {
        states: model.createF64Buffer(info.state_len),
        params: model.createF64Buffer(info.parameters.length),
        covariates: model.createF64Buffer(info.covariates.length),
        routes: model.createF64Buffer(info.route_len),
        derived: model.createF64Buffer(info.derived_len),
        stateScratch: model.createF64Buffer(info.state_len),
        outputs: model.createF64Buffer(info.output_len),
        routeScratch: model.createF64Buffer(info.route_len),
    };
    const views = Object.freeze({
        states: handles.states.view(),
        params: handles.params.view(),
        covariates: handles.covariates.view(),
        routes: handles.routes.view(),
        derived: handles.derived.view(),
        stateScratch: handles.stateScratch.view(),
        outputs: handles.outputs.view(),
        routeScratch: handles.routeScratch.view(),
    });
    zeroViews(views);

    const outputIndex = createNamedIndexResolver(info.outputs, "output");
    const initialized = {
        states: views.states.length === 0,
        params: views.params.length === 0,
        derived: views.derived.length === 0,
    };
    let freed = false;

    function assertOpen() {
        if (freed) {
            throw new Error(`Browser session for model \`${info.name}\` has already been freed`);
        }
    }

    function markInitialized(kind) {
        if (kind === "states" || kind === "params" || kind === "derived") {
            initialized[kind] = true;
        }
    }

    function setBuffer(kind, values, label) {
        assertOpen();
        writeArrayLikeToView(values, views[kind], label);
        markInitialized(kind);
        return views[kind];
    }

    function maybeSetBuffer(kind, values, label) {
        if (values !== undefined) {
            return setBuffer(kind, values, label);
        }
        return views[kind];
    }

    function ensureReady(kind, label) {
        if (!initialized[kind]) {
            throw new Error(
                `Missing ${label} for model \`${info.name}\`; set it on the session or pass it in the kernel inputs.`
            );
        }
    }

    function prepareBaseInputs(inputs) {
        maybeSetBuffer("states", inputs.states, "states");
        maybeSetBuffer("params", inputs.params, "parameters");
        maybeSetBuffer("covariates", inputs.covariates, "covariates");
        maybeSetBuffer("routes", inputs.routes, "routes");
    }

    function ensureDerivedReady(kernelName, time, inputs, options, invokeKernelInternal) {
        if (!kernelUsesDerivedInput(kernelName) || views.derived.length === 0) {
            return;
        }
        if (inputs.derived !== undefined) {
            setBuffer("derived", inputs.derived, "derived values");
            return;
        }
        if (options.derive === false) {
            ensureReady("derived", "derived values");
            return;
        }
        if (!hasKernel(model, "derive")) {
            ensureReady("derived", "derived values");
            return;
        }
        invokeKernelInternal("derive", { time }, { copy: false, derive: false });
    }

    function invokeKernelInternal(kernelName, inputs = {}, options = {}) {
        assertOpen();
        const normalizedKernelName = normalizeKernelName(kernelName);
        const kernel = model.kernels[normalizedKernelName];
        if (typeof kernel !== "function") {
            throw new Error(`Model \`${info.name}\` does not expose kernel \`${normalizedKernelName}\``);
        }

        prepareBaseInputs(inputs);
        const time = normalizeTime(inputs.time);
        if (kernelUsesStateInput(normalizedKernelName)) {
            ensureReady("states", "states");
        }
        if (views.params.length > 0) {
            ensureReady("params", "parameter values");
        }
        ensureDerivedReady(normalizedKernelName, time, inputs, options, invokeKernelInternal);

        const target = kernelOutputTarget(handles, views, info, normalizedKernelName);
        if (target.zeroBeforeCall) {
            target.view.fill(0);
        }
        kernel(
            time,
            handles.states.ptr,
            handles.params.ptr,
            handles.covariates.ptr,
            handles.routes.ptr,
            handles.derived.ptr,
            target.ptr
        );
        return options.copy === false
            ? target.view.subarray(0, target.length)
            : target.view.slice(0, target.length);
    }

    return Object.freeze({
        info,
        views,
        setStates(values) {
            return setBuffer("states", values, "states");
        },
        setParameters(values) {
            return setBuffer("params", values, "parameters");
        },
        setCovariates(values) {
            return setBuffer("covariates", values, "covariates");
        },
        setRoutes(values) {
            return setBuffer("routes", values, "routes");
        },
        setDerived(values) {
            return setBuffer("derived", values, "derived values");
        },
        hasKernel(name) {
            return hasKernel(model, name);
        },
        invokeKernel(name, inputs = {}, options = {}) {
            return invokeKernelInternal(name, inputs, { ...options, copy: options.copy ?? true });
        },
        invokeKernelView(name, inputs = {}, options = {}) {
            return invokeKernelInternal(name, inputs, { ...options, copy: false });
        },
        evaluateOutputs(inputs = {}, options = {}) {
            return invokeKernelInternal("outputs", inputs, { ...options, copy: options.copy ?? true });
        },
        evaluateOutputsView(inputs = {}, options = {}) {
            return invokeKernelInternal("outputs", inputs, { ...options, copy: false });
        },
        evaluateOutput(name, inputs = {}, options = {}) {
            const outputs = invokeKernelInternal("outputs", inputs, {
                ...options,
                copy: options.copy ?? true,
            });
            return outputs[outputIndex.required(name)];
        },
        free() {
            if (freed) {
                return;
            }
            freed = true;
            for (const handle of Object.values(handles)) {
                handle.free();
            }
        },
    });
}

/**
 * Augments the low-level loader result with named lookups plus one-off helpers.
 * Application code should prefer `model.createSession()` for repeated work or
 * `model.evaluateOutputs(...)` for simple one-off output evaluation.
 *
 * The raw low-level surface remains available on the returned object via
 * `model.kernels`, `model.memory`, `model.instance`, and `model.createF64Buffer(...)`.
 */
function createHighLevelPharmsolDslWasmModelApi(lowLevelModel) {
    const routeIndex = createNamedIndexResolver(lowLevelModel.info.routes, "route");
    const outputIndex = createNamedIndexResolver(lowLevelModel.info.outputs, "output");
    const covariateIndex = createNamedIndexResolver(lowLevelModel.info.covariates, "covariate");
    const parameterIndex = createParameterIndexResolver(lowLevelModel.info.parameters);

    return Object.freeze(Object.assign({}, lowLevelModel, {
        hasKernel(name) {
            return hasKernel(lowLevelModel, name);
        },
        routeIndex(name) {
            return routeIndex.optional(name);
        },
        requireRouteIndex(name) {
            return routeIndex.required(name);
        },
        outputIndex(name) {
            return outputIndex.optional(name);
        },
        requireOutputIndex(name) {
            return outputIndex.required(name);
        },
        covariateIndex(name) {
            return covariateIndex.optional(name);
        },
        requireCovariateIndex(name) {
            return covariateIndex.required(name);
        },
        parameterIndex(name) {
            return parameterIndex.optional(name);
        },
        requireParameterIndex(name) {
            return parameterIndex.required(name);
        },
        createSession() {
            return createPharmsolDslWasmSession(lowLevelModel);
        },
        evaluateOutputs(inputs = {}, options = {}) {
            const session = createPharmsolDslWasmSession(lowLevelModel);
            try {
                return session.evaluateOutputs(inputs, options);
            } finally {
                session.free();
            }
        },
        evaluateOutput(name, inputs = {}, options = {}) {
            const session = createPharmsolDslWasmSession(lowLevelModel);
            try {
                return session.evaluateOutput(name, inputs, options);
            } finally {
                session.free();
            }
        },
    }));
}
"#
        .to_string()
}

fn direct_wasm_browser_import_object_source() -> String {
    let unary_import_entries = DIRECT_WASM_UNARY_MATH_IMPORTS
        .iter()
        .map(|import| {
            let implementation = match import.name {
                "exp" => "(value) => Math.exp(value)",
                "ln" => "(value) => Math.log(value)",
                "log10" => "(value) => Math.log10(value)",
                "log2" => "(value) => Math.log2(value)",
                "round" => "(value) => Math.round(value)",
                "sin" => "(value) => Math.sin(value)",
                "cos" => "(value) => Math.cos(value)",
                "tan" => "(value) => Math.tan(value)",
                _ => unreachable!("unknown direct wasm unary import {}", import.name),
            };
            format!("    {}: {}", import.name, implementation)
        })
        .collect::<Vec<_>>();
    let binary_import_entries = DIRECT_WASM_BINARY_MATH_IMPORTS
        .iter()
        .map(|import| {
            let implementation = match import.name {
                "pow" => "(lhs, rhs) => Math.pow(lhs, rhs)",
                _ => unreachable!("unknown direct wasm binary import {}", import.name),
            };
            format!("    {}: {}", import.name, implementation)
        })
        .collect::<Vec<_>>();
    let import_entries = unary_import_entries
        .into_iter()
        .chain(binary_import_entries)
        .collect::<Vec<_>>()
        .join(",\n");

    format!(
        "const DIRECT_WASM_IMPORTS = Object.freeze({{\n  {module}: Object.freeze({{\n{entries}\n  }})\n}});",
        module = DIRECT_WASM_IMPORT_MODULE,
        entries = import_entries,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dsl::{
        DiagnosticPhase, DSL_LOWERING_GENERIC, DSL_PARSE_GENERIC, DSL_SEMANTIC_GENERIC,
    };

    const SIMPLE_SOURCE: &str = r#"
model = example_ode
kind = ode

params = ke, v
states = central
outputs = cp

infusion(iv) -> central

dx(central) = -ke * central

out(cp) = central / v ~ continuous()
"#;

    #[test]
    fn compile_module_source_to_wasm_module_returns_bytes_and_metadata() {
        let compiled = compile_module_source_to_wasm_module(SIMPLE_SOURCE, Some("example_ode"))
            .expect("compile wasm module");

        assert!(!compiled.wasm_bytes.is_empty());
        assert_eq!(compiled.metadata.model.name, "example_ode");
        assert_eq!(compiled.metadata.abi_version, WASM_API_VERSION);
        assert!(compiled.metadata.kernels.outputs);
    }

    #[test]
    fn wasm_compile_cache_reuses_identical_source_compiles() {
        let cache = WasmCompileCache::new(4);

        let first = cache
            .compile_module_source_to_wasm_module(SIMPLE_SOURCE, Some("example_ode"))
            .expect("compile first cached wasm module");
        let second = cache
            .compile_module_source_to_wasm_module(SIMPLE_SOURCE, Some("example_ode"))
            .expect("compile second cached wasm module");

        assert_eq!(cache.entry_count(), 1);
        assert_eq!(first, second);
    }

    #[test]
    fn wasm_compile_cache_evicts_oldest_entry_at_capacity() {
        let cache = WasmCompileCache::new(1);

        cache
            .compile_module_source_to_wasm_module(SIMPLE_SOURCE, Some("example_ode"))
            .expect("compile first cached wasm module");
        cache
            .compile_module_source_to_wasm_module(SIMPLE_SOURCE, Some("example_ode"))
            .expect("recompile first cached wasm module");
        cache
            .compile_module_source_to_wasm_module(
                r#"
model = second_ode
kind = ode

params = ke, v
states = central
outputs = cp

infusion(iv) -> central

dx(central) = -ke * central

out(cp) = central / v ~ continuous()
"#,
                Some("second_ode"),
            )
            .expect("compile second cached wasm module");

        assert_eq!(cache.entry_count(), 1);
    }

    #[test]
    fn compile_module_source_to_wasm_module_preserves_parse_diagnostic_structure() {
        let source = "model broken { kind ode outputs { cp = 1 + } }";
        let error = compile_module_source_to_wasm_module(source, None)
            .expect_err("invalid DSL should fail before wasm compilation");

        let diagnostic = error.diagnostic().expect("wasm should expose diagnostic");
        assert_eq!(diagnostic.phase, DiagnosticPhase::Parse);
        assert_eq!(diagnostic.code, DSL_PARSE_GENERIC);
        assert!(diagnostic.message.contains("expected expression"));

        let rendered = error
            .render_diagnostic(source)
            .expect("rendered diagnostic");
        assert!(rendered.contains("error[DSL1000]"), "{}", rendered);
        assert!(rendered.contains("expected expression"), "{}", rendered);

        let report = error
            .diagnostic_report("inline.dsl")
            .expect("diagnostic report");
        assert_eq!(report.source.name, "inline.dsl");
        assert_eq!(report.diagnostics[0].code, "DSL1000");
        assert_eq!(report.diagnostics[0].phase, "parse");
    }

    #[test]
    fn compile_module_source_to_wasm_module_preserves_semantic_diagnostic_structure() {
        let source = r#"
model = broken
kind = ode

states = central
outputs = cp

infusion(oral) -> central

dx(central) = rate(orla)

out(cp) = central ~ continuous()
"#;
        let error = compile_module_source_to_wasm_module(source, None)
            .expect_err("invalid semantic route reference should fail before wasm emission");

        let diagnostic = error.diagnostic().expect("wasm should expose diagnostic");
        assert_eq!(diagnostic.phase, DiagnosticPhase::Semantic);
        assert_eq!(diagnostic.code, DSL_SEMANTIC_GENERIC);
        assert!(diagnostic.message.contains("unknown route `orla`"));
        assert!(diagnostic
            .suggestions
            .iter()
            .any(|suggestion| suggestion.message.contains("did you mean `oral`?")));

        let rendered = error
            .render_diagnostic(source)
            .expect("rendered diagnostic");
        assert!(rendered.contains("error[DSL2000]"), "{}", rendered);
        assert!(rendered.contains("unknown route `orla`"), "{}", rendered);
        assert!(
            rendered.contains("suggestion: did you mean `oral`?"),
            "{}",
            rendered
        );

        let report = error
            .diagnostic_report("inline.dsl")
            .expect("diagnostic report");
        assert_eq!(report.diagnostics[0].code, "DSL2000");
        assert_eq!(report.diagnostics[0].phase, "semantic");
        assert!(!report.diagnostics[0].suggestions.is_empty());
    }

    #[test]
    fn compile_module_source_to_wasm_module_preserves_lowering_diagnostic_structure() {
        let source = r#"
model = broken
kind = ode

states = transit[4], central
outputs = cp

bolus(oral) -> transit[4]

dx(transit[0]) = -transit[0]
dx(transit[1]) = transit[0] - transit[1]
dx(transit[2]) = transit[1] - transit[2]
dx(transit[3]) = transit[2] - transit[3]
dx(central) = transit[3] - central

out(cp) = central ~ continuous()
"#;
        let error = compile_module_source_to_wasm_module(source, None)
            .expect_err("out-of-bounds route destination should fail during lowering");

        let diagnostic = error.diagnostic().expect("wasm should expose diagnostic");
        assert_eq!(diagnostic.phase, DiagnosticPhase::Lowering);
        assert_eq!(diagnostic.code, DSL_LOWERING_GENERIC);
        assert!(diagnostic
            .message
            .contains("route destination for `transit` indexes element 4"));

        let rendered = error
            .render_diagnostic(source)
            .expect("rendered diagnostic");
        assert!(rendered.contains("error[DSL3000]"), "{}", rendered);
        assert!(
            rendered.contains("route destination for `transit` indexes element 4"),
            "{}",
            rendered
        );

        let report = error
            .diagnostic_report("inline.dsl")
            .expect("diagnostic report");
        assert_eq!(report.diagnostics[0].code, "DSL3000");
        assert_eq!(report.diagnostics[0].phase, "lowering");
        assert_eq!(report.source.name, "inline.dsl");
    }

    #[test]
    fn browser_loader_supports_in_memory_wasm_bytes() {
        let loader = browser_loader_source();

        assert!(loader.contains("instantiatePharmsolDslWasmBytes"));
        assert!(loader.contains("normalizeWasmBytes"));
        assert!(loader.contains("WebAssembly.instantiate("));
    }

    #[test]
    fn browser_loader_exposes_high_level_runtime_wrapper() {
        let loader = browser_loader_source();

        assert!(loader.contains("export function createPharmsolDslWasmSession"));
        assert!(loader.contains("createHighLevelPharmsolDslWasmModelApi"));
        assert!(loader.contains("createSession()"));
        assert!(loader.contains("evaluateOutputs(inputs = {}, options = {})"));
        assert!(loader.contains("evaluateOutput(name, inputs = {}, options = {})"));
        assert!(loader.contains("invokeKernelView(name, inputs = {}, options = {})"));
        assert!(loader.contains("model.createF64Buffer(...)"));
    }

    #[test]
    fn browser_loader_source_is_stable_across_calls() {
        let first = browser_loader_source();
        let second = browser_loader_source();

        assert_eq!(first, second);
    }
}
