# Executing user-defined models on WebAssembly — analysis and design

October 29, 2025

This document analyzes the existing `exa` model-building and dynamic-loading approach in Pharmsol, explains why it is not usable on WebAssembly (WASM), and presents multiple design options to enable running user-defined models from within a WASM-hosted Pharmsol runtime. It discusses trade-offs, security, ABI proposals, testing strategies, and recommended next steps.

This is a technical engineering analysis intended to be a design blueprint. It intentionally avoids implementation code, and instead focuses on concrete architectures, precise interface sketches, hazards, and validation plans.

## Quick summary / recommendation

- The current `exa` approach (creating a temporary Rust project, running `cargo build`, producing a platform dynamic library, then using `libloading` to load symbols) cannot be used from a WASM target (browser or wasm32-unknown-unknown runtime) because it depends on process spawning, filesystem semantics, dynamic linking, and thread/process control not available in typical WASM hosts.
- Two primary workable approaches for WASM compatibility are: (A) interpret a serialized model representation (AST/bytecode/DSL) inside the WASM host; (B) accept precompiled WASM modules (created outside the WASM host) and run them with a well-defined, minimal ABI. Each has strong trade-offs. A hybrid can offer a pragmatic path: an interpreter as the default portable path plus an optional WASM-plugin pathway for advanced users.
- Recommendation: start with an interpreter/serialized-IR approach for maximum portability (works in browser and WASI), and define a companion, clearly versioned WASM plugin ABI for power users and server-side deployments where precompiled WASM modules can be uploaded/installed.

## Why the current `exa` cannot run on WASM

Key reasons:

1. Build-time tooling: `exa::build` shells out to `cargo` to create a template project and run `cargo build`. Running `cargo` requires a host OS with process spawning, filesystem with developer toolchain, and native toolchain availability. In browsers and many WASM runtimes this is impossible.

2. Dynamic linking: `exa::load` uses `libloading` and platform dynamic libraries (`.so`, `.dll`, `.dylib`) and relies on native ABI and dynamic linking at runtime. WASM runtimes (especially wasm32-unknown-unknown) do not support Unix-like dlopen of platform shared libraries. Even wasm-hosts that support dynamic linking (WASI with preopened files) differ significantly from native OS dynamic linking.

3. FFI and ownership: The code uses raw pointers, expects Rust ABI and cloned objects to cross library boundaries. WASM modules expose different ABIs (linear memory, function exports/imports). Passing complex Rust objects by pointer across a WASM boundary is fragile and often impossible without serialization glue.

4. Threads and blocking IO: The build process spawns threads to stream stdout/err and waits on child processes. Many WASM environments (browsers) do not support native threads or block the event loop differently.

Because of the above, the server-side native dynamic plugin model does not translate to a WASM-hosted environment without redesign.

## Use cases we must support (requirements)

1. Allow end users to define models (ODEs or other equation types) and execute them inside a WASM-hosted Pharmsol instance (browser and WASM runtimes) without requiring native cargo toolchain inside the runtime.
2. Preserve a reasonably high-performance execution where possible (some models are performance sensitive). Allow optional high-performance plugin paths.
3. Maintain safety and security: user models must be sandboxed (resource limits, no arbitrary host access unless explicitly granted).
4. Keep a small, stable host-plugin interface and version it.
5. Provide a migration path so existing `exa` users can adopt the wasm-capable approach.

Implicit constraints:

- Minimal or no native code compilation in the runtime. Compilation should happen outside the runtime or be avoided via interpretation.
- Deterministic (or at least consistent) behavior across platforms where possible.

## Candidate architectures (high level)

I. Interpreter / serialized IR (recommended default)

- Idea: convert the model text to a compact intermediate representation (IR), JSON AST, or small bytecode on the host (this can be done either offline or inside the non-WASM tooling), then ship that IR to the WASM runtime where a small interpreter executes it.

- Pros:
  - Works in all WASM hosts (browser, WASI, standalone runtimes).
  - No external toolchain or dynamic linking required inside the WASM module.
  - Can be secured and resource-limited easily (single-threaded, deterministic loops, step budgets).
  - Simpler lifecycle: the host (browser UI or server) supplies IR; the interpreter runs it.

- Cons:
  - Potentially slower than native code or compiled WASM modules (but can be optimized).
  - Must reimplement evaluation semantics for model expressions, numerical integration hooks, and any host APIs used by models.

II. Precompiled WASM modules from user (plugin-on-wasm)

- Idea: users compile their model to a small WASM module (using Rust or another language). Pharmsol running in WASM instantiates that module and connects by a well-defined ABI (exports/imports). Compilation occurs outside the Pharmsol WASM runtime (user's machine or a server-side build service).

- Pros:
  - Best performance; compiled code runs as native WASM in the runtime.
  - Allows complex user code without embedding an interpreter.

- Cons:
  - Requires the user to compile to WASM themselves, or an external build service.
  - ABI ergonomics are complex: sharing complex structures across the WASM boundary needs glue (shared linear memory, serialization, helper allocators).
  - Host must provide a precise, versioned import contract (logging, RNG, time, memory management).

III. Hybrid: interpreter as default + optional WASM plugin path

- Idea: implement the interpreter for general users and an optional plugin ABI for advanced users or server-based compilation pipelines. This covers both portability and perf.

IV. Host-compilation pipeline (server assisted)

- Idea: mirror existing `exa` server-side: accept user model text, run a server-side build pipeline to produce a WASM module (instead of native dynamic library), then deliver the `.wasm` to the client to instantiate. This removes the need to run `cargo` inside the browser.

- Pros:
  - Preserves compiled performance without requiring users to compile locally.
  - Centralizes build toolchain and security scanning.

- Cons:
  - Operational burden (CI/build infra), security (compiling untrusted code), distribution and versioning complexity.

## Detailed considerations and trade-offs

1) ABI and data exchange

- Simple serialized-model approach: exchange IR (JSON, CBOR, or MessagePack). The interpreter reads IR and returns results by JSON objects.
- WASM plugin approach: define a small C-like ABI with a fixed set of exported functions. Example minimal exports from plugin module:
  - `plugin_version() -> u32` (ABI version)
  - `create_model() -> u32` (returns an opaque handle)
  - `free_model(handle: u32)`
  - `step_model(handle: u32, t: f64, dt: f64, inputs_ptr: u32, inputs_len: u32, outputs_ptr: u32)`
  - `get_metadata_json(ptr_out: u32) -> u32` (returns pointer/length pair or writes into host-provided buffer)

- Memory management: require plugin to export an allocator or follow a simple convention (host provides memory, plugin uses host-provided functions to allocate). Or use a string/byte convention: exports with pointer and length encoded as two 32-bit values.

2) Passing complex Rust types

- Avoid trying to share Rust-specific types (Box, owned struct clones) across WASM boundary. Use a stable, language-neutral representation (JSON, CBOR) for metadata and parameters.

3) Host imports to plugin

- Plugins will likely need helper imports from the host: random numbers, logging, panic hooks (or traps), allocation, time. Minimally define these imports and keep them stable.

4) Security / sandboxing

- WebAssembly provides sandboxing but host must enforce memory, CPU time, and resource constraints. Approaches:
  - Use wasm runtimes (Wasmtime, Wasmer) with configurable memory limits and fuel (instruction count) to interrupt long-running modules.
  - In browsers, worker time-limits and cooperative stepping.
  - Reject exports/imports that give filesystem or network access unless explicitly trusted via WASI capabilities.

5) Determinism and numeric behavior

- Floating-point results may differ across hosts; document expected tolerances and avoid depending on platform-specific FP flags.

6) Threading and concurrency

- WASM threads are not yet universally available (shared memory / atomics). The wasm-capable module should not assume threads. If the host supports threads, the interpreter or plugin can optionally use them, gated behind feature detection.

7) Tooling and developer experience

- For plugin path: provide a `pharmsol-plugin` crate template that exports the standard ABI and instructions for compiling to wasm (cargo build --target wasm32-unknown-unknown or wasm32-wasi, or use wasm-bindgen if targeting JS). Provide examples for Rust and a plain C approach.
- For interpreter path: provide a serializer that converts existing model description (the same text used by `exa`) into IR. Keep the IR stable and versioned.

8) Size and startup cost

- Interpreter binary size depends on evaluator complexity. For browser deployments, keep interpreter lean (avoid heavy crates). For plugin path, each user-provided wasm module will increase download size; caching helps.

9) Compatibility with existing `exa` API

- `exa::build` and `exa::load` produce `E: Equation` and `Meta` clones. For wasm, design host-side shims that map from plugin/interpreter results into the same `Equation`/`Meta` trait surface used by the rest of Pharmsol. If the host runtime itself is built as WASM and shares the same Rust codebase, define a small adapter layer that converts the plugin or IR results into `Equation` implementations in the runtime.

10) Error reporting

- Prefer textual JSON errors with codes. Expose streaming logs during model compilation (if server-assisted build) and during instantiation; for interpreter model parsing, produce structured parse/semantics errors.

## ABI sketch for a precompiled WASM plugin

This sketch is intentionally conservative and minimal; if implemented, it should be strongly versioned.

- Module exports (names and semantics):

  - `plugin_abi_version() -> u32` — numeric ABI version (e.g., 1).
  - `plugin_name(ptr: u32, len: u32)` — optional name string (or return pointer/len to host).
  - `plugin_create(ptr: u32, len: u32) -> u32` — allocate and return a handle for a model instance created from a JSON blob at (ptr,len) or from internal code. Handle 0 reserved for null/error.
  - `plugin_free(handle: u32)` — free instance.
  - `plugin_step(handle: u32, t: f64, dt: f64, inputs_ptr: u32, inputs_len: u32, outputs_ptr: u32, outputs_len_ptr: u32) -> i32` — step or evaluate; input/output are serialized arrays or contiguous floats. Return 0 for success or negative error code.
  - `plugin_get_metadata(handle: u32, out_ptr_ptr: u32) -> i32` — write a JSON metadata blob to linear memory and return pointer/length via out_ptr_ptr.

- Host imports (the host should provide these):

  - `host_alloc(size: u32) -> u32` and `host_free(ptr: u32, size: u32)` — optional; otherwise plugin uses its own allocator.
  - `host_log(ptr: u32, len: u32, level: u32)` — optional logging.
  - `host_random_u32() -> u32` — for deterministic or host-provided RNG.

Notes:

- Use string/json for metadata to avoid sharing complex structs. This keeps the plugin language agnostic.
- Use u32 handles and linear memory offsets for safety.

## Interpreter / serialized IR proposal

Design a compact IR that expresses:

- Model metadata (parameters, state variables, initial values, parameter default values)
- Expressions (arithmetic, functions, accessors) — either as an AST or simple stack-based bytecode
- Event definitions or discontinuities (if Pharmsol supports them)

Representation options:

- JSON: human readable, easy to debug, larger size.
- CBOR / MessagePack: binary, smaller.
- Custom bytecode: most compact and efficient but takes more work to define and maintain.

Evaluation engine features:

- Expression evaluator: compile the AST to a sequence of instructions, then evaluate in a tight loop.
- Integrator interface: provide hooks for integrator state and allow the interpreter to evaluate derivatives; integrators live in the host and call into evaluator for right-hand side computation.
- Caching and JIT-like improvements: precompute evaluation order, constant folding, and expression inlining.

Why interpreter is attractive:

- Predictable: no external compilation step.
- Fast to iterate: developer can change model text and send new IR to the runtime without rebuilding.

Potential downsides:

- Interpreter complexity can grow if the model language is rich (user functions, closures). Keep the DSL bounded to maintain a fast interpreter.

## Migration path and compatibility

1. Define a model-IR serializer in the existing `exa::build` pipeline (native). Add a mode that produces IR instead of a native cdylib. This is low-effort and reuses existing parsing code.
2. Implement the interpreter in the WASM runtime to read IR and produce the `Equation` trait behaviors in the runtime. On native builds, the interpreter can be used as a fallback.
3. Define and publish the plugin WASM ABI and crate template for advanced users. Provide an example repository and CI workflow to produce valid `.wasm` plugins.
4. Keep `exa::load` semantics but offer new functions like `load_from_ir` and `load_wasm_plugin` that map to the same `Equation` / `Meta` surfaces.

## Testing strategy

- Unit tests: for IR generation and the interpreter expression evaluator. Use a battery of deterministic tests comparing interpreter outputs to native `exa` results.
- Integration tests: run a model end-to-end through host integrators on both native and wasm targets (use wasm-pack test harness or Wasmtime for server-side tests).
- Fuzzing: target parser and evaluator with malformed inputs to catch edge cases and panics.
- Performance benchmarks: compare interpreter vs plugin vs native compiled models; measure startup and per-step costs.

## Operational concerns and security

- If server-side compilation is offered, run untrusted compilations in isolated builder sandboxes and scan outputs for known-bad constructs. Prefer user-provided wasm modules or interpreter IR to avoid running arbitrary native build steps on shared infra.
- For wasm plugin hosting, enforce memory limits and instruction fuel limits (Wasmtime fuel, Wasmer middleware, or browser worker timeouts).

## Suggested project structure (conceptual)

- `src/exa/` — keep existing build/load for native platforms.
- `src/exa/ir.rs` — (new) IR definitions and serializer (no implementation here; just noting where it would live).
- `src/exa/interpreter/` — interpreter runtime for models.
- `src/exa/wasm_plugin_spec.md` — a short specification (could reference this document).

Note: the interpreter can be compiled for both native and wasm targets; keep it dependency-light for browser builds.

## Performance expectations

- Interpreter: expect some overhead relative to compiled native code. Reasonable targets: if evaluator is well optimized, derivative evaluation can be within 2–5x slower than compiled native code depending on expression complexity and integrator call frequency. Measurement required.
- WASM plugin: performance similar to native wasm-compiled code (good), but host bridging (serialization) can add overhead.

## Example migration scenarios (no code)

1. Browser: user enters model text in UI -> UI sends model text to server or local serializer -> IR (JSON) returned -> browser Pharmsol wasm runtime loads IR -> interpreter executes model.
2. Server (WASM runtime): Accept `.wasm` plugin from advanced user -> instantiate with Wasmtime with resource limits -> use plugin exports as model implementation.

## Versioning, compatibility, and future-proofing

- Version the IR and the plugin ABI separately. Include feature flags in the ABI (capabilities mask) so future extensions don't break older hosts.
- Consider aligning plugin ABI with WASI/component model as it stabilizes.

## Next steps (recommended minimal roadmap)

1. Add an `IR` serialization mode to the native `exa::build` pipeline so existing tooling can emit IR instead of or in addition to cdylib. (Low-risk, high-value.)
2. Implement a lightweight interpreter in the Pharmsol core, with an API `load_from_ir` that returns an `Equation`/`Meta` instance usable by existing integrators. Prioritize the feature set required by current users (params, states, derivatives, simple events).
3. Design and publish a versioned WASM plugin ABI, crate-template, and documentation for advanced users. Provide a CI-based example to compile a plugin to `.wasm`.
4. Add tests and a benchmark suite comparing native `exa` dynamic-loading, IR-interpreter, and wasm-plugin performance.

## Appendix: checklist of engineering and QA items

- [ ] IR schema definition (JSON Schema or protobuf)
- [ ] Parser changes to emit IR
- [ ] Interpreter design doc + micro-benchmarks
- [ ] WASM plugin ABI spec (short document)
- [ ] Crate template for compiling to `.wasm`
- [ ] Example demonstrations (browser and Wasmtime)
- [ ] Security review and sandbox configuration for server-side builds
- [ ] Documentation for end users and plugin authors

---

If you'd like, I can now proceed to:

- produce a formal `src/exa/wasm_plugin_spec.md` containing a precise ABI table and memory layout (more low-level and concrete), or
- implement the IR serializer and a first-pass interpreter prototype (in code), or
- draft the crate template and CI steps for producing WASM plugin artifacts.

Tell me which of the next steps you prefer and I will proceed. If you prefer, I can also update the repository with the spec file and a small README for plugin authors.
