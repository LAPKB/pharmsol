# Pharmsol WASM Plugin Specification

Version: 1.0 (draft)

Purpose
-------

This document specifies a minimal, versioned ABI for user-provided WebAssembly plugins that implement Pharmsol models. The goal is to allow advanced users to produce precompiled `.wasm` modules that can be safely instantiated by Pharmsol (in a browser or Wasmtime/Wasmer) and invoked to evaluate model behavior (derivatives, metadata, steps) without requiring recompilation of the host or linking native dynamic libraries.

Design goals
------------

- Minimal: small set of imports/exports to ease authoring across languages.
- Stable: versioned ABI to allow forward/backward compatibility.
- Language neutral: use linear memory + u32/u64 primitives and JSON for structured metadata.
- Safe: use opaque handles and pointer/length pairs instead of native pointers to Rust objects.
- Sandboxed: rely on WASM runtime to enforce limits; host should enforce extra limits (memory, fuel).

High-level contract
-------------------

- All plugin modules MUST export `plugin_abi_version` and `plugin_name`.
- Plugins MAY require specific host imports (logging, allocation helpers). The host will supply sensible defaults if imports are missing, if possible.
- The host will instantiate the plugin and then call `plugin_create` with an optional configuration blob (JSON). The plugin returns a small integer handle (non-zero) or zero to signal error.
- The host uses handles to call `plugin_step`, `plugin_get_metadata`, and `plugin_free`.

ABI versioning
--------------

- `plugin_abi_version()` -> u32
  - The plugin returns a u32 ABI version number. Host must refuse to load plugins with major versions it cannot support. Semantic versioning of the ABI is recommended (major increments break compatibility).

Exports (required)
------------------

All pointer and length types use 32-bit unsigned integers (u32) to index the module's linear memory. Handles are 32-bit unsigned integers (u32) with 0 reserved for invalid/null.

1. plugin_abi_version() -> u32

- Returns the ABI version implemented by the module.

2. plugin_name(ptr: u32, len: u32) -> u32

- Optional: write the plugin name string into host-provided buffer. Alternatively, plugin may return 0 and provide name via `plugin_get_metadata`.
- Semantics: host provides a pointer/len to memory it controls (or 0/0 to request size). If ptr==0 and len==0, plugin returns required size. If ptr!=0, plugin writes up to len bytes and returns actual written bytes or negative error code.

3. plugin_create(config_ptr: u32, config_len: u32) -> u32

- Create an instance of the model. `config` is a JSON blob (UTF-8) describing initial parameters or options. If both are 0, plugin uses built-in defaults.
- Returns a non-zero handle on success or 0 on error. For error details, the host should call `plugin_last_error` (see optional exports).

4. plugin_free(handle: u32) -> u32

- Free resources associated with a handle. Returns 0 for success, non-zero for error.

5. plugin_step(handle: u32, t: f64, dt: f64, inputs_ptr: u32, inputs_len: u32, outputs_ptr: u32, outputs_len_ptr: u32) -> i32

- Evaluate a step or compute derivatives for the model instance.
- `t` and `dt` are host-supplied time and timestep (floating point). The semantics of step are model-specific (integration step or single derivative evaluation); document clearly in plugin metadata.
- `inputs_ptr/inputs_len` point to an array of f64 values (packed little-endian) representing parameter values or exogenous inputs. The plugin may accept fewer or more inputs; any mismatch is an error.
- `outputs_ptr` is where the plugin writes resulting f64 outputs; `outputs_len_ptr` is a pointer in host memory where the plugin will write the number of f64 values it wrote (or required size when ptr was null).
- Return code: 0 success, negative values for defined error codes (see Error Codes).

6. plugin_get_metadata(handle: u32, out_ptr_ptr: u32) -> i32

- Return a JSON metadata blob describing the model: parameter names and ordering, state variable names, default values, units, equation kind, capabilities (events, stochastic), and ABI version.
- The plugin will allocate the JSON string in its linear memory and write a 64-bit pointer/length pair into the host-provided `out_ptr_ptr` (two consecutive u32 values: ptr then len). Alternatively, if the plugin implements `host_alloc`, it can call into the host's allocator instead.
- Return 0 success, negative error for failure.

Optional exports (recommended)
-----------------------------

1. plugin_last_error(handle: u32, out_ptr: u32, out_len: u32) -> i32

- Copy last error message string into the provided buffer. If out_ptr==0 and out_len==0, return required length.

2. plugin_supports_f64() -> u32

- Return 1 if plugin expects f64 for numerical buffers (recommended). Otherwise 0.

Host imports (recommended)
--------------------------

These function imports allow plugins to use host helpers rather than re-implementing allocators or logging. The host may choose to provide stubs.

1. host_alloc(size: u32) -> u32

- Allocate `size` bytes in the host's memory space accessible to the plugin. Returns pointer offset into host-supplied linear memory or 0 on failure. (Use only if the host and plugin share linear memory; otherwise plugin will allocate in its own memory.)

2. host_free(ptr: u32, size: u32)

- Free a host-allocated block.

3. host_log(ptr: u32, len: u32, level: u32)

- Host-provided logging helper. Plugin writes UTF-8 bytes to plugin memory and passes pointer,len. Level is user-defined (0=debug,1=info,2=warn,3=error).

4. host_random_u64() -> u64

- Provide randomness from host if needed. Plugins needing deterministic seeds should accept them via `plugin_create` config.

Error codes
-----------

- Return negative i32 values for errors to keep C-like convention.

- -1: Generic error
- -2: Invalid handle
- -3: Buffer too small / size mismatch (caller should retry with provided size)
- -4: Unsupported ABI version
- -5: Unsupported capability
- -6: Internal plugin panic/trap

Memory ownership and allocation patterns
--------------------------------------

- Prefer the linear-memory pointer/length convention for strings and blobs. The host will copy strings into plugin memory when calling functions, or the plugin will allocate and return pointers with lengths.
- To return dynamically created strings (like metadata JSON), plugin should allocate memory inside its own linear memory and write pointer/length into the host-supplied pointer slot. The host must be prepared to read and copy that data before the plugin frees it.

Security and sandboxing
-----------------------

- Plugins must not assume file system or network access unless launched with appropriate WASI capabilities. Hosts must opt-in to features and apply least privilege.
- Hosts must enforce memory limits and allow interrupting long-running plugins. Use Wasmtime's fuel mechanism or equivalent.

Compatibility notes
-------------------

- Always check `plugin_abi_version` before using other exports.
- Hosts should fallback to IR/interpreter-based execution when plugin ABI is unsupported.

Authoring guidelines
--------------------

1. Start a plugin with the minimal exports required to avoid host rejection.
2. Provide detailed metadata: parameter order, state order, units, capability flags (events, stochastic), and recommended recommended step semantics.
3. Use JSON for metadata to avoid tightly-coupled binary formats.

Build hints for Rust authors
---------------------------

- Build with `cdylib` or `--target wasm32-unknown-unknown` and avoid relying on `std` features that require WASI unless you target wasm32-wasi.
- Use `wasm-bindgen` only if you target JS and plan to use JS glue; otherwise prefer raw wasm exports with `#[no_mangle] extern "C"` functions and a small allocator like `wee_alloc`.

Example memory sequence (metadata retrieval)
-------------------------------------------

1. Host instantiates plugin and calls `plugin_get_metadata(instance_handle, host_out_ptr)` where `host_out_ptr` points to two consecutive u32 slots in host memory.
2. Plugin serializes JSON string into its linear memory at offset P and length L.
3. Plugin writes P and L to the two u32 slots at `host_out_ptr` and returns 0.
4. Host reads P and L via the Wasm instance memory view and copies the JSON blob to its own memory space. Host may then call `plugin_free_memory(P, L)` if the plugin offers such an export, or expect the plugin to free on `plugin_free`.

Troubleshooting
---------------

- If metadata size is unknown, host can call `plugin_get_metadata(handle, 0)` which should return the required size in a standard location or return -3 with the required size encoded in a convention (prefer the pointer/length return method described).

Examples and recipes
--------------------

- Example flow for a simple model plugin:
  1. `plugin_abi_version()` -> 1
  2. Host calls `plugin_create(0,0)` -> returns handle 1
  3. Host calls `plugin_get_metadata(1, out_ptr)` -> reads metadata JSON, learns parameter/state ordering
  4. Host calls `plugin_step(1, t, dt, inputs_ptr, inputs_len, out_ptr, out_len_ptr)` repeatedly to step/evaluate.

Specification lifecycle and version bumps
---------------------------------------

- Start version 1.0, keep ABI additive if possible. If a breaking change is required, increment major version and require host/plugin negotiation.

Next steps
----------

1. Add a concrete `pharmsol-plugin` crate template that exports the minimal ABI and demonstrates metadata and step implementations.
2. Add CI recipes for building `wasm32-unknown-unknown` and `wasm32-wasi` artifacts.
3. Implement host-side adapters in Pharmsol for instantiating a plugin, mapping metadata to the `Meta` type, and wrapping `plugin_step` as an `Equation` implementation that existing integrators can call.

Appendix: change log
--------------------

- 2025-10-29: Draft 1.0 created.
