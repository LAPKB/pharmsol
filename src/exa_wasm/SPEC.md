# exa_wasm — Interpreter + IR: SPEC, current state, gaps, and recommendations

Generated from reading the entire `src/exa_wasm` and `src/exa_wasm/interpreter` source.

This document is structured as:

- Short contract / goals
- IR format and loader contract
- Parser / AST / typechecker contract
- Evaluator semantics and dispatch contract
- Registry / runtime behavior
- Implemented features (what works today)
- Missing features / gaps to replicate Rust arithmetic/PKPD semantics
- Tests (what exists, what is missing, priorities)
- Detailed optimization recommendations (micro + architectural + WASM targets)
- Migration / next-steps and low-risk improvements

## Contract (inputs, outputs, success criteria)

- Inputs
  - JSON IR file (emitted by `emit_ir`) containing `ir_version`, `kind`, `params`, textual closures for `diffeq`, `lag`, `fa`, `init`, `out` and pre-extracted structured maps `lag_map`, `fa_map`.
  - Simulator vectors at runtime: `x` (states), `p` (params), `rateiv` (rate-in vectors), `t` (time scalar), `cov` (covariates object).
- Outputs
  - A registered runtime model (RegistryEntry) and an `ODE` wrapper with dispatch functions that the simulator uses: diffeq_dispatch, lag_dispatch, fa_dispatch, init_dispatch, out_dispatch.
  - Runtime return values via writes into `dx[]`, `x[]`, `y[]` through provided assignment closures during dispatch.
- Success criteria
  - The interpreter evaluates closure code deterministically and produces numerically equal results (within floating differences) compared to the equivalent native Rust ODE code for the same model text.
  - Loader rejects ill-formed IR (missing structured maps for lag/fa, index out of bounds, type errors, unknown identifiers in prelude) with informative errors.

## IR format and emitter (`build::emit_ir`)

- `emit_ir` produces a JSON object containing:
  - ir_version: "1.0"
  - kind: equation kind string (via E::kind())
  - params: vector of parameter names (strings)
  - diffeq/out/init/lag/fa: textual closures (strings) supplied by caller
  - lag_map/fa_map: structured HashMap<usize,String> extracted from textual macro bodies when present
  - prelude: not directly emitted; emit_ir extracts `lag_map` and `fa_map` to avoid runtime parsing of textual `lag!` and `fa!` macros
- Notes:
  - The runtime loader requires structured `lag_map` and `fa_map` fields (if textual macros are present in the IR, loader will reject unless maps exist). This is explicit loader behavior.

## Parser / AST / Typechecker

- Parser
  - `parser::tokenize` tokenizes numeric literals, booleans, identifiers, brackets, braces, parentheses, operators, and punctuation. Supports numeric exponent notation and recognizes `true`/`false` as booleans.
  - `parser::Parser` implements a recursive-descent parser supporting:
    - expressions: numbers, booleans, identifiers, indexed expressions (e.g., `x[0]`, `rateiv[ i ]`), function calls `f(a,b)`, method calls `obj.method(...)`, unary ops (`-`, `!`), binary ops (`+ - * / ^`, comparisons, `==, !=`, `&&, ||`), ternary `cond ? then : else`.
    - statements: expression-statement (`expr;`), assignment (`ident = expr;`), indexed assignment (`ident[expr] = expr;`), `if` with optional `else` and block or single-statement branches. It reads semicolons and braces.
- AST
  - `ast::Expr`, `ast::Stmt` capture parsed program structure. `Stmt::Assign(Lhs, Expr)` stores Lhs as `Ident` or `Indexed`.
- Typechecker
  - `typecheck` implements a conservative checker: numeric and boolean types, ensures indexed-assignment RHS are numeric, index expressions numeric, and attempts to detect obvious mistakes. It accepts numeric/bool coercions similar to evaluator semantics.

## Evaluator semantics (`eval.rs`)

- Runtime Value type: enum { Number(f64), Bool(bool) } with coercion rules:
  - `as_number()`: Bool -> 1.0/0.0, Number -> value
  - `as_bool()`: Number -> value != 0.0, Bool -> value
- Evaluator (`eval_expr`) implements:
  - Identifiers: resolves prefixed underscore names (return 0.0), locals map (prelude/assign locals), pmap-mapped parameters, `t` as time, covariates via interpolation when `cov` and `t` provided.
  - Indexed: resolves indexed names for `x`, `p/params`, `rateiv`. Performs bounds checks and sets runtime error when out-of-range.
  - Calls: evaluates arguments then `eval_call` handles builtin functions. Unknown function falls back to Number(0.0) (no runtime error).
  - Binary ops: arithmetic, comparisons, logical with short-circuit behaviour for `&&`, `||`.
  - Ternary: use `cond` coercion and evaluate appropriate branch.
  - MethodCall: treated as `eval_call(name)` with receiver as first arg.
- `eval_stmt` executes statements, manages `locals` HashMap for named locals, delegates indexed assignments to a closure provided by dispatchers (which perform safe write to dx,x,y or set runtime error on unsupported names).
- `eval_call` implements a set of builtin functions: exp, ln/log, log2/log10, sqrt, pow/powf, min/max, abs, floor, ceil, round, sin/cos/tan, plus `if` macro-like function used when parsing `if(expr, then, else)` calls — returns second or third argument based on first.

## Loader and `load_ir_ode` behavior

- Loads JSON, extracts `params` -> builds `pmap` param name -> index.
- Walks `diffeq`, `out`, `init` closures:
  - Prefer robust parsing: tries to extract closure body and parse with `Parser::parse_statements()`.
  - Runs `typecheck::check_statements()` and rejects IR with type errors.
  - If parsing fails, falls back to substring scanning to extract top-level indexed assignments (helpers `extract_all_assign`) and convert them to minimal AST `Assign` nodes.
- Prelude extraction: identifies simple non-indexed `name = expr;` assignments (used as locals) via `extract_prelude` via heuristics.
- `lag` and `fa`: loader expects structured `lag_map`/`fa_map` inside IR; will reject IR missing these fields if textual `lag`/`fa` is non-empty (loader no longer supports runtime textual parsing of `lag!{}` macros unless the `lag_map` exists).
- Validation: loader validates indexes, prelude references, fetch_params!/fetch_cov! macro bodies (basic checks), ensures at least some dx assignments exist.
- On success, constructs `RegistryEntry` containing parsed statements, lag/fa expressions, prelude list, pmap, nstates, nouteqs and registers it in `registry`.

## Registry / Dispatch contract

- Registry stores `RegistryEntry` in a global HashMap protected by a Mutex. Entries are referenced by generated `usize` ids.
- `CURRENT_EXPR_ID` is thread-local Cell<Option<usize>> used by dispatch functions to determine which entry to execute.
- Dispatch functions (`dispatch.rs`):
  - `diffeq_dispatch` runs prelude assignments producing locals, then executes `diffeq_stmts` using `eval_stmt` with an assign closure that allows `dx[index] = value` only. Unsupported indexed assignment names cause runtime error.
  - `out_dispatch`: executes `out_stmts` allowing writes to `y[index]` only.
  - `lag_dispatch`/`fa_dispatch`: evaluate entries in `lag`/`fa` maps using zeros for x/rateiv and return a HashMap<usize,T> of numeric results.
  - `init_dispatch`: executes `init_stmts` allowing writes to `x[index]`.
- Registry exposes: `register_entry`, `unregister_model`, `get_entry`, `ode_for_id` and helper functions to get/set current id and runtime error.

## Current implemented features (summary)

- Fully working tokenizer and parser for numeric and boolean expressions, calls, indexing, unary/binary ops, ternary, and `if` statement (with blocks/else).
- Conservative typechecker that catches common type mismatches and forbids assigning boolean to indexed state targets.
- Evaluator with the following key features:
  - numeric arithmetic (+ - \* / ^)
  - comparisons and boolean ops with short-circuiting
  - large set of math builtins: exp, log, ln, log2, log10, sqrt, pow/powf, min/max, abs, floor, ceil, round, sin, cos, tan
  - function-call semantics and method-call mapping (receiver passed as first arg)
  - identifier resolution: params via `pmap`, locals (prelude) and `t` time
  - covariate interpolation support (uses Covariates.interpolate when available)
  - indexed access for `x`, `p/params`, `rateiv` with bounds checks.
- Loader: robust multi-mode loader that prefers AST parsing but falls back to substring extraction for simple assignment patterns; prelude extraction and fetch macro validation exist.
- Registry and dispatch wiring: models are registered and produce an `ODE` with dispatch closures that the rest of simulator can call.
- Tests exist that exercise tokenizer, parser, loader fallback, and small loader behaviors.

## Concrete current tests (found in repository)

- `src/exa_wasm/mod.rs::tests`
  - `test_tokenize_and_parse_simple()` — tokenizes, parses simple expr and evaluates with dummy vectors.
  - `test_macro_parsing_load_ir()` — ensures emit_ir produces an IR loadable by `load_ir_ode` (uses `lag!{...}` macro parsing in emit_ir and loader).
  - `load_negative_tests::test_loader_errors_when_missing_structured_maps()` — asserts loader rejects IR that provides `lag`/`fa` textual form without `lag_map`/`fa_map`.
- `src/exa_wasm/interpreter/loader.rs::tests`
  - Tests for `extract_body` and parsing `if true/false` patterns, ensuring parser normalizes boolean literals and retains top-level `dx` assignment detection.
- `src/exa_wasm/interpreter/mod.rs` includes tests that exercise parser/eval integration.

## Gaps / Missing functionality (to get closer to full Rust-equivalent arithmetic semantics for PK/PD)

- Language features missing or limited
  - No loops (for/while) or `break`/`continue` constructs — many iterative PKPD constructs sometimes use loops for accumulation or vector operations.
  - No block-scoped `let` declarations beyond very small prelude heuristics; `extract_prelude` is conservative and the loader_helpers prelude extraction is a stub in places.
  - No support for compound assignment (+=, -=, etc.).
  - No support for full macros evaluated at runtime — macros are partially stripped, but complex macro bodies must be processed by emitter (emit_ir) into structured maps.
  - No user-land function definition; all functions are builtin only.
  - String handling is absent (not needed for arithmetic but relevant for diagnostics).
- Numeric & semantic gaps
  - No direct handling of NaN/Inf semantics or explicit domain errors (e.g., log of negative) — evaluator will produce f64 results per Rust but may not raise semantic runtime errors.
  - `eval_call` returns Number(0.0) for unknown functions with no runtime error — this hides mistakes (recommend change).
  - Limited builtins: missing many mathematical and special functions (erf, erfc, gamma, lgamma, erf_inv, special logistic forms, etc.) commonly used in PKPD.
  - No vectorized operations / broadcast: expressions that operate on vectors must be written explicitly with indices. No map/reduce primitives.
- Loader / IR gaps
  - Loader does substring scanning for fallbacks — fragile for complex code. The `loader_helpers` module contains stubs (extract_fetch_params, extract_prelude etc.) that are incomplete.
  - The runtime requires structured `lag_map` / `fa_map` in IR. emit_ir tries to produce them but tooling that emits IR must be dependable; otherwise loader rejects.
  - Pre-resolved param indices: while `pmap` exists on entry, expressions still contain identifier strings in AST rather than resolved index nodes; runtime resolves via pmap hash lookups on each identifier resolution.
- Performance / architecture
  - Evaluation uses boxed enums `Value` + recursion + many HashMap lookups for locals and pmap -> hot-path overhead.
  - Every identifier resolves via HashMap lookup; locals and pmap lookups happen at runtime repeatedly; branch mispredicts / hash overhead.
  - No bytecode or compact expression representation; AST walking is interpreted per-evaluation.
  - No precomputation (constant folding) beyond tokenization.
- Safety / ergonomics
  - `eval_call` swallowing unknown functions is a usability and correctness risk.
  - Runtime errors are stored thread-local but no structured diagnostics with expression positions or model id are emitted.

## Recommended missing features prioritized

High priority (for correctness and replication fidelity)

- Make unknown function calls produce loader or runtime error (not silent Number(0.0)). This will catch typos in IR and user errors.
- Fully implement macro extraction and prelude parsing in `loader_helpers` so loader does not rely on fragile substring heuristics. Emit resolved AST or bytecode from `emit_ir`.
- Resolve parameter identifier -> index mapping during load: transform identifier AST nodes representing params into a param-index variant (avoid hash lookup at runtime). Same for covariates and other well-known identifiers.
- Validate and canonicalize all index expressions at load time when possible (e.g., constant numeric indices), so runtime dispatch can avoid repeated checks.
- Replace textual-scanning-based helper heuristics with parser-driven extraction where possible (safer for complex code).
- Centralize evaluator's builtin lookup to use builtins.rs (we already use builtins in the typechecker; ensure eval and dispatch use the same single source of truth).
- Add unit tests specifically for loader_helpers functions (parse/macro/extraction/validation) to lock-in behavior.
- Add richer error reporting in loader to return structured loader errors (instead of just io::Error with a string) — implement a LoaderError enum that carries TypeError variants and positional info.

Medium priority (performance / robustness)

- Add constant folding and simple expression simplifications at load-time.
- Add a simple bytecode (or expression tree) compile step that converts AST into a compact opcode sequence. Implement a small fast interpreter for bytecode.
- Replace `Value` enum with raw f64 in arithmetic paths; booleans can be represented as 0.0/1.0 where appropriate and only coerced when needed — remove boxing in hot path.
- Convert locals from HashMap<String,f64> to an indexed local slot vector created at load-time (map local name -> slot index) and bind to a small Vec<f64> at runtime for O(1) access.

Lower priority (feature expansion)

- Add additional math builtins used in pharmacometrics: `erf`, `erfc`, `gamma`/`lgamma`, `beta`, special integrals, logistic and Hill functions, `sign`, `clamp`.
- Add explicit error handling primitives and optional runtime checks for domain errors.
- Add optional JIT or WASM codegen path: emit precompiled WASM modules (via Cranelift/wasmtime or hand-rolled emitter) for performance.

## Detailed optimization recommendations (nuanced)

These are grouped as quick wins, structural improvements, and advanced options.

1. Quick wins (safe, low risk)

- Change `eval_call` behavior: unknown function => set runtime error + return 0.0 or NaN — do not silently return 0.0. This is a correctness fix.
- Convert repeated HashMap lookups for `pmap`/locals into precomputed indices when possible:
  - When loading, scan AST for identifier usage: if identifier is a param -> replace with AST node ParamIndex(idx). For local names produced by `prelude` extraction, create local slots with indices and rewrite `Ident(name)` to `Local(slot)` where possible.
  - Keep a small structure per `RegistryEntry` describing local name->slot mapping.
- Local slots: replace `HashMap<String,f64>` with `Vec<f64>` and `HashMap<String,usize>` only at load-time; runtime uses direct indexing into the Vec.
- Replace `Value` enum in arithmetic evaluation with direct `f64` passing: the only places booleans are needed is logical operators and conditionals; implement `eval_expr_f64` in hot path that returns f64 and treat boolean contexts by test (value != 0.0). Keep a separate boolean evaluation path for `&&/||`.

2. Structural improvements (medium complexity)

- Implement an AST -> bytecode compiler:
  - Bytecode opcodes: PUSH_CONST(i), LOAD_PARAM(i), LOAD_X(i), LOAD_RATEIV(i), LOAD_LOCAL(i), LOAD_T, CALL_FN(idx), UNARY_NEG, BINARY_ADD, BINARY_MUL, CMP_LT, JUMP_IF_FALSE, ASSIGN_LOCAL(i), ASSIGN_INDEXED(base, idxSlotOrConst), ...
  - Pre-resolve function names to small function-table indices at load time to avoid string comparisons per-call.
  - Implement a small stack-based VM executor that executes opcodes efficiently using raw f64 and direct array accesses.
  - Generate specialized op sequences for `dx`/`x`/`y` assignments to remove runtime string comparison for assignment target.
- Implement constant folding & CSE at compile-time: fold arithmetic on constants and simple algebraic simplifications.
- Implement expression inlining for small functions (if/when user-defined functions are introduced) and partial-evaluation with param constants.

3. Advanced gains (higher risk / more work)

- WASM codegen: compile bytecode to WASM functions (either as textual .wat generation or via Cranelift) and instantiate a WASM module that exports the evaluate functions. This yields near-native speed in WASM hosts but increases code complexity.
- JIT to native code: with Cranelift generate machine code for hot expressions — requires careful memory/safety handling, but huge speedups are possible.
- SIMD / vectorization: for models that do repeated elementwise ops across vectors, provide a vectorized runtime or generate WASM SIMD instructions.

4. Memory and concurrency

- Ensure registry APIs allow safe concurrency: current EXPR_REGISTRY uses Mutex; consider RwLock if reads dominate.
- Provide lifecycle APIs: `drop_model(id)` and ensure no lingering references; add reference counts if ODE objects can outlive registry removal.

5. Numeric stability

- Use f64 consistently but consider `fma` (fused multiply-add) via libm if available for certain patterns.
- Add optional runtime checks for over/underflow and domain errors that can be enabled by a debug mode when running models.

## Tests: what exists, what to add (granular)

Existing tests (detected):

- Parser & tokenizer correctness: many tests in `interpreter/loader.rs` and module-level tests.
- Loader negative test: missing structured maps rejection.
- Parser/If normalization tests: ensure `true` => `1.0` and `false` => `0.0`, and `if` constructs parsed and converted properly.

Missing tests (priority ordered)

1. High priority correctness tests

- Numeric equivalence tests: For a set of representative models, compare outputs of native Rust ODE vs exa_wasm ODE for a range of times and parameter vectors. (Property-based or fixture-based)
- Unknown function handling: ensure loader/runtime errors for unknown function names (after implementing the fix above).
- Parameter resolution: ensure params referred in code map to correct p indices and produce same numeric results as native extraction.
- Index bounds: negative/large indices should produce loader or runtime errors.
- Prelude ordering: test cases where prelude assignment depends on earlier prelude variables; ensure order respected.

2. Medium priority behavioral tests

- Logical short-circuit: ensure `&&`/`||` do not evaluate RHS when LHS decides.
- Ternary and `if()` builtin parity: ensure both mechanisms yield same results.
- Covariate interpolation behavior: tests covering valid/invalid times and missing covariate names.
- Lag/fa maps: ensure `lag_map` values are used and loader rejects textual-only forms.

3. Performance & regression tests

- Microbenchmarks: measure hot path eval time for simple arithmetic expressions vs AST bytecode VM vs native function pointer version.
- Stress tests for registry: many load/unload cycles to check for leaks and correctness.

4. Fuzz / edge cases

- Random expression fuzzing to ensure parser doesn't panic and loader returns acceptable error messages.
- Numeric edge cases: division by zero, log negative, pow with non-integer exponents of negative values — ensure predictable behavior or documented errors.

Suggested test harness additions

- A small test runner that loads a set of model pairs (native and IR) and asserts predictions and likelihoods match within tolerance — this can be used in CI.
- Use `approx` crate for floating comparisons with relative and absolute tolerances.

## Low-risk, high-value immediate changes (implementation steps)

1. Change `eval_call` to report unknown function names as errors.
2. Implement param-id -> ParamIndex AST node and rewrite AST at load-time to resolve `Ident` representing parameters.
3. Replace locals HashMap with Vec slots and a local-name->slot map produced at load-time.
4. Add unit tests to assert that unknown functions trigger loader/runtime errors.

## Longer-term plan (roadmap)

- Phase 1 (0-2 weeks): correctness fixes and small refactors
  - Unknown function error, param resolution, local slots, implement more loader_helpers to remove substring heuristics.
  - Add tests that assert numeric parity for a few canonical ODEs.
- Phase 2 (2-6 weeks): interpreter performance
  - AST -> bytecode compiler, VM runtime, constant folding, pre-resolved function table.
  - Add microbenchmarks and CI perf checks.
- Phase 3 (6+ weeks): WASM/native code generation
  - Emit precompiled WASM modules for hot models and add runtime switches: interpret vs wasm vs native.
  - Investigate JIT via Cranelift for server-side/back-end tooling.

## Developer notes and rationale

- The current code prioritizes correctness and simplicity over raw performance: AST parsing and `eval_expr` recursion are straightforward and robust, and loader performs conservative validation to avoid silent miscompilation.
- The main friction points are runtime hash lookups and string-based resolution of identifiers, and the presence of fallback substring parsing in loader which is fragile for complex closures.
- An incremental approach (resolve param/local names at load-time and add a small bytecode interpreter) yields excellent benefit/cost ratio before pursuing WASM or JIT compilation.

## Recommended SPEC additions to the IR (future)

- Add resolved metadata fields per expression emitted by `emit_ir`:
  - `pmap` (already present at loader) but also an AST/bytecode serialization (e.g., base64 compressed bytecode) so the runtime doesn't need to re-parse expressions.
  - `funcs`: list of builtin functions used so loader can validate and map to indexes.
  - `locals`: prelude local names and evaluation order.
  - `constants`: constant table for deduping floats.

## Security / safety considerations

- Evaluating arbitrary IR should be considered untrusted input if IR comes from external sources. Prefer to validate and sandbox execution. The current interpreter runs in-process with no sandboxing; emitting compiled WASM to a WASM runtime (wasmtime) provides stronger isolation if needed.

## Quick checklist summary (what changed / what to do next)

- I inspected and documented all files in `src/exa_wasm` and `src/exa_wasm/interpreter`.
- Next, implement the high-priority fixes described above (unknown-function errors, AST param resolution, local slot mapping) and add the corresponding unit tests.

---

If you'd like, I can:

- Open a follow-up PR that implements the first low-risk fixes (unknown function -> error, param resolution rewrite, local slots), with unit tests and benchmarks.
- Or, generate the initial bytecode VM design and a minimal implementation for one expression type (binary arithmetic) so you can see the performance improvement baseline.

Tell me which follow-up you prefer and I'll implement it (I will update the todo list and write the code + tests).
