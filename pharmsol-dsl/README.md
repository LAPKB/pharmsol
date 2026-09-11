# pharmsol-dsl

`pharmsol-dsl` compiles pharmsol DSL model source into a ready-to-run form.

Use this crate when you need to work with model source as data:

- parse DSL text into syntax nodes
- inspect spans and diagnostics
- analyze names and types into a checked model
- compile validated models into the ready-to-run form used by runtime backends

Do not use this crate for JIT compilation or `Subject`-based prediction helpers. Those workflows stay in `pharmsol::dsl` in the main `pharmsol` crate.

## Main Pipeline

The one-shot pipeline is `compile_model` or `compile_module`, which fails with
the unified `DslError`:

```rust
use pharmsol_dsl::compile_model;

let source = r#"
name = bimodal_ke
kind = ode

params = ke, v
states = central
outputs = cp

infusion(iv) -> central

dx(central) = -ke * central
out(cp) = central / v
"#;

let execution = compile_model(source).expect("model compiles");

assert_eq!(execution.name, "bimodal_ke");
assert_eq!(execution.metadata.routes.len(), 1);
assert_eq!(execution.metadata.outputs.len(), 1);
```

The staged pipeline is available when you need the intermediate representations:

1. `parse_model` or `parse_module`
2. `analyze_model` or `analyze_module`
3. `compile_analyzed_model` or `compile_analyzed_module`

The main public modules are:

- `syntax` for the syntax tree
- `diagnostic` for spans, codes, and rendered reports
- `analysis` for the analyzed, fully checked model
- `execution` for the ready-to-run model consumed by the runtime backend

The parser accepts both canonical `model { ... }` source and the authoring
shorthand used by the `pharmsol` examples.

## Runtime-only utility functions

The DSL supports the same effect functions as the Rust API:

- `estimate_effect_2(u, v, alpha, h1, h2)` takes five numeric arguments.
- `estimate_effect_3(a, b, c, alpha12, alpha13, alpha23, alpha123, h1, h2, h3)` takes ten numeric arguments.

These functions calculate a combined effect using supplied interaction
coefficients; they do not fit those coefficients. Both return real values.
`pharmsol-dsl` validates and lowers these calls separately from mathematical
intrinsics but does not evaluate them while folding constants. Execute models
using these calls through the `pharmsol` runtime, which supplies host callbacks
to the Rust implementations.

## Errors

Every stage reports errors with source spans and renders an annotated report
when printed:

```text
error[DSL2000]: unknown identifier `missing_state`
  --> line 3, column 11
  |
3 | out(cp) = missing_state
  |           ^^^^^^^^^^^^^ unknown identifier `missing_state`
```

`DslError::phase` identifies the failing stage, `DslError::diagnostics`
exposes the structured diagnostics, and `DslError::diagnostic_report` produces
a JSON-serializable report for editors and tooling.

## Boundary With `pharmsol`

`pharmsol-dsl` owns the source-to-execution compiler and its data structures.

`pharmsol::dsl` re-exports that compiler surface and adds the runtime-facing APIs for backend selection, artifact loading, and prediction execution.

Use `pharmsol-dsl` when you are building tooling, validation, migration, or your own backend. Use `pharmsol::dsl` when you want a complete source-to-runtime workflow.
