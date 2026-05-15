# pharmsol-dsl

`pharmsol-dsl` is the backend-neutral frontend crate for the pharmsol DSL.

Use this crate when you need to work with model source as data:

- parse DSL text into syntax nodes
- inspect spans and diagnostics
- analyze names and types into typed IR
- lower validated models into the execution model used by runtime backends

Do not use this crate for JIT compilation, native AoT export or load, WASM runtime loading, or `Subject`-based prediction helpers. Those workflows stay in `pharmsol::dsl` in the main `pharmsol` crate.

## Main Pipeline

The public pipeline is:

1. `parse_model` or `parse_module`
2. `analyze_model` or `analyze_module`
3. `lower_typed_model` or `lower_typed_module`

The main public modules are:

- `ast` for syntax-level nodes
- `diagnostic` for spans, codes, and rendered reports
- `ir` for the typed intermediate representation
- `execution` for the lowered execution model shared by JIT, AoT, and WASM backends

The parser accepts both canonical `model { ... }` source and the authoring shorthand used by the `pharmsol` examples.

## Small Example

```rust
use pharmsol_dsl::{analyze_model, lower_typed_model, parse_model};

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

let syntax = parse_model(source).expect("model parses");
let typed = analyze_model(&syntax).expect("model analyzes");
let execution = lower_typed_model(&typed).expect("model lowers");

assert_eq!(execution.name, "bimodal_ke");
assert_eq!(execution.metadata.routes.len(), 1);
assert_eq!(execution.metadata.outputs.len(), 1);
```

## Boundary With `pharmsol`

`pharmsol-dsl` owns the frontend pipeline and its data structures.

`pharmsol::dsl` re-exports that frontend surface and adds the runtime-facing APIs for backend selection, artifact loading, and prediction execution.

Use `pharmsol-dsl` when you are building tooling, validation, migration, or your own backend. Use `pharmsol::dsl` when you want a complete source-to-runtime workflow.
