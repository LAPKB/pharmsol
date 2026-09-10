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

dx(central) = infusion(iv) - ke * central
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

## Inputs And Initial Conditions

ODE authoring uses `bolus(input) * scale` and `infusion(input) * scale` in the
`dx(state)` RHS. Routes are inferred. ODE route declarations and `fa(input)`
assignments are no longer accepted; `lag(input)` remains available. Analytical
and SDE models keep their route APIs.

Bolus scales and lag may use parameters, constants, covariates, time, and
transitively event-safe derived values, but not states or infusion rates. Each
bolus input has one additive term and one destination. Infusion scales are
ordinary continuous RHS expressions, and the same infusion rate may be used in
more than one derivative. No additional input term is injected automatically.

```text
oral_scale = f * wt / 70
lag(oral) = tlag

dx(gut) = bolus(oral) * oral_scale - ka * gut
dx(central) = infusion(iv) * iv_scale + ka * gut - ke * central
```

`bolus[oral]` and `infusion[iv]` are also accepted. A bolus is a discrete dose,
not a continuous rate: its term lowers into the existing route and dose-scaling
machinery. The execution ABI, simulator, and normalized route representation
stay unchanged. Lag and scale keep their existing evaluation timing.

To migrate an ODE authoring model:

- Remove the `bolus(input) -> state` and `infusion(input) -> state` declarations.
- Add the corresponding input term to the destination derivative.
- Move `fa(input)` into the bolus term as a scale, using derived values as needed.
- Replace explicit `rate(input)` calls with `infusion(input)`.
- Keep `lag` and initial conditions unchanged. Put conditional bolus scaling in
  an event-safe derived value, with one bolus term outside the conditional.

The `ode!` macro likewise removes `routes` and `fa` fields. Write
`bolus[oral] * oral_scale` and `infusion[iv] * iv_scale` inside its `diffeq`
closure; the macro binds them to the existing simulator input vectors. Both
three-argument `|x, t, dx|` and five-argument `|x, p, t, dx, cov|` closures remain.

Model initial conditions run before the first event of every occasion.

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
