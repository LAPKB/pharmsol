# pharmsol-dsl

`pharmsol-dsl` is the extraction target for the backend-neutral frontend of the pharmsol DSL.

The crate is introduced as an internal workspace member so the codebase can move to a clean engine / DSL split without duplicating workflows or breaking the current `pharmsol::dsl` user-facing API mid-migration.

## Current Status

Slices 1 through 7 have moved the shared frontend data modules, parsing frontend, semantic analysis, and execution lowering here, rewired `pharmsol` backend modules to consume that frontend directly, and cleaned up frontend test ownership:

- AST types
- diagnostics and spans
- typed IR
- lexer
- parser
- authoring desugaring used by the parser
- semantic analysis and semantic diagnostics
- execution lowering and execution model types
- frontend-only integration tests and authoring fixtures

`pharmsol::dsl` now acts as a deliberate compatibility façade: it re-exports the frontend surface from this crate while keeping runtime compilation, artifact loading, and execution wrappers in `pharmsol`.

## Planned Ownership

The crate will own the backend-neutral frontend pipeline:

- AST and syntax types
- authoring desugaring
- diagnostics and spans
- lexical analysis
- parse entrypoints
- typed IR
- execution IR and lowering
- parse / analyze / lower entrypoints

The crate will not own runtime-facing APIs such as JIT, AoT, WASM loading, or `Subject`-based prediction wrappers in the initial extraction.

## Migration Rule

Until the move is complete, `pharmsol::dsl` remains the compatibility façade.

That means:

- backend code continues to live in `pharmsol`
- frontend modules move here slice by slice
- user-facing import churn is deferred until the architecture is stable

## Transitional Note

The temporary lexer bridge from Slice 1 is gone. Frontend-only authoring fixtures now live under `pharmsol-dsl/tests/fixtures/dsl`, while the shared structured-block corpus remains under `tests/fixtures/dsl` because both crates still consume it.
