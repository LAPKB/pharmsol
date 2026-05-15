# browser-compile-bridge

This crate packages the browser-side DSL compiler as a standalone `wasm32-unknown-unknown` module.

It lives under `pharmsol-dsl/` because it is coupled to the DSL browser compile surface, while still depending on `pharmsol` for the current `dsl-wasm-compile` backend API.

It is used by:

- the browser E2E harness in `tests/browser-e2e/`
- the bundled browser playground asset shipped in `pharmsol-examples/examples/dsl_browser_playground/`

## Build

```bash
rustup target add wasm32-unknown-unknown
cargo build --manifest-path pharmsol-dsl/browser-compile-bridge/Cargo.toml --target wasm32-unknown-unknown --release --target-dir target/browser-compile-bridge-dist
```

Output:

```text
target/browser-compile-bridge-dist/wasm32-unknown-unknown/release/browser_compile_bridge.wasm
```

## Refresh The Bundled pharmsol-examples Asset

```bash
cp target/browser-compile-bridge-dist/wasm32-unknown-unknown/release/browser_compile_bridge.wasm ../pharmsol-examples/examples/dsl_browser_playground/browser_compile_bridge.wasm
```
