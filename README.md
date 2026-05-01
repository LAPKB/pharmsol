# pharmsol

[![Build](https://github.com/LAPKB/pharmsol/actions/workflows/build.yml/badge.svg)](https://github.com/LAPKB/pharmsol/actions/workflows/build.yml)
[![Documentation](https://github.com/LAPKB/pharmsol/actions/workflows/docs.yml/badge.svg)](https://github.com/LAPKB/pharmsol/actions/workflows/docs.yml)
[![crates.io](https://img.shields.io/crates/v/pharmsol.svg)](https://crates.io/crates/pharmsol)

A high-performance Rust library for pharmacokinetic/pharmacodynamic (PK/PD) simulation using analytical solutions, ordinary differential equations (ODEs), or stochastic differential equations (SDEs).

## Installation

Add `pharmsol` to `Cargo.toml`:

```bash
cargo add pharmsol
```

## Quick Start

Most Rust-first workflows start with one of the equation macros: `analytical!`,
`ode!`, or `sde!`. Here is a simple one-compartment IV infusion model using `analytical!`:

```rust
use pharmsol::prelude::*;

let analytical = analytical! {
    name: "one_cmt_iv",
    params: [ke, v],
    states: [central],
    outputs: [cp],
    routes: {
        infusion(iv) -> central,
    },
    structure: one_compartment,
    out: |x, _p, _t, _cov, y| {
        y[cp] = x[central] / v;
    },
};

let subject = Subject::builder("patient_001")
    .infusion(0.0, 500.0, "iv", 0.5)
    .missing_observation(0.5, "cp")
    .missing_observation(1.0, "cp")
    .missing_observation(2.0, "cp")
    .missing_observation(4.0, "cp")
    .build();

let predictions = analytical
    .estimate_predictions(&subject, &[1.022, 194.0])
    .unwrap();
```

## Modeling Surfaces

Here is the same one-compartment IV setup written as an ODE:

```rust
use pharmsol::prelude::*;

let ode = ode! {
    name: "one_cmt_iv",
    params: [ke, v],
    states: [central],
    outputs: [cp],
    routes: {
        infusion(iv) -> central,
    },
    diffeq: |x, _p, _t, dx, _cov| {
        dx[central] = -ke * x[central];
    },
    out: |x, _p, _t, _cov, y| {
        y[cp] = x[central] / v;
    },
};
```

See [examples/analytical_readme.rs](examples/analytical_readme.rs),
[examples/ode_readme.rs](examples/ode_readme.rs),
[examples/sde_readme.rs](examples/sde_readme.rs),
[examples/analytical_vs_ode.rs](examples/analytical_vs_ode.rs), and
[examples/compare_solvers.rs](examples/compare_solvers.rs). For migration-oriented notes,
see [docs/analytical-authoring-migration.md](docs/analytical-authoring-migration.md) and
[docs/ode-authoring-migration.md](docs/ode-authoring-migration.md).

### Built-In Analytical Kernels

- [x] One-compartment with IV infusion
- [x] One-compartment with IV infusion and oral absorption
- [x] Two-compartment with IV infusion
- [x] Two-compartment with IV infusion and oral absorption
- [x] Three-compartment with IV infusion
- [x] Three-compartment with IV infusion and oral absorption

## DSL and Runtime Targets

If the model needs to be loaded or compiled at runtime, pharmsol also provides a DSL with
the same broad modeling coverage: ODE, analytical, and SDE authoring. The DSL can target
an in-process JIT runtime, native ahead-of-time artifacts, or WASM bundles depending on
how you want to ship and execute the model.

- `dsl-jit`: compile DSL source into a runtime model inside the current process.
- `dsl-aot` and `dsl-aot-load`: emit a native artifact and load it later.
- `dsl-wasm`: compile and execute portable WASM model artifacts.

See [examples/dsl_runtime_jit.rs](examples/dsl_runtime_jit.rs) for the in-repo JIT flow.
The companion `pharmsol-examples` crate includes end-to-end native AOT and WASM runtime
examples.

## Performance

Analytical solutions provide 20-33× speedups compared to equivalent ODE formulations. See [benchmarks](benches/) for details.

## Non-Compartmental Analysis (NCA)

pharmsol includes a complete NCA module for calculating standard pharmacokinetic parameters.

```rust
use pharmsol::prelude::*;
use pharmsol::nca::NCAOptions;

let subject = Subject::builder("patient_001")
    .bolus(0.0, 100.0, "oral")  // 100 mg oral dose
    .observation(0.5, 5.0, "cp")
    .observation(1.0, 10.0, "cp")
    .observation(2.0, 8.0, "cp")
    .observation(4.0, 4.0, "cp")
    .observation(8.0, 2.0, "cp")
    .build();

let result = subject.nca(&NCAOptions::default()).expect("NCA failed");

println!("Cmax: {:.2}", result.exposure.cmax);
println!("Tmax: {:.2} h", result.exposure.tmax);
println!("AUClast: {:.2}", result.exposure.auc_last);

if let Some(ref term) = result.terminal {
    println!("Half-life: {:.2} h", term.half_life);
}
```

**Supported NCA Parameters:**

- Exposure: Cmax, Tmax, Clast, Tlast, AUClast, AUCinf, tlag
- Terminal: λz, t½, MRT
- Clearance: CL/F, Vz/F, Vss
- IV-specific: C0 (back-extrapolation), Vd
- Steady-state: AUCtau, Cmin, Cavg, fluctuation, swing

## Links

- [API Documentation](https://lapkb.github.io/pharmsol/pharmsol/)
- [Examples](examples/)
