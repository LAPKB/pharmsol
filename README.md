# pharmsol

[![Build](https://github.com/LAPKB/pharmsol/actions/workflows/build.yml/badge.svg)](https://github.com/LAPKB/pharmsol/actions/workflows/build.yml)
[![Documentation](https://github.com/LAPKB/pharmsol/actions/workflows/docs.yml/badge.svg)](https://github.com/LAPKB/pharmsol/actions/workflows/docs.yml)
[![crates.io](https://img.shields.io/crates/v/pharmsol.svg)](https://crates.io/crates/pharmsol)

A high-performance Rust library for pharmacokinetic/pharmacodynamic (PK/PD) simulation using analytical solutions, ordinary differential equations (ODEs), or stochastic differential equations (SDEs).

## Installation

Add `pharmsol` to your `Cargo.toml`, either manually or using

```bash
cargo add pharmsol
```

## Quick Start

```rust
use pharmsol::*;

// Create a subject with an IV infusion and observations
let subject = Subject::builder("patient_001")
    .infusion(0.0, 500.0, 0, 0.5)  // 500 units over 0.5 hours
    .observation(0.5, 1.645, 0)
    .observation(1.0, 1.216, 0)
    .observation(2.0, 0.462, 0)
    .observation(4.0, 0.063, 0)
    .build();

// Define parameters: ke (elimination rate), v (volume)
let ke = 1.022;
let v = 194.0;

// Use the built-in one-compartment analytical solution
let analytical = equation::Analytical::new(
    one_compartment,
    |_p, _t, _cov| {},
    |_p, _t, _cov| lag! {},
    |_p, _t, _cov| fa! {},
    |_p, _t, _cov, _x| {},
    |x, p, _t, _cov, y| {
        fetch_params!(p, _ke, v);
        y[0] = x[0] / v;  // Concentration = Amount / Volume
    },
    (1, 1),  // (compartments, outputs)
);

// Get predictions
let predictions = analytical.estimate_predictions(&subject, &vec![ke, v]).unwrap();
```

## ODE-Based Models

For custom or complex models, define your own ODEs:

```rust
use pharmsol::*;

let ode = equation::ODE::new(
    |x, p, _t, dx, _b, rateiv, _cov| {
        fetch_params!(p, ke, _v);
        // One-compartment model with IV infusion support
        dx[0] = -ke * x[0] + rateiv[0];
    },
    |_p, _t, _cov| lag! {},
    |_p, _t, _cov| fa! {},
    |_p, _t, _cov, _x| {},
    |x, p, _t, _cov, y| {
        fetch_params!(p, _ke, v);
        y[0] = x[0] / v;
    },
    (1, 1),
);
```

## Supported Analytical Models

- [x] One-compartment with IV infusion
- [x] One-compartment with IV infusion and oral absorption
- [x] Two-compartment with IV infusion
- [x] Two-compartment with IV infusion and oral absorption
- [x] Three-compartment with IV infusion
- [x] Three-compartment with IV infusion and oral absorption

## Performance

Analytical solutions provide 20-33× speedups compared to equivalent ODE formulations. See [benchmarks](benches/) for details.

## Non-Compartmental Analysis (NCA)

pharmsol includes a complete NCA module for calculating standard pharmacokinetic parameters.

```rust
use pharmsol::prelude::*;
use pharmsol::nca::NCAOptions;

let subject = Subject::builder("patient_001")
    .bolus(0.0, 100.0, 0)  // 100 mg oral dose
    .observation(0.5, 5.0, 0)
    .observation(1.0, 10.0, 0)
    .observation(2.0, 8.0, 0)
    .observation(4.0, 4.0, 0)
    .observation(8.0, 2.0, 0)
    .build();

let results = subject.nca(&NCAOptions::default(), 0);
let result = results[0].as_ref().expect("NCA failed");

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

# Links

- [API Documentation](https://lapkb.github.io/pharmsol/pharmsol/)
- [Examples](examples/)
