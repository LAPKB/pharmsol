# Pharmsol

[![Build](https://github.com/LAPKB/pharmsol/actions/workflows/build.yml/badge.svg)](https://github.com/LAPKB/pharmsol/actions/workflows/build.yml)
[![Documentation](https://github.com/LAPKB/pharmsol/actions/workflows/docs.yml/badge.svg)](https://github.com/LAPKB/pharmsol/actions/workflows/docs.yml)
![crates.io](https://img.shields.io/crates/v/pharmsol.svg)

Simulate PK/PD profiles using ODE and analytical models.

## Example

ODE based model.

```rust
use pharmsol::*;

let subject = data::Subject::builder("id1")
    .bolus(0.0, 100.0, 0)
    .repeat(2, 0.5)
    .observation(0.5, 0.1, 0)
    .observation(1.0, 0.4, 0)
    .observation(2.0, 1.0, 0)
    .observation(2.5, 1.1, 0)
    .build();
println!("{subject:#?}");
let ode = simulator::Equation::new_ode(
    //Difussion Equations
    |x, p, _t, dx, _rateiv, _cov| {
        fetch_cov!(cov, t,);
        fetch_params!(p, ka, ke, _tlag, _v);
        dx[0] = -ka * x[0];
        dx[1] = ka * x[0] - ke * x[1];
    },
    // Lag definition (In this case boluses on dx[0] will be delayed by `tlag`)
    |p| {
        fetch_params!(p, _ka, _ke, tlag, _v);
        lag! {0=>tlag}
    },
    // No bio-availability
    |_p| fa! {},
    // Default initial conditions (0.0,0.0)
    |_p, _t, _cov, _x| {},
    // Output Equations
    |x, p, _t, _cov, y| {
        fetch_params!(p, _ka, _ke, _tlag, v);
        y[0] = x[1] / v;
    },
    (2, 1),
);

let op = ode.simulate_subject(&subject, &vec![0.3, 0.5, 0.1, 70.0], false);
println!("{op:#?}");
```

Analytic based model.

```rust
use pharmsol::*;
let analytical = simulator::Equation::new_analytical(
    one_compartment_with_absorption,
    |_p, _cov| {},
    |p| {
        fetch_params!(p, _ka, _ke, tlag, _v);
        lag! {0=>tlag}
    },
    |_p| fa! {},
    |_p, _t, _cov, _x| {},
    |x, p, _t, _cov, y| {
        fetch_params!(p, _ka, _ke, _tlag, v);
        y[0] = x[1] / v;
    },
    (2, 1),
);
let op = analytical.simulate_subject(&subject, &vec![0.3, 0.5, 0.1, 70.0]);
println!("{op:#?}");
```

## Supported analytical models

We are working to support all the standard analytical models.

- [x] One-compartment with IV infusion
- [x] One-compartment with IV infusion and oral absorption
- [x] Two-compartment with IV infusion
- [x] Two-compartment with IV infusion and oral absorption
- [ ] Three-compartmental models
