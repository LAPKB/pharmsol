# pharmsol

[![Build](https://github.com/LAPKB/pharmsol/actions/workflows/build.yml/badge.svg)](https://github.com/LAPKB/pharmsol/actions/workflows/build.yml)
[![Documentation](https://github.com/LAPKB/pharmsol/actions/workflows/docs.yml/badge.svg)](https://github.com/LAPKB/pharmsol/actions/workflows/docs.yml)
[![crates.io](https://img.shields.io/crates/v/pharmsol.svg)](https://crates.io/crates/pharmsol)

Simulate PK/PD profiles using ordinary and stochastic differential equations, or analytical models.

## Example

ODE based model.

```rust
    use pharmsol::*;

    // Subject data can be generated using the builder pattern
    let subject = Subject::builder("id1")
        .bolus(0.0, 100.0, 0)
        .repeat(2, 0.5)
        .observation(0.5, 0.1, 0)
        .observation(1.0, 0.4, 0)
        .observation(2.0, 1.0, 0)
        .observation(2.5, 1.1, 0)
        .covariate("wt", 0.0, 80.0)
        .covariate("wt", 1.0, 83.0)
        .covariate("age", 0.0, 25.0)
        .build();

    let ode = equation::ODE::new(
        |x, p, t, dx, _rateiv, cov| {
            // The following are helper functions to fetch parameters and covariates
            fetch_cov!(cov, t, _wt, _age);
            fetch_params!(p, ka, ke, _tlag, _v);

            // The ODEs are defined here
            dx[0] = -ka * x[0];
            dx[1] = ka * x[0] - ke * x[1];
        },
        |p| {
            fetch_params!(p, _ka, _ke, tlag, _v);
            lag! {0=>tlag}
        },
        |_p, _t, _cov| fa! {},
        |_p, _t, _cov, _x| {},
        |x, p, _t, _cov, y| {
            fetch_params!(p, _ka, _ke, _tlag, v);
            // This equation specifies the output, e.g. the measured concentrations
            y[0] = x[1] / v;
        },
        (2, 1),
    );

    let op = ode.estimate_predictions(&subject, &vec![0.3, 0.5, 0.1, 70.0]);
    // println!("{op:#?}");
    let _ = op.run();
```

Analytic based model.

```rust
use pharmsol::*;
let analytical = equation::Analytical::new(
    one_compartment_with_absorption,
    |_p, _cov| {},
    |p| {
        fetch_params!(p, _ka, _ke, tlag, _v);
        lag! {0=>tlag}
    },
    |_p, _t, _cov| fa! {},
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

[Documentation](https://lapkb.github.io/pharmsol/pharmsol/)
[Benchmarks](https://lapkb.github.io/pharmsol/dev/bench/)
