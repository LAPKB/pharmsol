# Pharmsol

[![Build](https://github.com/LAPKB/pharmsol/actions/workflows/build.yml/badge.svg)](https://github.com/LAPKB/pharmsol/actions/workflows/build.yml)
[![Documentation](https://github.com/LAPKB/pharmsol/actions/workflows/docs.yml/badge.svg)](https://github.com/LAPKB/pharmsol/actions/workflows/docs.yml)
![crates.io](https://img.shields.io/crates/v/pharmsol.svg)

Solve pharmacokinetic models using differential equations and their analytical solutions!

## Example
ODE based model.
```Rust
Equation::new_ode(
    |x, p, _t, dx, rateiv, _cov| {
        fetch_cov!(cov, t,);
        fetch_params!(p, ka, ke, _tlag, _v);
        dx[0] = -ka * x[0];
        dx[1] = ka * x[0] - ke * x[1];
    },
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
```

Analytic based model.

```Rust
Equation::new_analytical(
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
```


## Supported analytical models
We are working to support all the standard analytical models. 

-  [x] One-compartment with IV infusion
-  [x] One-compartment with IV infusion and oral absorption
-  [x] Two-compartment with IV infusion
-  [x] Two-compartment with IV infusion and oral absorption
-  [ ] Three-compartmental models

 
