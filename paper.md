---
title: "pharmsol: A high-performance Rust library for pharmacokinetic/pharmacodynamic modeling and simulation"
tags:
  - Rust
  - pharmacokinetics
  - pharmacodynamics
  - modeling
  - simulation
  - ODE
  - compartmental models
authors:
  - name: Julián D. Otálvaro
    orcid: 0000-0000-0000-0000
    affiliation: 1
  - name: Markus Hovd
    orcid: 0000-0002-6077-0934
    affiliation: 2
affiliations:
  - name: LAPKB
    index: 1
  - name: Department of Transplantation Medicine, Oslo University Hospital
    index: 2
date: 14 June 2024
bibliography: paper.bib
---

# Summary

pharmsol is a modern, high-performance library for pharmacokinetic/pharmacodynamic (PK/PD) modeling and simulation implemented in Rust. It provides a flexible, declarative approach to defining, solving, and analyzing compartmental models through a carefully designed API that balances expressivity with computational efficiency. The library addresses significant performance challenges in PK/PD modeling through a systems-based architecture that leverages Rust's zero-cost abstractions and memory safety guarantees while maintaining an accessible interface for researchers and practitioners.

The design of pharmsol emphasizes three core components: (1) a hierarchical data model for representing subject data, dosing regimens, and time-varying covariates; (2) a flexible equation system supporting numerical ODE and SDE-based solutions and closed-form analytical implementations of standard compartmental models; and (3) an efficient simulation framework that optimizes computational resources through strategic memory management and parallelization. This architecture enables rapid specification and simulation of complex dosing scenarios, sophisticated handling of inter-individual variability, and efficient solutions for both standard and custom PK/PD models.

pharmsol serves as a foundation for building robust pharmacometric workflows, offering a significant performance advantage over existing tools while making advanced modeling techniques accessible to researchers across the pharmaceutical development pipeline.

# Statement of Need

Pharmacokinetic and pharmacodynamic simulation faces increasing complexity as drug development workflows incorporate more sophisticated dosing regimens, mechanistic models, and individualized approaches. **pharmsol** addresses these challenges by providing a high-performance simulation library with three specialized backends (analytical, ODE, and SDE) for efficient execution of PK/PD simulations.

Unlike comprehensive pharmacometric platforms such as NONMEM [@beal1989nonmem], Phoenix NLME [@phoenixnlme], or Monolix [@monolix], pharmsol is purpose-built as a simulation engine that pharmacometricians and modelers can leverage to rapidly execute simulations for individuals or populations with pre- and user defined models. While the library includes basic functionality for individual parameter fitting, its primary focus is on delivering a fully open-source solution that empowers users to inspect, modify, and extend the simulation capabilities without licensing constraints.

This focused approach addresses several critical gaps in the current ecosystem:

1. **Open-Source Accessibility**  
   Many established simulation tools are proprietary and require expensive licenses, which can hinder accessibility, reproducibility, and community-driven innovation. By being fully open source, pharmsol enables researchers and developers worldwide—including those in resource‐limited environments—to freely use, audit, and improve the simulation engine, thereby promoting transparency and reproducibility in pharmacometrics.

2. **Memory safety and security**  
   Most established simulation tools rely on legacy codebases written in languages that lack memory safety guarantees. According to security agencies, memory safety issues account for approximately 70% of software vulnerabilities [@cisa2023memorysafe; @nsa2022adopting]. As pharmaceutical simulations increasingly process sensitive patient data and support critical decision-making, these security considerations become essential.

3. **Balancing Model Expressivity, Efficiency and Accuracy**  
   Current simulation approaches frequently impose unnecessary tradeoffs between model expressivity and computational efficiency. Custom mechanisms in R-based tools like nlmixr [@fidler2019nlmixr] provide flexibility but incur significant performance penalties, while faster tools often constrain model structures to predefined templates [mentre2005prediction].

pharmsol addresses these challenges through three specialized backends:

- **Analytical Backend**: Provides closed-form solutions for standard compartmental models, offering maximum performance for common PK structures.
- **ODE Backend**: Enables simulation of custom differential equation systems with optimized numerical methods.
- **SDE Backend**: Supports stochastic differential equations for systems with inherent variability.

The library's open-source architecture (available at [https://github.com/LAPKB/pharmsol](https://github.com/LAPKB/pharmsol)) is designed for extensibility, allowing additional backends to be added as pharmacometric methods evolve. To our knowledge, pharmsol represents the first open-source, high-performance, memory-safe simulation library specifically tailored for pharmacology applications.

Additional technical capabilities address specific simulation challenges:

- **Efficient Time-Varying Covariate Handling**: pharmsol implements an optimized approach to incorporating time-dependent patient characteristics within simulation workflows [@svensson2016topics].
- **Complex Dosing Regimen Support**: The library provides an intuitive interface for specifying adaptive dosing, combination therapies, or complex administration patterns [@mager2001target; @ribba2017methodologies].
- **Parallel Simulation Capabilities**: pharmsol automatically leverages multi-core architectures without requiring custom code development [@elmokadem2019parallel].

By providing these capabilities in a memory-safe language with performance comparable to C/C++, pharmsol fills a critical gap in the pharmacometric software ecosystem—enabling researchers to rapidly simulate complex models without compromising security or computational efficiency, while interfacing with existing modeling tools for parameter estimation and population analysis.

# Implementation and Architecture

pharmsol is built around three primary modules that work together to provide a comprehensive PK/PD modeling system: the data module, the equation module, and the simulator module. This modular design separates concerns while maintaining efficient interoperability between components.

## Data Module

The data module implements a hierarchical data structure that models the typical organization of pharmacometric data:

```
Data → Subject → Occasion → Event (Bolus, Infusion, Observation)
```

### Subject and Event Structure

At the core of the data module are strongly-typed event structures representing different occurrences in a PK/PD study:

- **Bolus**: Instantaneous drug administration, characterized by time, amount, and input compartment
- **Infusion**: Continuous administration over time, including duration and rate information
- **Observation**: Measurements of drug concentration or effects, including time, value, and output equation

This structured approach enables type safety throughout the simulation pipeline while maintaining an intuitive conceptual model that maps directly to clinical study designs.

### Covariate System

The covariate system handles time-varying subject characteristics through a flexible piecewise function approach:

- Supports both linear interpolation between timepoints and constant (carry-forward) values
- Memory-efficient sparse representation storing only change points rather than dense time series
- On-demand interpolation during simulation to minimize computational overhead

### Builder Pattern

A fluent interface implemented through the builder pattern provides an intuitive way to construct complex dosing regimens:

```rust
let subject = Subject::builder("patient_001")
    .bolus(0.0, 100.0, 0)       // 100 mg oral dose at time 0
    .repeat(6, 24.0)            // Repeat daily for 6 more days
    .observation(1.0, 5.2, 0)   // Blood sample at 1 hour
    .covariate("weight", 0.0, 70.0)
    .covariate("weight", 24.0, 72.0)  // Linear interpolation between
    .build();
```

This approach reduces code verbosity while maintaining the expressivity needed for complex clinical scenarios.

## Equation Module

The equation module provides the mathematical foundation for representing PK/PD systems through two complementary approaches: Ordinary Differential Equations (ODEs) and analytical (closed-form) solutions.

### ODE Implementation

The ODE component allows specification of arbitrary differential equation systems through a callback-based interface:

```rust
let ode = equation::ODE::new(
    |x, p, t, dx, _rateiv, cov| {  // State derivative function
        fetch_cov!(cov, t, weight);
        fetch_params!(p, ka, ke, v);
        dx[0] = -ka * x[0];
        dx[1] = ka * x[0] - ke * x[1];
    },
    // Additional components for lag time, bioavailability, etc.
    |x, p, _t, _cov, y| {  // Output transformation
        fetch_params!(p, _ka, _ke, v);
        y[0] = x[1] / v;  // Convert amount to concentration
    },
    (2, 1),  // 2 states, 1 output
);
```

This approach leverages Rust's zero-cost abstractions to maintain high performance despite the flexibility of the callback interface.

### Analytical Solutions

For standard compartmental models, pharmsol provides optimized closed-form solutions:

- One-compartment models (with/without absorption)
- Two-compartment models (with/without absorption)
- Support for various administration routes and dosing patterns

These analytical solutions maintain the same interface as ODE-based models, allowing seamless interchangeability while providing significant computational advantages for supported model structures.

### Parameter Handling

Access to model parameters and covariates is simplified through a macro system that provides syntactic sugar without runtime overhead:

- `fetch_params!`: Extracts named parameters from parameter vectors
- `fetch_cov!`: Retrieves and interpolates covariates at specific time points
- `lag!`: Defines absorption lag times for different compartments
- `fa!`: Specifies bioavailability factors

## Simulator Module

The simulator module provides the computational engine that applies mathematical models to subject data to generate predictions.

### Unified Interface

A trait-based approach provides a common interface for different solving methods:

- `Equation` trait defines the public API for simulation and prediction
- `EquationTypes` handles associated types for state vectors and prediction outputs
- `EquationPriv` encapsulates implementation details

This design enables code that works with any equation type, whether ODE-based, analytical, or stochastic.

### Simulation Workflow

The core simulation process follows a systematic approach:

1. Initialize the model state for each subject occasion
2. Process events chronologically (dosing and observations)
3. Apply the mathematical model to update state between events
4. Generate predictions at observation time points

This event-driven approach efficiently handles complex dosing regimens and sparse observation patterns.

### Performance Optimizations

Multiple optimization strategies ensure pharmsol delivers high performance:

- **Memoization and Caching**: Simulation results are cached and reused when parameters remain constant
- **Numerical Algorithm Selection**: Appropriate solvers are selected based on system characteristics, with BDF methods for stiff systems
- **Memory Management**: Pre-allocation and reuse of buffers minimizes allocations in critical paths
- **Concurrency**: Thread-safe components enable parallel simulation of multiple subjects

## Error Handling and Validation

pharmsol implements comprehensive error handling through Rust's type system:

- Detailed error types provide context for debugging and diagnostics
- Validation of inputs ensures consistent behavior and early failure
- Numerical stability checks prevent common computational issues

The library includes an extensive test suite that verifies numerical accuracy against analytical solutions and reference implementations.

# Performance Evaluation

pharmsol demonstrates significant performance advantages over comparable tools, particularly for complex models and large datasets. Internal benchmarks show speedups of 10-100x compared to equivalent R implementations for typical PK/PD workflows.

Key performance characteristics include:

- Linear scaling with the number of subjects through parallel processing
- Efficient handling of stiff systems common in PK/PD modeling
- Minimal memory overhead during simulation
- Fast caching strategy for repeated evaluations with similar parameters

These performance benefits are particularly impactful for computationally intensive tasks such as population modeling, Bayesian approaches, and simulation-based trial design.

# Example Usage

The following example demonstrates a complete workflow for simulating a two-compartment PK model with first-order absorption:

```rust
use pharmsol::*;

// Define subject with dosing, observations, and covariates
let subject = Subject::builder("id1")
    .bolus(0.0, 100.0, 0)           // 100 mg oral dose at time 0
    .observation(0.5, 0.1, 0)       // Observation at 0.5 hours
    .observation(1.0, 0.4, 0)       // Observation at 1 hour
    .observation(2.0, 1.0, 0)       // Observation at 2 hours
    .covariate("weight", 0.0, 80.0) // Subject weighs 80 kg
    .build();

// Define two-compartment model with first-order absorption
let ode = equation::ODE::new(
    |x, p, _t, dx, _rateiv, _cov| {
        fetch_params!(p, ka, ke, k12, k21, _tlag, _v);

        // First-order absorption
        dx[0] = -ka * x[0];

        // Central compartment
        dx[1] = ka * x[0] - (ke + k12) * x[1] + k21 * x[2];

        // Peripheral compartment
        dx[2] = k12 * x[1] - k21 * x[2];
    },
    |p| {
        fetch_params!(p, _ka, _ke, _k12, _k21, tlag, _v);
        lag! {0=>tlag}  // Apply lag time to absorption compartment
    },
    |_p| fa! {},  // Default bioavailability
    |_p, _t, _cov, _x| {},  // No special initialization
    |x, p, _t, _cov, y| {
        fetch_params!(p, _ka, _ke, _k12, _k21, _tlag, v);
        y[0] = x[1] / v;  // Convert amount to concentration
    },
    (3, 1),  // 3 states, 1 output
);

// Parameter values: ka, ke, k12, k21, tlag, v
let params = vec![0.8, 0.15, 0.05, 0.03, 0.1, 70.0];

// Generate predictions
let predictions = ode.estimate_predictions(&subject, &params);

// Access the results
for pred in predictions.flat_predictions() {
    println!("Time: {}, Concentration: {}", pred.time(), pred.prediction());
}
```

This example demonstrates the declarative style enabled by pharmsol, where complex PK models can be concisely expressed while maintaining readability and performance.

# Conclusion and Future Work

pharmsol provides a robust foundation for pharmacokinetic and pharmacodynamic modeling, emphasizing performance, flexibility, and usability. By leveraging Rust's systems programming capabilities, the library achieves computational efficiency without sacrificing expressive power or safety. As an open-source project (https://github.com/LAPKB/pharmsol), pharmsol welcomes community contributions to expand its capabilities and application domains.

Future development priorities include:

1. Expanded analytical solutions for additional compartmental models
2. Enhanced parameter estimation capabilities using gradient-based methods
3. Integration with Bayesian workflows for uncertainty quantification
4. Additional import/export capabilities for interoperability with established tools
5. Development of higher-level abstractions for common modeling patterns
6. New specialized backends for emerging computational approaches

The library's flexible, extensible architecture is specifically designed to accommodate new simulation backends as computational techniques evolve. While pharmsol currently focuses on simulation rather than comprehensive population modeling, its foundations provide the necessary building blocks for researchers who wish to implement such capabilities in the future.

pharmsol aims to support the evolving needs of pharmacometric research by providing a modern, efficient platform that can adapt to the increasing complexity of pharmaceutical development while remaining accessible through its open-source licensing model.

# Acknowledgements

We acknowledge contributions from the pharmacometrics community and the support of the Laboratory for Applied PK/PD Modeling and Bayesian Analytics. We also thank the Rust ecosystem developers whose libraries form the foundation of pharmsol's numerical capabilities.

# References
