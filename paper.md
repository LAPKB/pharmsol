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
  - name: Laboratory of Applied Pharmacokinetics and Bioinformatics, Childrens Hospital of Los Angeles, Los Angeles, USA
    index: 1
  - name: Department of Transplantation Medicine, Oslo University Hospital, Oslo, Norway
    index: 2
date: 14 June 2024
bibliography: paper.bib
---

# Summary

`pharmsol` is a modern, high-performance library for pharmacokinetic/pharmacodynamic (PK/PD) modeling and simulation written in Rust. It provides a flexible, declarative approach to defining, solving, and analyzing compartmental models. The library addresses significant performance challenges in PK/PD modeling through a systems-based architecture that leverages Rust's zero-cost abstractions and memory safety guarantees while maintaining an accessible interface for researchers and practitioners.


# Statement of Need

Pharmacokinetic and pharmacodynamic simulation faces increasing complexity as drug development workflows incorporate more sophisticated dosing regimens, mechanistic models, and individualized approaches. `pharmsol` addresses these challenges by providing a high-performance simulation library with three specialized backends (analytical, ODE, and SDE) for efficient execution of PK/PD simulations.

Unlike comprehensive pharmacometric platforms such as NONMEM [@beal1989nonmem], Phoenix NLME [@phoenixnlme], or Monolix [@monolix], `pharmsol` is purpose-built as a simulation engine that pharmacometricians and modelers can leverage to rapidly execute simulations for individuals or populations with pre- and user-defined models. While the library includes basic functionality for individual parameter fitting, its primary focus is on delivering a fully open-source solution that empowers users to inspect, modify, and extend the simulation capabilities without licensing constraints.

pharmsol addresses these challenges through three specialized backends:

- **Analytical Backend**: Provides closed-form solutions for standard compartmental models, offering maximum performance for common PK structures.
- **ODE Backend**: Enables simulation of custom differential equation systems with optimized numerical methods.
- **SDE Backend**: Supports stochastic differential equations for systems with inherent variability.

The library's open-source architecture (available at [https://github.com/LAPKB/pharmsol](https://github.com/LAPKB/pharmsol)) is designed for extensibility, allowing additional backends to be added as pharmacometric methods evolve. To our knowledge, pharmsol represents the first open-source pharmacometric simulator written in Rust.

# Architecture

`pharmsol` is built around three primary modules that work together to provide a comprehensive PK/PD modeling system: the data module, the equation module, and the simulator module. This modular design separates concerns while maintaining efficient interoperability between components.

## Data Module

The data module implements a hierarchical data structure that models the typical organization of pharmacometric data:

```
Data → Subject → Occasion → Event (Bolus, Infusion, Observation)
```

Data is a collection of subjects, which may have one or more occasions, i.e. pharmacokinetic investigations separated by time. Each occasion consists of one or more events, e.g. an instantaneous dose (bolus), infusions of drug, or observed concentrations at given times.

`pharmsol` also provides methods to read data in the Pmetrics data format. In the future, we also aim to provide parsers for all common data formats, such as those used by NONMEM, Monolix, and others.

## Equation Module

The equation module provides the mathematical foundation for representing PK/PD systems through two complementary approaches: Ordinary Differential Equations (ODEs) and analytical (closed-form) solutions.

### Analytical Solutions

For standard compartmental models, pharmsol provides optimized closed-form solutions:

- One-compartment models (with/without absorption)
- Two-compartment models (with/without absorption)
- Support for various administration routes and dosing patterns

These analytical solutions maintain the same interface as ODE-based models, allowing seamless interchangeability while providing significant computational advantages for supported model structures.


# Conclusion and Future Work


pharmsol aims to support the evolving needs of pharmacometric research by providing a modern, efficient platform that can adapt to the increasing complexity of pharmaceutical development while remaining accessible through its open-source licensing model.

# Acknowledgements

We acknowledge the intellectual ontributions to the package by members of the Laboratory of Applied Pharmacokinetics and Bioinformations (LAPKB), and feedback from the pharmacokinetics research group at the University of Oslo.

We are especially grateful to the authors of packages on which `pharmsol` relies, in particular Martin Robinson (diffsol), Sarah Quinones (faer), and Mossa Reimert (extendr).


# References