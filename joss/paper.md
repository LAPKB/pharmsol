---
title: "pharmsol: A high-performance Rust library for pharmacokinetic/pharmacodynamic modeling and simulation"
tags:
  - Rust
  - pharmacokinetics
  - pharmacodynamics
  - modeling
  - simulation
  - compartmental models
authors:
  - name: Julián D. Otálvaro
    orcid: 0000-0001-5202-1645
    affiliation: 1
  - name: Markus Hovd
    orcid: 0000-0002-6077-0934
    affiliation: 2
  - name: Walter M. Yamada
    orcid: 0000-0003-3512-9202
    affiliation: 1
  - name: Michael Neely
    orcid: 0000-0002-1675-8276
    affiliation: 1
affiliations:
  - name: Laboratory of Applied Pharmacokinetics and Bioinformatics, Children's Hospital of Los Angeles, Los Angeles, USA
    index: 1
  - name: Department of Transplantation Medicine, Oslo University Hospital, Oslo, Norway
    index: 2
date: 12 December 2025
bibliography: paper.bib
---

# Summary

`pharmsol` is a library for pharmacokinetic/pharmacodynamic (PK/PD) simulation written in Rust. It provides the necessary tools and frameworks for defining, solving, and analyzing compartmental models, with support for differential equations, their analytical solutions, and experimental support for stochastic differential equations. Written in Rust, the library aims to provide researchers and developers with a framework for pharmacokinetic simulation in a memory-safe and performant language. The library is distributed via crates.io with comprehensive API documentation, usage examples, and a test suite validated through continuous integration.

# Statement of Need

Pharmacokinetic and pharmacodynamic modeling and simulation are computationally intense when applied to modern, complex, and sophisticated dosing regimens, mechanistic models, and individualized approaches. Unlike comprehensive pharmacometric platforms such as NONMEM [@nonmem], Phoenix NLME [@phoenix], or Monolix [@monolix], `pharmsol` is purpose-built as a simulation engine that pharmacometricians can leverage to rapidly execute simulations for individuals or populations with pre- and user-defined models.

As a fully open-source solution, `pharmsol` empowers users to inspect, modify, and extend the simulation capabilities without licensing constraints. Users can define custom models by specifying their own differential equations as closures, or use the provided analytical solutions for standard compartmental models. Additionally, `pharmsol` can be integrated in more user-friendly languages such as R using `extendr` [@extendr], making it accessible to pharmacometricians who may prefer higher-level interfaces.

# Data format

`pharmsol` is designed around a hierarchical data structure that models the typical organization of pharmacometric data. The primary data struct, `Data`, is a collection of `Subject`s, which may have one or more `Occasion`s, i.e. separate pharmacokinetic investigations. Each occasion consists of one or more `Event`s, e.g. an instantaneous dose (bolus), infusions of drug, or observed concentrations at given times.

```text
Data → Subject → Occasion → Event (Bolus, Infusion, Observation)
```

Currently, `pharmsol` provides methods to parse the Pmetrics [@pmetrics] data format. In the future, we aim to also support additional formats, such as those used by NONMEM, Monolix [@monolix], and more.

# Supported equation formats

The equation module provides the mathematical foundation for simulating PK/PD output with three model equation solver types: analytical solutions, ordinary differential equations, and experimental support for stochastic differential equations.

## Analytical Solutions

For standard compartmental models, `pharmsol` provides closed-form solutions for one- and two-compartment models, with and without oral absorption. These have been verified against their differential equation counterparts. Benchmarks demonstrate 20-33× speedups compared to equivalent ODE formulations without loss of precision (see repository benchmarks for details). Additional analytical solutions will be added in future versions.

## Ordinary Differential Equations

For more complex or non-standard models, `pharmsol` supports user-defined ordinary differential equations (ODEs). The numerical integration is performed using the `diffsol` library [@diffsol], which provides efficient BDF solvers suitable for the stiff systems often encountered in pharmacometric modeling.

## Stochastic Differential Equations

Experimental support for stochastic differential equations (SDEs) is available using the Euler-Maruyama method. SDEs allow modeling of within-subject variability as a continuous stochastic process. However, particular care should be taken if applying SDEs in a non-parametric approach to population pharmacokinetic modeling, such as when using the non-parametric adaptive grid algorithm (NPAG) [@npag] for parameter estimation.

# Conclusion and Future Work

`pharmsol` aims to support the evolving needs of pharmacometric research by providing a modern, efficient platform that can adapt to the increasing complexity of pharmaceutical development while remaining accessible through its open-source licensing model. Future development will focus on additional analytical model implementations, support for common data formats used by other pharmacometric software, non-compartmental analysis and continued performance improvements.

# Acknowledgements

We acknowledge the intellectual contributions to the package by members of the Laboratory of Applied Pharmacokinetics and Bioinformatics (LAPKB), and feedback from the pharmacokinetics research group at the University of Oslo.

We are especially grateful to the authors of the packages on which `pharmsol` relies, in particular Martin Robinson (diffsol), Sarah Quinones (faer), and Mossa Reimert (extendr). Their help and discussions are much appreciated.

# References
