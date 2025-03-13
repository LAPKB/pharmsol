# Contributing
Thank you for your interest in contributing to `pharmsol`! This document provides guidelines and instructions for contributing to the project.

## Code of Conduct

Contributions to the project are expected to follow the [Rust Code of Conduct](https://www.rust-lang.org/policies/code-of-conduct).

## How to contribute

As an open-source project, we welcome contributions from everyone adhering to the code of conduct. Contributions can take many forms, including bug reports, feature requests, documentation improvements, and code contributions.

When reporting a bug or requesting a feature, please be as detailed as possible. Include information about the environment, the expected behavior, and the actual behavior. If possible, include a minimal code example that reproduces the issue.

Code contributions should be submitted as pull requests, and may be submitted for review at any time. Please ensure that code is well-documented, tested, and formatted. The project follows the standard Rust code style. Please ensure your code is formatted using `rustfmt`.

## Implementing a new backend

`pharmsol` is designed to be extensible with new simulation backends.

1. Implement the appropriate traits (`Equation`, `EquationTypes`, etc.)
2. Add comprehensive tests comparing with analytical solutions when possible
3. Include performance benchmarks
4. Document the mathematical approach and implementation details
5. Provide example usage in the documentation

## License

`pharmsol` is licensed under the GPL-3.0 license. By contributing to the project, you agree to license your contributions under the same license.

## Getting Help

If you have any questions about contributing, please open an issue or reach out to the maintainers.