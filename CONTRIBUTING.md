# Contributing to PharmSol

Thank you for your interest in contributing to PharmSol! This document provides guidelines and instructions for contributing to the project.

## Code of Conduct

This project follows a Code of Conduct that all contributors are expected to adhere to. By participating, you are expected to uphold this code.

## How to Contribute

### Reporting Bugs

If you find a bug, please create an issue in the GitHub repository with the following information:

- A clear, descriptive title
- A detailed description of the bug
- Steps to reproduce the behavior
- Expected behavior
- Screenshots (if applicable)
- Environment details (OS, Rust version, etc.)

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion, include:

- A clear, descriptive title
- A detailed description of the proposed enhancement
- An explanation of why this enhancement would be useful
- Possible implementation details or approaches

### Code Contributions

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Implement your changes
4. Add tests for your changes
5. Ensure all tests pass (`cargo test`)
6. Format your code with rustfmt (`cargo fmt`)
7. Check your code with clippy (`cargo clippy`)
8. Commit your changes (`git commit -m 'Add some amazing feature'`)
9. Push to the branch (`git push origin feature/amazing-feature`)
10. Open a Pull Request

## Development Guidelines

### Code Style

PharmSol follows the standard Rust code style. Please ensure your code is formatted using `rustfmt`:

```bash
cargo fmt
```

And check for common issues using `clippy`:

```bash
cargo clippy --all-targets --all-features
```

### Documentation

- All public API functions, types, and traits should be documented with rustdoc comments
- Include examples in documentation when appropriate
- Document complex algorithms or non-obvious behavior

### Testing

- Write unit tests for new functionality
- Ensure all tests pass before submitting pull requests
- Consider adding performance tests for performance-critical code

### New Backends

PharmSol is designed to be extensible with new simulation backends. If you're contributing a new backend:

1. Implement the appropriate traits (`Equation`, `EquationTypes`, etc.)
2. Add comprehensive tests comparing with analytical solutions when possible
3. Include performance benchmarks
4. Document the mathematical approach and implementation details
5. Provide example usage in the documentation

## Extending PharmSol

### Adding New Models

To add new predefined models:

1. Implement the model in a new module or extend an existing one
2. Follow the same interface patterns as existing models
3. Add comprehensive tests
4. Document the model equations and assumptions

### Improving Performance

Performance optimizations are always welcome. When contributing performance improvements:

1. Include before/after benchmarks
2. Explain your approach and reasoning
3. Ensure correctness is maintained with tests

## Getting Help

If you have questions about contributing, please open an issue or reach out to the maintainers.

Thank you for contributing to PharmSol!
