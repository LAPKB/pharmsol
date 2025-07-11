# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.13.1](https://github.com/LAPKB/pharmsol/compare/v0.13.0...v0.13.1) - 2025-07-09

### Other

- hms
- suggestions by @mhovd
- remove indicatif
- remove indicatif
- remove indicatif, implemented a simple progress tracker that is compatible with R non-TTY terminal

## [0.13.0](https://github.com/LAPKB/pharmsol/compare/v0.12.0...v0.13.0) - 2025-07-03

### Added

- Added support for covariates on the fa, and lag blocks ([#87](https://github.com/LAPKB/pharmsol/pull/87))

## [0.12.0](https://github.com/LAPKB/pharmsol/compare/v0.11.1...v0.12.0) - 2025-06-25

### Fixed

- This commit properly handles multiple events at the sime time, avoiding an error by the ODE solver

### Other

- Improve error handling

## [0.11.1](https://github.com/LAPKB/pharmsol/compare/v0.11.0...v0.11.1) - 2025-06-19

### Fixed

- Infusions were not being handled correctly by the analytic solver. The old approach made a big jump without taking into account the discontinuities introduced by the infusions. This new approach detects the points in time where those discontinuities happen and solve the analytic model step by step.

## [0.11.0](https://github.com/LAPKB/pharmsol/compare/v0.10.0...v0.11.0) - 2025-06-17

### Added

- env variable PHARMSOL_LOCAL_EXA controls the execution of EXA via a remote or local phamrsol dependency. Thanks to @mhovd
- Exa now supports Analytical equations too

### Fixed

- params are fetched as mut
- fix, EqnKind is not behind exa anymore

### Other

- Apply suggestions from code review

## [0.10.0](https://github.com/LAPKB/pharmsol/compare/v0.9.1...v0.10.0) - 2025-06-11

### Added

- Add events to occasions
- Expose functions for number of states and outeqs
- Add support for multiple error models  ([#65](https://github.com/LAPKB/pharmsol/pull/65))

## [0.9.1](https://github.com/LAPKB/pharmsol/compare/v0.9.0...v0.9.1) - 2025-05-22

### Other

- Update README.md

## [0.9.0](https://github.com/LAPKB/pharmsol/compare/v0.8.0...v0.9.0) - 2025-05-19

### Added

- Improve error handling ([#58](https://github.com/LAPKB/pharmsol/pull/58))

### Other

- Update criterion requirement from 0.5.1 to 0.6.0 ([#62](https://github.com/LAPKB/pharmsol/pull/62))
- Benchmark with self-hosted runners ([#60](https://github.com/LAPKB/pharmsol/pull/60))

## [0.7.8](https://github.com/LAPKB/pharmsol/compare/v0.7.7...v0.7.8) - 2025-03-11

### Fixed

- fix infusions for SDEs ([#27](https://github.com/LAPKB/pharmsol/pull/27))

### Other

- Update cached requirement from 0.54.0 to 0.55.1 ([#29](https://github.com/LAPKB/pharmsol/pull/29))
- Bump peaceiris/actions-gh-pages from 3 to 4 ([#33](https://github.com/LAPKB/pharmsol/pull/33))
- Bump actions/cache from 2 to 4 ([#31](https://github.com/LAPKB/pharmsol/pull/31))
- Bump actions/checkout from 2 to 4 ([#32](https://github.com/LAPKB/pharmsol/pull/32))
- Allow dependabot to update github actions ([#25](https://github.com/LAPKB/pharmsol/pull/25))

## [0.7.7](https://github.com/LAPKB/pharmsol/compare/v0.7.6...v0.7.7) - 2025-02-23

### Other

- CI for releases ([#23](https://github.com/LAPKB/pharmsol/pull/23))
- Add ODE benchmarks ([#19](https://github.com/LAPKB/pharmsol/pull/19))
- Update ndarray requirement from 0.15.6 to 0.16.1 ([#13](https://github.com/LAPKB/pharmsol/pull/13))
