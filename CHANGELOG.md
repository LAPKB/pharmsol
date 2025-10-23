# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.19.0](https://github.com/LAPKB/pharmsol/compare/v0.18.1...v0.19.0) - 2025-10-23

### Other

- Add method to get censor status for prediction ([#146](https://github.com/LAPKB/pharmsol/pull/146))

## [0.18.1](https://github.com/LAPKB/pharmsol/compare/v0.18.0...v0.18.1) - 2025-10-23

### Other

- diffsol 0.7

## [0.18.0](https://github.com/LAPKB/pharmsol/compare/v0.17.2...v0.18.0) - 2025-10-21

### Added

- Add support for censoring (LLOQ and ULOQ) ([#140](https://github.com/LAPKB/pharmsol/pull/140))
- add support to EXA
- Doses are now defined in the model
- Initial for the new censoring column in the Pmetrics format.

### Other

- Remove dbg statements
- set the local flag
- set the local flag
- debug
- Trying to handle the windows case correctly
- Force diffsol's version
- Bump nalgebra ([#144](https://github.com/LAPKB/pharmsol/pull/144))
- Were we are going, there are no maps... This commit removes mappings as their functionality is now replaced by having the model define the bolus inputs inside them
- if-else
- Exhaustive match instead of if-else block
- Add setters
- Make cens column optional
- Censoring likelihood calculation
- Update argmin requirement from 0.10.0 to 0.11.0 ([#135](https://github.com/LAPKB/pharmsol/pull/135))

## [0.17.2](https://github.com/LAPKB/pharmsol/compare/v0.17.1...v0.17.2) - 2025-09-30

### Added

- Implement into Data for Subject(s) ([#131](https://github.com/LAPKB/pharmsol/pull/131))

### Other

- moved the optim module into pharmsol, the idea is that the SppOptimizer and the EffectOptimizer both can be directly used in the model and because of this they belong in pharmsol
- Bump actions/upload-pages-artifact from 3 to 4 ([#124](https://github.com/LAPKB/pharmsol/pull/124))

## [0.17.1](https://github.com/LAPKB/pharmsol/compare/v0.17.0...v0.17.1) - 2025-09-06

### Added

- make mappings optional

### Fixed

- Add missing implementation for occasion on Prediction ([#126](https://github.com/LAPKB/pharmsol/pull/126))

## [0.17.0](https://github.com/LAPKB/pharmsol/compare/v0.16.0...v0.17.0) - 2025-09-03

### Added

- Adds the occasion index to events and predictions ([#123](https://github.com/LAPKB/pharmsol/pull/123))

## [0.16.0](https://github.com/LAPKB/pharmsol/compare/v0.15.0...v0.16.0) - 2025-08-14

### Added

- Update API for covariates ([#108](https://github.com/LAPKB/pharmsol/pull/108))
- Add mapping support to boluses ([#119](https://github.com/LAPKB/pharmsol/pull/119))

### Other

- Implicit lifetimes where elided ([#120](https://github.com/LAPKB/pharmsol/pull/120))
- Add helper function to check if error model should be optimized ([#122](https://github.com/LAPKB/pharmsol/pull/122))
- Bump actions/checkout from 4 to 5 ([#117](https://github.com/LAPKB/pharmsol/pull/117))

## [0.15.0](https://github.com/LAPKB/pharmsol/compare/v0.14.1...v0.14.2) - 2025-08-10

### Added

- Change how SDE predictions are calculated ([#115](https://github.com/LAPKB/pharmsol/pull/115))
- Update API for Data ([#102](https://github.com/LAPKB/pharmsol/pull/102))
- Support fixed and variable scalars for error models ([#106](https://github.com/LAPKB/pharmsol/pull/106))

### Fixed

- Fix test missing Some ([#114](https://github.com/LAPKB/pharmsol/pull/114))
- Handle NA in Pmetrics files ([#109](https://github.com/LAPKB/pharmsol/pull/109))

### Other

- Update diffsol to 0.6.5 ([#94](https://github.com/LAPKB/pharmsol/pull/94))
- Adding covariate benchmarks ([#107](https://github.com/LAPKB/pharmsol/pull/107))
- Update criterion and cached ([#105](https://github.com/LAPKB/pharmsol/pull/105))

## [0.14.1](https://github.com/LAPKB/pharmsol/compare/v0.14.0...v0.14.1) - 2025-07-25

### Other

- quiet-lloq
- Update CI for building and testing ([#97](https://github.com/LAPKB/pharmsol/pull/97))

## [0.14.0](https://github.com/LAPKB/pharmsol/compare/v0.13.1...v0.14.0) - 2025-07-21

### Other

- cleaning up
- renaming get_error_model to error_model
- some documentation
- error propagation
- blq->lloq
- blq->llq
- support for BLQ

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
