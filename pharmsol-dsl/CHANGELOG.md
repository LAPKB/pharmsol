# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- Breaking: add runtime-only `get_e2(u, v, alpha, h1, h2)` with exact arity five; the runtime computes `w = alpha * u * v` and compiler constants reject the call.

## [0.28.8](https://github.com/LAPKB/pharmsol/compare/pharmsol-dsl-v0.28.7...pharmsol-dsl-v0.28.8) - 2026-08-18

### Fixed

- numeric equality in DSL ([#342](https://github.com/LAPKB/pharmsol/pull/342))

## [0.28.7](https://github.com/LAPKB/pharmsol/compare/pharmsol-dsl-v0.28.6...pharmsol-dsl-v0.28.7) - 2026-08-12

### Other

- Remove WASM ([#335](https://github.com/LAPKB/pharmsol/pull/335))
- Feat/if braces ([#330](https://github.com/LAPKB/pharmsol/pull/330))

## [0.28.5](https://github.com/LAPKB/pharmsol/compare/pharmsol-dsl-v0.28.4...pharmsol-dsl-v0.28.5) - 2026-08-06

### Fixed

- Allow boluses and infusions to go to the same comparment ([#326](https://github.com/LAPKB/pharmsol/pull/326))

## [0.28.3](https://github.com/LAPKB/pharmsol/compare/pharmsol-dsl-v0.28.2...pharmsol-dsl-v0.28.3) - 2026-07-28

### Added

- Support time in the DSL ([#310](https://github.com/LAPKB/pharmsol/pull/310))
- Rename DSL types and add additional safeguards ([#302](https://github.com/LAPKB/pharmsol/pull/302))

## [0.28.0](https://github.com/LAPKB/pharmsol/compare/pharmsol-dsl-v0.27.1...pharmsol-dsl-v0.27.2) - 2026-07-07

### Fixed

- DSL tests ([#271](https://github.com/LAPKB/pharmsol/pull/271))

## [0.27.1](https://github.com/LAPKB/pharmsol/compare/pharmsol-dsl-v0.27.0...pharmsol-dsl-v0.27.1) - 2026-05-14

### Added

- Metadata contract ([#262](https://github.com/LAPKB/pharmsol/pull/262))

## [0.27.0](https://github.com/LAPKB/pharmsol/releases/tag/pharmsol-dsl-v0.27.0) - 2026-05-14

### Added

- Domain Specific Language (DSL) using JIT or AOT ([#252](https://github.com/LAPKB/pharmsol/pull/252))

### Other

- Use cargo workspaces ([#260](https://github.com/LAPKB/pharmsol/pull/260))
- Rename support points to parameters ([#251](https://github.com/LAPKB/pharmsol/pull/251))
- add missing macros ([#253](https://github.com/LAPKB/pharmsol/pull/253))
