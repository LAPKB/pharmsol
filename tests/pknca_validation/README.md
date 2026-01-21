# PKNCA Cross-Validation Framework

This framework validates pharmsol's NCA implementation against PKNCA (the gold-standard R package) using a **clean-room approach**:

1. **Test cases are independently designed** based on pharmacokinetic principles
2. **PKNCA serves as an oracle** - we run it to get expected values
3. **pharmsol results are compared** against these expected values

## Directory Structure

```
tests/pknca_validation/
├── README.md                    # This file
├── generate_expected.R          # R script to run PKNCA and save expected values
├── expected_values.json         # Generated expected outputs from PKNCA
├── test_scenarios.json          # Test case definitions (inputs)
└── validation_tests.rs          # Rust tests that compare pharmsol vs expected
```

## Usage

### Step 1: Generate Expected Values (requires R + PKNCA)

```bash
cd tests/pknca_validation
Rscript generate_expected.R
```

This creates `expected_values.json` with PKNCA's outputs.

### Step 2: Run Validation Tests

```bash
cargo test pknca_validation
```

## Test Scenarios

Test cases are designed to cover:

| Category         | Scenarios                                             |
| ---------------- | ----------------------------------------------------- |
| **Basic PK**     | Single-dose oral, IV bolus, IV infusion               |
| **AUC Methods**  | Linear, lin-up/log-down, lin-log                      |
| **Lambda-z**     | Various terminal phase slopes, different point counts |
| **BLQ Handling** | Zero, LOQ/2, exclude, positional                      |
| **C0 Methods**   | Back-extrapolation, observed, first conc              |
| **Edge Cases**   | Sparse data, flat profiles, noisy data                |

## Validation Results

**Current Status: 100% match (194/194 parameters)**

| Metric                       | Value          |
| ---------------------------- | -------------- |
| Exact matches                | 194/194 (100%) |
| Known convention differences | 0              |
| Unexpected failures          | 0              |

All NCA parameters computed by pharmsol match PKNCA v0.12.1 exactly.

## Legal Note

This framework does NOT copy PKNCA code or tests. Test scenarios are independently
designed based on pharmacokinetic theory. PKNCA is used only as a reference
implementation to validate numerical accuracy.
