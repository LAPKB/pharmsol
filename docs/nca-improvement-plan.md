# pharmsol NCA Improvement Plan

## How pharmsol Currently Selects AUC Intervals — Code Examples

### 1. AUClast — Entire occasion profile, first observation to Tlast

The default and always-computed AUC. Covers the full observation profile within the occasion, from the first time point to the last positive concentration (`Tlast`).

```rust
use pharmsol::prelude::*;
use pharmsol::nca::NCA;

// Single occasion, single dose — AUClast covers [0h, 8h] (Tlast)
let subject = Subject::builder("pt1")
    .bolus(0.0, 100.0, 0)          // dose at t=0
    .observation(0.5, 5.0, 0)
    .observation(1.0, 10.0, 0)      // Cmax
    .observation(2.0, 8.0, 0)
    .observation(4.0, 4.0, 0)
    .observation(8.0, 2.0, 0)       // Tlast (last positive conc)
    .observation(12.0, 0.0, 0)      // BLQ — excluded from AUClast
    .build();

let result = subject.nca(&NCAOptions::default())?;
// result.exposure.auc_last covers integral from t=0.5 to t=8.0
// result.exposure.auc_inf_obs = auc_last + Clast/λz (if λz estimable)
```

**Internally** (`src/data/observation.rs` L272–L275):
```rust
pub fn auc_last(&self, method: &AUCMethod) -> Result<f64, ObservationError> {
    let end = self.tlast_idx + 1;  // index of last positive concentration
    auc::auc(&self.times[..end], &self.concentrations[..end], method)
}
```

### 2. Partial AUC — Single user-specified `[start, end]` window

Controlled by `NCAOptions::auc_interval`. Only one interval can be specified.

```rust
let opts = NCAOptions {
    auc_interval: Some((0.0, 4.0)),  // AUC from t=0 to t=4
    ..Default::default()
};
let result = subject.nca(&opts)?;
// result.exposure.auc_partial = Some(AUC over [0, 4])
```

**Internally** (`src/nca/analyze.rs` L178–L181):
```rust
let auc_partial = options
    .auc_interval
    .map(|(start, end)| profile.auc_interval(start, end, &options.auc_method))
    .transpose()?;
```

The `auc_interval` function (`src/data/auc.rs` L251) uses **linear interpolation only** at boundaries:
```rust
let c1 = if t1 < start {
    interpolate_linear(times, values, start)  // Always linear, even with LinUpLogDown
} else {
    values[i - 1]
};
```

### 3. AUCτ (Steady-state) — Fixed window `[0, tau]`

Controlled by `NCAOptions::tau`. Computes AUC over the dosing interval.

```rust
// Steady-state: multiple doses at q12h, analyze last dosing interval
let subject = Subject::builder("pt1")
    .bolus(0.0, 100.0, 0)
    .observation(0.5, 20.0, 0)    // ... observations in first interval ...
    .reset()                       // <-- occasion boundary
    .bolus(12.0, 100.0, 0)
    .observation(12.5, 22.0, 0)
    .observation(14.0, 15.0, 0)
    .observation(18.0, 6.0, 0)
    .observation(24.0, 2.0, 0)
    .build();

let opts = NCAOptions {
    tau: Some(12.0),  // dosing interval = 12h
    ..Default::default()
};

// Analyzes the FIRST occasion only (subject.nca)
// or ALL occasions (subject.nca_all)
let results = subject.nca_all(&opts);
// results[1].steady_state.auc_tau = AUC over [0, 12] within occasion 2
```

**Internally** (`src/nca/analyze.rs` L358):
```rust
let auc_tau = profile.auc_interval(0.0, tau, &options.auc_method)?;
```

> **Note:** Time within the occasion is absolute (not zeroed to dose time). If the dose is at t=12 and observations are at t=12.5, 14, 18, 24, the interval `[0, 12]` would miss all data. The user must know that time is absolute within the occasion.

### 4. Multi-dose intervals — Explicit `dose_times` Vec

Controlled by `NCAOptions::dose_times`. Must be manually provided — **not** extracted from actual dose events in the data.

```rust
// Single occasion with multiple doses (no .reset() — all in one occasion)
let subject = Subject::builder("pt1")
    .bolus(0.0, 100.0, 0)
    .observation(1.0, 10.0, 0)
    .observation(6.0, 3.0, 0)
    .bolus(8.0, 100.0, 0)           // second dose
    .observation(9.0, 12.0, 0)
    .observation(14.0, 4.0, 0)
    .bolus(16.0, 100.0, 0)          // third dose
    .observation(17.0, 11.0, 0)
    .observation(24.0, 2.0, 0)
    .build();

let opts = NCAOptions {
    dose_times: Some(vec![0.0, 8.0, 16.0]),  // Must be manually specified!
    ..Default::default()
};

let result = subject.nca(&opts)?;
// result.multi_dose.auc_intervals = [AUC[0,8], AUC[8,16], AUC[16,24]]
// result.multi_dose.cmax_intervals = [Cmax in [0,8], Cmax in [8,16], Cmax in [16,24]]
// result.multi_dose.tmax_intervals = [Tmax in [0,8], Tmax in [8,16], Tmax in [16,24]]
```

**Internally** (`src/nca/analyze.rs` L389–L409):
```rust
for i in 0..n {
    let start = sorted_times[i];
    let end = if i + 1 < n {
        sorted_times[i + 1]          // next dose time
    } else {
        last_obs_time                 // last observation for final interval
    };
    auc_intervals.push(profile.auc_interval(start, end, &options.auc_method)?);
}
```

**The main AUClast/AUCinf is still computed over the entire occasion**, regardless of `dose_times`. Multi-dose intervals are stored separately in `result.multi_dose`.

### 5. Multi-subject / multi-occasion — Per-occasion, parallel

```rust
let data = Data::builder(vec![subject_a, subject_b, subject_c])
    .build();

// Flat results: one per occasion across all subjects
let all: Vec<Result<NCAResult, _>> = data.nca_all(&NCAOptions::default());

// Grouped: results keyed by subject
let grouped: Vec<SubjectNCAResult> = data.nca_grouped(&NCAOptions::default());
// grouped[0].subject_id = "A"
// grouped[0].occasions = [Ok(result_occ1), Ok(result_occ2)]
```

Each occasion is **fully independent**. Dose info comes from the occasion's own events. Different subjects can have different numbers of occasions, different dose schedules, different observation times — all handled naturally because there is no cross-occasion or cross-subject data sharing.

### What pharmsol **cannot** do today

```rust
// ❌ Cannot specify multiple arbitrary intervals
let opts = NCAOptions {
    auc_interval: Some((0.0, 4.0)),   // only ONE interval allowed
    // No way to also request [0, 24] and [0, Inf] simultaneously
    ..Default::default()
};

// ❌ Cannot auto-detect dose intervals from data
let subject = Subject::builder("pt1")
    .bolus(0.0, 100.0, 0)
    .bolus(12.0, 100.0, 0)
    .observation(1.0, 10.0, 0)
    .observation(13.0, 10.0, 0)
    .build();
// The dose times [0, 12] are IN the data, but NCA won't use them
// unless you manually set dose_times: Some(vec![0.0, 12.0])

// ❌ Cannot request different parameters per interval
// (e.g., AUClast for [0,24] and AUCinf for [0,Inf])

// ❌ Cannot use log-linear interpolation at interval boundaries
// (always linear, even when AUCMethod is LinUpLogDown)

// ❌ No AUCall variant
// (AUClast ends at last positive conc; no extension to next BLQ point)
```

---

## Issues Found — Classification

### Bugs

Issues where current behavior produces incorrect or inconsistent results.

#### B1. Linear-only interpolation at interval boundaries (inconsistent with selected AUC method)

- **Location:** `src/data/auc.rs` L284–L291 (`auc_interval`)
- **Problem:** When `AUCMethod::LinUpLogDown` is selected, the integration segments use log-linear trapezoids for descending portions, but the concentrations at interval boundary points are **always** interpolated linearly via `interpolate_linear()`. This is internally inconsistent — the boundary concentration should be interpolated using the same method as the segments. On a descending log-linear portion, linear interpolation overestimates the concentration, producing an AUC that is too high at the boundary.
- **PKNCA reference:** PKNCA's `interp.extrap.conc()` uses log-linear interpolation when the method is "lin up/log down" and the profile is descending at that point.
- **Impact:** Moderate. Affects `auc_partial`, `auc_tau`, and all multi-dose interval AUCs whenever boundaries don't coincide with observed data points on a descending curve.

#### B2. AUCτ uses absolute time `[0, tau]` instead of `[dose_time, dose_time + tau]`

- **Location:** `src/nca/analyze.rs` L358
- **Problem:** `compute_steady_state` computes `profile.auc_interval(0.0, tau, ...)`, which assumes the observation profile starts at time 0 relative to dose. But when occasions contain events with absolute times (e.g., occasion 2 starting at t=12h with dose at t=12), the interval `[0, 12]` is empty — the data lives at `[12, 24]`. The profile's time array preserves the original absolute times from the occasion events.
- **Workaround:** Users must ensure occasions are constructed so that times are relative to the dose (i.e., each occasion starts at t=0). This is not documented and is easy to get wrong.
- **Impact:** High for multi-occasion subjects where occasion times are absolute.

#### B3. Multi-dose `dose_times` not auto-extracted from occasion events

- **Location:** `src/nca/analyze.rs` L370–L375
- **Problem:** The dose times for multi-dose interval computation come exclusively from `NCAOptions::dose_times`. The actual bolus/infusion events within the occasion are **ignored** for interval partitioning, even though they contain exactly the information needed. This means users must manually duplicate dose-schedule information that is already in the data — violating DRY and inviting mismatches.
- **Classification:** This is **both** a bug (if you consider that NCA on a multi-dose occasion should partition by doses) and a missing feature. Classified as a bug because the occasion's dose events are used for `total_dose()` and `route()` but silently ignored for interval partitioning, which is inconsistent.
- **Impact:** High. Without explicit `dose_times`, a multi-dose occasion computes AUClast over the entire profile, which is pharmacokinetically meaningless for multiple doses.

---

### Improvements

Features that are missing or could be enhanced, but where current behavior is not strictly wrong.

#### I1. Support multiple arbitrary intervals with per-interval parameter selection

- **Current:** `auc_interval: Option<(f64, f64)>` — single interval, all parameters always computed.
- **Desired:** `Vec<IntervalSpec>` where each entry has `(start, end)` and a set of parameters to compute. Intervals can overlap. Different parameters can be requested per interval (e.g., AUClast for `[0, 24]`, AUCinf for `[0, Inf]`).
- **PKNCA reference:** PKNCA uses a data-frame of intervals with boolean columns per parameter. This is the core of its flexibility.
- **Impact:** Major UX improvement. Eliminates the need for multiple NCA calls with different options.

#### I2. Auto-detect dosing intervals from data (like PKNCA's `choose.auc.intervals`)

- **Current:** Dose times must be explicitly provided in `NCAOptions::dose_times`.
- **Desired:** When `dose_times` is `None`, auto-extract dose event times from the occasion's bolus/infusion events and generate intervals from consecutive dose times. For single-dose occasions, use default intervals `[0, Tlast]` and `[0, Inf]`.
- **PKNCA reference:** `choose.auc.intervals()` + `find.tau()` auto-generates intervals from dose data, handling single-dose, multi-dose regular, and multi-dose irregular schedules.
- **Impact:** Major. Removes the most common source of user error.

#### I3. Auto-detect τ for steady-state analysis

- **Current:** `tau` must be manually specified in `NCAOptions`.
- **Desired:** When dose events are present, detect regular dosing intervals automatically (port PKNCA's `find.tau()` logic). If τ is detected and the subject has enough dose history, compute steady-state parameters automatically.
- **Impact:** Moderate. Reduces configuration burden.

#### I4. Add AUCall variant

- **Current:** Only AUClast (to last positive concentration) and AUCinf (extrapolated to infinity).
- **Desired:** AUCall = AUClast + linear trapezoid from Tlast to the first BLQ after Tlast. This is a commonly requested regulatory metric.
- **PKNCA reference:** `pk.calc.auc.all()` — uses the same trapezoidal rule but extends one step beyond Tlast to the next BLQ.
- **Impact:** Low-moderate. Straightforward to implement.

#### I5. Zero time to interval/dose start for reported parameters

- **Current:** Tmax, Tlast, etc. are reported in absolute time within the occasion.
- **Desired:** When computing NCA for an interval `[start, end]`, time-referenced parameters (Tmax, Tlast, Tfirst) should be reported relative to the interval start. E.g., for an interval `[144, 168]` (day 7), Tmax=146h should be reported as 2h.
- **PKNCA reference:** PKNCA explicitly zeroes time: `time - interval$start[1]` in `pk.nca.interval()`.
- **Impact:** High for interpretability in multi-dose/steady-state studies.

#### I6. Full-profile interpolation for `aucint`-style computation

- **Current:** Interval-boundary interpolation uses only the data within the interval (the occasion's observation profile).
- **Desired:** For steady-state τ intervals, allow using the full subject profile (all observations across all occasions) for interpolation/extrapolation at boundaries. This is important when no observation exists at exactly the boundary time.
- **PKNCA reference:** `aucint` functions receive `conc.group` / `time.group` (full group data) in addition to interval-filtered data.
- **Impact:** Low-moderate. Only matters for steady-state intervals where boundary interpolation needs λz extrapolation.

#### I7. Ka estimation (absorption rate constant)

- **Location:** Already noted as a TODO in `src/nca/analyze.rs` L9.
- **Desired:** Implement Wagner-Nelson or flip-flop detection for extravascular routes.
- **Impact:** Low. Nice-to-have for advanced NCA.

#### I8. Accumulation ratio computation

- **Current:** `SteadyStateParams::accumulation` is always `None` with a comment "Would need single-dose reference."
- **Desired:** When both single-dose and steady-state occasions exist for the same subject, compute accumulation ratio = AUCτ,ss / AUCτ,sd.
- **Impact:** Low-moderate. Requires cross-occasion data sharing, which is a design change.

---

## Priority & Implementation Order

| Priority | ID | Type | Effort | Description |
|----------|-----|------|--------|-------------|
| **P0** | B1 | Bug | Small | Fix interpolation at interval boundaries to match AUC method |
| **P0** | B2 | Bug | Small | Fix AUCτ to use dose-relative time (or document/enforce relative occasion times) |
| **P1** | B3 | Bug | Medium | Auto-extract dose times from occasion events for multi-dose intervals |
| **P1** | I2 | Improvement | Medium | Auto-generate intervals from dose data (choose.auc.intervals equivalent) |
| **P1** | I5 | Improvement | Small | Zero time to interval start for Tmax/Tlast/Tfirst |
| **P2** | I1 | Improvement | Large | Support multiple arbitrary intervals with per-interval parameter requests |
| **P2** | I4 | Improvement | Small | Add AUCall variant |
| **P2** | I3 | Improvement | Medium | Auto-detect τ from dose events |
| **P3** | I6 | Improvement | Medium | Full-profile interpolation for aucint-style computation |
| **P3** | I8 | Improvement | Medium | Cross-occasion accumulation ratio |
| **P3** | I7 | Improvement | Large | Ka estimation (Wagner-Nelson) |

---

## Implementation Notes

### B1 — Fix: Method-consistent interpolation

Add `interpolate_log_linear()` to `src/data/auc.rs` and use it in `auc_interval()` when the AUC method is `LinUpLogDown` and the profile is descending at the interpolation point. Fallback to linear when concentrations are ascending, zero, or equal.

### B2 — Fix: Dose-relative AUCτ

Two options:
1. **Detect dose time from occasion** and compute `[dose_time, dose_time + tau]` instead of `[0, tau]`. This is the cleaner fix.
2. **Normalize occasion times** so they start at 0 (the dose time). This is a larger change affecting occasion construction.

Option 1 is preferred — change `compute_steady_state` to accept the occasion's first dose time and offset the interval.

### B3 + I2 — Auto-detect dose intervals

When `dose_times` is `None`:
1. Extract bolus/infusion event times from the occasion.
2. If single dose → default intervals = `[dose_time, Tlast]` and `[dose_time, Inf]`.
3. If multiple doses → generate `[dose_i, dose_{i+1}]` intervals.
4. If regular spacing → set τ automatically.

This collapses B3, I2, and I3 into one coherent feature.

### I1 — Multiple arbitrary intervals

Replace:
```rust
pub auc_interval: Option<(f64, f64)>,
```
with:
```rust
pub intervals: Vec<IntervalSpec>,
```
where:
```rust
pub struct IntervalSpec {
    pub start: f64,
    pub end: f64,    // f64::INFINITY for AUCinf
    pub params: HashSet<NCAParam>,  // which parameters to compute
}
```

This is a breaking API change and should be versioned accordingly.
