use serde::{Deserialize, Deserializer, Serialize};
use std::{
    collections::{BTreeMap, HashMap},
    fmt,
    sync::atomic::{AtomicU64, Ordering},
};
use thiserror::Error;

const NO_LEFT_CONTINUITY_TIME: u64 = u64::MAX;

/// Error type for covariate operations
#[derive(Error, Debug, Clone, Serialize, Deserialize)]
pub enum CovariateError {
    #[error("No segments available for interpolation")]
    MissingSegments,
    #[error(
        "Covariate `{name}` has a non-finite ODE observation: time = {time:?}, value = {value:?}"
    )]
    NonFiniteObservation { name: String, time: f64, value: f64 },
    #[error("Covariate `{name}` has duplicate ODE observations at time {time:?}")]
    DuplicateObservation { name: String, time: f64 },
}

/// Method used to interpolate covariate values between observations
#[derive(Serialize, Clone, Debug, Deserialize)]
pub enum Interpolation {
    /// Linear interpolation between two points with slope and intercept
    Linear { slope: f64, intercept: f64 },
    /// Constant value carried forward
    CarryForward { value: f64 },
}

/// A segment of a piecewise interpolation function for a covariate
///
/// Each segment defines how to interpolate values within its time range.
#[derive(Clone, Debug)]
struct CovariateSegment {
    from: f64,
    to: Option<f64>,
    method: Interpolation,
}

impl CovariateSegment {
    /// Create a new covariate segment
    ///
    /// # Arguments
    ///
    /// * `from` - Start time of the segment
    /// * `to` - End time of the segment (None for unbounded)
    /// * `method` - Interpolation method to use within this segment
    pub(crate) fn new(from: f64, to: Option<f64>, method: Interpolation) -> Self {
        CovariateSegment { from, to, method }
    }

    /// Interpolate the covariate value at a specific time within this segment
    ///
    /// Returns None if the time is outside the segment's range.
    #[inline]
    fn interpolate(&self, time: f64) -> Option<f64> {
        if !self.in_interval(time) {
            return None;
        }

        match self.method {
            Interpolation::Linear { slope, intercept } => Some(slope * time + intercept),
            Interpolation::CarryForward { value } => Some(value),
        }
    }

    /// Evaluate the segment at its right endpoint without changing the public
    /// right-continuous interpolation rule.
    #[inline]
    fn interpolate_at_end(&self, time: f64) -> Option<f64> {
        if self.to != Some(time) {
            return None;
        }

        match self.method {
            Interpolation::Linear { slope, intercept } => Some(slope * time + intercept),
            Interpolation::CarryForward { value } => Some(value),
        }
    }

    /// Check if a given time is within this segment's interval
    #[inline]
    fn in_interval(&self, time: f64) -> bool {
        self.from <= time && self.to.is_none_or(|to| time < to)
    }
}

/// A time-varying covariate built from source observations.
///
/// Source observations are retained exactly. Interpolation segments are rebuilt
/// whenever those observations or the interpolation mode change.
#[derive(Serialize, Debug)]
pub struct Covariate {
    /// The name of the covariate
    name: String,
    /// Original time-value observations
    observations: Vec<(f64, f64)>,
    /// Segments representing the covariate's value over time
    #[serde(skip)]
    segments: Vec<CovariateSegment>,
    /// Flag to indicate if this covariate should always use carry-forward interpolation
    fixed: bool,
    /// Session-local boundary used to select the segment ending at an exact knot.
    ///
    /// This is deliberately atomic so public covariates remain `Send + Sync`.
    /// `Clone` resets it because continuity belongs to one solver session.
    #[serde(skip)]
    left_continuity_time: AtomicU64,
}

impl Clone for Covariate {
    fn clone(&self) -> Self {
        Self {
            name: self.name.clone(),
            observations: self.observations.clone(),
            segments: self.segments.clone(),
            fixed: self.fixed,
            left_continuity_time: AtomicU64::new(NO_LEFT_CONTINUITY_TIME),
        }
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CovariateData {
    name: String,
    observations: Vec<(f64, f64)>,
    fixed: bool,
}

impl<'de> Deserialize<'de> for Covariate {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let data = CovariateData::deserialize(deserializer)?;
        let mut observations = data.observations;
        for (time, value) in &observations {
            if !time.is_finite() || !value.is_finite() {
                return Err(serde::de::Error::custom(
                    "covariate observations must contain finite times and values",
                ));
            }
        }
        observations.sort_by(|left, right| left.0.total_cmp(&right.0));
        if let Some(duplicate) = observations.windows(2).find(|pair| pair[0].0 == pair[1].0) {
            return Err(serde::de::Error::custom(format!(
                "duplicate covariate observation at time {}",
                duplicate[0].0
            )));
        }

        let mut covariate = Self {
            name: data.name,
            observations,
            segments: Vec::new(),
            fixed: data.fixed,
            left_continuity_time: AtomicU64::new(NO_LEFT_CONTINUITY_TIME),
        };
        covariate.build_segments();
        Ok(covariate)
    }
}

impl Covariate {
    /// Create a new covariate with the given name
    ///
    /// # Arguments
    ///
    /// * `name` - The name of the covariate
    /// * `fixed` - Whether this covariate should use carry-forward interpolation
    pub fn new(name: String, fixed: bool) -> Self {
        Covariate {
            name,
            observations: Vec::new(),
            segments: Vec::new(),
            fixed,
            left_continuity_time: AtomicU64::new(NO_LEFT_CONTINUITY_TIME),
        }
    }

    /// Add an observation, updating an existing value at the same time.
    pub fn add_observation(&mut self, time: f64, value: f64) {
        if let Some(existing) = self
            .observations
            .iter_mut()
            .find(|observation| observation.0 == time)
        {
            existing.1 = value;
        } else {
            self.observations.push((time, value));
        }
        self.build_segments();
    }

    /// Update an observation, returning whether the time was present.
    pub fn update_observation(&mut self, time: f64, new_value: f64) -> bool {
        let Some(existing) = self
            .observations
            .iter_mut()
            .find(|observation| observation.0 == time)
        else {
            return false;
        };
        existing.1 = new_value;
        self.build_segments();
        true
    }

    /// Remove an observation at a specific time.
    pub fn remove_observation(&mut self, time: f64) -> bool {
        let initial_len = self.observations.len();
        self.observations
            .retain(|observation| observation.0 != time);
        if self.observations.len() == initial_len {
            false
        } else {
            self.build_segments();
            true
        }
    }

    /// Get all source observations as time-value pairs.
    pub fn observations(&self) -> Vec<(f64, f64)> {
        self.observations.clone()
    }

    /// Validate source observations before they are used by an ODE callback.
    pub(crate) fn validate_for_ode(&self) -> Result<(), CovariateError> {
        for &(time, value) in &self.observations {
            if !time.is_finite() || !value.is_finite() {
                return Err(CovariateError::NonFiniteObservation {
                    name: self.name.clone(),
                    time,
                    value,
                });
            }
        }

        if let Some(pair) = self
            .observations
            .windows(2)
            .find(|pair| pair[0].0 == pair[1].0)
        {
            return Err(CovariateError::DuplicateObservation {
                name: self.name.clone(),
                time: pair[0].0,
            });
        }

        Ok(())
    }

    /// Return exact ODE integration boundaries introduced by this covariate.
    ///
    /// LOCF contributes only observation times whose carried value changes.
    /// Linear interpolation contributes every observation knot, including the
    /// first and last knots, because endpoint clamping and slope changes can
    /// change the RHS derivative there. These are exact f64 times; no tolerance
    /// based deduplication is used.
    pub(crate) fn ode_breakpoint_times(&self) -> Result<Vec<f64>, CovariateError> {
        self.validate_for_ode()?;

        let mut breakpoints: Vec<f64> = if self.fixed {
            self.observations
                .windows(2)
                .filter(|pair| pair[0].1 != pair[1].1)
                .map(|pair| pair[1].0)
                .collect()
        } else {
            self.observations.iter().map(|&(time, _)| time).collect()
        };
        breakpoints.sort_by(f64::total_cmp);
        breakpoints.dedup();
        Ok(breakpoints)
    }

    /// Return exact knots where the covariate value itself changes at the
    /// right-hand side. Linear knots are intentionally absent: their values
    /// are continuous even when the time derivative changes, so they need an
    /// integration stop but not a state/RHS discontinuity restart.
    pub(crate) fn ode_discontinuity_times(&self) -> Result<Vec<f64>, CovariateError> {
        self.validate_for_ode()?;

        let mut discontinuities = if self.fixed {
            self.observations
                .windows(2)
                .filter(|pair| pair[0].1 != pair[1].1)
                .map(|pair| pair[1].0)
                .collect()
        } else {
            Vec::new()
        };
        discontinuities.sort_by(f64::total_cmp);
        discontinuities.dedup();
        Ok(discontinuities)
    }

    fn set_left_continuity_time(&self, time: Option<f64>) {
        let encoded = time.map_or(NO_LEFT_CONTINUITY_TIME, f64::to_bits);
        self.left_continuity_time.store(encoded, Ordering::Relaxed);
    }

    fn left_continuity_time(&self) -> Option<f64> {
        let encoded = self.left_continuity_time.load(Ordering::Relaxed);
        (encoded != NO_LEFT_CONTINUITY_TIME).then(|| f64::from_bits(encoded))
    }

    /// Rebuild interpolation segments from the source observations.
    fn build_segments(&mut self) {
        self.observations
            .sort_by(|left, right| left.0.total_cmp(&right.0));
        self.segments.clear();

        for (index, current) in self.observations.iter().enumerate() {
            let next = self.observations.get(index + 1);
            let end = next.map(|observation| observation.0);

            let method = if self.fixed {
                Interpolation::CarryForward { value: current.1 }
            } else if let Some(next) = next {
                let slope = (next.1 - current.1) / (next.0 - current.0);
                Interpolation::Linear {
                    slope,
                    intercept: current.1 - slope * current.0,
                }
            } else {
                Interpolation::CarryForward { value: current.1 }
            };
            self.segments
                .push(CovariateSegment::new(current.0, end, method));
        }
    }

    /// Interpolate between observations, carrying endpoint values outside their range.
    #[inline]
    pub fn interpolate(&self, time: f64) -> Result<f64, CovariateError> {
        if self.segments.is_empty() {
            return Err(CovariateError::MissingSegments);
        }

        if self.left_continuity_time() == Some(time) {
            if let Some(value) = self
                .segments
                .iter()
                .find_map(|segment| segment.interpolate_at_end(time))
            {
                return Ok(value);
            }
        }

        if let Some(value) = self
            .segments
            .iter()
            .find_map(|segment| segment.interpolate(time))
        {
            return Ok(value);
        }

        if let Some(first) = self.observations.first() {
            if time < first.0 {
                return Ok(first.1);
            }
        }
        if let Some(last) = self.observations.last() {
            if time >= last.0 {
                return Ok(last.1);
            }
        }

        Err(CovariateError::MissingSegments)
    }

    /// Get the name of the covariate
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Set the covariate as fixed (use carry-forward interpolation)
    ///
    /// This is useful when you want to treat a time-varying covariate as constant
    /// using carry-forward interpolation, which is common in pharmacokinetic modeling.
    pub fn set_fixed(&mut self, fixed: bool) {
        self.fixed = fixed;
        self.build_segments();
    }

    /// Check if this covariate is set to use carry-forward interpolation
    pub fn fixed(&self) -> bool {
        self.fixed
    }
}

impl fmt::Display for Covariate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Covariate '{}':", self.name)?;
        for (index, segment) in self.segments.iter().enumerate() {
            let to_str = segment.to.map_or("∞".to_string(), |t| format!("{:.2}", t));
            write!(
                f,
                "  Segment {}: from {:.2} to {}, ",
                index + 1,
                segment.from,
                to_str
            )?;
            match &segment.method {
                Interpolation::Linear { slope, intercept } => {
                    writeln!(
                        f,
                        "Linear, Slope: {:.2}, Intercept: {:.2}",
                        slope, intercept
                    )
                }
                Interpolation::CarryForward { value } => {
                    writeln!(f, "Carry Forward, Value: {:.2}", value)
                }
            }?;
        }
        Ok(())
    }
}

/// A collection of [Covariate]s
///
/// This struct provides methods to manage multiple covariates and retrieve
/// interpolated values for all covariates at specific time points.
#[derive(Serialize, Clone, Debug, Deserialize)]
pub struct Covariates {
    covariates: BTreeMap<String, Covariate>,
}

impl Default for Covariates {
    fn default() -> Self {
        Covariates::new()
    }
}

impl Covariates {
    /// Create a new empty collection of covariates
    pub fn new() -> Self {
        Covariates {
            covariates: BTreeMap::new(),
        }
    }

    /// Create covariates from Pmetrics raw observations
    pub(crate) fn from_row_observations(
        raw_observations: &HashMap<String, Vec<(f64, f64)>>,
    ) -> Self {
        let mut covariates = Covariates::new();

        for (key, observations) in raw_observations {
            let (name, fixed) = key
                .strip_suffix('!')
                .map_or_else(|| (key.as_str(), false), |name| (name, true));
            let mut covariate = Covariate::new(name.to_string(), fixed);
            for &(time, value) in observations {
                covariate.add_observation(time, value);
            }
            covariates.add_covariate(name.to_string(), covariate);
        }

        covariates
    }

    /// Get all covariates in this collection
    pub fn covariates(&self) -> HashMap<String, &Covariate> {
        self.covariates
            .iter()
            .map(|(k, v)| (k.clone(), v))
            .collect()
    }

    /// Validate all source observations before ODE construction or callbacks.
    pub(crate) fn validate_for_ode(&self) -> Result<(), CovariateError> {
        for covariate in self.covariates.values() {
            covariate.validate_for_ode()?;
        }
        Ok(())
    }

    /// Collect exact covariate discontinuity and derivative-knot times.
    pub(crate) fn ode_breakpoint_times(&self) -> Result<Vec<f64>, CovariateError> {
        let mut breakpoints = Vec::new();
        for covariate in self.covariates.values() {
            breakpoints.extend(covariate.ode_breakpoint_times()?);
        }
        breakpoints.sort_by(f64::total_cmp);
        breakpoints.dedup();
        Ok(breakpoints)
    }

    /// Collect exact covariate value-change times that require a solver
    /// history/Jacobian restart. Linear knots are integration boundaries only.
    pub(crate) fn ode_discontinuity_times(&self) -> Result<Vec<f64>, CovariateError> {
        let mut discontinuities = Vec::new();
        for covariate in self.covariates.values() {
            discontinuities.extend(covariate.ode_discontinuity_times()?);
        }
        discontinuities.sort_by(f64::total_cmp);
        discontinuities.dedup();
        Ok(discontinuities)
    }

    /// Set the session-local left-continuity boundary on every covariate.
    pub(crate) fn set_left_continuity_time(&self, time: Option<f64>) {
        for covariate in self.covariates.values() {
            covariate.set_left_continuity_time(time);
        }
    }

    /// Produce a content-based hash of all covariates.
    ///
    /// The internal `BTreeMap` guarantees deterministic iteration order.
    pub fn hash(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = ahash::AHasher::default();
        for (name, covariate) in &self.covariates {
            name.hash(&mut hasher);
            covariate.fixed.hash(&mut hasher);
            for (time, value) in &covariate.observations {
                time.to_bits().hash(&mut hasher);
                value.to_bits().hash(&mut hasher);
            }
        }
        hasher.finish()
    }

    /// Add a covariate to the collection
    ///
    /// This method allows you to add a new covariate with a specific name and its associated data.
    pub fn add_covariate(&mut self, name: String, covariate: Covariate) {
        self.covariates.insert(name, covariate);
    }

    /// Get access to a specific covariate by name
    pub fn get_covariate(&self, name: &str) -> Option<&Covariate> {
        self.covariates.get(name)
    }

    /// Get access to a specific covariate by name
    pub fn get_covariate_mut(&mut self, name: &str) -> Option<&mut Covariate> {
        self.covariates.get_mut(name)
    }

    /// Remove a covariate by name
    pub fn remove_covariate(&mut self, name: &str) -> Option<Covariate> {
        self.covariates.remove(name)
    }

    /// Add an observation to a covariate, creating the covariate if it doesn't exist
    ///
    /// If a value already exists at the specified time, it will update that value silently
    pub fn add_observation(&mut self, name: &str, time: f64, value: f64) {
        if let Some(covariate) = self.covariates.get_mut(name) {
            covariate.add_observation(time, value);
        } else {
            let mut covariate = Covariate::new(name.to_string(), false);
            covariate.add_observation(time, value);
            self.covariates.insert(name.to_string(), covariate);
        }
    }

    /// Update an observation for a specific covariate.
    pub fn update_observation(&mut self, name: &str, time: f64, new_value: f64) -> bool {
        self.covariates
            .get_mut(name)
            .is_some_and(|covariate| covariate.update_observation(time, new_value))
    }

    /// Remove an observation from a specific covariate
    pub fn remove_observation(&mut self, name: &str, time: f64) -> bool {
        if let Some(covariate) = self.covariates.get_mut(name) {
            covariate.remove_observation(time)
        } else {
            false
        }
    }

    /// Set a covariate as fixed (use carry-forward interpolation)
    ///
    /// This is a common operation in pharmacokinetic modeling where you want
    /// to treat a covariate as constant.
    pub fn set_covariate_fixed(&mut self, name: &str, fixed: bool) -> bool {
        if let Some(covariate) = self.covariates.get_mut(name) {
            covariate.set_fixed(fixed);
            true
        } else {
            false
        }
    }

    /// Convert all covariates to a HashMap of values at a specific time
    ///
    /// # Arguments
    ///
    /// * `time` - The time at which to interpolate all covariate values
    ///
    /// # Returns
    ///
    /// A HashMap mapping covariate names to their interpolated values at the specified time
    pub fn to_hashmap(&mut self, time: f64) -> Result<HashMap<String, f64>, CovariateError> {
        self.covariates
            .iter_mut()
            .map(|(name, covariate)| {
                covariate
                    .interpolate(time)
                    .map(|value| (name.clone(), value))
            })
            .collect()
    }
}

impl fmt::Display for Covariates {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Covariates:")?;
        for covariate in self.covariates.values() {
            writeln!(f, "{}", covariate)?;
        }
        Ok(())
    }
}

mod tests {
    #[allow(unused_imports)]
    use super::*;

    #[test]
    fn test_covariate_linear_interpolation() {
        let segment = CovariateSegment {
            from: 0.0,
            to: Some(10.0),
            method: Interpolation::Linear {
                slope: 1.0,
                intercept: 0.0,
            },
        };

        assert_eq!(segment.interpolate(0.0), Some(0.0));
        assert_eq!(segment.interpolate(5.0), Some(5.0));
        assert_eq!(segment.interpolate(10.0), None);
        assert_eq!(segment.interpolate(15.0), None);
    }

    #[test]
    fn test_covariate_carry_forward() {
        let segment = CovariateSegment {
            from: 0.0,
            to: Some(10.0),
            method: Interpolation::CarryForward { value: 5.0 },
        };

        assert_eq!(segment.interpolate(0.0), Some(5.0));
        assert_eq!(segment.interpolate(5.0), Some(5.0));
        assert_eq!(segment.interpolate(10.0), None);
        assert_eq!(segment.interpolate(15.0), None);
    }

    #[test]
    fn test_covariates() {
        let mut covariates = Covariates::new();

        // Create a covariate with observations
        let mut covariate1 = Covariate::new("covariate1".to_string(), false);
        covariate1.add_observation(0.0, 0.0);
        covariate1.add_observation(10.0, 10.0);

        covariates.add_covariate("covariate1".to_string(), covariate1);

        assert_eq!(
            covariates
                .get_covariate("covariate1")
                .unwrap()
                .interpolate(0.0)
                .unwrap(),
            0.0
        );
        assert_eq!(
            covariates
                .get_covariate("covariate1")
                .unwrap()
                .interpolate(5.0)
                .unwrap(),
            (5.0)
        );
        assert_eq!(
            covariates
                .get_covariate("covariate1")
                .unwrap()
                .interpolate(10.0)
                .unwrap(),
            (10.0)
        );
        assert_eq!(
            covariates
                .get_covariate("covariate1")
                .unwrap()
                .interpolate(15.0)
                .unwrap(),
            (10.0)
        );
    }

    #[test]
    fn test_covariate_data_new_api() {
        // Test the new API for collecting raw data and building segments
        let mut covariates = Covariates::new();

        // Add some raw observations
        covariates.add_observation("weight", 0.0, 70.0);
        covariates.add_observation("weight", 12.0, 72.0);
        covariates.add_observation("weight", 24.0, 75.0);
        covariates.add_observation("age", 0.0, 35.0);

        // Fixed covariate
        covariates.set_covariate_fixed("age", true);

        // Test weight interpolation (should be linear)
        let weight_cov = covariates.get_covariate("weight").unwrap();
        assert_eq!(weight_cov.interpolate(0.0).unwrap(), 70.0);
        assert_eq!(weight_cov.interpolate(6.0).unwrap(), 71.0); // Linear interpolation
        assert_eq!(weight_cov.interpolate(12.0).unwrap(), 72.0);
        assert_eq!(weight_cov.interpolate(18.0).unwrap(), 73.5); // Linear interpolation
        assert_eq!(weight_cov.interpolate(24.0).unwrap(), 75.0);
        assert_eq!(weight_cov.interpolate(30.0).unwrap(), 75.0); // Carry forward after last observation

        // Test age (fixed covariate, should be carry forward)
        let age_cov = covariates.get_covariate("age").unwrap();
        assert_eq!(age_cov.interpolate(0.0).unwrap(), 35.0);
        assert_eq!(age_cov.interpolate(12.0).unwrap(), 35.0); // Carry forward
        assert_eq!(age_cov.interpolate(100.0).unwrap(), 35.0); // Carry forward to infinity
    }

    #[test]
    fn covariate_deserialization_rebuilds_segments_from_observations() {
        let json = r#"{
            "name": "wt",
            "observations": [[10.0, 10.0], [0.0, 0.0]],
            "fixed": false
        }"#;
        let covariate: Covariate = serde_json::from_str(json).unwrap();
        assert_eq!(covariate.observations(), [(0.0, 0.0), (10.0, 10.0)]);
        assert_eq!(covariate.interpolate(5.0).unwrap(), 5.0);

        let serialized = serde_json::to_string(&covariate).unwrap();
        assert!(!serialized.contains("segments"));
        let round_tripped: Covariate = serde_json::from_str(&serialized).unwrap();
        assert_eq!(round_tripped.interpolate(5.0).unwrap(), 5.0);
    }

    #[test]
    fn covariate_deserialization_rejects_invalid_observations() {
        let duplicate = r#"{
            "name": "wt",
            "observations": [[0.0, 70.0], [0.0, 71.0]],
            "fixed": false
        }"#;
        assert!(serde_json::from_str::<Covariate>(duplicate)
            .unwrap_err()
            .to_string()
            .contains("duplicate covariate observation"));

        let derived_segments = r#"{
            "name": "wt",
            "observations": [[0.0, 70.0]],
            "segments": [],
            "fixed": false
        }"#;
        assert!(serde_json::from_str::<Covariate>(derived_segments)
            .unwrap_err()
            .to_string()
            .contains("unknown field `segments`"));

        let nonfinite = r#"{
            "name": "wt",
            "observations": [[1e400, 70.0]],
            "fixed": false
        }"#;
        assert!(serde_json::from_str::<Covariate>(nonfinite).is_err());
    }

    #[test]
    fn test_covariate_data_update_functionality() {
        let mut covariates = Covariates::new();

        // Add initial observations
        covariates.add_observation("bmi", 0.0, 25.0);
        covariates.add_observation("bmi", 12.0, 26.0);

        // Test initial interpolation
        assert_eq!(
            covariates
                .get_covariate("bmi")
                .unwrap()
                .interpolate(6.0)
                .unwrap(),
            25.5
        );

        // Update an observation
        assert!(covariates.update_observation("bmi", 12.0, 27.0));
        assert!(!covariates.update_observation("bmi", 18.0, 99.0));
        assert!(!covariates.update_observation("missing", 12.0, 99.0));

        // Test updated interpolation
        assert_eq!(
            covariates
                .get_covariate("bmi")
                .unwrap()
                .interpolate(6.0)
                .unwrap(),
            26.0
        ); // Should be different now
        assert_eq!(
            covariates
                .get_covariate("bmi")
                .unwrap()
                .interpolate(12.0)
                .unwrap(),
            27.0
        ); // Updated value

        // Add a new observation
        covariates.add_observation("bmi", 24.0, 28.0);

        assert_eq!(
            covariates
                .get_covariate("bmi")
                .unwrap()
                .interpolate(18.0)
                .unwrap(),
            27.5
        );
    }

    #[test]
    fn test_row_observation_parsing() {
        let mut raw_observations: HashMap<String, Vec<(f64, f64)>> = HashMap::new();
        raw_observations.insert("weight".to_string(), vec![(0.0, 70.0), (12.0, 72.0)]);
        raw_observations.insert("age!".to_string(), vec![(0.0, 35.0)]);

        let covariates = Covariates::from_row_observations(&raw_observations);

        // Weight should use linear interpolation
        let weight_cov = covariates.get_covariate("weight").unwrap();
        assert_eq!(weight_cov.interpolate(6.0).unwrap(), (71.0));

        // Age should use carry forward (fixed covariate)
        let age_cov = covariates.get_covariate("age").unwrap();
        assert_eq!(age_cov.interpolate(0.0).unwrap(), (35.0));
        assert_eq!(age_cov.interpolate(100.0).unwrap(), (35.0));
    }

    #[test]
    fn test_pmetrics_csv_covariate_interpolation() {
        use crate::data::parser::pmetrics::read_pmetrics;

        // Read the test CSV file with weight data
        let data_result = read_pmetrics("src/tests/data/covariate_test.csv");
        assert!(
            data_result.is_ok(),
            "Failed to read CSV file: {:?}",
            data_result.err()
        );

        let data = data_result.unwrap();

        // Get the first subject
        let binding = data.subjects();
        let subject1 = binding.first().expect("Should have at least one subject");

        // Get the covariates for subject 1
        let covariates = subject1.occasions().first().unwrap().covariates();

        // Header names are normalized to lowercase.
        let wt_cov = covariates
            .get_covariate("wt")
            .expect("wt covariate should exist");

        // Test interpolation at observation times
        assert_eq!(
            wt_cov.interpolate(0.0).unwrap(),
            70.0,
            "Weight at time 0 should be 70.0"
        );
        assert_eq!(
            wt_cov.interpolate(24.0).unwrap(),
            72.0,
            "Weight at time 24 should be 72.0"
        );
        assert_eq!(
            wt_cov.interpolate(48.0).unwrap(),
            74.0,
            "Weight at time 48 should be 74.0"
        );

        // Test linear interpolation between observations
        let interpolated_value = wt_cov.interpolate(12.0).unwrap();
        assert!(
            (interpolated_value - 70.4).abs() < 1e-8,
            "Weight at time 12 should be approximately 70.4 (linear interpolation), got {}",
            interpolated_value
        );
        assert_eq!(
            wt_cov.interpolate(36.0).unwrap(),
            73.0,
            "Weight at time 36 should be 73.0 (linear interpolation)"
        );

        // Test carry forward after last observation
        assert_eq!(
            wt_cov.interpolate(60.0).unwrap(),
            74.0,
            "Weight at time 60 should be 74.0 (carry forward)"
        );

        // Get the second subject
        let binding = data.subjects();
        let subject2 = binding.get(1).expect("Should have a second subject");
        let covariates2 = subject2.occasions().first().unwrap().covariates();
        let wt_cov2 = covariates2
            .get_covariate("wt")
            .expect("wt covariate should exist for subject 2");

        // Test subject 2 weight interpolation
        assert_eq!(
            wt_cov2.interpolate(0.0).unwrap(),
            65.0,
            "Subject 2 weight at time 0 should be 65.0"
        );
        assert_eq!(
            wt_cov2.interpolate(18.0).unwrap(),
            66.0,
            "Subject 2 weight at time 18 should be 66.0 (linear interpolation)"
        );
        assert_eq!(
            wt_cov2.interpolate(48.0).unwrap(),
            69.0,
            "Subject 2 weight at time 48 should be 69.0"
        );
    }

    #[test]
    fn covariates_hash_deterministic() {
        let mut covs = Covariates::new();
        let mut cov = Covariate::new("wt".into(), false);
        cov.add_observation(0.0, 70.0);
        cov.add_observation(24.0, 72.0);
        covs.add_covariate("wt".into(), cov);
        assert_eq!(covs.hash(), covs.hash());
    }

    #[test]
    fn covariates_hash_differs_on_value_change() {
        let mut covs_a = Covariates::new();
        let mut cov_a = Covariate::new("wt".into(), false);
        cov_a.add_observation(0.0, 70.0);
        covs_a.add_covariate("wt".into(), cov_a);

        let mut covs_b = Covariates::new();
        let mut cov_b = Covariate::new("wt".into(), false);
        cov_b.add_observation(0.0, 80.0);
        covs_b.add_covariate("wt".into(), cov_b);

        assert_ne!(covs_a.hash(), covs_b.hash());
    }

    #[test]
    fn covariates_hash_differs_on_name() {
        let mut covs_a = Covariates::new();
        let mut cov_a = Covariate::new("wt".into(), false);
        cov_a.add_observation(0.0, 70.0);
        covs_a.add_covariate("wt".into(), cov_a);

        let mut covs_b = Covariates::new();
        let mut cov_b = Covariate::new("ht".into(), false);
        cov_b.add_observation(0.0, 70.0);
        covs_b.add_covariate("ht".into(), cov_b);

        assert_ne!(covs_a.hash(), covs_b.hash());
    }

    #[test]
    fn covariates_hash_includes_fixed_semantics() {
        let mut linear = Covariates::new();
        let mut linear_covariate = Covariate::new("age".into(), false);
        linear_covariate.add_observation(0.0, 40.0);
        linear.add_covariate("age".into(), linear_covariate);

        let mut fixed = Covariates::new();
        let mut fixed_covariate = Covariate::new("age".into(), true);
        fixed_covariate.add_observation(0.0, 40.0);
        fixed.add_covariate("age".into(), fixed_covariate);

        assert_ne!(linear.hash(), fixed.hash());
    }

    #[test]
    fn locf_interpolation_uses_left_segment_only_at_active_boundary() {
        let mut covariate = Covariate::new("rate".into(), true);
        covariate.add_observation(0.0, 1.0);
        covariate.add_observation(1.0, 2.0);

        assert_eq!(covariate.interpolate(1.0).unwrap(), 2.0);
        covariate.set_left_continuity_time(Some(1.0));
        assert_eq!(covariate.interpolate(1.0).unwrap(), 1.0);
        assert_eq!(covariate.interpolate(1.0 + f64::EPSILON).unwrap(), 2.0);
        covariate.set_left_continuity_time(None);
        assert_eq!(covariate.interpolate(1.0).unwrap(), 2.0);
    }

    #[test]
    fn locf_breakpoints_exclude_repeated_equal_values() {
        let mut covariate = Covariate::new("rate".into(), true);
        covariate.add_observation(0.0, 1.0);
        covariate.add_observation(1.0, 1.0);
        covariate.add_observation(2.0, 2.0);
        covariate.add_observation(3.0, 2.0);
        covariate.add_observation(4.0, 3.0);

        assert_eq!(covariate.ode_breakpoint_times().unwrap(), [2.0, 4.0]);
    }

    #[test]
    fn linear_breakpoints_include_endpoint_and_interior_knots_exactly() {
        let mut covariate = Covariate::new("rate".into(), false);
        covariate.add_observation(0.1, 1.0);
        covariate.add_observation(1.1, 2.0);
        covariate.add_observation(2.1, 4.0);

        assert_eq!(covariate.ode_breakpoint_times().unwrap(), [0.1, 1.1, 2.1]);
    }

    #[test]
    fn ode_validation_rejects_nonfinite_observations() {
        let mut covariate = Covariate::new("rate".into(), true);
        covariate.add_observation(f64::NAN, 1.0);

        assert!(matches!(
            covariate.ode_breakpoint_times(),
            Err(CovariateError::NonFiniteObservation { .. })
        ));
    }

    #[test]
    fn covariates_remain_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}

        assert_send_sync::<Covariate>();
        assert_send_sync::<Covariates>();
    }

    #[test]
    fn covariate_clone_resets_session_continuity_marker() {
        let mut original = Covariate::new("rate".into(), true);
        original.add_observation(0.0, 1.0);
        original.add_observation(1.0, 2.0);
        original.set_left_continuity_time(Some(1.0));

        let clone = original.clone();
        assert_eq!(original.interpolate(1.0).unwrap(), 1.0);
        assert_eq!(clone.interpolate(1.0).unwrap(), 2.0);
    }
}
