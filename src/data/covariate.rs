use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, HashMap},
    fmt,
};
use thiserror::Error;

/// Error type for covariate operations
#[derive(Error, Debug, Clone, Serialize, Deserialize)]
pub enum CovariateError {
    #[error("Observation already exists at time {time}")]
    ObservationExists { time: f64 },
    #[error("Covariate build required")]
    BuildRequired,
    #[error("No segments available for interpolation")]
    MissingSegments,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CovariateObservation {
    time: f64,
    value: f64,
}

impl CovariateObservation {
    /// Create a new covariate observation
    pub(crate) fn new(time: f64, value: f64) -> Self {
        CovariateObservation { time, value }
    }

    /// Get the time of this observation
    pub fn time(&self) -> f64 {
        self.time
    }

    /// Get the value of this observation
    pub fn value(&self) -> f64 {
        self.value
    }
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
#[derive(Serialize, Clone, Debug, Deserialize)]
struct CovariateSegment {
    from: f64,
    to: f64,
    method: Interpolation,
}

impl CovariateSegment {
    /// Create a new covariate segment
    ///
    /// # Arguments
    ///
    /// * `from` - Start time of the segment
    /// * `to` - End time of the segment
    /// * `method` - Interpolation method to use within this segment
    pub(crate) fn new(from: f64, to: f64, method: Interpolation) -> Self {
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

    /// Check if a given time is within this segment's interval
    #[inline]
    fn in_interval(&self, time: f64) -> bool {
        self.from <= time && time < self.to
    }
}

/// A time-varying covariate consisting of raw observations and computed segments
///
/// The covariate holds raw observations and builds interpolated segments on demand.
/// This allows for dynamic updates to the observations and automatic rebuilding of segments.
#[derive(Serialize, Clone, Debug, Deserialize)]
pub struct Covariate {
    name: String,
    observations: Vec<CovariateObservation>,
    segments: Vec<CovariateSegment>,
    segments_dirty: bool, // Flag to track if segments need rebuilding
    /// Flag to indicate if this covariate should always use carry-forward interpolation
    fixed: bool,
}

impl Covariate {
    /// Create a new covariate with the given name and observations
    ///
    /// # Arguments
    ///
    /// * `name` - The name of the covariate
    /// * `observations` - Vector of raw observations
    /// * `fixed` - Whether this covariate should use carry-forward interpolation
    pub fn new(name: String, fixed: bool) -> Self {
        let mut covariate = Covariate {
            name,
            observations: Vec::new(),
            segments: Vec::new(),
            segments_dirty: true,
            fixed,
        };
        covariate.build_segments();
        covariate
    }

    /// Add an observation to this covariate
    pub fn add_observation(&mut self, time: f64, value: f64) -> Result<(), CovariateError> {
        // Check if observation already exists at this time
        if self.observations.iter().any(|obs| obs.time == time) {
            return Err(CovariateError::ObservationExists { time });
        }

        self.observations
            .push(CovariateObservation::new(time, value));
        self.segments_dirty = true;
        Ok(())
    }

    /// Update an observation at a specific time
    pub fn update_observation(&mut self, time: f64, new_value: f64) -> bool {
        if let Some(obs) = self.observations.iter_mut().find(|obs| obs.time == time) {
            obs.value = new_value;
            self.segments_dirty = true;
            true
        } else {
            false
        }
    }

    /// Remove an observation at a specific time
    pub fn remove_observation(&mut self, time: f64) -> bool {
        let initial_len = self.observations.len();
        self.observations.retain(|obs| obs.time != time);
        if self.observations.len() < initial_len {
            self.segments_dirty = true;
            true
        } else {
            false
        }
    }

    /// Get all raw observations
    pub fn observations(&self) -> &[CovariateObservation] {
        &self.observations
    }

    /// Build segments from raw observations
    fn build_segments(&mut self) {
        self.segments.clear();

        if self.observations.is_empty() {
            self.segments_dirty = false;
            return;
        }

        // Sort observations by time
        let mut observations = self.observations.clone();
        observations.sort_by(|a, b| a.time.partial_cmp(&b.time).unwrap());

        for i in 0..observations.len() {
            let current_obs = &observations[i];
            let next_obs = observations.get(i + 1);
            let to_time = next_obs.map_or(f64::INFINITY, |next| next.time);

            if self.fixed {
                // Use CarryForward for fixed covariates
                self.segments.push(CovariateSegment::new(
                    current_obs.time,
                    to_time,
                    Interpolation::CarryForward {
                        value: current_obs.value,
                    },
                ));
            } else if let Some(next) = next_obs {
                if next.time == current_obs.time {
                    self.segments.push(CovariateSegment::new(
                        current_obs.time,
                        next.time,
                        Interpolation::CarryForward {
                            value: current_obs.value,
                        },
                    ));
                } else {
                    let slope = (next.value - current_obs.value) / (next.time - current_obs.time);
                    self.segments.push(CovariateSegment::new(
                        current_obs.time,
                        next.time,
                        Interpolation::Linear {
                            slope,
                            intercept: current_obs.value - slope * current_obs.time,
                        },
                    ));
                }
            }
        }
        self.segments_dirty = false;
    }

    /// Build the covariate segments
    ///
    /// Must be called before using the covariate to ensure segments are up-to-date.
    fn build(&mut self) {
        if self.segments_dirty {
            self.build_segments();
        }
    }

    /// Sort segments by their start time
    pub fn sort_segments(&mut self) {
        self.build();
        self.segments
            .sort_by(|a, b| a.from.partial_cmp(&b.from).unwrap());
    }

    /// Interpolate the covariate value at a specific time
    ///
    /// Returns the interpolated value if the time falls within any segment's range,
    /// otherwise returns the last known observation value. This method requires mutable access to ensure segments are built.
    ///
    /// This method is optimized for sequential access patterns common in ODE solvers
    /// by caching the last used segment index.
    #[inline]
    pub fn interpolate(&self, time: f64) -> Result<f64, CovariateError> {
        if self.segments_dirty {
            return Err(CovariateError::BuildRequired);
        }

        // If no segments are available, return the last observation's value if any
        if self.segments.is_empty() {
            return Err(CovariateError::MissingSegments);
        }

        // Search for the correct segment
        if let Some(value) = self
            .segments
            .iter()
            .find(|&segment| segment.in_interval(time))
            .and_then(|segment| segment.interpolate(time))
        {
            return Ok(value);
        }
        // If no segment contains this time, handle edge cases
        if let Some(first_obs) = self.observations.first() {
            if time < first_obs.time {
                // Time is before first observation - carry first value backwards
                return Ok(first_obs.value);
            }
        }

        if let Some(last_obs) = self.observations.last() {
            if time >= last_obs.time {
                // Time is after last observation - carry last value forward
                return Ok(last_obs.value);
            }
        }

        // Fallback: if we reach here, something went wrong
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
        self.segments_dirty = true;
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
            write!(
                f,
                "  Segment {}: from {:.2} to {:.2}, ",
                index + 1,
                segment.from,
                segment.to
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

    pub fn build(&mut self) {
        for covariate in self.covariates.values_mut() {
            covariate.build();
        }
    }

    /// Create covariates from Pmetrics raw observations
    pub fn from_pmetrics_observations(
        raw_observations: &HashMap<String, Vec<(f64, Option<f64>)>>,
    ) -> Self {
        let mut covariates = Covariates::new();

        for (key, occurrences) in raw_observations {
            let is_fixed = key.ends_with('!');
            let name = if is_fixed {
                key.trim_end_matches('!').to_string()
            } else {
                key.clone()
            };

            let mut observations = Vec::new();
            for &(time, value_opt) in occurrences {
                if let Some(value) = value_opt {
                    observations.push(CovariateObservation::new(time, value));
                }
            }

            if !observations.is_empty() {
                let covariate = Covariate::new(name.clone(), is_fixed);
                covariates.add_covariate(name, covariate);
            }
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
    pub fn add_observation(
        &mut self,
        name: &str,
        time: f64,
        value: f64,
    ) -> Result<(), CovariateError> {
        if let Some(covariate) = self.covariates.get_mut(name) {
            covariate.add_observation(time, value).map_err(|e| e.into())
        } else {
            let mut covariate = Covariate::new(name.to_string(), false);
            covariate
                .add_observation(time, value)
                .map_err(|e| e.into())?;
            self.covariates.insert(name.to_string(), covariate);
            Ok(())
        }
    }

    /// Update an observation for a specific covariate
    pub fn update_observation(&mut self, name: &str, time: f64, new_value: f64) -> bool {
        if let Some(covariate) = self.covariates.get_mut(name) {
            covariate.update_observation(time, new_value)
        } else {
            false
        }
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
            to: 10.0,
            method: Interpolation::Linear {
                slope: 1.0,
                intercept: 0.0,
            },
        };

        assert_eq!(segment.interpolate(0.0), Some(0.0));
        assert_eq!(segment.interpolate(5.0), Some(5.0));
        assert_eq!(segment.interpolate(10.0), Some(10.0));
        assert_eq!(segment.interpolate(15.0), None);
    }

    #[test]
    fn test_covariate_carry_forward() {
        let segment = CovariateSegment {
            from: 0.0,
            to: 10.0,
            method: Interpolation::CarryForward { value: 5.0 },
        };

        assert_eq!(segment.interpolate(0.0), Some(5.0));
        assert_eq!(segment.interpolate(5.0), Some(5.0));
        assert_eq!(segment.interpolate(10.0), Some(5.0));
        assert_eq!(segment.interpolate(15.0), None);
    }

    #[test]
    fn test_covariates() {
        let mut covariates = Covariates::new();

        // Create a covariate with observations
        let observations = vec![
            CovariateObservation::new(0.0, 0.0),
            CovariateObservation::new(10.0, 10.0),
        ];
        let mut covariate1 = Covariate::new("covariate1".to_string(), false);
        for obs in observations {
            covariate1.add_observation(obs.time, obs.value).unwrap();
        }

        covariates.add_covariate("covariate1".to_string(), covariate1);
        covariates.build();

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
        covariates.add_observation("weight", 0.0, 70.0).unwrap();
        covariates.add_observation("weight", 12.0, 72.0).unwrap();
        covariates.add_observation("weight", 24.0, 75.0).unwrap();
        covariates.add_observation("age", 0.0, 35.0).unwrap();

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
    fn test_covariate_data_update_functionality() {
        let mut covariates = Covariates::new();

        // Add initial observations
        covariates.add_observation("bmi", 0.0, 25.0).unwrap();
        covariates.add_observation("bmi", 12.0, 26.0).unwrap();

        // Test initial interpolation
        assert_eq!(
            covariates
                .get_covariate("bmi")
                .unwrap()
                .interpolate(6.0)
                .unwrap(),
            (25.5)
        );

        // Update an observation
        assert!(covariates.update_observation("bmi", 12.0, 27.0));

        // Test updated interpolation
        assert_eq!(
            covariates
                .get_covariate("bmi")
                .unwrap()
                .interpolate(6.0)
                .unwrap(),
            (26.0)
        ); // Should be different now
        assert_eq!(
            covariates
                .get_covariate("bmi")
                .unwrap()
                .interpolate(12.0)
                .unwrap(),
            (27.0)
        ); // Updated value

        // Add a new observation
        covariates.add_observation("bmi", 24.0, 28.0).unwrap();
        assert_eq!(
            covariates
                .get_covariate("bmi")
                .unwrap()
                .interpolate(18.0)
                .unwrap(),
            (27.5)
        );
    }

    #[test]
    fn test_individual_segment_updates() {
        let mut covariates = Covariates::new();
        covariates.add_observation("test_cov", 0.0, 10.0).unwrap();
        covariates.add_observation("test_cov", 10.0, 20.0).unwrap();

        // Initial interpolation should be linear
        assert_eq!(
            covariates
                .get_covariate("test_cov")
                .unwrap()
                .interpolate(5.0)
                .unwrap(),
            (15.0)
        );
    }

    #[test]
    fn test_pmetrics_format_parsing() {
        // Test parsing from Pmetrics-style format with "!" for fixed covariates
        let mut raw_observations: HashMap<String, Vec<(f64, Option<f64>)>> = HashMap::new();
        raw_observations.insert(
            "weight".to_string(),
            vec![(0.0, Some(70.0)), (12.0, Some(72.0))],
        );
        raw_observations.insert("age!".to_string(), vec![(0.0, Some(35.0))]); // Fixed covariate

        let covariates = Covariates::from_pmetrics_observations(&raw_observations);

        // Weight should use linear interpolation
        let weight_cov = covariates.get_covariate("weight").unwrap();
        assert_eq!(weight_cov.interpolate(6.0).unwrap(), (71.0));

        // Age should use carry forward (fixed covariate)
        let age_cov = covariates.get_covariate("age").unwrap();
        assert_eq!(age_cov.interpolate(0.0).unwrap(), (35.0));
        assert_eq!(age_cov.interpolate(100.0).unwrap(), (35.0));
    }
}
