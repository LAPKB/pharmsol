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
    #[error("No segments available for interpolation")]
    MissingSegments,
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

    /// Get the original observation time (same as 'from' for observation-based segments)
    fn time(&self) -> f64 {
        self.from
    }

    /// Get the original observation value
    fn value(&self) -> f64 {
        match self.method {
            Interpolation::Linear { slope, intercept } => slope * self.from + intercept,
            Interpolation::CarryForward { value } => value,
        }
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

/// A time-varying covariate consisting of computed segments
///
/// The covariate holds interpolated segments that are rebuilt whenever observations are modified.
/// Original observation data is stored within the segments themselves.
#[derive(Serialize, Clone, Debug, Deserialize)]
pub struct Covariate {
    /// The name of the covariate
    name: String,
    /// Segments representing the covariate's value over time
    segments: Vec<CovariateSegment>,
    /// Flag to indicate if this covariate should always use carry-forward interpolation
    fixed: bool,
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
            segments: Vec::new(),
            fixed,
        }
    }

    /// Extract original observations from segments
    fn get_observations(&self) -> Vec<(f64, f64)> {
        let mut observations: Vec<(f64, f64)> = self
            .segments
            .iter()
            .map(|segment| (segment.time(), segment.value()))
            .collect();

        // Remove duplicates and sort by time
        observations.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        observations.dedup_by(|a, b| a.0 == b.0);
        observations
    }

    /// Add an observation to this covariate
    ///
    /// If an observation already exists at this time, it will update that value instead of adding a new one.
    pub fn add_observation(&mut self, time: f64, value: f64) {
        // If an observation already exists at this time, update it instead of adding a new one
        if let Some(existing_segment) = self.segments.iter_mut().find(|seg| seg.time() == time) {
            // Update the existing observation's value
            existing_segment.method = Interpolation::CarryForward { value };
            self.build_segments();
        }

        // Add a temporary segment to store the new observation
        self.segments.push(CovariateSegment::new(
            time,
            time,
            Interpolation::CarryForward { value },
        ));

        // Rebuild all segments
        self.build_segments();
    }

    /// Update an observation at a specific time
    pub fn update_observation(&mut self, time: f64, new_value: f64) {
        // Remove the old observation and add the new one
        let removed = self.remove_observation(time);
        if removed {
            // Add the updated observation
            self.add_observation(time, new_value)
        }
    }

    /// Remove an observation at a specific time
    pub fn remove_observation(&mut self, time: f64) -> bool {
        let initial_len = self.segments.len();
        self.segments.retain(|seg| seg.time() != time);
        if self.segments.len() < initial_len {
            self.build_segments();
            true
        } else {
            false
        }
    }

    /// Get all raw observations as time-value pairs
    pub fn observations(&self) -> Vec<(f64, f64)> {
        self.get_observations()
    }

    /// Build segments from raw observations
    fn build_segments(&mut self) {
        // Get observations from current segments
        let observations = self.get_observations();

        // Clear segments and rebuild
        self.segments.clear();

        if observations.is_empty() {
            return;
        }

        for i in 0..observations.len() {
            let current_obs = &observations[i];
            let next_obs = observations.get(i + 1);
            let to_time = next_obs.map_or(f64::INFINITY, |next| next.0);

            if self.fixed {
                // Use CarryForward for fixed covariates
                self.segments.push(CovariateSegment::new(
                    current_obs.0,
                    to_time,
                    Interpolation::CarryForward {
                        value: current_obs.1,
                    },
                ));
            } else if let Some(next) = next_obs {
                let slope = (next.1 - current_obs.1) / (next.0 - current_obs.0);
                self.segments.push(CovariateSegment::new(
                    current_obs.0,
                    next.0,
                    Interpolation::Linear {
                        slope,
                        intercept: current_obs.1 - slope * current_obs.0,
                    },
                ));
            } else {
                // Single observation, not fixed - create a CarryForward segment to infinity
                self.segments.push(CovariateSegment::new(
                    current_obs.0,
                    f64::INFINITY,
                    Interpolation::CarryForward {
                        value: current_obs.1,
                    },
                ));
            }
        }
    }

    /// Interpolate the covariate value at a specific time
    ///
    /// Returns the interpolated value if the time falls within any segment's range,
    /// otherwise returns the last known observation value.
    ///
    /// This method is optimized for sequential access patterns common in ODE solvers
    /// by caching the last used segment index.
    #[inline]
    pub fn interpolate(&self, time: f64) -> Result<f64, CovariateError> {
        // If no segments are available, return error
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
        let observations = self.get_observations();
        if let Some(first_obs) = observations.first() {
            if time < first_obs.0 {
                // Time is before first observation - carry first value backwards
                return Ok(first_obs.1);
            }
        }

        if let Some(last_obs) = observations.last() {
            if time >= last_obs.0 {
                // Time is after last observation - carry last value forward
                return Ok(last_obs.1);
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

    /// Create covariates from Pmetrics raw observations
    pub(crate) fn from_pmetrics_observations(
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

            let mut covariate = Covariate::new(name.clone(), is_fixed);
            for &(time, value_opt) in occurrences {
                if let Some(value) = value_opt {
                    covariate.add_observation(time, value);
                }
            }

            if !covariate.segments.is_empty() {
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

    /// Update an observation for a specific covariate
    pub fn update_observation(&mut self, name: &str, time: f64, new_value: f64) -> bool {
        if let Some(covariate) = self.covariates.get_mut(name) {
            covariate.update_observation(time, new_value);
            true
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
        assert_eq!(segment.interpolate(10.0), None);
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
        let subject1 = binding.get(0).expect("Should have at least one subject");

        // Get the covariates for subject 1
        let covariates = subject1.occasions().first().unwrap().covariates();

        // Verify that WT covariate exists
        let wt_cov = covariates
            .get_covariate("wt")
            .expect("WT covariate should exist");

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
            .expect("WT covariate should exist for subject 2");

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
}
