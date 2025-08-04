use serde::{Deserialize, Serialize};
use std::{collections::HashMap, fmt};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CovariateObservation {
    pub time: f64,
    pub value: f64,
    pub fixed: bool,
}

impl CovariateObservation {
    /// Create a new covariate observation
    pub fn new(time: f64, value: f64, fixed: bool) -> Self {
        CovariateObservation { time, value, fixed }
    }
}

/// Method used to interpolate covariate values between observations
#[derive(serde::Serialize, Clone, Debug, Deserialize)]
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
pub struct CovariateSegment {
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

    /// Update the interpolation method for this segment
    pub fn set_interpolation(&mut self, method: Interpolation) {
        self.method = method;
    }

    /// Interpolate the covariate value at a specific time within this segment
    ///
    /// Returns None if the time is outside the segment's range.
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
    fn in_interval(&self, time: f64) -> bool {
        self.from <= time && time <= self.to
    }

    /// Get the start time of the segment
    pub fn from(&self) -> f64 {
        self.from
    }

    /// Get the end time of the segment
    pub fn to(&self) -> f64 {
        self.to
    }

    /// Get the interpolation method used in this segment
    pub fn method(&self) -> &Interpolation {
        &self.method
    }
}

/// A time-varying covariate consisting of raw observations and built segments
///
/// The covariate holds raw observations and builds interpolated segments on demand.
/// This allows for dynamic updates to the observations and automatic rebuilding of segments.
#[derive(serde::Serialize, Clone, Debug, Deserialize)]
pub struct Covariate {
    name: String,
    observations: Vec<CovariateObservation>,
    segments: Vec<CovariateSegment>,
    segments_dirty: bool, // Flag to track if segments need rebuilding
}

impl Covariate {
    /// Create a new covariate with the given name and observations
    ///
    /// # Arguments
    ///
    /// * `name` - The name of the covariate
    /// * `observations` - Vector of raw observations
    pub fn new(name: String, observations: Vec<CovariateObservation>) -> Self {
        let mut covariate = Covariate {
            name,
            observations,
            segments: Vec::new(),
            segments_dirty: true,
        };
        covariate.build_segments();
        covariate
    }

    /// Create a new empty covariate with just a name
    pub fn new_empty(name: String) -> Self {
        Covariate {
            name,
            observations: Vec::new(),
            segments: Vec::new(),
            segments_dirty: false,
        }
    }

    /// Add an observation to this covariate
    pub fn add_observation(&mut self, time: f64, value: f64, fixed: bool) {
        self.observations
            .push(CovariateObservation::new(time, value, fixed));
        self.segments_dirty = true;
    }

    /// Add an observation directly
    pub fn add_observation_direct(&mut self, observation: CovariateObservation) {
        self.observations.push(observation);
        self.segments_dirty = true;
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

        // If only one observation, add a single segment to infinity
        if observations.len() == 1 {
            let obs = &observations[0];
            self.segments.push(CovariateSegment::new(
                obs.time,
                f64::INFINITY,
                Interpolation::CarryForward { value: obs.value },
            ));
            self.segments_dirty = false;
            return;
        }

        let mut last_value: Option<(&CovariateObservation, &CovariateObservation)> = None;
        for i in 0..observations.len() {
            let current_obs = &observations[i];
            let next_obs = observations.get(i + 1);
            let to_time = next_obs.map_or(f64::INFINITY, |next| next.time);

            if current_obs.fixed {
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
                last_value = Some((current_obs, next));
            } else if let Some((_, last_obs)) = last_value {
                // Extend the last linear segment to infinity if no more segments are available
                self.segments.push(CovariateSegment::new(
                    last_obs.time,
                    f64::INFINITY,
                    Interpolation::CarryForward {
                        value: last_obs.value,
                    },
                ));
            }
        }
        self.segments_dirty = false;
    }

    /// Ensure segments are up to date
    fn ensure_segments_built(&mut self) {
        if self.segments_dirty {
            self.build_segments();
        }
    }

    /// Add a segment to this covariate (legacy method for compatibility)
    pub(crate) fn add_segment(&mut self, segment: CovariateSegment) {
        self.segments.push(segment);
        self.segments_dirty = false; // Assume manual segment management
    }

    /// Update the interpolation method for a segment at a specific time
    pub fn update_segment_interpolation(&mut self, time: f64, new_method: Interpolation) -> bool {
        self.ensure_segments_built();
        if let Some(segment) = self.segments.iter_mut().find(|s| s.in_interval(time)) {
            segment.method = new_method;
            true
        } else {
            false
        }
    }

    /// Remove a segment that contains the given time
    pub fn remove_segment_at(&mut self, time: f64) -> bool {
        self.ensure_segments_built();
        let initial_len = self.segments.len();
        self.segments.retain(|s| !s.in_interval(time));
        self.segments.len() < initial_len
    }

    /// Replace all segments with new ones (useful for manual segment management)
    pub fn replace_segments(&mut self, new_segments: Vec<CovariateSegment>) {
        self.segments = new_segments;
        self.segments_dirty = false; // Assume manual segment management
    }

    /// Sort segments by their start time
    pub fn sort_segments(&mut self) {
        self.ensure_segments_built();
        self.segments
            .sort_by(|a, b| a.from.partial_cmp(&b.from).unwrap());
    }

    /// Interpolate the covariate value at a specific time
    ///
    /// Returns the interpolated value if the time falls within any segment's range,
    /// otherwise returns None. This method requires mutable access to ensure segments are built.
    pub fn interpolate(&mut self, time: f64) -> Option<f64> {
        self.ensure_segments_built();
        self.segments
            .iter()
            .find(|&segment| segment.in_interval(time))
            .and_then(|segment| segment.interpolate(time))
    }

    /// Interpolate the covariate value at a specific time (immutable version)
    ///
    /// Returns the interpolated value if the time falls within any segment's range,
    /// otherwise returns None. If segments are dirty, returns None.
    pub fn interpolate_immutable(&self, time: f64) -> Option<f64> {
        if self.segments_dirty {
            return None;
        }
        self.segments
            .iter()
            .find(|&segment| segment.in_interval(time))
            .and_then(|segment| segment.interpolate(time))
    }

    /// Get the name of the covariate
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get all segments in this covariate
    pub fn segments(&mut self) -> Vec<&CovariateSegment> {
        self.ensure_segments_built();
        self.segments.iter().collect()
    }

    /// Get mutable access to segments (for advanced operations)
    pub fn segments_mut(&mut self) -> &mut Vec<CovariateSegment> {
        self.ensure_segments_built();
        &mut self.segments
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

/// A collection of named covariates
///
/// This struct provides methods to manage multiple covariates and retrieve
/// interpolated values for all covariates at specific time points.
#[derive(serde::Serialize, Clone, Debug, Deserialize)]
pub struct Covariates {
    // Mapping from covariate name to its segments
    // FIXME: this hashmap have a key, covariate also has a name field, we are never
    // checking if the name in the hashmap is the same as the name in the covariate
    covariates: HashMap<String, Covariate>,
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
            covariates: HashMap::new(),
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
                    observations.push(CovariateObservation::new(time, value, is_fixed));
                }
            }

            if !observations.is_empty() {
                let covariate = Covariate::new(name.clone(), observations);
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
    pub(crate) fn add_covariate(&mut self, name: String, covariate: Covariate) {
        self.covariates.insert(name, covariate);
    }

    /// Get a specific covariate by name
    pub fn get_covariate(&self, name: &str) -> Option<&Covariate> {
        self.covariates.get(name)
    }

    /// Get mutable access to a specific covariate by name
    pub fn get_covariate_mut(&mut self, name: &str) -> Option<&mut Covariate> {
        self.covariates.get_mut(name)
    }

    /// Remove a covariate by name
    pub fn remove_covariate(&mut self, name: &str) -> Option<Covariate> {
        self.covariates.remove(name)
    }

    /// Update the interpolation method for a segment at a specific time for a given covariate
    pub fn update_covariate_segment(
        &mut self,
        name: &str,
        time: f64,
        new_method: Interpolation,
    ) -> bool {
        if let Some(covariate) = self.covariates.get_mut(name) {
            covariate.update_segment_interpolation(time, new_method)
        } else {
            false
        }
    }

    /// Add an observation to a covariate, creating the covariate if it doesn't exist
    pub fn add_observation(&mut self, name: &str, time: f64, value: f64, fixed: bool) {
        if let Some(covariate) = self.covariates.get_mut(name) {
            covariate.add_observation(time, value, fixed);
        } else {
            let mut covariate = Covariate::new_empty(name.to_string());
            covariate.add_observation(time, value, fixed);
            self.covariates.insert(name.to_string(), covariate);
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

    /// Convert all covariates to a HashMap of values at a specific time
    ///
    /// # Arguments
    ///
    /// * `time` - The time at which to interpolate all covariate values
    ///
    /// # Returns
    ///
    /// A HashMap mapping covariate names to their interpolated values at the specified time
    pub fn to_hashmap(&mut self, time: f64) -> HashMap<String, f64> {
        self.covariates
            .iter_mut()
            .filter_map(|(name, covariate)| {
                covariate
                    .interpolate(time)
                    .map(|value| (name.clone(), value))
            })
            .collect()
    }

    /// Convert all covariates to a HashMap of values at a specific time (immutable version)
    ///
    /// # Arguments
    ///
    /// * `time` - The time at which to interpolate all covariate values
    ///
    /// # Returns
    ///
    /// A HashMap mapping covariate names to their interpolated values at the specified time.
    /// Covariates with dirty segments will be skipped.
    pub fn to_hashmap_immutable(&self, time: f64) -> HashMap<String, f64> {
        self.covariates
            .iter()
            .filter_map(|(name, covariate)| {
                covariate
                    .interpolate_immutable(time)
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
            CovariateObservation::new(0.0, 0.0, false),
            CovariateObservation::new(10.0, 10.0, false),
        ];
        let mut covariate1 = Covariate::new("covariate1".to_string(), observations);

        // Add a carry forward segment manually
        covariate1.add_segment(CovariateSegment {
            from: 10.0,
            to: 20.0,
            method: Interpolation::CarryForward { value: 10.0 },
        });

        covariates.add_covariate("covariate1".to_string(), covariate1);

        assert_eq!(
            covariates
                .get_covariate_mut("covariate1")
                .unwrap()
                .interpolate(0.0),
            Some(0.0)
        );
        assert_eq!(
            covariates
                .get_covariate_mut("covariate1")
                .unwrap()
                .interpolate(5.0),
            Some(5.0)
        );
        assert_eq!(
            covariates
                .get_covariate_mut("covariate1")
                .unwrap()
                .interpolate(10.0),
            Some(10.0)
        );
        assert_eq!(
            covariates
                .get_covariate_mut("covariate1")
                .unwrap()
                .interpolate(15.0),
            Some(10.0)
        );
    }

    #[test]
    fn test_covariate_data_new_api() {
        // Test the new API for collecting raw data and building segments
        let mut covariates = Covariates::new();

        // Add some raw observations
        covariates.add_observation("weight", 0.0, 70.0, false);
        covariates.add_observation("weight", 12.0, 72.0, false);
        covariates.add_observation("weight", 24.0, 75.0, false);
        covariates.add_observation("age", 0.0, 35.0, true); // Fixed covariate

        // Test weight interpolation (should be linear)
        let weight_cov = covariates.get_covariate_mut("weight").unwrap();
        assert_eq!(weight_cov.interpolate(0.0), Some(70.0));
        assert_eq!(weight_cov.interpolate(6.0), Some(71.0)); // Linear interpolation
        assert_eq!(weight_cov.interpolate(12.0), Some(72.0));
        assert_eq!(weight_cov.interpolate(18.0), Some(73.5)); // Linear interpolation
        assert_eq!(weight_cov.interpolate(24.0), Some(75.0));
        assert_eq!(weight_cov.interpolate(30.0), Some(75.0)); // Carry forward after last observation

        // Test age (fixed covariate, should be carry forward)
        let age_cov = covariates.get_covariate_mut("age").unwrap();
        assert_eq!(age_cov.interpolate(0.0), Some(35.0));
        assert_eq!(age_cov.interpolate(12.0), Some(35.0)); // Carry forward
        assert_eq!(age_cov.interpolate(100.0), Some(35.0)); // Carry forward to infinity
    }

    #[test]
    fn test_covariate_data_update_functionality() {
        let mut covariates = Covariates::new();

        // Add initial observations
        covariates.add_observation("bmi", 0.0, 25.0, false);
        covariates.add_observation("bmi", 12.0, 26.0, false);

        // Test initial interpolation
        assert_eq!(
            covariates
                .get_covariate_mut("bmi")
                .unwrap()
                .interpolate(6.0),
            Some(25.5)
        );

        // Update an observation
        assert!(covariates.update_observation("bmi", 12.0, 27.0));

        // Test updated interpolation
        assert_eq!(
            covariates
                .get_covariate_mut("bmi")
                .unwrap()
                .interpolate(6.0),
            Some(26.0)
        ); // Should be different now
        assert_eq!(
            covariates
                .get_covariate_mut("bmi")
                .unwrap()
                .interpolate(12.0),
            Some(27.0)
        ); // Updated value

        // Add a new observation
        covariates.add_observation("bmi", 24.0, 28.0, false);
        assert_eq!(
            covariates
                .get_covariate_mut("bmi")
                .unwrap()
                .interpolate(18.0),
            Some(27.5)
        );
    }

    #[test]
    fn test_individual_segment_updates() {
        let mut covariates = Covariates::new();
        covariates.add_observation("test_cov", 0.0, 10.0, false);
        covariates.add_observation("test_cov", 10.0, 20.0, false);

        // Initial interpolation should be linear
        assert_eq!(
            covariates
                .get_covariate_mut("test_cov")
                .unwrap()
                .interpolate(5.0),
            Some(15.0)
        );

        // Update the interpolation method for a specific segment
        assert!(covariates.update_covariate_segment(
            "test_cov",
            5.0,
            Interpolation::CarryForward { value: 12.0 }
        ));

        // Now the interpolation should use the new method
        assert_eq!(
            covariates
                .get_covariate_mut("test_cov")
                .unwrap()
                .interpolate(5.0),
            Some(12.0)
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

        let mut covariates = Covariates::from_pmetrics_observations(&raw_observations);

        // Weight should use linear interpolation
        let weight_cov = covariates.get_covariate_mut("weight").unwrap();
        assert_eq!(weight_cov.interpolate(6.0), Some(71.0));

        // Age should use carry forward (fixed covariate)
        let age_cov = covariates.get_covariate_mut("age").unwrap();
        assert_eq!(age_cov.interpolate(0.0), Some(35.0));
        assert_eq!(age_cov.interpolate(100.0), Some(35.0));
    }
}
