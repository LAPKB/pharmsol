use serde::{Deserialize, Serialize};
use std::{collections::HashMap, fmt};

#[derive(Clone, Debug, Serialize, Deserialize)]
struct CovariateObservation {
    name: String,
    time: f64,
    value: f64,
    fixed: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CovariateData {
    data: Vec<CovariateObservation>,
}

impl CovariateData {
    /// Create a new empty CovariateData
    pub fn new() -> Self {
        CovariateData { data: Vec::new() }
    }

    /// Add an observation to the covariate data
    pub fn add_observation(&mut self, name: impl Into<String>, time: f64, value: f64, fixed: bool) {
        self.data.push(CovariateObservation {
            name: name.into(),
            time,
            value,
            fixed,
        });
    }

    /// Get all observations in this covariate data
    fn observations(&self) -> &[CovariateObservation] {
        &self.data
    }

    pub fn to_covariates(&self) -> Covariates {
        // First get all observations grouped by name
        let mut grouped: HashMap<String, Vec<CovariateObservation>> = HashMap::new();
        for obs in &self.data {
            grouped
                .entry(obs.name.clone())
                .or_default()
                .push((*obs).clone());
        }

        // Sort each group by time
        for obs_list in grouped.values_mut() {
            obs_list.sort_by(|a, b| a.time.partial_cmp(&b.time).unwrap());
        }

        // Create covariates from the grouped observations
        let mut covariates = Covariates::new();
        for (name, observations) in grouped {
            let mut segments = Vec::new();
            let mut last_time = None;
            let mut last_value = None;

            for obs in observations {
                if let Some(from) = last_time {
                    if let Some(value) = last_value {
                        // Create a segment from the last observation to this one
                        segments.push(CovariateSegment::new(
                            from,
                            obs.time,
                            Interpolation::Linear {
                                slope: (obs.value - value) / (obs.time - from),
                                intercept: value - ((obs.value - value) / (obs.time - from)) * from,
                            },
                        ));
                    }
                }
                last_time = Some(obs.time);
                last_value = Some(obs.value);
            }
        }
        covariates
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

/// A time-varying covariate consisting of multiple segments
///
/// The covariate provides interpolated values across its entire time range by
/// combining multiple segments with different interpolation methods.
#[derive(serde::Serialize, Clone, Debug, Deserialize)]
pub struct Covariate {
    name: String,
    segments: Vec<CovariateSegment>,
}

impl Covariate {
    /// Create a new covariate with the given name and segments
    ///
    /// # Arguments
    ///
    /// * `name` - The name of the covariate
    /// * `segments` - Vector of covariate segments defining the interpolation
    pub(crate) fn new(name: String, segments: Vec<CovariateSegment>) -> Self {
        Covariate { name, segments }
    }

    /// Add a segment to this covariate
    pub(crate) fn add_segment(&mut self, segment: CovariateSegment) {
        self.segments.push(segment);
    }

    /// Interpolate the covariate value at a specific time
    ///
    /// Returns the interpolated value if the time falls within any segment's range,
    /// otherwise returns None.
    pub fn interpolate(&self, time: f64) -> Option<f64> {
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
    pub fn segments(&self) -> Vec<&CovariateSegment> {
        self.segments.iter().collect()
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

    /// Convert all covariates to a HashMap of values at a specific time
    ///
    /// # Arguments
    ///
    /// * `time` - The time at which to interpolate all covariate values
    ///
    /// # Returns
    ///
    /// A HashMap mapping covariate names to their interpolated values at the specified time
    pub fn to_hashmap(&self, time: f64) -> HashMap<String, f64> {
        self.covariates
            .iter()
            .map(|(name, covariate)| (name.clone(), covariate.interpolate(time).unwrap()))
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
        let mut covariates = Covariates {
            covariates: HashMap::new(),
        };
        covariates.covariates.insert(
            "covariate1".to_string(),
            Covariate::new(
                "covariate1".to_string(),
                vec![
                    CovariateSegment {
                        from: 0.0,
                        to: 10.0,
                        method: Interpolation::Linear {
                            slope: 1.0,
                            intercept: 0.0,
                        },
                    },
                    CovariateSegment {
                        from: 10.0,
                        to: 20.0,
                        method: Interpolation::CarryForward { value: 10.0 },
                    },
                ],
            ),
        );

        assert_eq!(
            covariates
                .get_covariate("covariate1")
                .unwrap()
                .interpolate(0.0),
            Some(0.0)
        );
        assert_eq!(
            covariates
                .get_covariate("covariate1")
                .unwrap()
                .interpolate(5.0),
            Some(5.0)
        );
        assert_eq!(
            covariates
                .get_covariate("covariate1")
                .unwrap()
                .interpolate(10.0),
            Some(10.0)
        );
        assert_eq!(
            covariates
                .get_covariate("covariate1")
                .unwrap()
                .interpolate(15.0),
            Some(10.0)
        );
    }
}
