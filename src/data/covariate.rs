use serde::Deserialize;
use std::{collections::HashMap, fmt};

#[derive(Clone, Debug, Deserialize)]
pub(crate) enum InterpolationMethod {
    Linear { slope: f64, intercept: f64 },
    CarryForward { value: f64 },
}

/// A [CovariateSegment] is a segment of the piece-wise interpolation of a [Covariate]
#[derive(Clone, Debug, Deserialize)]
pub(crate) struct CovariateSegment {
    from: f64,
    to: f64,
    method: InterpolationMethod,
}

impl CovariateSegment {
    pub(crate) fn new(from: f64, to: f64, method: InterpolationMethod) -> Self {
        CovariateSegment { from, to, method }
    }

    fn interpolate(&self, time: f64) -> Option<f64> {
        if !self.in_interval(time) {
            return None;
        }

        match self.method {
            InterpolationMethod::Linear { slope, intercept } => Some(slope * time + intercept),
            InterpolationMethod::CarryForward { value } => Some(value),
        }
    }

    fn in_interval(&self, time: f64) -> bool {
        self.from <= time && time <= self.to
    }
}

/// A [Covariate] is a collection of [CovariateSegment]s, which allows for interpolation of covariate values
#[derive(Clone, Debug, Deserialize)]
pub struct Covariate {
    name: String,
    segments: Vec<CovariateSegment>,
}

impl Covariate {
    pub(crate) fn new(name: String, segments: Vec<CovariateSegment>) -> Self {
        Covariate { name, segments }
    }
    pub(crate) fn add_segment(&mut self, segment: CovariateSegment) {
        self.segments.push(segment);
    }
    // Check that no segments are overlapping
    // fn check(&self) -> bool {
    //     let mut sorted = self.segments.clone();
    //     sorted.sort_by(|a, b| a.from.partial_cmp(&b.from).unwrap());
    //     for i in 0..sorted.len() - 1 {
    //         if sorted[i].to > sorted[i + 1].from {
    //             return false;
    //         }
    //     }
    //     true
    // }
}

impl Covariate {
    pub fn interpolate(&self, time: f64) -> Option<f64> {
        self.segments
            .iter()
            .find(|&segment| segment.in_interval(time))
            .and_then(|segment| segment.interpolate(time))
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
                InterpolationMethod::Linear { slope, intercept } => {
                    writeln!(
                        f,
                        "Linear, Slope: {:.2}, Intercept: {:.2}",
                        slope, intercept
                    )
                }
                InterpolationMethod::CarryForward { value } => {
                    writeln!(f, "Carry Forward, Value: {:.2}", value)
                }
            }?;
        }
        Ok(())
    }
}

/// [Covariates] is a collection of [Covariate]
#[derive(Clone, Debug, Deserialize)]
pub struct Covariates {
    // Mapping from covariate name to its segments
    covariates: HashMap<String, Covariate>,
}

impl Default for Covariates {
    fn default() -> Self {
        Covariates::new()
    }
}

impl Covariates {
    pub fn new() -> Self {
        Covariates {
            covariates: HashMap::new(),
        }
    }

    pub(crate) fn add_covariate(&mut self, name: String, covariate: Covariate) {
        self.covariates.insert(name, covariate);
    }

    pub fn get_covariate(&self, name: &str) -> Option<&Covariate> {
        self.covariates.get(name)
    }
    // fn get_covariate_names(&self) -> Vec<String> {
    //     self.covariates.keys().cloned().collect()
    // }
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
            method: InterpolationMethod::Linear {
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
            method: InterpolationMethod::CarryForward { value: 5.0 },
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
                        method: InterpolationMethod::Linear {
                            slope: 1.0,
                            intercept: 0.0,
                        },
                    },
                    CovariateSegment {
                        from: 10.0,
                        to: 20.0,
                        method: InterpolationMethod::CarryForward { value: 10.0 },
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

    /* #[test]
    fn test_infusions() {
        let infusions = vec![
            Infusion {
                time: 0.0,
                amount: 100.0,
                input: 1,
                duration: 1.0,
            },
            Infusion {
                time: 2.0,
                amount: 50.0,
                input: 1,
                duration: 1.0,
            },
            Infusion {
                time: 3.0,
                amount: 50.0,
                input: 2,
                duration: 1.0,
            },
        ];

        let infusions = Infusions::from_infusions(infusions);

        assert_eq!(infusions.get_rate_at_time(1, 0.0), 100.0);
        assert_eq!(infusions.get_rate_at_time(1, 0.5), 100.0);
        assert_eq!(infusions.get_rate_at_time(1, 1.0), 0.0);
        assert_eq!(infusions.get_rate_at_time(1, 1.5), 50.0);
        assert_eq!(infusions.get_rate_at_time(1, 2.0), 50.0);
        assert_eq!(infusions.get_rate_at_time(1, 2.5), 50.0);
        assert_eq!(infusions.get_rate_at_time(1, 3.0), 0.0);
        assert_eq!(infusions.get_rate_at_time(1, 3.5), 0.0);
        assert_eq!(infusions.get_rate_at_time(2, 3.0), 50.0);
    } */
}
