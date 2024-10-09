use crate::data::*;
use serde::de::{MapAccess, Visitor};
use serde::{de, Deserialize, Deserializer, Serialize};
use std::collections::HashMap;

use std::fmt;
use std::str::FromStr;
use thiserror::Error;

/// Custom error type for the module
#[allow(private_interfaces)]
#[derive(Error, Debug)]
pub enum PmetricsError {
    #[error("CSV error: {0}")]
    ReadError(#[from] csv::Error),
    #[error("Parse error: {0}")]
    SerdeError(#[from] serde::de::value::Error),
    #[error("Unknown EVID: {evid} for ID {id} at time {time}")]
    UnknownEvid { evid: isize, id: String, time: f64 },
    #[error("Observation OUT is missing for {id} at time {time}")]
    MissingObservationOut { id: String, time: f64 },
    #[error("Observation OUTEQ is missing in for {id} at time {time}")]
    MissingObservationOuteq { id: String, time: f64 },
    #[error("Infusion amount (DOSE) is missing for {id} at time {time}")]
    MissingInfusionDose { id: String, time: f64 },
    #[error("Infusion compartment (INPUT) is missing for {id} at time {time}")]
    MissingInfusionInput { id: String, time: f64 },
    #[error("Infusion duration (DUR) is missing for {id} at time {time}")]
    MissingInfusionDur { id: String, time: f64 },
    #[error("Bolus amount (DOSE) is missing for {id} at time {time}")]
    MissingBolusDose { id: String, time: f64 },
    #[error("Bolus compartment (INPUT) is missing for {id} at time {time}")]
    MissingBolusInput { id: String, time: f64 },
}

/// Read a Pmetrics datafile and convert it to a [Data] object
///
/// For specific details, see the [Row] struct.
#[allow(dead_code)]
pub fn read_pmetrics(path: impl Into<String>) -> Result<Data, PmetricsError> {
    let path = path.into();

    let mut reader = csv::ReaderBuilder::new()
        .comment(Some(b'#'))
        .has_headers(true)
        .from_path(path)?;

    // Convert headers to lowercase
    let headers = reader
        .headers()?
        .iter()
        .map(|h| h.to_lowercase())
        .collect::<Vec<_>>();
    reader.set_headers(csv::StringRecord::from(headers));

    // This is the object we are building, which can be converted to [Data]
    let mut subjects: Vec<Subject> = Vec::new();

    // Read the datafile into a hashmap of rows by ID
    let mut rows_map: HashMap<String, Vec<Row>> = HashMap::new();
    for row_result in reader.deserialize() {
        let row: Row = row_result?;

        rows_map.entry(row.id.clone()).or_default().push(row);
    }

    // For each ID, we ultimately create a [Subject] object
    for (id, rows) in rows_map {
        // Split rows into vectors of rows, creating the occasions
        let split_indices: Vec<usize> = rows
            .iter()
            .enumerate()
            .filter_map(|(i, row)| if row.evid == 4 { Some(i) } else { None })
            .collect();

        let mut block_rows_vec = Vec::new();
        let mut start = 0;
        for &split_index in &split_indices {
            let end = split_index;
            if start < rows.len() {
                block_rows_vec.push(&rows[start..end]);
            }
            start = end;
        }

        if start < rows.len() {
            block_rows_vec.push(&rows[start..]);
        }

        let block_rows: Vec<Vec<Row>> = block_rows_vec.iter().map(|block| block.to_vec()).collect();
        let mut occasions: Vec<Occasion> = Vec::new();
        for (block_index, rows) in block_rows.clone().iter().enumerate() {
            // Collector for all events
            let mut events: Vec<Event> = Vec::new();
            // Collector for covariates
            let mut covariates = Covariates::new();

            // Parse events
            for row in rows.clone() {
                match row.parse_events() {
                    Ok(ev) => events.extend(ev),
                    Err(e) => {
                        // dbg!(&row);
                        // dbg!(&e);
                        return Err(e);
                    }
                }
            }

            // Parse covariates
            let mut cloned_rows = rows.clone();
            cloned_rows.retain(|row| !row.covs.is_empty());

            // Collect all covariates by name
            let mut observed_covariates: HashMap<String, Vec<(f64, Option<f64>)>> = HashMap::new();
            for row in &cloned_rows {
                for (key, value) in &row.covs {
                    if let Some(val) = value {
                        observed_covariates
                            .entry(key.clone())
                            .or_default()
                            .push((row.time, Some(*val)));
                    }
                }
            }

            // Create segments for each covariate
            for (key, mut occurrences) in observed_covariates {
                occurrences.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
                let is_fixed = key.ends_with('!');

                // If it's a fixed covariate, modify the name to remove "!"
                let name = if is_fixed {
                    key.trim_end_matches('!').to_string()
                } else {
                    key.clone()
                };

                let mut covariate = Covariate::new(name.clone(), vec![]);

                // If only one occurence, add a single segment to infinity
                if occurrences.len() == 1 {
                    let (time, value) = occurrences[0];
                    covariate.add_segment(CovariateSegment::new(
                        time,
                        f64::INFINITY,
                        InterpolationMethod::CarryForward {
                            value: value.unwrap(),
                        },
                    ));
                    covariates.add_covariate(name, covariate);
                    continue;
                }

                let mut last_value = None;
                for i in 0..occurrences.len() {
                    let (time, value) = occurrences[i];
                    let next_occurrence = occurrences.get(i + 1);
                    let to_time =
                        next_occurrence.map_or(f64::INFINITY, |&(next_time, _)| next_time);

                    if is_fixed {
                        // Use CarryForward for fixed covariates
                        covariate.add_segment(CovariateSegment::new(
                            time,
                            to_time,
                            InterpolationMethod::CarryForward {
                                value: value.unwrap(),
                            },
                        ));
                    } else if let Some((next_time, next_value)) = next_occurrence {
                        if let Some(current_value) = value {
                            if *next_time == time {
                                covariate.add_segment(CovariateSegment::new(
                                    time,
                                    *next_time,
                                    InterpolationMethod::CarryForward {
                                        value: current_value,
                                    },
                                ));
                            } else {
                                let slope =
                                    (next_value.unwrap() - current_value) / (next_time - time);
                                covariate.add_segment(CovariateSegment::new(
                                    time,
                                    *next_time,
                                    InterpolationMethod::Linear {
                                        slope,
                                        intercept: current_value - slope * time,
                                    },
                                ));
                            }

                            last_value = Some((next_time, next_value));
                        }
                    } else if let Some((last_time, last_value)) = last_value {
                        // Extend the last linear segment to infinity if no more segments are available
                        covariate.add_segment(CovariateSegment::new(
                            *last_time,
                            f64::INFINITY,
                            InterpolationMethod::CarryForward {
                                value: last_value.unwrap(),
                            },
                        ));
                    }
                }
                covariates.add_covariate(name, covariate)
            }
            // Create the block
            let mut occasion = Occasion::new(events, covariates, block_index);
            occasion.sort();
            occasions.push(occasion);
        }

        let subject = Subject::new(id, occasions);
        subjects.push(subject);
    }

    // Sort subjects alphabetically by ID to get consistent ordering
    subjects.sort_by(|a, b| a.id().cmp(b.id()));
    let data = Data::new(subjects);

    Ok(data)
}

/// A [Row] represents a row in the Pmetrics data format
#[derive(Deserialize, Debug, Serialize, Default, Clone)]
#[serde(rename_all = "lowercase")]
struct Row {
    /// Subject ID
    id: String,
    /// Event type
    evid: isize,
    /// Event time
    time: f64,
    /// Infusion duration
    #[serde(deserialize_with = "deserialize_option_f64")]
    dur: Option<f64>,
    /// Dose amount
    #[serde(deserialize_with = "deserialize_option_f64")]
    dose: Option<f64>,
    /// Additional doses
    #[serde(deserialize_with = "deserialize_option_isize")]
    addl: Option<isize>,
    /// Dosing interval
    #[serde(deserialize_with = "deserialize_option_f64")]
    ii: Option<f64>,
    /// Input compartment
    #[serde(deserialize_with = "deserialize_option_usize")]
    input: Option<usize>,
    /// Observed value
    #[serde(deserialize_with = "deserialize_option_f64")]
    out: Option<f64>,
    /// Corresponding output equation for the observation
    #[serde(deserialize_with = "deserialize_option_usize")]
    outeq: Option<usize>,
    /// First element of the error polynomial
    #[serde(deserialize_with = "deserialize_option_f64")]
    c0: Option<f64>,
    /// Second element of the error polynomial
    #[serde(deserialize_with = "deserialize_option_f64")]
    c1: Option<f64>,
    /// Third element of the error polynomial
    #[serde(deserialize_with = "deserialize_option_f64")]
    c2: Option<f64>,
    /// Fourth element of the error polynomial
    #[serde(deserialize_with = "deserialize_option_f64")]
    c3: Option<f64>,
    /// All other columns are covariates
    #[serde(deserialize_with = "deserialize_covs", flatten)]
    covs: HashMap<String, Option<f64>>,
}

impl Row {
    /// Get the error polynomial coefficients
    fn get_errorpoly(&self) -> Option<(f64, f64, f64, f64)> {
        match (self.c0, self.c1, self.c2, self.c3) {
            (Some(c0), Some(c1), Some(c2), Some(c3)) => Some((c0, c1, c2, c3)),
            _ => None,
        }
    }
    fn parse_events(self) -> Result<Vec<Event>, PmetricsError> {
        let mut events: Vec<Event> = Vec::new();

        match self.evid {
            0 => events.push(Event::Observation(Observation::new(
                self.time,
                self.out
                    .ok_or_else(|| PmetricsError::MissingObservationOut {
                        id: self.id.clone(),
                        time: self.time,
                    })?,
                self.outeq
                    .ok_or_else(|| PmetricsError::MissingObservationOuteq {
                        id: self.id.clone(),
                        time: self.time,
                    })?
                    - 1,
                self.get_errorpoly(),
                self.out == Some(-99.0),
            ))),
            1 | 4 => {
                let event = if self.dur.unwrap_or(0.0) > 0.0 {
                    Event::Infusion(Infusion::new(
                        self.time,
                        self.dose
                            .ok_or_else(|| PmetricsError::MissingInfusionDose {
                                id: self.id.clone(),
                                time: self.time,
                            })?,
                        self.input
                            .ok_or_else(|| PmetricsError::MissingInfusionInput {
                                id: self.id.clone(),
                                time: self.time,
                            })?
                            - 1,
                        self.dur.ok_or_else(|| PmetricsError::MissingInfusionDur {
                            id: self.id.clone(),
                            time: self.time,
                        })?,
                    ))
                } else {
                    Event::Bolus(Bolus::new(
                        self.time,
                        self.dose.ok_or_else(|| PmetricsError::MissingBolusDose {
                            id: self.id.clone(),
                            time: self.time,
                        })?,
                        self.input.ok_or_else(|| PmetricsError::MissingBolusInput {
                            id: self.id,
                            time: self.time,
                        })? - 1,
                    ))
                };
                if self.addl.is_some() && self.ii.is_some() {
                    if self.addl.unwrap_or(0) != 0 && self.ii.unwrap_or(0.0) > 0.0 {
                        let mut ev = event.clone();
                        let interval = &self.ii.unwrap().abs();
                        let repetitions = &self.addl.unwrap().abs();
                        let direction = &self.addl.unwrap().signum();

                        for _ in 0..*repetitions {
                            ev.inc_time((*direction as f64) * interval);
                            events.push(ev.clone());
                        }
                    }
                }
                events.push(event);
            }
            _ => {
                return Err(PmetricsError::UnknownEvid {
                    evid: self.evid,
                    id: self.id.clone(),
                    time: self.time,
                });
            }
        };
        Ok(events)
    }
}

/// Deserialize Option<T> from a string
fn deserialize_option<'de, T, D>(deserializer: D) -> Result<Option<T>, D::Error>
where
    D: Deserializer<'de>,
    T: FromStr,
    T::Err: std::fmt::Display,
{
    let s: String = Deserialize::deserialize(deserializer)?;
    if s.is_empty() || s == "." {
        Ok(None)
    } else {
        T::from_str(&s).map(Some).map_err(serde::de::Error::custom)
    }
}

fn deserialize_option_f64<'de, D>(deserializer: D) -> Result<Option<f64>, D::Error>
where
    D: Deserializer<'de>,
{
    deserialize_option::<f64, D>(deserializer)
}

fn deserialize_option_usize<'de, D>(deserializer: D) -> Result<Option<usize>, D::Error>
where
    D: Deserializer<'de>,
{
    deserialize_option::<usize, D>(deserializer)
}

fn deserialize_option_isize<'de, D>(deserializer: D) -> Result<Option<isize>, D::Error>
where
    D: Deserializer<'de>,
{
    deserialize_option::<isize, D>(deserializer)
}

fn deserialize_covs<'de, D>(deserializer: D) -> Result<HashMap<String, Option<f64>>, D::Error>
where
    D: Deserializer<'de>,
{
    struct CovsVisitor;

    impl<'de> Visitor<'de> for CovsVisitor {
        type Value = HashMap<String, Option<f64>>;

        fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
            formatter.write_str(
                "a map of string keys to optionally floating-point numbers or placeholders",
            )
        }

        fn visit_map<M>(self, mut map: M) -> Result<Self::Value, M::Error>
        where
            M: MapAccess<'de>,
        {
            let mut covs = HashMap::new();
            while let Some((key, value)) = map.next_entry::<String, serde_json::Value>()? {
                let opt_value = match value {
                    serde_json::Value::String(s) => match s.as_str() {
                        "" => None,
                        "." => None,
                        _ => match s.parse::<f64>() {
                            Ok(val) => Some(val),
                            Err(_) => {
                                return Err(de::Error::custom(
                                    "expected a floating-point number or empty string",
                                ))
                            }
                        },
                    },
                    serde_json::Value::Number(n) => Some(n.as_f64().unwrap()),
                    _ => return Err(de::Error::custom("expected a string or number")),
                };
                covs.insert(key, opt_value);
            }
            Ok(covs)
        }
    }

    deserializer.deserialize_map(CovsVisitor)
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_addl() {
        let data = read_pmetrics("src/tests/data/addl_test.csv");

        assert!(data.is_ok(), "Failed to parse data");

        let data = data.unwrap();
        let subjects = data.get_subjects();
        let first_subject = subjects.get(0).unwrap();
        let second_subject = subjects.get(1).unwrap();
        let s1_occasions = first_subject.occasions();
        let s2_occasions = second_subject.occasions();
        let first_scenario = s1_occasions.get(0).unwrap();
        let second_scenario = s2_occasions.get(0).unwrap();

        let s1_times = first_scenario
            .events()
            .iter()
            .map(|e| e.get_time())
            .collect::<Vec<_>>();

        // Negative ADDL, observations shifted forward

        assert_eq!(
            s1_times,
            vec![-120.0, -108.0, -96.0, -84.0, -72.0, -60.0, -48.0, -36.0, -24.0, -12.0, 0.0, 9.0]
        );

        let s2_times = second_scenario
            .events()
            .iter()
            .map(|e| e.get_time())
            .collect::<Vec<_>>();

        // Positive ADDL, no shift in observations

        assert_eq!(
            s2_times,
            vec![0.0, 9.0, 12.0, 24.0, 36.0, 48.0, 60.0, 72.0, 84.0, 96.0, 108.0, 120.0]
        );
    }
}
