use crate::data::*;
use serde::de::{MapAccess, Visitor};
use serde::{de, Deserialize, Deserializer};
use std::collections::HashMap;
use std::fmt;

use std::str::FromStr;
use thiserror::Error;

#[derive(Debug)]
struct PmetricsRow {
    id: String,
    evid: usize,
    time: f64,
    dur: Option<f64>,
    dose: Option<f64>,
    addl: Option<usize>,
    ii: Option<f64>,
    input: Option<usize>,
    out: Option<f64>,
    outeq: Option<usize>,
    c0: Option<f64>,
    c1: Option<f64>,
    c2: Option<f64>,
    c3: Option<f64>,
    // Covariates is provided as a HashMap, with keys (name, time) and value (f64)
    covs: HashMap<String, f64>,
}

// Define specific errors for Pmetrics parsing
#[derive(Error, Debug, PartialEq)]
pub enum PmetricsError {
    #[error("Missing required field: {0}")]
    MissingField(String),
    #[error("Invalid value for field {field}: {message}")]
    InvalidValue { field: String, message: String },
    #[error("CSV parsing error: {0}")]
    CsvError(String), // Wrap underlying CSV errors
    #[error("Unknown EVID: {0}")]
    UnknownEvid(usize),
}

// Implement From<csv::Error> for PmetricsError
impl From<csv::Error> for PmetricsError {
    fn from(err: csv::Error) -> Self {
        PmetricsError::CsvError(err.to_string())
    }
}

struct PmetricsVisitor;

impl<'de> Visitor<'de> for PmetricsVisitor {
    type Value = PmetricsRow;

    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.write_str("a map representing a Pmetrics CSV row")
    }

    fn visit_map<M>(self, mut map: M) -> Result<Self::Value, M::Error>
    where
        M: MapAccess<'de>,
    {
        let mut id: Option<String> = None;
        let mut evid: Option<usize> = None;
        let mut time: Option<f64> = None;
        let mut dur: Option<f64> = None;
        let mut dose: Option<f64> = None;
        let mut addl: Option<usize> = None;
        let mut ii: Option<f64> = None;
        let mut input: Option<usize> = None;
        let mut out: Option<f64> = None;
        let mut outeq: Option<usize> = None;
        let mut c0: Option<f64> = None;
        let mut c1: Option<f64> = None;
        let mut c2: Option<f64> = None;
        let mut c3: Option<f64> = None;
        let mut covs: HashMap<String, f64> = HashMap::new();

        // Helper to parse optional fields that might be empty strings
        fn parse_optional<'de, T, E>(value: Option<&str>) -> Result<Option<T>, E>
        where
            T: FromStr,
            E: de::Error,
            T::Err: fmt::Display,
        {
            // TODO: Allows "." as empty
            match value {
                Some(s) if !s.is_empty() => s
                    .parse::<T>()
                    .map(Some)
                    .map_err(|e| E::custom(format!("Failed to parse value '{}': {}", s, e))),
                _ => Ok(None),
            }
        }

        while let Some((key, value)) = map.next_entry::<String, Option<String>>()? {
            let value_str = value.as_deref(); // Get Option<&str>
            match key.to_lowercase().as_str() {
                "id" => id = value_str.map(String::from),
                "evid" => evid = parse_optional(value_str)?,
                "time" => time = parse_optional(value_str)?,
                "dur" => dur = parse_optional(value_str)?,
                "dose" => dose = parse_optional(value_str)?,
                "addl" => addl = parse_optional(value_str)?,
                "ii" => ii = parse_optional(value_str)?,
                "input" => input = parse_optional(value_str)?,
                "out" => out = parse_optional(value_str)?,
                "outeq" => outeq = parse_optional(value_str)?,
                "c0" => c0 = parse_optional(value_str)?,
                "c1" => c1 = parse_optional(value_str)?,
                "c2" => c2 = parse_optional(value_str)?,
                "c3" => c3 = parse_optional(value_str)?,

                // Collect any other columns as covariates
                other_key => {
                    if let Some(val_float) = parse_optional::<f64, M::Error>(value_str)? {
                        covs.insert(other_key.to_string(), val_float);
                    }
                    // Decide how to handle non-f64 covariates if needed
                    // else { warn!("Covariate '{}' has non-numeric value '{:?}', skipping", other_key, value_str); }
                }
            }
        }

        // Check for required fields
        let id = id.ok_or_else(|| de::Error::missing_field("id"))?;
        let evid = evid.ok_or_else(|| de::Error::missing_field("evid"))?;
        let time = time.ok_or_else(|| de::Error::missing_field("time"))?;

        Ok(PmetricsRow {
            id,
            evid,
            time,
            dur,
            dose,
            addl,
            ii,
            input,
            out,
            outeq,
            c0,
            c1,
            c2,
            c3,
            covs,
        })
    }
}

// Implement Deserialize manually using the visitor
impl<'de> Deserialize<'de> for PmetricsRow {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_map(PmetricsVisitor)
    }
}

// Conversion from PmetricsRow to Vec<Event>, returning PmetricsError on failure
impl TryFrom<PmetricsRow> for Vec<Event> {
    type Error = PmetricsError;

    fn try_from(row: PmetricsRow) -> Result<Self, Self::Error> {
        let covariates = if row.covs.is_empty() {
            None
        } else {
            Some(row.covs)
        };
        match row.evid {
            // Observation Event (EVID 0)
            0 => {
                let time = row.time;
                let value = row
                    .out
                    .ok_or_else(|| PmetricsError::MissingField("out".to_string()))?;
                let outeq = row
                    .outeq
                    .ok_or_else(|| PmetricsError::MissingField("outeq".to_string()))?;

                // Create the errorpoly tuple
                let errorpoly = match (row.c0, row.c1, row.c2, row.c3) {
                    (Some(c0), Some(c1), Some(c2), Some(c3)) => Some((c0, c1, c2, c3)),
                    _ => None,
                };

                // If out == -99, set ignore to TRUE
                let ignore = if value == -99.0 { true } else { false };

                let obs = Observation::new(time, value, outeq, errorpoly, ignore);
                Ok(vec![Event::Observation(obs)])
            }
            // Dosing Event (EVID 1 or EVID 4)
            1 | 4 => {
                // Minimum information
                let time = row.time;
                let amount = row
                    .dose
                    .ok_or_else(|| PmetricsError::MissingField("dose".to_string()))?;
                let input = row
                    .input
                    .ok_or_else(|| PmetricsError::MissingField("input".to_string()))?;

                // If dur is None or 0, set to None
                let dur = if let Some(duration) = row.dur {
                    if duration > 0.0 {
                        Some(duration)
                    } else {
                        None
                    }
                } else {
                    None
                };

                // For addl and ii, both must be provided, otherwise they are both none.
                // ii must be > 0.0
                let ii = if let Some(interval) = row.ii {
                    if interval > 0.0 {
                        Some(interval)
                    } else {
                        None
                    }
                } else {
                    None
                };
                let addl = if let Some(additional) = row.addl {
                    if additional > 0 {
                        Some(additional)
                    } else {
                        None
                    }
                } else {
                    None
                };

                let mut events: Vec<Event> = Vec::new();

                match addl {
                    Some(addl) => {
                        // Create additional events
                        for i in 0..addl {
                            let time = time + (i as f64) * ii.unwrap_or(0.0);
                            // If duration is some, we have an infusion
                            match dur {
                                Some(duration) => {
                                    let infusion = Infusion::new(time, amount, input, duration);
                                    events.push(Event::Infusion(infusion));
                                }
                                None => {
                                    // If duration is None, we have a bolus
                                    let bolus = Bolus::new(time, amount, input);
                                    events.push(Event::Bolus(bolus));
                                }
                            }
                        }
                    }
                    None => {
                        match dur {
                            Some(duration) => {
                                let infusion = Infusion::new(time, amount, input, duration);
                                events.push(Event::Infusion(infusion));
                            }
                            None => {
                                // If duration is None, we have a bolus
                                let bolus = Bolus::new(time, amount, input);
                                events.push(Event::Bolus(bolus));
                            }
                        }
                    }
                }

                Ok(events)
            }
            // Unknown EVID
            _ => Err(PmetricsError::UnknownEvid(row.evid)),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::parser::pmetrics::PmetricsRow;

    #[test]
    fn test_parse_dose() {
        let data = r#"
            id,evid,time,dur,dose,addl,ii,input,out,outeq,c0,c1,c2,c3
            1,1,0.0,0.0,100.0,0,0,1,0.0,1,0.0,0.0,0.0,0.0
            1,2,24.0,,100.0,,24,,1.5,,,
            2,1,12.0,,50.0,,12,,2.5,,,
        "#;

        let parsed_data: Vec<PmetricsRow> = serde_json::from_str(data).unwrap();
        assert_eq!(parsed_data.len(), 3);
    }
}
