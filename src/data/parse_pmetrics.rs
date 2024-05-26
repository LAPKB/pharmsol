use crate::data::*;
use serde::de::{MapAccess, Visitor};
use serde::{de, Deserialize, Deserializer, Serialize};
use std::collections::HashMap;
use std::path::Path;
use std::str::FromStr;
use std::{error::Error, fmt};

/// Read a Pmetrics datafile and convert it to a [Data] object
///
/// For specific details, see the [Row] struct.
#[allow(dead_code)]
pub fn read_pmetrics(path: &Path) -> Result<Data, Box<dyn Error>> {
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
                let event: Event = Event::from(row);
                events.push(event);
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

                let mut last_value = None;
                for i in 0..occurrences.len() {
                    let (time, value) = occurrences[i];
                    let next_occurrence = occurrences.get(i + 1);
                    let to_time =
                        next_occurrence.map_or(f64::INFINITY, |&(next_time, _)| next_time);

                    if is_fixed {
                        // Use CarryForward for fixed covariates
                        covariate.add_segment(CovariateSegment {
                            from: time,
                            to: to_time,
                            method: InterpolationMethod::CarryForward {
                                value: value.unwrap(),
                            },
                        });
                    } else if let Some(next) = next_occurrence {
                        // Linear interpolation for non-fixed covariates
                        let (next_time, next_value) = next;
                        if let Some(current_value) = value {
                            let slope = (next_value.unwrap() - current_value) / (next_time - time);
                            covariate.add_segment(CovariateSegment {
                                from: time,
                                to: *next_time,
                                method: InterpolationMethod::Linear {
                                    slope,
                                    intercept: current_value - slope * time,
                                },
                            });
                            last_value = Some((next_time, next_value));
                        }
                    } else if let Some((last_time, last_value)) = last_value {
                        // Extend the last linear segment to infinity if no more segments are available
                        covariate.add_segment(CovariateSegment {
                            from: *last_time,
                            to: f64::INFINITY,
                            method: InterpolationMethod::CarryForward {
                                value: last_value.unwrap(),
                            },
                        });
                    }
                }
                covariates.add_covariate(name, covariate)
            }
            // Create the block
            let occasion = Occasion::new(events, covariates, block_index);
            occasions.push(occasion);
        }

        let subject = Subject { id, occasions };
        subjects.push(subject);
    }

    // Sort subjects alphabetically by ID to get consistent ordering
    subjects.sort_by(|a, b| a.id.cmp(&b.id));
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
    #[serde(deserialize_with = "deserialize_option_isize")]
    ii: Option<isize>,
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
}

///
impl From<Row> for Event {
    fn from(row: Row) -> Self {
        match row.evid {
            0 => Event::Observation(Observation {
                time: row.time,
                value: row.out.expect("Observation OUT is missing"),
                outeq: row.outeq.expect("Observation OUTEQ is missing") - 1,
                errorpoly: row.get_errorpoly(),
                ignore: row.out == Some(-99.0),
            }),
            1 | 4 => {
                if row.dur.unwrap_or(0.0) > 0.0 {
                    Event::Infusion(Infusion {
                        time: row.time,
                        amount: row.dose.expect("Infusion amount (DOSE) is missing"),
                        input: row.input.expect("Infusion compartment (INPUT) is missing") - 1,
                        duration: row.dur.expect("Infusion duration (DUR) is missing"),
                    })
                } else {
                    Event::Bolus(Bolus {
                        time: row.time,
                        amount: row.dose.expect("Bolus amount (DOSE) is missing"),
                        input: row.input.expect("Bolus compartment (INPUT) is missing") - 1,
                    })
                }
            }
            _ => panic!(
                "Unknown EVID: {} for ID {} at time {}",
                row.evid, row.id, row.time
            ),
        }
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

// #[cfg(test)]
// mod tests {
//     use super::*;

//     #[test]
//     fn test_read_pmetrics() {
//         let path = std::path::Path::new("examples/data/bimodal_ke_blocks.csv");
//         let data = read_pmetrics(&path).unwrap();

//         assert_eq!(data.nsubjects(), 1);
//         assert_eq!(data.nobs(), 30);

//         println!("{}", data);
//     }
// }
