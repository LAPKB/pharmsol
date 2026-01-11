use crate::{data::*, PharmsolError};
use csv::WriterBuilder;
use serde::de::{MapAccess, Visitor};
use serde::{de, Deserialize, Deserializer, Serialize};
use std::collections::HashMap;

use std::fmt;
use std::str::FromStr;
use thiserror::Error;

/// Custom error type for the module
#[allow(private_interfaces)]
#[derive(Error, Debug, Clone)]
pub enum PmetricsError {
    /// Error encountered when reading CSV data
    #[error("CSV error: {0}")]
    CSVError(String),
    /// Error during data deserialization
    #[error("Parse error: {0}")]
    SerdeError(String),
    /// Encountered an unknown EVID value
    #[error("Unknown EVID: {evid} for ID {id} at time {time}")]
    UnknownEvid { evid: isize, id: String, time: f64 },
    /// Required observation value (OUT) is missing
    #[error("Observation OUT is missing for {id} at time {time}")]
    MissingObservationOut { id: String, time: f64 },
    /// Required observation output equation (OUTEQ) is missing
    #[error("Observation OUTEQ is missing in for {id} at time {time}")]
    MissingObservationOuteq { id: String, time: f64 },
    /// Required infusion dose amount is missing
    #[error("Infusion amount (DOSE) is missing for {id} at time {time}")]
    MissingInfusionDose { id: String, time: f64 },
    /// Required infusion input compartment is missing
    #[error("Infusion compartment (INPUT) is missing for {id} at time {time}")]
    MissingInfusionInput { id: String, time: f64 },
    /// Required infusion duration is missing
    #[error("Infusion duration (DUR) is missing for {id} at time {time}")]
    MissingInfusionDur { id: String, time: f64 },
    /// Required bolus dose amount is missing
    #[error("Bolus amount (DOSE) is missing for {id} at time {time}")]
    MissingBolusDose { id: String, time: f64 },
    /// Required bolus input compartment is missing
    #[error("Bolus compartment (INPUT) is missing for {id} at time {time}")]
    MissingBolusInput { id: String, time: f64 },
}

/// Read a Pmetrics datafile and convert it to a [Data] object
///
/// This function parses a Pmetrics-formatted CSV file and constructs a [Data] object containing the structured
/// pharmacokinetic/pharmacodynamic data. The function handles various data formats including doses, observations,
/// and covariates.
///
/// # Arguments
///
/// * `path` - The path to the Pmetrics CSV file
///
/// # Returns
///
/// * `Result<Data, PmetricsError>` - A result containing either the parsed [Data] object or an error
///
/// # Example
///
/// ```rust,no_run
/// use pharmsol::prelude::data::read_pmetrics;
///
/// let data = read_pmetrics("path/to/pmetrics_data.csv").unwrap();
/// println!("Number of subjects: {}", data.subjects().len());
/// ```
///
/// # Format details
///
/// The Pmetrics format expects columns like ID, TIME, EVID, DOSE, DUR, etc. The function will:
/// - Convert all headers to lowercase for case-insensitivity
/// - Group rows by subject ID
/// - Create occasions based on EVID=4 events
/// - Parse covariates and create appropriate interpolations
/// - Handle additional doses via ADDL and II fields
///
/// For specific column definitions, see the `Row` struct.
#[allow(dead_code)]
pub fn read_pmetrics(path: impl Into<String>) -> Result<Data, PmetricsError> {
    let path = path.into();

    let mut reader = csv::ReaderBuilder::new()
        .comment(Some(b'#'))
        .has_headers(true)
        .from_path(&path)
        .map_err(|e| PmetricsError::CSVError(e.to_string()))?;
    // Convert headers to lowercase
    let headers = reader
        .headers()
        .map_err(|e| PmetricsError::CSVError(e.to_string()))?
        .iter()
        .map(|h| h.to_lowercase())
        .collect::<Vec<_>>();
    reader.set_headers(csv::StringRecord::from(headers));

    // Parse CSV rows and convert to NormalizedRows
    let mut normalized_rows: Vec<super::normalized::NormalizedRow> = Vec::new();
    for row_result in reader.deserialize() {
        let row: Row = row_result.map_err(|e| PmetricsError::CSVError(e.to_string()))?;
        normalized_rows.push(row.to_normalized());
    }

    // Use the shared build_data logic
    super::normalized::build_data(normalized_rows)
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
    /// Censoring output
    #[serde(default, deserialize_with = "deserialize_option_censor")]
    cens: Option<Censor>,
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
    /// Convert this Row to a NormalizedRow for parsing
    fn to_normalized(&self) -> super::normalized::NormalizedRow {
        super::normalized::NormalizedRow {
            id: self.id.clone(),
            time: self.time,
            evid: self.evid as i32,
            dose: self.dose,
            dur: self.dur,
            addl: self.addl.map(|a| a as i64),
            ii: self.ii,
            input: self.input,
            // Treat -99 as missing value (Pmetrics convention)
            out: self
                .out
                .and_then(|v| if v == -99.0 { None } else { Some(v) }),
            outeq: self.outeq,
            cens: self.cens,
            c0: self.c0,
            c1: self.c1,
            c2: self.c2,
            c3: self.c3,
            covariates: self
                .covs
                .iter()
                .filter_map(|(k, v)| v.map(|val| (k.clone(), val)))
                .collect(),
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
    if s.is_empty() || s == "." || s == "NA" {
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

fn deserialize_option_censor<'de, D>(deserializer: D) -> Result<Option<Censor>, D::Error>
where
    D: Deserializer<'de>,
{
    let s: String = Deserialize::deserialize(deserializer)?;
    if s.is_empty() || s == "." || s == "NA" {
        Ok(None)
    } else {
        match s.as_str() {
            "1" | "bloq" => Ok(Some(Censor::BLOQ)),
            "0" | "none" => Ok(Some(Censor::None)),
            "-1" | "aloq" => Ok(Some(Censor::ALOQ)),
            _ => Err(serde::de::Error::custom(format!(
                "Expected one of 1/-1/0 or bloq/aloq/none), got {}",
                s
            ))),
        }
    }
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

impl Data {
    /// Write the dataset to a file in Pmetrics format
    ///
    /// # Arguments
    ///
    /// * `file` - The file to write to
    pub fn write_pmetrics(&self, file: &std::fs::File) -> Result<(), PharmsolError> {
        let mut writer = WriterBuilder::new().has_headers(true).from_writer(file);

        writer
            .write_record([
                "ID", "EVID", "TIME", "DUR", "DOSE", "ADDL", "II", "INPUT", "OUT", "OUTEQ", "CENS",
                "C0", "C1", "C2", "C3",
            ])
            .map_err(|e| PharmsolError::OtherError(e.to_string()))?;

        for subject in self.subjects() {
            for occasion in subject.occasions() {
                for event in occasion.process_events(None, false) {
                    match event {
                        Event::Observation(obs) => {
                            let time = obs.time().to_string();
                            let value = obs
                                .value()
                                .map_or_else(|| ".".to_string(), |v| v.to_string());
                            let outeq = (obs.outeq() + 1).to_string();
                            let censor = match obs.censoring() {
                                Censor::None => "0".to_string(),
                                Censor::BLOQ => "1".to_string(),
                                Censor::ALOQ => "-1".to_string(),
                            };
                            let (c0, c1, c2, c3) = obs
                                .errorpoly()
                                .map(|poly| {
                                    let (c0, c1, c2, c3) = poly.coefficients();
                                    (
                                        c0.to_string(),
                                        c1.to_string(),
                                        c2.to_string(),
                                        c3.to_string(),
                                    )
                                })
                                .unwrap_or_else(|| {
                                    (
                                        ".".to_string(),
                                        ".".to_string(),
                                        ".".to_string(),
                                        ".".to_string(),
                                    )
                                });

                            // Write each field individually
                            writer
                                .write_record([
                                    subject.id(),
                                    &"0".to_string(),
                                    &time,
                                    &".".to_string(),
                                    &".".to_string(),
                                    &".".to_string(),
                                    &".".to_string(),
                                    &".".to_string(),
                                    &value,
                                    &outeq,
                                    &censor,
                                    &c0,
                                    &c1,
                                    &c2,
                                    &c3,
                                ])
                                .map_err(|e| PharmsolError::OtherError(e.to_string()))?;
                        }
                        Event::Infusion(inf) => {
                            writer
                                .write_record([
                                    subject.id(),
                                    &"1".to_string(),
                                    &inf.time().to_string(),
                                    &inf.duration().to_string(),
                                    &inf.amount().to_string(),
                                    &".".to_string(),
                                    &".".to_string(),
                                    &(inf.input() + 1).to_string(),
                                    &".".to_string(),
                                    &".".to_string(),
                                    &".".to_string(),
                                    &".".to_string(),
                                    &".".to_string(),
                                    &".".to_string(),
                                    &".".to_string(),
                                ])
                                .map_err(|e| PharmsolError::OtherError(e.to_string()))?;
                        }
                        Event::Bolus(bol) => {
                            writer
                                .write_record([
                                    subject.id(),
                                    &"1".to_string(),
                                    &bol.time().to_string(),
                                    &"0".to_string(),
                                    &bol.amount().to_string(),
                                    &".".to_string(),
                                    &".".to_string(),
                                    &(bol.input() + 1).to_string(),
                                    &".".to_string(),
                                    &".".to_string(),
                                    &".".to_string(),
                                    &".".to_string(),
                                    &".".to_string(),
                                    &".".to_string(),
                                    &".".to_string(),
                                ])
                                .map_err(|e| PharmsolError::OtherError(e.to_string()))?;
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::{Censor, ErrorPoly, SubjectBuilderExt};
    use csv::ReaderBuilder;
    use std::io::Cursor;
    use tempfile::NamedTempFile;

    #[test]
    fn test_addl() {
        let data = read_pmetrics("src/tests/data/addl_test.csv");

        assert!(data.is_ok(), "Failed to parse data");

        let data = data.unwrap();
        let subjects = data.subjects();
        let first_subject = subjects.first().unwrap();
        let second_subject = subjects.get(1).unwrap();
        let s1_occasions = first_subject.occasions();
        let s2_occasions = second_subject.occasions();
        let first_scenario = s1_occasions.first().unwrap();
        let second_scenario = s2_occasions.first().unwrap();

        let s1_times = first_scenario
            .events()
            .iter()
            .map(|e| e.time())
            .collect::<Vec<_>>();

        // Negative ADDL, observations shifted forward

        assert_eq!(
            s1_times,
            vec![-120.0, -108.0, -96.0, -84.0, -72.0, -60.0, -48.0, -36.0, -24.0, -12.0, 0.0, 9.0]
        );

        let s2_times = second_scenario
            .events()
            .iter()
            .map(|e| e.time())
            .collect::<Vec<_>>();

        // Positive ADDL, no shift in observations

        assert_eq!(
            s2_times,
            vec![0.0, 9.0, 12.0, 24.0, 36.0, 48.0, 60.0, 72.0, 84.0, 96.0, 108.0, 120.0]
        );
    }

    #[test]
    fn write_pmetrics_preserves_infusion_input() {
        let subject = Subject::builder("writer")
            .infusion(0.0, 200.0, 2, 1.0)
            .observation(1.0, 0.0, 0)
            .build();
        let data = Data::new(vec![subject]);

        let file = NamedTempFile::new().unwrap();
        data.write_pmetrics(file.as_file()).unwrap();

        let contents = std::fs::read_to_string(file.path()).unwrap();
        let mut reader = ReaderBuilder::new()
            .has_headers(true)
            .from_reader(Cursor::new(contents));

        let infusion_row = reader
            .records()
            .filter_map(Result::ok)
            .find(|record| record.get(3) != Some("0"))
            .expect("infusion row missing");

        assert_eq!(infusion_row.get(7), Some("3"));
    }

    #[test]
    fn write_pmetrics_preserves_censoring_and_errorpoly() {
        let subject = Subject::builder("writer")
            .observation_with_error(
                0.0,
                2.5,
                0,
                ErrorPoly::new(0.1, 0.2, 0.3, 0.4),
                Censor::BLOQ,
            )
            .censored_observation(1.0, 3.5, 1, Censor::ALOQ)
            .build();
        let data = Data::new(vec![subject]);

        let file = NamedTempFile::new().unwrap();
        data.write_pmetrics(file.as_file()).unwrap();

        let contents = std::fs::read_to_string(file.path()).unwrap();
        let mut reader = ReaderBuilder::new()
            .has_headers(true)
            .from_reader(Cursor::new(contents));

        let mut observations: Vec<_> = reader
            .records()
            .filter_map(Result::ok)
            .filter(|record| record.get(1) == Some("0"))
            .collect();

        assert_eq!(observations.len(), 2, "expected two observation rows");

        let first = observations.remove(0);
        assert_eq!(first.get(10), Some("1"));
        assert_eq!(first.get(11), Some("0.1"));
        assert_eq!(first.get(12), Some("0.2"));
        assert_eq!(first.get(13), Some("0.3"));
        assert_eq!(first.get(14), Some("0.4"));

        let second = observations.remove(0);
        assert_eq!(second.get(10), Some("-1"));
        assert_eq!(second.get(11), Some("."));
        assert_eq!(second.get(14), Some("."));
    }
}
