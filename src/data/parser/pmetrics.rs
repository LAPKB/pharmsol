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
/// For specific column definitions, see the [Row] struct.
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

    // This is the object we are building, which can be converted to [Data]
    // Read the datafile into a hashmap of rows by ID
    let mut rows_map: HashMap<String, Vec<Row>> = HashMap::new();
    let mut subjects: Vec<Subject> = Vec::new();
    for row_result in reader.deserialize() {
        let row: Row = row_result.map_err(|e| PmetricsError::CSVError(e.to_string()))?;

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

            // Parse covariates - collect raw observations
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

            // Parse the raw covariate observations and build covariates
            let covariates = Covariates::from_pmetrics_observations(&observed_covariates);

            // Create the occasion
            let mut occasion = Occasion::new(block_index);
            events.iter_mut().for_each(|e| e.set_occasion(block_index));
            occasion.events = events;
            occasion.covariates = covariates;
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
    /// Get the error polynomial coefficients
    fn get_errorpoly(&self) -> Option<ErrorPoly> {
        match (self.c0, self.c1, self.c2, self.c3) {
            (Some(c0), Some(c1), Some(c2), Some(c3)) => Some(ErrorPoly::new(c0, c1, c2, c3)),
            _ => None,
        }
    }
    fn parse_events(self) -> Result<Vec<Event>, PmetricsError> {
        let mut events: Vec<Event> = Vec::new();

        match self.evid {
            0 => events.push(Event::Observation(Observation::new(
                self.time,
                if self.out == Some(-99.0) {
                    None
                } else {
                    self.out
                },
                self.outeq
                    .ok_or_else(|| PmetricsError::MissingObservationOuteq {
                        id: self.id.clone(),
                        time: self.time,
                    })?
                    - 1,
                self.get_errorpoly(),
                0,
                self.cens.unwrap_or(Censor::None),
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
                        0,
                    ))
                } else {
                    Event::Bolus(Bolus::new(
                        self.time,
                        self.dose.ok_or_else(|| PmetricsError::MissingBolusDose {
                            id: self.id.clone(),
                            time: self.time,
                        })?,
                        self.input.ok_or(PmetricsError::MissingBolusInput {
                            id: self.id,
                            time: self.time,
                        })? - 1,
                        0,
                    ))
                };
                if self.addl.is_some()
                    && self.ii.is_some()
                    && self.addl.unwrap_or(0) != 0
                    && self.ii.unwrap_or(0.0) > 0.0
                {
                    let mut ev = event.clone();
                    let interval = &self.ii.unwrap().abs();
                    let repetitions = &self.addl.unwrap().abs();
                    let direction = &self.addl.unwrap().signum();

                    for _ in 0..*repetitions {
                        ev.inc_time((*direction as f64) * interval);
                        events.push(ev.clone());
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
                "ID", "EVID", "TIME", "DUR", "DOSE", "ADDL", "II", "INPUT", "OUT", "OUTEQ", "C0",
                "C1", "C2", "C3",
            ])
            .map_err(|e| PharmsolError::OtherError(e.to_string()))?;

        for subject in self.subjects() {
            for occasion in subject.occasions() {
                for event in occasion.process_events(None, false) {
                    match event {
                        Event::Observation(obs) => {
                            // Write each field individually
                            writer
                                .write_record([
                                    subject.id(),
                                    &"0".to_string(),
                                    &obs.time().to_string(),
                                    &".".to_string(),
                                    &".".to_string(),
                                    &".".to_string(),
                                    &".".to_string(),
                                    &".".to_string(),
                                    &obs.value().map_or(".".to_string(), |v| v.to_string()),
                                    &(obs.outeq() + 1).to_string(),
                                    &".".to_string(),
                                    &".".to_string(),
                                    &".".to_string(),
                                    &".".to_string(),
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
}
