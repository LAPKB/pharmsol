//! Pmetrics CSV parsing and export helpers.
//!
//! This module reads and writes the Pmetrics-style tabular format while keeping
//! pharmsol's public input and output labels intact.
//!
//! `INPUT` and `OUTEQ` values are parsed as labels, not rewritten to dense
//! indices. Named values such as `iv` and `cp` are preserved exactly, and
//! numeric values such as `1` are preserved as numeric-looking labels.

use crate::data::*;
use csv::{ReaderBuilder, StringRecord};
use serde::de::{MapAccess, Visitor};
use serde::{de, Deserialize, Deserializer, Serialize};
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::str::FromStr;

use crate::data::row::{build_data, DataError, DataRow};

pub(super) const CORE_HEADERS: [&str; 15] = [
    "ID", "EVID", "TIME", "DUR", "DOSE", "ADDL", "II", "INPUT", "OUT", "OUTEQ", "CENS", "C0", "C1",
    "C2", "C3",
];

/// Read a Pmetrics CSV file into [`Data`].
///
/// Use [`read_pmetrics`] when the source file already follows the usual
/// Pmetrics column convention instead of mapping the file into [`DataRow`]
/// values yourself.
///
/// The parser normalizes header names to lowercase, preserves `INPUT` and
/// `OUTEQ` as public labels, expands `ADDL` dosing rows through the shared row
/// ingestion path, and groups rows into occasions using `EVID=4`.
///
/// All columns not claimed by the core Pmetrics schema are treated as
/// covariates. Column names are read without regard to capitalization. A
/// covariate header ending in `!` selects carry-forward behavior; otherwise its
/// values are interpolated. The same covariate cannot be declared in both
/// forms.
///
/// `ADDL`/`II` doses are expanded while reading. Export writes the expanded
/// doses as individual rows. Negative `ADDL` remains supported for `EVID=1`,
/// but not for an `EVID=4` reset whose identity would be lost during expansion.
/// `OUT=-99` represents a missing observation, so `-99` cannot be preserved as
/// a real observed value.
///
/// # Arguments
///
/// * `path` - Path to the Pmetrics CSV file
///
/// # Returns
///
/// A parsed [`Data`] object or a [`DataError`] if the file cannot be read or a
/// required row field is missing.
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
/// # Expected columns
///
/// The core columns are `ID`, `TIME`, `EVID`, `DOSE`, `DUR`, `ADDL`,
/// `II`, `INPUT`, `OUT`, `OUTEQ`, `CENS`, and optional `C0..C3` error
/// coefficients.
///
/// All other numeric columns are treated as covariates.
///
/// # Parsing behavior
///
/// The parser will:
/// - Convert all headers to lowercase for case-insensitivity
/// - Group rows by subject ID
/// - Create occasions based on EVID=4 events
/// - Parse covariates and create appropriate interpolations
/// - Handle additional doses via ADDL and II fields
/// - Preserve raw `INPUT` and `OUTEQ` labels as strings until model resolution
/// - Treat `OUT=-99` as a missing observation value, matching the common
///   Pmetrics convention
///
/// For specific column definitions, see the `Row` struct.
#[allow(dead_code)]
pub fn read_pmetrics(path: impl Into<String>) -> Result<Data, DataError> {
    let bytes =
        std::fs::read(path.into()).map_err(|error| DataError::CSVError(error.to_string()))?;
    Data::from_pmetrics_csv_bytes(&bytes)
}

impl Data {
    /// Read Pmetrics CSV bytes into a dataset.
    pub fn from_pmetrics_csv_bytes(bytes: &[u8]) -> Result<Data, DataError> {
        let mut reader = ReaderBuilder::new()
            .comment(Some(b'#'))
            .has_headers(true)
            .from_reader(bytes);
        let original_headers = reader
            .headers()
            .map_err(|error| DataError::CSVError(error.to_string()))?
            .clone();
        let mut core_names = HashSet::new();
        let mut covariate_forms = HashMap::<String, bool>::new();
        let mut headers = Vec::with_capacity(original_headers.len());

        for header in &original_headers {
            if let Some(core) = CORE_HEADERS
                .iter()
                .find(|core| core.eq_ignore_ascii_case(header))
            {
                let name = core.to_ascii_lowercase();
                if !core_names.insert(name.clone()) {
                    return Err(DataError::InvalidPmetricsData(format!(
                        "duplicate core header `{name}`"
                    )));
                }
                headers.push(name);
                continue;
            }

            validate_covariate_header(header)?;
            let fixed = header.ends_with('!');
            let base = header.strip_suffix('!').unwrap_or(header);
            let name = normalize_covariate_name(base);
            if let Some(previous_fixed) = covariate_forms.insert(name.clone(), fixed) {
                let message = if previous_fixed == fixed {
                    format!("duplicate covariate column `{name}`")
                } else {
                    format!("covariate `{name}` is declared both with and without trailing !")
                };
                return Err(DataError::InvalidPmetricsData(message));
            }
            headers.push(if fixed { format!("{name}!") } else { name });
        }
        reader.set_headers(StringRecord::from(headers));

        let mut data_rows = Vec::new();
        for row_result in reader.deserialize() {
            let row: Row = row_result.map_err(|error| DataError::CSVError(error.to_string()))?;
            row.validate()?;
            data_rows.push(row.to_datarow());
        }
        build_data(data_rows)
    }
}

pub(super) fn normalize_covariate_name(name: &str) -> String {
    name.to_lowercase()
}

pub(super) fn validate_covariate_header(header: &str) -> Result<(), DataError> {
    let base = header.strip_suffix('!').unwrap_or(header);
    if base.is_empty()
        || base.contains('!')
        || base.chars().any(char::is_control)
        || CORE_HEADERS
            .iter()
            .any(|core| core.eq_ignore_ascii_case(base))
    {
        return Err(DataError::InvalidPmetricsData(format!(
            "reserved or ambiguous covariate column `{header}`"
        )));
    }
    Ok(())
}

fn ensure_finite(value: f64, field: &str, id: &str) -> Result<(), DataError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(DataError::NonFiniteValue {
            field: field.to_string(),
            id: id.to_string(),
        })
    }
}

/// One row from a Pmetrics file after serde deserialization.
#[derive(Deserialize, Debug, Serialize, Default, Clone)]
#[serde(rename_all = "lowercase")]
struct Row {
    /// Subject ID
    id: String,
    /// Event type
    evid: i64,
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
    /// Input label from the `INPUT` column
    #[serde(deserialize_with = "deserialize_option_route_label")]
    input: Option<InputLabel>,
    /// Observed value
    #[serde(deserialize_with = "deserialize_option_f64")]
    out: Option<f64>,
    /// Output label from the `OUTEQ` column
    #[serde(deserialize_with = "deserialize_option_output_label")]
    outeq: Option<OutputLabel>,
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
    fn validate(&self) -> Result<(), DataError> {
        ensure_finite(self.time, "TIME", &self.id)?;
        for (field, value) in [
            ("DUR", self.dur),
            ("DOSE", self.dose),
            ("II", self.ii),
            ("OUT", self.out),
            ("C0", self.c0),
            ("C1", self.c1),
            ("C2", self.c2),
            ("C3", self.c3),
        ] {
            if let Some(value) = value {
                ensure_finite(value, field, &self.id)?;
            }
        }
        for (name, value) in &self.covs {
            if let Some(value) = value {
                ensure_finite(*value, name, &self.id)?;
            }
        }

        let coefficients = [self.c0, self.c1, self.c2, self.c3];
        let present = coefficients.iter().filter(|value| value.is_some()).count();
        if present != 0 && present != coefficients.len() {
            return Err(DataError::InvalidPmetricsData(format!(
                "partial error polynomial for {} at time {}",
                self.id, self.time
            )));
        }

        if self.addl.is_some_and(|addl| addl != 0) && !self.ii.is_some_and(|ii| ii > 0.0) {
            return Err(DataError::InvalidPmetricsData(format!(
                "nonzero ADDL for {} at time {} requires a positive II",
                self.id, self.time
            )));
        }
        if self.evid == 4 && self.addl.is_some_and(|addl| addl < 0) {
            return Err(DataError::InvalidPmetricsData(format!(
                "EVID=4 row for {} at time {} cannot use negative ADDL",
                self.id, self.time
            )));
        }

        match self.evid {
            1 | 4 if self.dur.is_some_and(|duration| duration < 0.0) => {
                return Err(DataError::InvalidPmetricsData(format!(
                    "dose row for {} at time {} contains a negative duration",
                    self.id, self.time
                )));
            }
            4 if self.dose.is_none() || self.input.is_none() => {
                return Err(DataError::InvalidPmetricsData(format!(
                    "EVID=4 row for {} at time {} must contain a dose and INPUT",
                    self.id, self.time
                )));
            }
            0 | 1 | 4 => {}
            unsupported => {
                return Err(DataError::InvalidPmetricsData(format!(
                    "unsupported EVID={unsupported} for subject {} at time {}",
                    self.id, self.time
                )));
            }
        }
        Ok(())
    }

    fn to_datarow(&self) -> DataRow {
        DataRow {
            id: self.id.clone(),
            time: self.time,
            evid: self.evid as i32,
            dose: self.dose,
            dur: self.dur,
            addl: self.addl.map(|a| a as i64),
            ii: self.ii,
            input: self.input.clone(),
            // Treat -99 as missing, matching the common Pmetrics convention.
            out: self.out.filter(|&value| value != -99.0),
            outeq: self.outeq.clone(),
            cens: self.cens,
            c0: self.c0,
            c1: self.c1,
            c2: self.c2,
            c3: self.c3,
            covariates: self
                .covs
                .iter()
                .filter_map(|(key, value)| value.map(|value| (key.clone(), value)))
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

fn deserialize_option_route_label<'de, D>(deserializer: D) -> Result<Option<InputLabel>, D::Error>
where
    D: Deserializer<'de>,
{
    deserialize_option::<String, D>(deserializer).map(|value| value.map(InputLabel::from))
}

fn deserialize_option_output_label<'de, D>(deserializer: D) -> Result<Option<OutputLabel>, D::Error>
where
    D: Deserializer<'de>,
{
    deserialize_option::<String, D>(deserializer).map(|value| value.map(OutputLabel::from))
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
                        "." | "NA" => None,
                        _ => match s.parse::<f64>() {
                            Ok(val) => Some(val),
                            Err(_) => {
                                return Err(de::Error::custom(
                                    "expected a floating-point number or empty string",
                                ))
                            }
                        },
                    },
                    serde_json::Value::Number(number) => {
                        Some(number.as_f64().ok_or_else(|| {
                            de::Error::custom("expected a finite floating-point number")
                        })?)
                    }
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
            .infusion(0.0, 200.0, 3, 1.0) // input=3 (1-indexed)
            .observation(1.0, 0.0, 1) // outeq=1 (1-indexed)
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
            .find(|record| record.get(1) == Some("1") && record.get(3) != Some("0"))
            .expect("infusion row missing");

        assert_eq!(infusion_row.get(7), Some("3")); // Written as-is (1-indexed)
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

    #[test]
    fn read_pmetrics_preserves_named_route_and_output_labels() {
        let file = NamedTempFile::new().unwrap();
        std::fs::write(
            file.path(),
            "ID,EVID,TIME,DUR,DOSE,ADDL,II,INPUT,OUT,OUTEQ,CENS,C0,C1,C2,C3\npt1,1,0,1,100,.,.,iv,.,.,.,.,.,.,.\npt1,0,1,.,.,.,.,.,42,cp,0,.,.,.,.\n",
        )
        .unwrap();

        let data = read_pmetrics(file.path().display().to_string()).unwrap();
        let events = data.subjects()[0].occasions()[0].events();

        match &events[0] {
            Event::Infusion(infusion) => assert_eq!(infusion.input().as_str(), "iv"),
            _ => panic!("expected infusion event"),
        }

        match &events[1] {
            Event::Observation(observation) => assert_eq!(observation.outeq().as_str(), "cp"),
            _ => panic!("expected observation event"),
        }
    }

    #[test]
    fn read_pmetrics_preserves_numeric_labels_as_strings() {
        let file = NamedTempFile::new().unwrap();
        std::fs::write(
            file.path(),
            "ID,EVID,TIME,DUR,DOSE,ADDL,II,INPUT,OUT,OUTEQ,CENS,C0,C1,C2,C3\npt1,1,0,.,100,.,.,1,.,.,.,.,.,.,.\npt1,0,1,.,.,.,.,.,42,1,0,.,.,.,.\n",
        )
        .unwrap();

        let data = read_pmetrics(file.path().display().to_string()).unwrap();
        let events = data.subjects()[0].occasions()[0].events();

        match &events[0] {
            Event::Bolus(bolus) => assert_eq!(bolus.input().as_str(), "1"),
            _ => panic!("expected bolus event"),
        }

        match &events[1] {
            Event::Observation(observation) => assert_eq!(observation.outeq().as_str(), "1"),
            _ => panic!("expected observation event"),
        }
    }
}
