//! Pmetrics CSV parsing and export helpers.
//!
//! This module reads and writes the Pmetrics-style tabular format while keeping
//! pharmsol's public input and output labels intact.
//!
//! `INPUT` and `OUTEQ` values are parsed as labels, not rewritten to dense
//! indices. Named values such as `iv` and `cp` are preserved exactly, and
//! numeric values such as `1` are preserved as numeric-looking labels.

use crate::data::*;
use ::csv::{ReaderBuilder, StringRecord};
use serde::de::{MapAccess, Visitor};
use serde::{de, Deserialize, Deserializer};
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::str::FromStr;

mod csv;
mod row;

#[cfg(test)]
#[path = "tests.rs"]
mod csv_tests;

pub use row::{build_data, DataError, DataRow, DataRowBuilder};

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(usize)]
pub(super) enum CoreColumn {
    Id,
    Evid,
    Time,
    Dur,
    Dose,
    Addl,
    Ii,
    Input,
    Out,
    Outeq,
    Cens,
    C0,
    C1,
    C2,
    C3,
}

impl CoreColumn {
    pub(super) const ALL: [Self; 15] = [
        Self::Id,
        Self::Evid,
        Self::Time,
        Self::Dur,
        Self::Dose,
        Self::Addl,
        Self::Ii,
        Self::Input,
        Self::Out,
        Self::Outeq,
        Self::Cens,
        Self::C0,
        Self::C1,
        Self::C2,
        Self::C3,
    ];
    const REQUIRED: [Self; 3] = [Self::Id, Self::Evid, Self::Time];
    pub(super) const COUNT: usize = Self::ALL.len();

    pub(super) const fn index(self) -> usize {
        self as usize
    }

    pub(super) const fn header(self) -> &'static str {
        match self {
            Self::Id => "ID",
            Self::Evid => "EVID",
            Self::Time => "TIME",
            Self::Dur => "DUR",
            Self::Dose => "DOSE",
            Self::Addl => "ADDL",
            Self::Ii => "II",
            Self::Input => "INPUT",
            Self::Out => "OUT",
            Self::Outeq => "OUTEQ",
            Self::Cens => "CENS",
            Self::C0 => "C0",
            Self::C1 => "C1",
            Self::C2 => "C2",
            Self::C3 => "C3",
        }
    }

    fn from_header(header: &str) -> Option<Self> {
        Self::ALL
            .into_iter()
            .find(|column| column.header().eq_ignore_ascii_case(header))
    }
}

pub(super) fn core_headers() -> impl ExactSizeIterator<Item = &'static str> {
    CoreColumn::ALL.into_iter().map(CoreColumn::header)
}

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
/// doses as individual rows. For `EVID=4`, positive `ADDL` resets at the base
/// dose time and negative `ADDL` resets at the earliest expanded dose time.
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
/// `ID`, `EVID`, and `TIME` are required. The remaining core columns are
/// `DOSE`, `DUR`, `ADDL`, `II`, `INPUT`, `OUT`, `OUTEQ`, `CENS`, and optional
/// `C0..C3` error coefficients.
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
        let mut core_columns = HashSet::new();
        let mut covariate_forms = HashMap::<String, bool>::new();
        let mut headers = Vec::with_capacity(original_headers.len());

        for header in &original_headers {
            if let Some(column) = CoreColumn::from_header(header) {
                let name = column.header().to_ascii_lowercase();
                if !core_columns.insert(column) {
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
        for required in CoreColumn::REQUIRED {
            if !core_columns.contains(&required) {
                return Err(DataError::InvalidPmetricsData(format!(
                    "missing required core header `{}`",
                    required.header()
                )));
            }
        }
        reader.set_headers(StringRecord::from(headers));

        let mut data_rows = Vec::new();
        for row_result in reader.deserialize() {
            let row: Row = row_result.map_err(|error| DataError::CSVError(error.to_string()))?;
            data_rows.push(row.into_datarow());
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
        || CoreColumn::from_header(base).is_some()
    {
        return Err(DataError::InvalidPmetricsData(format!(
            "reserved or ambiguous covariate column `{header}`"
        )));
    }
    Ok(())
}

/// One row from a Pmetrics file after serde deserialization.
#[derive(Deserialize)]
#[serde(rename_all = "lowercase")]
struct Row {
    /// Subject ID
    id: String,
    /// Event type
    evid: i32,
    /// Event time
    time: f64,
    /// Infusion duration
    #[serde(default, deserialize_with = "deserialize_option_f64")]
    dur: Option<f64>,
    /// Dose amount
    #[serde(default, deserialize_with = "deserialize_option_f64")]
    dose: Option<f64>,
    /// Additional doses
    #[serde(default, deserialize_with = "deserialize_option_i64")]
    addl: Option<i64>,
    /// Dosing interval
    #[serde(default, deserialize_with = "deserialize_option_f64")]
    ii: Option<f64>,
    /// Input label from the `INPUT` column
    #[serde(default, deserialize_with = "deserialize_option_route_label")]
    input: Option<InputLabel>,
    /// Observed value
    #[serde(default, deserialize_with = "deserialize_option_f64")]
    out: Option<f64>,
    /// Output label from the `OUTEQ` column
    #[serde(default, deserialize_with = "deserialize_option_output_label")]
    outeq: Option<OutputLabel>,
    /// Censoring output
    #[serde(default, deserialize_with = "deserialize_option_censor")]
    cens: Option<Censor>,
    /// First element of the error polynomial
    #[serde(default, deserialize_with = "deserialize_option_f64")]
    c0: Option<f64>,
    /// Second element of the error polynomial
    #[serde(default, deserialize_with = "deserialize_option_f64")]
    c1: Option<f64>,
    /// Third element of the error polynomial
    #[serde(default, deserialize_with = "deserialize_option_f64")]
    c2: Option<f64>,
    /// Fourth element of the error polynomial
    #[serde(default, deserialize_with = "deserialize_option_f64")]
    c3: Option<f64>,
    /// All other columns are covariates
    #[serde(deserialize_with = "deserialize_covs", flatten)]
    covs: HashMap<String, Option<f64>>,
}

impl Row {
    fn into_datarow(self) -> DataRow {
        DataRow {
            id: self.id,
            time: self.time,
            evid: self.evid,
            dose: self.dose,
            dur: self.dur,
            addl: self.addl,
            ii: self.ii,
            input: self.input,
            // Treat -99 as missing, matching the common Pmetrics convention.
            out: self.out.filter(|&value| value != -99.0),
            outeq: self.outeq,
            cens: self.cens,
            c0: self.c0,
            c1: self.c1,
            c2: self.c2,
            c3: self.c3,
            covariates: self
                .covs
                .into_iter()
                .filter_map(|(key, value)| value.map(|value| (key, value)))
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

fn deserialize_option_i64<'de, D>(deserializer: D) -> Result<Option<i64>, D::Error>
where
    D: Deserializer<'de>,
{
    deserialize_option::<i64, D>(deserializer)
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
