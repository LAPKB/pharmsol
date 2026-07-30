//! Pmetrics CSV parsing and export helpers.
//!
//! This module reads and writes the Pmetrics-style tabular format while keeping
//! pharmsol's public input and output labels intact.
//!
//! `INPUT` and `OUTEQ` values are parsed as labels, not rewritten to dense
//! indices. Named values such as `iv` and `cp` are preserved exactly, and
//! numeric values such as `1` are preserved as numeric-looking labels.

use crate::{data::*, PharmsolError};
use csv::{ReaderBuilder, StringRecord, Terminator, WriterBuilder};
use serde::de::{MapAccess, Visitor};
use serde::{de, Deserialize, Deserializer, Serialize};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt;
use std::fs::File;
use std::io::{Cursor, Read, Write};
use std::str::FromStr;

use crate::data::row::{build_data, DataError, DataRow};

const CORE_HEADERS: [&str; 15] = [
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
/// covariates.
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
/// The canonical columns are `ID`, `TIME`, `EVID`, `DOSE`, `DUR`, `ADDL`,
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
    let file = File::open(path.into()).map_err(|error| DataError::CSVError(error.to_string()))?;
    read_pmetrics_reader(file, false)
}

/// Read canonical `pmetrics-csv.v1` bytes from a stream.
pub fn read_pmetrics_csv_v1<R: Read>(mut reader: R) -> Result<Data, DataError> {
    let mut bytes = Vec::new();
    reader
        .read_to_end(&mut bytes)
        .map_err(|error| DataError::CSVError(error.to_string()))?;

    if bytes.starts_with(&[0xEF, 0xBB, 0xBF]) {
        return Err(DataError::InvalidCanonicalFormat(
            "UTF-8 BOM is not permitted".to_string(),
        ));
    }
    std::str::from_utf8(&bytes).map_err(|error| {
        DataError::InvalidCanonicalFormat(format!("input is not UTF-8: {error}"))
    })?;
    if bytes.contains(&b'\r') {
        return Err(DataError::InvalidCanonicalFormat(
            "line endings must be LF".to_string(),
        ));
    }
    if !bytes.ends_with(b"\n") {
        return Err(DataError::InvalidCanonicalFormat(
            "input must end with LF".to_string(),
        ));
    }

    read_pmetrics_reader(Cursor::new(bytes), true)
}

fn read_pmetrics_reader<R: Read>(reader: R, canonical: bool) -> Result<Data, DataError> {
    let mut builder = ReaderBuilder::new();
    builder.has_headers(true);
    if !canonical {
        builder.comment(Some(b'#'));
    }
    let mut reader = builder.from_reader(reader);
    let original_headers = reader
        .headers()
        .map_err(|error| DataError::CSVError(error.to_string()))?
        .clone();
    if canonical {
        validate_canonical_headers(&original_headers)?;
    }

    let headers = original_headers
        .iter()
        .enumerate()
        .map(|(index, header)| {
            if canonical && index >= CORE_HEADERS.len() {
                header.to_string()
            } else {
                header.to_lowercase()
            }
        })
        .collect::<Vec<_>>();
    reader.set_headers(StringRecord::from(headers));

    let canonical_covariates = if canonical {
        {
            original_headers
                .iter()
                .skip(CORE_HEADERS.len())
                .map(ToString::to_string)
                .collect::<Vec<_>>()
        }
    } else {
        Default::default()
    };
    let mut seen_covariates = HashSet::new();
    let mut data_rows = Vec::new();
    let mut canonical_state = canonical.then(CanonicalReadState::default);
    for row_result in reader.deserialize() {
        let row: Row = row_result.map_err(|error| DataError::CSVError(error.to_string()))?;
        if let Some(state) = canonical_state.as_mut() {
            row.validate_canonical()?;
            state.observe(&row)?;
            seen_covariates.extend(
                row.covs
                    .iter()
                    .filter(|(_, value)| value.is_some())
                    .map(|(name, _)| name.clone()),
            );
        }
        data_rows.push(row.to_datarow(canonical));
    }
    if let Some(state) = canonical_state {
        state.finish()?;
    }
    for covariate in canonical_covariates {
        if !seen_covariates.contains(&covariate) {
            return Err(DataError::InvalidCanonicalFormat(format!(
                "covariate column `{covariate}` has no observations"
            )));
        }
    }
    build_data(data_rows)
}

fn validate_canonical_headers(headers: &StringRecord) -> Result<(), DataError> {
    if headers.len() < CORE_HEADERS.len() {
        return Err(DataError::InvalidCanonicalFormat(format!(
            "expected at least {} columns, found {}",
            CORE_HEADERS.len(),
            headers.len()
        )));
    }
    for (actual, expected) in headers.iter().zip(CORE_HEADERS) {
        if actual != expected {
            return Err(DataError::InvalidCanonicalFormat(format!(
                "expected core column `{expected}`, found `{actual}`"
            )));
        }
    }

    let mut previous: Option<&str> = None;
    let mut folded = HashSet::new();
    for header in headers.iter().skip(CORE_HEADERS.len()) {
        validate_covariate_header(header)?;
        if previous.is_some_and(|name| name >= header) {
            return Err(DataError::InvalidCanonicalFormat(
                "covariate columns must be unique and lexically sorted".to_string(),
            ));
        }
        let base = header.strip_suffix('!').unwrap_or(header);
        if !folded.insert(base.to_lowercase()) {
            return Err(DataError::InvalidCanonicalFormat(format!(
                "ambiguous covariate column `{header}`"
            )));
        }
        previous = Some(header);
    }
    Ok(())
}

fn validate_covariate_header(header: &str) -> Result<(), DataError> {
    let base = header.strip_suffix('!').unwrap_or(header);
    if base.is_empty()
        || base.ends_with('!')
        || base.contains(['\r', '\n', '\0'])
        || CORE_HEADERS
            .iter()
            .any(|core| core.eq_ignore_ascii_case(base))
    {
        return Err(DataError::InvalidCanonicalFormat(format!(
            "reserved or ambiguous covariate column `{header}`"
        )));
    }
    Ok(())
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

#[derive(Default)]
struct CanonicalReadState {
    subject_id: Option<String>,
    boundary_time: Option<f64>,
    first_row_time: Option<f64>,
    last_row_key: Option<(f64, u8)>,
}

impl CanonicalReadState {
    fn observe(&mut self, row: &Row) -> Result<(), DataError> {
        if self.subject_id.as_deref() != Some(row.id.as_str()) {
            if let Some(previous_id) = self.subject_id.as_deref() {
                self.finish_occasion()?;
                if previous_id >= row.id.as_str() {
                    return Err(DataError::InvalidCanonicalFormat(format!(
                        "subject `{}` is not after `{previous_id}` in lexical order",
                        row.id
                    )));
                }
            }
            if row.evid != 4 {
                return Err(DataError::InvalidCanonicalFormat(format!(
                    "subject `{}` does not start with an EVID=4 boundary",
                    row.id
                )));
            }
            self.subject_id = Some(row.id.clone());
            self.start_occasion(row.time);
            return Ok(());
        }

        if row.evid == 4 {
            self.finish_occasion()?;
            self.start_occasion(row.time);
            return Ok(());
        }

        let rank = match row.evid {
            0 => 1,
            1 if row.dur.is_some_and(|duration| duration > 0.0) => 3,
            1 => 2,
            2 => 4,
            _ => unreachable!("canonical EVID was validated before ordering"),
        };
        if self.first_row_time.is_none() {
            self.first_row_time = Some(row.time);
        }
        if let Some((previous_time, previous_rank)) = self.last_row_key {
            let order = previous_time
                .total_cmp(&row.time)
                .then_with(|| previous_rank.cmp(&rank));
            if order.is_gt() || (order.is_eq() && rank == 4) {
                return Err(DataError::InvalidCanonicalFormat(format!(
                    "rows for subject `{}` are not in canonical occasion order",
                    row.id
                )));
            }
        }
        self.last_row_key = Some((row.time, rank));
        Ok(())
    }

    fn start_occasion(&mut self, boundary_time: f64) {
        self.boundary_time = Some(boundary_time);
        self.first_row_time = None;
        self.last_row_key = None;
    }

    fn finish_occasion(&self) -> Result<(), DataError> {
        let Some(boundary_time) = self.boundary_time else {
            return Ok(());
        };
        let expected_time = self.first_row_time.unwrap_or(0.0);
        if boundary_time.to_bits() != expected_time.to_bits() {
            return Err(DataError::InvalidCanonicalFormat(format!(
                "occasion boundary for `{}` has time {boundary_time}, expected {expected_time}",
                self.subject_id.as_deref().unwrap_or_default()
            )));
        }
        Ok(())
    }

    fn finish(self) -> Result<(), DataError> {
        self.finish_occasion()
    }
}

impl Row {
    fn validate_canonical(&self) -> Result<(), DataError> {
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
            return Err(DataError::InvalidCanonicalFormat(format!(
                "partial error polynomial for {} at time {}",
                self.id, self.time
            )));
        }

        let has_dose_fields = self.dur.is_some()
            || self.dose.is_some()
            || self.addl.is_some()
            || self.ii.is_some()
            || self.input.is_some();
        let has_observation_fields =
            self.out.is_some() || self.outeq.is_some() || self.cens.is_some() || present != 0;
        let has_covariates = self.covs.values().any(Option::is_some);

        match self.evid {
            0 if has_dose_fields || has_covariates => {
                return Err(DataError::InvalidCanonicalFormat(format!(
                    "observation row for {} at time {} contains dose or covariate fields",
                    self.id, self.time
                )));
            }
            1 if has_observation_fields
                || has_covariates
                || self.addl.is_some()
                || self.ii.is_some()
                || self.dur.is_none()
                || self.dur.is_some_and(|duration| duration < 0.0) =>
            {
                return Err(DataError::InvalidCanonicalFormat(format!(
                    "dose row for {} at time {} is not explicit",
                    self.id, self.time
                )));
            }
            2 if has_dose_fields || has_observation_fields || !has_covariates => {
                return Err(DataError::MalformedCovariateRow {
                    id: self.id.clone(),
                    time: self.time,
                });
            }
            4 if has_dose_fields || has_observation_fields || has_covariates => {
                return Err(DataError::MalformedBoundaryRow {
                    id: self.id.clone(),
                    time: self.time,
                });
            }
            0 | 1 | 2 | 4 => {}
            unsupported => {
                return Err(DataError::InvalidCanonicalFormat(format!(
                    "unsupported EVID={unsupported} for {} at time {}",
                    self.id, self.time
                )));
            }
        }
        Ok(())
    }

    /// Convert this Row to a DataRow for parsing
    fn to_datarow(&self, canonical: bool) -> DataRow {
        DataRow {
            id: self.id.clone(),
            time: self.time,
            evid: self.evid as i32,
            dose: self.dose,
            dur: self.dur,
            addl: self.addl.map(|a| a as i64),
            ii: self.ii,
            input: self.input.clone(),
            // Treat -99 as missing only in the legacy Pmetrics dialect.
            out: if canonical {
                self.out
            } else {
                self.out.filter(|&value| value != -99.0)
            },
            outeq: self.outeq.clone(),
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

#[derive(Debug)]
struct CanonicalCovariate {
    name: String,
    header: String,
}

#[derive(Debug)]
struct CanonicalRow {
    time: f64,
    rank: u8,
    sequence: usize,
    fields: Vec<String>,
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

fn ensure_label(label: &str, field: &str, id: &str) -> Result<(), DataError> {
    if label.is_empty() || label == "." || label == "NA" || label.contains('\r') {
        Err(DataError::InvalidCanonicalFormat(format!(
            "{field} label `{label}` for {id} is reserved as missing"
        )))
    } else {
        Ok(())
    }
}

fn collect_covariate_schema(data: &Data) -> Result<Vec<CanonicalCovariate>, DataError> {
    let mut fixed_by_name = BTreeMap::<String, bool>::new();
    let mut folded_names = HashMap::<String, String>::new();

    for subject in data.subjects() {
        for occasion in subject.occasions() {
            for (key, covariate) in occasion.covariates().covariates() {
                if key != covariate.name() {
                    return Err(DataError::InvalidCanonicalFormat(format!(
                        "covariate key `{key}` does not match name `{}`",
                        covariate.name()
                    )));
                }
                validate_covariate_header(&key)?;
                if key.ends_with('!') {
                    return Err(DataError::InvalidCanonicalFormat(format!(
                        "covariate name `{key}` reserves trailing ! for fixed covariates"
                    )));
                }
                let folded = key.to_lowercase();
                if let Some(existing) = folded_names.insert(folded, key.clone()) {
                    if existing != key {
                        return Err(DataError::InvalidCanonicalFormat(format!(
                            "covariate names `{existing}` and `{key}` are ambiguous"
                        )));
                    }
                }
                if covariate.observations().is_empty() {
                    return Err(DataError::InvalidCanonicalFormat(format!(
                        "covariate `{key}` for subject `{}` occasion {} has no observations",
                        subject.id(),
                        occasion.index()
                    )));
                }
                if let Some(existing) = fixed_by_name.insert(key.clone(), covariate.fixed()) {
                    if existing != covariate.fixed() {
                        return Err(DataError::InvalidCanonicalFormat(format!(
                            "covariate `{key}` has inconsistent fixed semantics"
                        )));
                    }
                }
            }
        }
    }

    let mut schema = fixed_by_name
        .into_iter()
        .map(|(name, fixed)| CanonicalCovariate {
            header: if fixed {
                format!("{name}!")
            } else {
                name.clone()
            },
            name,
        })
        .collect::<Vec<_>>();
    schema.sort_by(|left, right| left.header.cmp(&right.header));
    Ok(schema)
}

fn empty_row(id: &str, evid: i32, time: f64, covariate_count: usize) -> Vec<String> {
    let mut fields = vec![".".to_string(); CORE_HEADERS.len() + covariate_count];
    fields[0] = id.to_string();
    fields[1] = evid.to_string();
    fields[2] = time.to_string();
    fields
}

fn event_rank(event: &Event) -> u8 {
    match event {
        Event::Observation(_) => 1,
        Event::Bolus(_) => 2,
        Event::Infusion(_) => 3,
    }
}

fn event_row(id: &str, event: &Event, covariate_count: usize) -> Result<Vec<String>, DataError> {
    ensure_finite(event.time(), "TIME", id)?;
    let mut fields = empty_row(
        id,
        match event {
            Event::Observation(_) => 0,
            Event::Bolus(_) | Event::Infusion(_) => 1,
        },
        event.time(),
        covariate_count,
    );

    match event {
        Event::Observation(observation) => {
            if let Some(value) = observation.value() {
                ensure_finite(value, "OUT", id)?;
                fields[8] = value.to_string();
            }
            let outeq = observation.outeq().to_string();
            ensure_label(&outeq, "OUTEQ", id)?;
            fields[9] = outeq;
            fields[10] = match observation.censoring() {
                Censor::None => "0",
                Censor::BLOQ => "1",
                Censor::ALOQ => "-1",
            }
            .to_string();
            if let Some(error) = observation.errorpoly() {
                let coefficients = error.coefficients();
                for (offset, value) in [
                    coefficients.0,
                    coefficients.1,
                    coefficients.2,
                    coefficients.3,
                ]
                .into_iter()
                .enumerate()
                {
                    ensure_finite(value, &format!("C{offset}"), id)?;
                    fields[11 + offset] = value.to_string();
                }
            }
        }
        Event::Bolus(bolus) => {
            ensure_finite(bolus.amount(), "DOSE", id)?;
            let input = bolus.input().to_string();
            ensure_label(&input, "INPUT", id)?;
            fields[3] = "0".to_string();
            fields[4] = bolus.amount().to_string();
            fields[7] = input;
        }
        Event::Infusion(infusion) => {
            ensure_finite(infusion.duration(), "DUR", id)?;
            ensure_finite(infusion.amount(), "DOSE", id)?;
            if infusion.duration() <= 0.0 {
                return Err(DataError::InvalidCanonicalFormat(format!(
                    "infusion duration for {id} must be greater than zero"
                )));
            }
            let input = infusion.input().to_string();
            ensure_label(&input, "INPUT", id)?;
            fields[3] = infusion.duration().to_string();
            fields[4] = infusion.amount().to_string();
            fields[7] = input;
        }
    }
    Ok(fields)
}

impl Data {
    /// Write canonical `pmetrics-csv.v1` bytes to a stream.
    pub fn write_pmetrics_csv_v1<W: Write>(&self, writer: W) -> Result<(), DataError> {
        let schema = collect_covariate_schema(self)?;
        let mut headers = CORE_HEADERS
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>();
        headers.extend(schema.iter().map(|covariate| covariate.header.clone()));

        let mut csv = WriterBuilder::new()
            .has_headers(false)
            .terminator(Terminator::Any(b'\n'))
            .from_writer(writer);
        csv.write_record(&headers)
            .map_err(|error| DataError::CSVError(error.to_string()))?;

        let mut subjects = self.subjects();
        subjects.sort_by(|left, right| left.id().cmp(right.id()));
        for pair in subjects.windows(2) {
            if pair[0].id() == pair[1].id() {
                return Err(DataError::InvalidCanonicalFormat(format!(
                    "duplicate subject ID `{}`",
                    pair[0].id()
                )));
            }
        }

        for subject in subjects {
            if subject.id().contains('\r') {
                return Err(DataError::InvalidCanonicalFormat(format!(
                    "subject ID `{}` contains a carriage return",
                    subject.id()
                )));
            }
            if subject.occasions().is_empty() {
                return Err(DataError::InvalidCanonicalFormat(format!(
                    "subject `{}` has no occasions",
                    subject.id()
                )));
            }
            for (occasion_index, occasion) in subject.occasions().iter().enumerate() {
                if occasion.index() != occasion_index {
                    return Err(DataError::InvalidCanonicalFormat(format!(
                        "subject `{}` has nonsequential occasion index {}",
                        subject.id(),
                        occasion.index()
                    )));
                }

                let events = occasion.events();
                for event in events {
                    if event.occasion() != occasion_index {
                        return Err(DataError::InvalidCanonicalFormat(format!(
                            "subject `{}` has an event assigned to occasion {} inside occasion {}",
                            subject.id(),
                            event.occasion(),
                            occasion_index
                        )));
                    }
                }
                for pair in events.windows(2) {
                    let order = pair[0]
                        .time()
                        .total_cmp(&pair[1].time())
                        .then_with(|| event_rank(&pair[0]).cmp(&event_rank(&pair[1])));
                    if order.is_gt() {
                        return Err(DataError::InvalidCanonicalFormat(format!(
                            "events for subject `{}` occasion {} are not in canonical order",
                            subject.id(),
                            occasion_index
                        )));
                    }
                }

                let mut rows = Vec::new();
                for (sequence, event) in events.iter().enumerate() {
                    rows.push(CanonicalRow {
                        time: event.time(),
                        rank: event_rank(event),
                        sequence,
                        fields: event_row(subject.id(), event, schema.len())?,
                    });
                }

                let covariates = occasion.covariates().covariates();
                let mut covariate_rows = HashMap::<u64, CanonicalRow>::new();
                for (column, canonical) in schema.iter().enumerate() {
                    let Some(covariate) = covariates.get(&canonical.name) else {
                        continue;
                    };
                    for (time, value) in covariate.observations() {
                        ensure_finite(time, &format!("{} time", canonical.name), subject.id())?;
                        ensure_finite(value, &canonical.name, subject.id())?;
                        let next_sequence = rows.len() + covariate_rows.len();
                        let row =
                            covariate_rows
                                .entry(time.to_bits())
                                .or_insert_with(|| CanonicalRow {
                                    time,
                                    rank: 4,
                                    sequence: next_sequence,
                                    fields: empty_row(subject.id(), 2, time, schema.len()),
                                });
                        row.fields[CORE_HEADERS.len() + column] = value.to_string();
                    }
                }
                rows.extend(covariate_rows.into_values());
                rows.sort_by(|left, right| {
                    left.time
                        .total_cmp(&right.time)
                        .then_with(|| left.rank.cmp(&right.rank))
                        .then_with(|| left.sequence.cmp(&right.sequence))
                });

                let boundary_time = rows.first().map_or(0.0, |row| row.time);
                ensure_finite(boundary_time, "TIME", subject.id())?;
                csv.write_record(empty_row(subject.id(), 4, boundary_time, schema.len()))
                    .map_err(|error| DataError::CSVError(error.to_string()))?;
                for row in rows {
                    csv.write_record(row.fields)
                        .map_err(|error| DataError::CSVError(error.to_string()))?;
                }
            }
        }

        csv.flush()
            .map_err(|error| DataError::CSVError(error.to_string()))
    }

    /// Return canonical `pmetrics-csv.v1` bytes.
    pub fn to_pmetrics_csv_v1(&self) -> Result<Vec<u8>, DataError> {
        let mut bytes = Vec::new();
        self.write_pmetrics_csv_v1(&mut bytes)?;
        Ok(bytes)
    }

    /// Write the dataset to a file in canonical Pmetrics format.
    pub fn write_pmetrics(&self, file: &File) -> Result<(), PharmsolError> {
        self.write_pmetrics_csv_v1(file)
            .map_err(PharmsolError::from)
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
