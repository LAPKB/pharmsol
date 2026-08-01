//! Pmetrics CSV byte encoder.

use super::pmetrics::{normalize_covariate_name, validate_covariate_header, CORE_HEADERS};
use crate::data::row::DataError;
use crate::data::{Censor, Data, Event, Occasion, Subject};
use crate::PharmsolError;
use csv::{Terminator, WriterBuilder};
use std::collections::BTreeMap;
use std::fs::File;
use std::io::Write;

mod column {
    pub const ID: usize = 0;
    pub const EVID: usize = 1;
    pub const TIME: usize = 2;
    pub const DUR: usize = 3;
    pub const DOSE: usize = 4;
    pub const INPUT: usize = 7;
    pub const OUT: usize = 8;
    pub const OUTEQ: usize = 9;
    pub const CENS: usize = 10;
    pub const C0: usize = 11;
}

#[derive(Clone, Copy)]
enum PmetricsEvid {
    Observation,
    Dose,
    ResetDose,
}

impl PmetricsEvid {
    fn as_str(self) -> &'static str {
        match self {
            Self::Observation => "0",
            Self::Dose => "1",
            Self::ResetDose => "4",
        }
    }
}

#[derive(Debug)]
struct PmetricsCovariateColumn {
    name: String,
    header: String,
}

#[derive(Debug)]
struct PmetricsCsvRow {
    time: f64,
    fields: Vec<String>,
}

fn unrepresentable(message: impl Into<String>) -> DataError {
    DataError::UnrepresentablePmetricsData(message.into())
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
        Err(unrepresentable(format!(
            "{field} label `{label}` for {id} is reserved as missing"
        )))
    } else {
        Ok(())
    }
}

fn collect_covariate_schema(data: &Data) -> Result<Vec<PmetricsCovariateColumn>, DataError> {
    let mut fixed_by_name = BTreeMap::<String, bool>::new();

    for subject in data.subjects() {
        for occasion in subject.occasions() {
            let mut keys_by_name = BTreeMap::<String, String>::new();
            for (key, covariate) in occasion.covariates().covariates() {
                if normalize_covariate_name(&key) != normalize_covariate_name(covariate.name()) {
                    return Err(unrepresentable(format!(
                        "covariate key `{key}` does not match name `{}`",
                        covariate.name()
                    )));
                }
                validate_covariate_header(&key).map_err(|error| match error {
                    DataError::InvalidPmetricsData(message) => unrepresentable(message),
                    other => other,
                })?;
                if key.ends_with('!') {
                    return Err(unrepresentable(format!(
                        "covariate name `{key}` reserves trailing ! for fixed covariates"
                    )));
                }

                let name = normalize_covariate_name(&key);
                if let Some(existing) = keys_by_name.insert(name.clone(), key.clone()) {
                    return Err(unrepresentable(format!(
                        "covariates `{existing}` and `{key}` for subject `{}` occasion {} both map to `{name}`",
                        subject.id(),
                        occasion.index()
                    )));
                }
                if covariate.observations().is_empty() {
                    return Err(unrepresentable(format!(
                        "covariate `{key}` for subject `{}` occasion {} has no observations",
                        subject.id(),
                        occasion.index()
                    )));
                }
                if let Some(existing) = fixed_by_name.insert(name.clone(), covariate.fixed()) {
                    if existing != covariate.fixed() {
                        return Err(unrepresentable(format!(
                            "covariate `{name}` has inconsistent fixed settings"
                        )));
                    }
                }
            }
        }
    }

    Ok(fixed_by_name
        .into_iter()
        .map(|(name, fixed)| PmetricsCovariateColumn {
            header: if fixed {
                format!("{name}!")
            } else {
                name.clone()
            },
            name,
        })
        .collect())
}

fn empty_row(id: &str, evid: PmetricsEvid, time: f64, covariate_count: usize) -> Vec<String> {
    let mut fields = vec![".".to_string(); CORE_HEADERS.len() + covariate_count];
    fields[column::ID] = id.to_string();
    fields[column::EVID] = evid.as_str().to_string();
    fields[column::TIME] = time.to_string();
    fields
}

fn event_row(id: &str, event: &Event, covariate_count: usize) -> Result<Vec<String>, DataError> {
    ensure_finite(event.time(), "TIME", id)?;
    let evid = match event {
        Event::Observation(_) => PmetricsEvid::Observation,
        Event::Bolus(_) | Event::Infusion(_) => PmetricsEvid::Dose,
    };
    let mut fields = empty_row(id, evid, event.time(), covariate_count);

    match event {
        Event::Observation(observation) => {
            fields[column::OUT] = match observation.value() {
                Some(value) => {
                    ensure_finite(value, "OUT", id)?;
                    if value == -99.0 {
                        return Err(unrepresentable(format!(
                            "observation OUT=-99 for {id} at time {} is reserved for missing data",
                            observation.time()
                        )));
                    }
                    value.to_string()
                }
                None => "-99".to_string(),
            };
            let outeq = observation.outeq().to_string();
            ensure_label(&outeq, "OUTEQ", id)?;
            fields[column::OUTEQ] = outeq;
            fields[column::CENS] = match observation.censoring() {
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
                    fields[column::C0 + offset] = value.to_string();
                }
            }
        }
        Event::Bolus(bolus) => {
            ensure_finite(bolus.amount(), "DOSE", id)?;
            let input = bolus.input().to_string();
            ensure_label(&input, "INPUT", id)?;
            fields[column::DUR] = "0".to_string();
            fields[column::DOSE] = bolus.amount().to_string();
            fields[column::INPUT] = input;
        }
        Event::Infusion(infusion) => {
            ensure_finite(infusion.duration(), "DUR", id)?;
            ensure_finite(infusion.amount(), "DOSE", id)?;
            if infusion.duration() <= 0.0 {
                return Err(unrepresentable(format!(
                    "infusion duration for {id} must be greater than zero"
                )));
            }
            let input = infusion.input().to_string();
            ensure_label(&input, "INPUT", id)?;
            fields[column::DUR] = infusion.duration().to_string();
            fields[column::DOSE] = infusion.amount().to_string();
            fields[column::INPUT] = input;
        }
    }
    Ok(fields)
}

fn validate_subject(subject: &Subject) -> Result<(), DataError> {
    if subject.id().starts_with('#') {
        return Err(unrepresentable(format!(
            "subject ID `{}` cannot start with #",
            subject.id()
        )));
    }
    if subject.id().contains('\r') {
        return Err(unrepresentable(format!(
            "subject ID `{}` contains a carriage return",
            subject.id()
        )));
    }
    if subject.occasions().is_empty() {
        return Err(unrepresentable(format!(
            "subject `{}` has no occasions",
            subject.id()
        )));
    }
    Ok(())
}

fn validate_occasion(
    subject: &Subject,
    occasion_index: usize,
    occasion: &Occasion,
) -> Result<(), DataError> {
    if occasion.index() != occasion_index {
        return Err(unrepresentable(format!(
            "subject `{}` has nonsequential occasion index {}",
            subject.id(),
            occasion.index()
        )));
    }

    let events = occasion.events();
    if events.is_empty() {
        return Err(unrepresentable(format!(
            "subject `{}` occasion {occasion_index} has no dose or observation rows",
            subject.id()
        )));
    }
    for event in events {
        if event.occasion() != occasion_index {
            return Err(unrepresentable(format!(
                "subject `{}` has an event assigned to occasion {} inside occasion {occasion_index}",
                subject.id(),
                event.occasion()
            )));
        }
    }
    if events
        .windows(2)
        .any(|pair| pair[0].cmp_time_then_type(&pair[1]).is_gt())
    {
        return Err(unrepresentable(format!(
            "events for subject `{}` occasion {occasion_index} must have nondecreasing times, with observations before doses at equal times",
            subject.id()
        )));
    }

    if occasion_index == 0 {
        return Ok(());
    }

    let first = events.first().expect("nonempty occasion checked above");
    if !matches!(first, Event::Bolus(_) | Event::Infusion(_)) || first.time() != 0.0 {
        return Err(unrepresentable(format!(
            "subject `{}` occasion {occasion_index} cannot be represented: later occasions must begin with a dose at time 0",
            subject.id()
        )));
    }

    let has_bolus = events
        .iter()
        .any(|event| event.time() == 0.0 && matches!(event, Event::Bolus(_)));
    let has_infusion = events
        .iter()
        .any(|event| event.time() == 0.0 && matches!(event, Event::Infusion(_)));
    if has_bolus && has_infusion {
        return Err(unrepresentable(format!(
            "subject `{}` occasion {occasion_index} has both bolus and infusion doses at time 0, so the reset dose is ambiguous",
            subject.id()
        )));
    }
    Ok(())
}

fn encode_occasion(
    subject: &Subject,
    occasion_index: usize,
    occasion: &Occasion,
    schema: &[PmetricsCovariateColumn],
) -> Result<Vec<PmetricsCsvRow>, DataError> {
    validate_occasion(subject, occasion_index, occasion)?;

    let mut rows = Vec::with_capacity(occasion.events().len());
    for (sequence, event) in occasion.events().iter().enumerate() {
        let mut fields = event_row(subject.id(), event, schema.len())?;
        if occasion_index > 0 && sequence == 0 {
            fields[column::EVID] = PmetricsEvid::ResetDose.as_str().to_string();
        }
        rows.push(PmetricsCsvRow {
            time: event.time(),
            fields,
        });
    }

    let covariates = occasion.covariates().covariates();
    for (column_index, csv_covariate) in schema.iter().enumerate() {
        let Some(covariate) = covariates.iter().find_map(|(key, covariate)| {
            (normalize_covariate_name(key) == csv_covariate.name).then_some(*covariate)
        }) else {
            continue;
        };
        for (time, value) in covariate.observations() {
            ensure_finite(time, &format!("{} time", csv_covariate.name), subject.id())?;
            ensure_finite(value, &csv_covariate.name, subject.id())?;
            let mut matched_event = false;
            for row in &mut rows {
                if row.time == time {
                    row.fields[CORE_HEADERS.len() + column_index] = value.to_string();
                    matched_event = true;
                }
            }
            if !matched_event {
                return Err(unrepresentable(format!(
                    "covariate `{}` for subject `{}` occasion {occasion_index} at time {time} has no dose or observation row",
                    csv_covariate.name,
                    subject.id()
                )));
            }
        }
    }

    Ok(rows)
}

impl Data {
    /// Return the dataset as Pmetrics CSV bytes.
    ///
    /// Every occasion must contain real events. Each occasion after the first
    /// must begin with an unambiguous dose at time zero, and every covariate
    /// observation must share a time with a dose or observation row. Doses
    /// expanded from `ADDL`/`II` input are written as individual rows.
    ///
    /// Missing observations are written as `OUT=-99`; a real value of `-99`
    /// cannot be represented.
    pub fn to_pmetrics_csv_bytes(&self) -> Result<Vec<u8>, DataError> {
        let schema = collect_covariate_schema(self)?;
        let mut headers = CORE_HEADERS
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>();
        headers.extend(schema.iter().map(|covariate| covariate.header.clone()));

        let mut bytes = Vec::new();
        let mut csv = WriterBuilder::new()
            .has_headers(false)
            .terminator(Terminator::Any(b'\n'))
            .from_writer(&mut bytes);
        csv.write_record(&headers)
            .map_err(|error| DataError::CSVError(error.to_string()))?;

        let mut subjects = self.subjects();
        subjects.sort_by(|left, right| left.id().cmp(right.id()));
        for pair in subjects.windows(2) {
            if pair[0].id() == pair[1].id() {
                return Err(unrepresentable(format!(
                    "duplicate subject ID `{}`",
                    pair[0].id()
                )));
            }
        }

        for subject in subjects {
            validate_subject(subject)?;
            for (occasion_index, occasion) in subject.occasions().iter().enumerate() {
                for row in encode_occasion(subject, occasion_index, occasion, &schema)? {
                    csv.write_record(row.fields)
                        .map_err(|error| DataError::CSVError(error.to_string()))?;
                }
            }
        }

        csv.flush()
            .map_err(|error| DataError::CSVError(error.to_string()))?;
        drop(csv);
        Ok(bytes)
    }

    /// Write the same bytes returned by [`Data::to_pmetrics_csv_bytes`].
    pub fn write_pmetrics(&self, file: &File) -> Result<(), PharmsolError> {
        let bytes = self.to_pmetrics_csv_bytes().map_err(PharmsolError::from)?;
        let mut output = file;
        output
            .write_all(&bytes)
            .map_err(|error| PharmsolError::OtherError(error.to_string()))
    }
}
