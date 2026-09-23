//! Pmetrics CSV byte encoder.

use super::{
    core_headers, normalize_covariate_name, validate_covariate_header, CoreColumn, DataError,
};
use crate::data::{Censor, Data, Event, Occasion, Subject};
use crate::PharmsolError;
use csv::{Terminator, WriterBuilder};
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{Seek, SeekFrom, Write};

#[derive(Clone, Copy)]
enum PmetricsEvid {
    Observation,
    Dose,
    Covariates,
    Reset,
    ResetDose,
}

impl PmetricsEvid {
    fn as_str(self) -> &'static str {
        match self {
            Self::Observation => "0",
            Self::Dose => "1",
            Self::Covariates => "2",
            Self::Reset => "3",
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
    let mut fields = vec![".".to_string(); CoreColumn::COUNT + covariate_count];
    fields[CoreColumn::Id.index()] = id.to_string();
    fields[CoreColumn::Evid.index()] = evid.as_str().to_string();
    fields[CoreColumn::Time.index()] = time.to_string();
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
            fields[CoreColumn::Out.index()] = match observation.value() {
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
            fields[CoreColumn::Outeq.index()] = outeq;
            fields[CoreColumn::Cens.index()] = match observation.censoring() {
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
                    fields[CoreColumn::C0.index() + offset] = value.to_string();
                }
            }
        }
        Event::Bolus(bolus) => {
            ensure_finite(bolus.amount(), "DOSE", id)?;
            let input = bolus.input().to_string();
            ensure_label(&input, "INPUT", id)?;
            fields[CoreColumn::Dur.index()] = "0".to_string();
            fields[CoreColumn::Dose.index()] = bolus.amount().to_string();
            fields[CoreColumn::Input.index()] = input;
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
            fields[CoreColumn::Dur.index()] = infusion.duration().to_string();
            fields[CoreColumn::Dose.index()] = infusion.amount().to_string();
            fields[CoreColumn::Input.index()] = input;
        }
    }
    Ok(fields)
}

fn validate_subject(subject: &Subject) -> Result<(), DataError> {
    if subject.id().is_empty() {
        return Err(unrepresentable("subject ID cannot be empty"));
    }
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
            "events for subject `{}` occasion {occasion_index} must have nondecreasing times, with observations before doses and boluses before infusions at equal times",
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
    for event in occasion.events() {
        let fields = event_row(subject.id(), event, schema.len())?;
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
            let mut matched_row = false;
            for row in &mut rows {
                if row.time == time {
                    row.fields[CoreColumn::COUNT + column_index] = value.to_string();
                    matched_row = true;
                }
            }
            if !matched_row {
                let mut fields =
                    empty_row(subject.id(), PmetricsEvid::Covariates, time, schema.len());
                fields[CoreColumn::COUNT + column_index] = value.to_string();
                rows.push(PmetricsCsvRow { time, fields });
            }
        }
    }

    rows.sort_by(|left, right| left.time.total_cmp(&right.time));
    // Separate occasions without inventing a dose. Empty occasions also need a row.
    if occasion_index > 0 || rows.is_empty() {
        let time = rows.first().map_or(0.0, |row| row.time);
        if let Some(dose) = rows
            .first_mut()
            .filter(|row| row.fields[CoreColumn::Evid.index()] == PmetricsEvid::Dose.as_str())
        {
            dose.fields[CoreColumn::Evid.index()] = PmetricsEvid::ResetDose.as_str().to_string();
        } else {
            rows.insert(
                0,
                PmetricsCsvRow {
                    time,
                    fields: empty_row(subject.id(), PmetricsEvid::Reset, time, schema.len()),
                },
            );
        }
    }
    Ok(rows)
}

impl Data {
    /// Return the dataset as Pmetrics CSV bytes.
    ///
    /// Occasions need not contain or start with a dose. Boundaries are written
    /// as EVID=3 (reset only), or EVID=4 when the first row is a dose.
    /// Empty occasions are preserved with an EVID=3 row at time zero.
    /// Covariate times without a dose or observation are written as EVID=2 rows.
    /// Doses expanded from `ADDL`/`II` input are written as individual rows.
    ///
    /// Missing observations are written as `OUT=-99`; a real value of `-99`
    /// cannot be represented.
    pub fn to_pmetrics_csv_bytes(&self) -> Result<Vec<u8>, DataError> {
        let schema = collect_covariate_schema(self)?;
        let mut headers = core_headers().map(ToString::to_string).collect::<Vec<_>>();
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

    /// Replace the file contents with the bytes from [`Data::to_pmetrics_csv_bytes`].
    pub fn write_pmetrics(&self, file: &File) -> Result<(), PharmsolError> {
        let bytes = self.to_pmetrics_csv_bytes().map_err(PharmsolError::from)?;
        file.set_len(0)
            .map_err(|error| PharmsolError::OtherError(error.to_string()))?;
        let mut output = file;
        output
            .seek(SeekFrom::Start(0))
            .and_then(|_| output.write_all(&bytes))
            .map_err(|error| PharmsolError::OtherError(error.to_string()))
    }
}
