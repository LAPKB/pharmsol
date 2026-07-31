//! Pmetrics CSV byte encoder.

use super::pmetrics::{validate_covariate_header, CORE_HEADERS};
use crate::data::row::DataError;
use crate::data::{Censor, Data, Event};
use csv::{Terminator, WriterBuilder};
use std::collections::{BTreeMap, HashMap};

#[derive(Debug)]
struct CsvCovariate {
    name: String,
    header: String,
}

#[derive(Debug)]
struct CsvRow {
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
        Err(DataError::InvalidPmetricsData(format!(
            "{field} label `{label}` for {id} is reserved as missing"
        )))
    } else {
        Ok(())
    }
}

fn collect_covariate_schema(data: &Data) -> Result<Vec<CsvCovariate>, DataError> {
    let mut fixed_by_name = BTreeMap::<String, bool>::new();
    let mut folded_names = HashMap::<String, String>::new();

    for subject in data.subjects() {
        for occasion in subject.occasions() {
            for (key, covariate) in occasion.covariates().covariates() {
                if key != covariate.name() {
                    return Err(DataError::InvalidPmetricsData(format!(
                        "covariate key `{key}` does not match name `{}`",
                        covariate.name()
                    )));
                }
                validate_covariate_header(&key)?;
                if key.ends_with('!') {
                    return Err(DataError::InvalidPmetricsData(format!(
                        "covariate name `{key}` reserves trailing ! for fixed covariates"
                    )));
                }
                let folded = key.to_lowercase();
                if let Some(existing) = folded_names.insert(folded, key.clone()) {
                    if existing != key {
                        return Err(DataError::InvalidPmetricsData(format!(
                            "covariate names `{existing}` and `{key}` are ambiguous"
                        )));
                    }
                }
                if covariate.has_legacy_unmarked_linear_segments() {
                    return Err(DataError::LegacyLinearCovariate {
                        name: key.clone(),
                        id: subject.id().clone(),
                        occasion: occasion.index(),
                    });
                }
                if covariate.observations().is_empty() {
                    return Err(DataError::InvalidPmetricsData(format!(
                        "covariate `{key}` for subject `{}` occasion {} has no observations",
                        subject.id(),
                        occasion.index()
                    )));
                }
                if let Some(existing) = fixed_by_name.insert(key.clone(), covariate.fixed()) {
                    if existing != covariate.fixed() {
                        return Err(DataError::InvalidPmetricsData(format!(
                            "covariate `{key}` has inconsistent fixed settings"
                        )));
                    }
                }
            }
        }
    }

    let mut schema = fixed_by_name
        .into_iter()
        .map(|(name, fixed)| CsvCovariate {
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
                return Err(DataError::InvalidPmetricsData(format!(
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
    /// Return the complete dataset as deterministic Pmetrics CSV bytes.
    pub fn as_bytes(&self) -> Result<Vec<u8>, DataError> {
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
                return Err(DataError::InvalidPmetricsData(format!(
                    "duplicate subject ID `{}`",
                    pair[0].id()
                )));
            }
        }

        for subject in subjects {
            if subject.id().contains('\r') {
                return Err(DataError::InvalidPmetricsData(format!(
                    "subject ID `{}` contains a carriage return",
                    subject.id()
                )));
            }
            if subject.occasions().is_empty() {
                return Err(DataError::InvalidPmetricsData(format!(
                    "subject `{}` has no occasions",
                    subject.id()
                )));
            }
            for (occasion_index, occasion) in subject.occasions().iter().enumerate() {
                if occasion.index() != occasion_index {
                    return Err(DataError::InvalidPmetricsData(format!(
                        "subject `{}` has nonsequential occasion index {}",
                        subject.id(),
                        occasion.index()
                    )));
                }

                let events = occasion.events();
                for event in events {
                    if event.occasion() != occasion_index {
                        return Err(DataError::InvalidPmetricsData(format!(
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
                        return Err(DataError::InvalidPmetricsData(format!(
                            "events for subject `{}` occasion {} are not in Pmetrics order",
                            subject.id(),
                            occasion_index
                        )));
                    }
                }

                let mut rows = Vec::new();
                for (sequence, event) in events.iter().enumerate() {
                    rows.push(CsvRow {
                        time: event.time(),
                        rank: event_rank(event),
                        sequence,
                        fields: event_row(subject.id(), event, schema.len())?,
                    });
                }

                let covariates = occasion.covariates().covariates();
                let mut covariate_rows = HashMap::<u64, CsvRow>::new();
                for (column, csv_covariate) in schema.iter().enumerate() {
                    let Some(covariate) = covariates.get(&csv_covariate.name) else {
                        continue;
                    };
                    for (time, value) in covariate.observations() {
                        ensure_finite(time, &format!("{} time", csv_covariate.name), subject.id())?;
                        ensure_finite(value, &csv_covariate.name, subject.id())?;
                        let next_sequence = rows.len() + covariate_rows.len();
                        let row = covariate_rows
                            .entry(time.to_bits())
                            .or_insert_with(|| CsvRow {
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
            .map_err(|error| DataError::CSVError(error.to_string()))?;
        drop(csv);
        Ok(bytes)
    }
}
