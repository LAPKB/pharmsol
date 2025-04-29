use crate::data::*;
use serde::de::{MapAccess, Visitor};
use serde::{de, Deserialize, Deserializer};
use std::collections::HashMap;
use std::fmt;
use std::fs::File;
use std::io::Read;
use std::path::Path;
use std::str::FromStr;
use thiserror::Error;

/// Defines the structure of data in the Pmetrics data format
#[derive(Debug, Clone)]
struct PmetricsRow {
    // Subject ID
    id: String,
    // Event ID (EVID)
    evid: usize,
    // Time of the event
    time: f64,
    // Duration of the event (optional)
    dur: Option<f64>,
    // Dose amount (optional)
    dose: Option<f64>,
    // Additional doses (optional)
    addl: Option<usize>,
    // Dosing interval (optional)
    ii: Option<f64>,
    // Input type (optional)
    input: Option<usize>,
    // Output value (optional)
    out: Option<f64>,
    // Output equation (optional)
    outeq: Option<usize>,
    // Error polynomial coefficients (optional)
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
            // Treat both empty strings and "." as empty values
            match value {
                Some(s) if !s.is_empty() && s != "." => s
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

/// Deserialize PmetricsRow data from a CSV file into the Data struct
///
/// # Arguments
///
/// * `path` - Path to the CSV file containing Pmetrics data
///
/// # Returns
///
/// Result with the deserialized Data struct or a PmetricsError
pub fn from_csv<P: AsRef<Path>>(path: P) -> Result<Data, PmetricsError> {
    let file = File::open(path).map_err(|e| PmetricsError::CsvError(e.to_string()))?;
    from_reader(file)
}

/// Deserialize PmetricsRow data from a reader into the Data struct
///
/// # Arguments
///
/// * `rdr` - Any type that implements Read trait containing Pmetrics CSV data
///
/// # Returns
///
/// Result with the deserialized Data struct or a PmetricsError
pub fn from_reader<R: Read>(rdr: R) -> Result<Data, PmetricsError> {
    let mut csv_reader = csv::ReaderBuilder::new()
        .has_headers(true)
        .trim(csv::Trim::All)
        .from_reader(rdr);

    // Prepare a HashMap to store subjects with their occasions
    let mut subjects_map: HashMap<
        String,
        Vec<(usize, Vec<Event>, HashMap<String, Vec<(f64, f64)>>)>,
    > = HashMap::new();

    // Track occasion indices for each subject
    let mut subject_occasions: HashMap<String, usize> = HashMap::new();

    // Process each row in the CSV file
    for record_result in csv_reader.deserialize() {
        let record: PmetricsRow = record_result?;
        let subject_id = record.id.clone();

        // Check if this is a new occasion based on EVID=4
        if record.evid == 4 {
            // Increment the occasion index for this subject if EVID=4
            let current_idx = subject_occasions.get(&subject_id).copied().unwrap_or(0);
            let new_idx = current_idx + 1;
            subject_occasions.insert(subject_id.clone(), new_idx);
        } else if !subject_occasions.contains_key(&subject_id) {
            // First record for this subject
            subject_occasions.insert(subject_id.clone(), 0);
        }

        // Get the occasion for this subject
        let occasion_idx = subject_occasions[&subject_id];

        // Convert the PmetricsRow to events
        let events = Vec::<Event>::try_from(record.clone())?;

        // Get or create the subject's data
        let subject_data = subjects_map
            .entry(subject_id.clone())
            .or_insert_with(Vec::new);

        // Find the occasion for this subject
        let occasion_data = if let Some(occ) = subject_data
            .iter_mut()
            .find(|(idx, _, _)| *idx == occasion_idx)
        {
            occ
        } else {
            // Create a new occasion for this subject
            subject_data.push((occasion_idx, Vec::new(), HashMap::new()));
            subject_data.last_mut().unwrap()
        };

        // Add the events to the occasion
        occasion_data.1.extend(events);

        // Process covariates
        for (name, value) in record.covs {
            let time_values = occasion_data.2.entry(name).or_insert_with(Vec::new);
            time_values.push((record.time, value));
        }
    }

    // Convert the collected data into Subject and Occasion objects
    let mut data_subjects = Vec::new();

    for (subject_id, occasions) in subjects_map {
        let mut subject_occasions = Vec::new();

        for (idx, events, covariate_map) in occasions {
            // Convert covariate map to Covariates structure
            let mut covariates = Covariates::new();

            // Process each covariate name and its time-value pairs
            for (name, mut time_points) in covariate_map {
                if time_points.is_empty() {
                    continue;
                }

                let mut covariate = Covariate::new(name.clone(), Vec::new());

                // Sort time points by time
                time_points.sort_by(|(t1, _), (t2, _)| t1.partial_cmp(t2).unwrap());

                // Create segments between time points with linear interpolation
                for i in 0..time_points.len() - 1 {
                    let (t1, v1) = time_points[i];
                    let (t2, v2) = time_points[i + 1];

                    // Calculate linear interpolation parameters
                    let slope = (v2 - v1) / (t2 - t1);
                    let intercept = v1 - slope * t1;

                    // Create and add the segment to the covariate
                    let segment = CovariateSegment::new(
                        t1,
                        t2,
                        InterpolationMethod::Linear { slope, intercept },
                    );
                    covariate.add_segment(segment);
                }

                // Add a carry-forward segment for the last time point
                if let Some(&(last_time, last_value)) = time_points.last() {
                    let segment = CovariateSegment::new(
                        last_time,
                        f64::INFINITY,
                        InterpolationMethod::CarryForward { value: last_value },
                    );
                    covariate.add_segment(segment);
                }

                // Add the completed covariate to the covariates collection
                covariates.add_covariate(name, covariate);
            }

            let mut occasion = Occasion::new(events, covariates, idx);
            occasion.sort();
            subject_occasions.push(occasion);
        }

        let subject = Subject::new(subject_id, subject_occasions);
        data_subjects.push(subject);
    }

    Ok(Data::new(data_subjects))
}

impl Data {
    pub fn read_pmetrics<P: AsRef<Path>>(path: P) -> Result<Data, PmetricsError> {
        from_csv(path)
    }

    pub fn read_pmetrics_csv_from_reader<R: Read>(rdr: R) -> Result<Data, PmetricsError> {
        from_reader(rdr)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn test_parse_dose() {
        let data = r#"
            id,evid,time,dur,dose,addl,ii,input,out,outeq,c0,c1,c2,c3
            1,1,0.0,0.0,100.0,0,0,1,0.0,1,0.0,0.0,0.0,0.0
            1,2,24.0,,100.0,,24,,1.5,,,
            2,1,12.0,,50.0,,12,,2.5,,,
        "#;

        let parsed_data: Vec<PmetricsRow> = serde_json::from_str(data).unwrap();
        dbg!(&parsed_data);
        assert_eq!(parsed_data.len(), 3);
    }

    #[test]
    fn test_deserialize_data_from_csv() {
        // Create a sample CSV in Pmetrics format
        let csv_data = "\
ID,EVID,TIME,DUR,DOSE,ADDL,II,INPUT,OUT,OUTEQ,weight,age,C0,C1,C2,C3
1,1,0.0,0,100.0,0,0,1,.,.,70,30,.,.,.,. 
1,0,1.0,.,.,.,.,.,10.0,1,70,30,0.1,0,0,0
1,0,2.0,.,.,.,.,.,8.0,1,72,30,0.1,0,0,0
1,1,24.0,0,100.0,0,0,1,.,.,75,30,.,.,.,. 
1,0,25.0,.,.,.,.,.,12.0,1,75,30,0.1,0,0,0
1,0,26.0,.,.,.,.,.,9.0,1,75,30,0.1,0,0,0
2,1,0.0,1.5,150.0,0,0,1,.,.,65,25,.,.,.,. 
2,0,2.0,.,.,.,.,.,15.0,1,65,25,0.1,0,0,0
2,0,4.0,.,.,.,.,.,12.0,1,67,25,0.1,0,0,0
2,1,24.0,1.5,150.0,0,0,1,.,.,68,25,.,.,.,. 
2,0,26.0,.,.,.,.,.,18.0,1,68,25,0.1,0,0,0
2,0,28.0,.,.,.,.,.,14.0,1,68,25,0.1,0,0,0";

        // Create a reader from the CSV data
        let cursor = Cursor::new(csv_data);

        // Deserialize the data
        let data = from_reader(cursor).unwrap();

        // Verify the structure
        assert_eq!(data.len(), 2, "Should have 2 subjects");

        // Check subject 1
        let subject1 = data.get_subject("1").unwrap();
        assert_eq!(
            subject1.occasions().len(),
            1,
            "Subject 1 should have 1 occasion"
        );

        let occasion = subject1.occasions()[0];

        // Check events
        let events = occasion.events();
        assert_eq!(events.len(), 6, "Subject 1 should have 6 events");

        // Check covariates
        let covariates = occasion.get_covariates().unwrap();
        assert!(
            covariates.get_covariate("weight").is_some(),
            "Subject 1 should have weight covariate"
        );
        assert!(
            covariates.get_covariate("age").is_some(),
            "Subject 1 should have age covariate"
        );

        // Check subject 2
        let subject2 = data.get_subject("2").unwrap();
        assert_eq!(
            subject2.occasions().len(),
            1,
            "Subject 2 should have 1 occasion"
        );

        let occasion = subject2.occasions()[0];

        // Check events
        let events = occasion.events();
        assert_eq!(events.len(), 6, "Subject 2 should have 6 events");

        // Check covariates
        let covariates = occasion.get_covariates().unwrap();
        assert!(
            covariates.get_covariate("weight").is_some(),
            "Subject 2 should have weight covariate"
        );
        assert!(
            covariates.get_covariate("age").is_some(),
            "Subject 2 should have age covariate"
        );

        // Check specific covariate values
        let weight_cov = covariates.get_covariate("weight").unwrap();
        assert_eq!(
            weight_cov.interpolate(0.0).unwrap(),
            65.0,
            "Subject 2 weight at time 0.0 should be 65.0"
        );
        assert_eq!(
            weight_cov.interpolate(24.0).unwrap(),
            68.0,
            "Subject 2 weight at time 24.0 should be 68.0"
        );
    }

    #[test]
    fn test_deserialize_data_from_csv_occasions() {
        // Create a sample CSV in Pmetrics format
        let csv_data = "\
ID,EVID,TIME,DUR,DOSE,ADDL,II,INPUT,OUT,OUTEQ,weight,age,C0,C1,C2,C3
1,1,0.0,0,100.0,0,0,1,.,.,70,30,.,.,.,. 
1,0,1.0,.,.,.,.,.,10.0,1,70,30,0.1,0,0,0
1,0,2.0,.,.,.,.,.,8.0,1,72,30,0.1,0,0,0
1,1,24.0,0,100.0,0,0,1,.,.,75,30,.,.,.,. 
1,0,25.0,.,.,.,.,.,12.0,1,75,30,0.1,0,0,0
1,0,26.0,.,.,.,.,.,9.0,1,75,30,0.1,0,0,0
1,4,0.0,1.5,150.0,0,0,1,.,.,65,25,.,.,.,. 
1,0,2.0,.,.,.,.,.,15.0,1,65,25,0.1,0,0,0
1,0,4.0,.,.,.,.,.,12.0,1,67,25,0.1,0,0,0
1,1,24.0,1.5,150.0,0,0,1,.,.,68,25,.,.,.,. 
1,0,26.0,.,.,.,.,.,18.0,1,68,25,0.1,0,0,0
1,0,28.0,.,.,.,.,.,14.0,1,68,25,0.1,0,0,0";

        // Create a reader from the CSV data
        let cursor = Cursor::new(csv_data);

        // Deserialize the data
        let data = from_reader(cursor).unwrap();

        // Verify the structure
        assert_eq!(data.len(), 1, "Should have 1 subject");

        let subject = data.get_subject("1").unwrap();
        assert_eq!(
            subject.occasions().len(),
            2,
            "Subject 1 should have 2 occasions"
        );

        assert_eq!(
            subject.occasions()[0].events().len(),
            6,
            "Occasion 1 should have 6 events"
        );
        assert_eq!(
            subject.occasions()[1].events().len(),
            6,
            "Occasion 2 should have 6 events"
        );
    }
}
