//! Row representation of [Data] for flexible parsing
//!
//! # Example
//!
//! ```rust
//! use pharmsol::data::parser::DataRow;
//!
//! // Create a dosing row with ADDL expansion
//! let row = DataRow::builder("subject_1", 0.0)
//!     .evid(1)
//!     .dose(100.0)
//!     .input(1)
//!     .addl(3)   // 3 additional doses
//!     .ii(12.0)  // 12 hours apart
//!     .build();
//!
//! let events = row.into_events().unwrap();
//! assert_eq!(events.len(), 4); // Original + 3 additional doses
//! ```
//!

use crate::data::*;
use std::collections::HashMap;
use thiserror::Error;

/// A format-agnostic representation of a single data row
///
/// This struct represents the canonical fields needed to create pharmsol Events.
/// Consumers construct this from their source data (regardless of column names),
/// then call [`into_events()`](DataRow::into_events) to get properly parsed
/// Events with full ADDL expansion, EVID handling, censoring, etc.
///
/// # Fields
///
/// All fields use Pmetrics conventions:
/// - `input` and `outeq` are **1-indexed** (kept as-is, user must size arrays accordingly)
/// - `evid`: 0=observation, 1=dose, 4=reset/new occasion
/// - `addl`: positive=forward in time, negative=backward in time
///
/// # Example
///
/// ```rust
/// use pharmsol::data::parser::DataRow;
///
/// // Observation row
/// let obs = DataRow::builder("pt1", 1.0)
///     .evid(0)
///     .out(25.5)
///     .outeq(1)
///     .build();
///
/// // Dosing row with negative ADDL (doses before time 0)
/// let dose = DataRow::builder("pt1", 0.0)
///     .evid(1)
///     .dose(100.0)
///     .input(1)
///     .addl(-10)  // 10 doses BEFORE time 0
///     .ii(12.0)
///     .build();
///
/// let events = dose.into_events().unwrap();
/// // Events at times: -120, -108, -96, ..., -12, 0
/// assert_eq!(events.len(), 11);
/// ```
#[derive(Debug, Clone, Default)]
pub struct DataRow {
    /// Subject identifier (required)
    pub id: String,
    /// Event time (required)
    pub time: f64,
    /// Event type: 0=observation, 1=dose, 4=reset/new occasion
    pub evid: i32,
    /// Dose amount (for EVID=1)
    pub dose: Option<f64>,
    /// Infusion duration (if > 0, dose is infusion; otherwise bolus)
    pub dur: Option<f64>,
    /// Additional doses count (positive=forward, negative=backward in time)
    pub addl: Option<i64>,
    /// Interdose interval for ADDL
    pub ii: Option<f64>,
    /// Input compartment
    pub input: Option<usize>,
    /// Observed value (for EVID=0)
    pub out: Option<f64>,
    /// Output equation number
    pub outeq: Option<usize>,
    /// Censoring indicator
    pub cens: Option<Censor>,
    /// Error polynomial coefficients
    pub c0: Option<f64>,
    /// Error polynomial coefficients
    pub c1: Option<f64>,
    /// Error polynomial coefficients
    pub c2: Option<f64>,
    /// Error polynomial coefficients
    pub c3: Option<f64>,
    /// Covariate values at this time point
    pub covariates: HashMap<String, f64>,
}

impl DataRow {
    /// Create a new builder for constructing a DataRow
    ///
    /// # Arguments
    ///
    /// * `id` - Subject identifier
    /// * `time` - Event time
    ///
    /// # Example
    ///
    /// ```rust
    /// use pharmsol::data::parser::DataRow;
    ///
    /// let row = DataRow::builder("patient_001", 0.0)
    ///     .evid(1)
    ///     .dose(100.0)
    ///     .input(1)
    ///     .build();
    /// ```
    pub fn builder(id: impl Into<String>, time: f64) -> DataRowBuilder {
        DataRowBuilder::new(id, time)
    }

    /// Get error polynomial if all coefficients are present
    fn get_errorpoly(&self) -> Option<ErrorPoly> {
        match (self.c0, self.c1, self.c2, self.c3) {
            (Some(c0), Some(c1), Some(c2), Some(c3)) => Some(ErrorPoly::new(c0, c1, c2, c3)),
            _ => None,
        }
    }

    /// Convert this row into pharmsol Events
    ///
    /// This method contains all the complex parsing logic:
    /// - EVID interpretation (0=observation, 1=dose, 4=reset)
    /// - ADDL/II expansion (both positive and negative directions)
    /// - Infusion vs bolus detection based on DUR
    /// - Censoring and error polynomial handling
    ///
    /// # ADDL Expansion
    ///
    /// When `addl` and `ii` are both specified:
    /// - **Positive ADDL**: Additional doses are placed *after* the base time
    ///   - Example: time=0, addl=3, ii=12 → doses at 12, 24, 36, then 0
    /// - **Negative ADDL**: Additional doses are placed *before* the base time
    ///   - Example: time=0, addl=-3, ii=12 → doses at -36, -24, -12, then 0
    ///
    /// # Returns
    ///
    /// A vector of Events. A single row may produce multiple events when ADDL is used.
    ///
    /// # Errors
    ///
    /// Returns [`DataError`] if required fields are missing for the given EVID:
    /// - EVID=0: Requires `outeq`
    /// - EVID=1: Requires `dose` and `input`; if `dur > 0`, it's an infusion
    ///
    /// # Example
    ///
    /// ```rust
    /// use pharmsol::data::parser::DataRow;
    ///
    /// let row = DataRow::builder("pt1", 0.0)
    ///     .evid(1)
    ///     .dose(100.0)
    ///     .input(1)
    ///     .addl(2)
    ///     .ii(24.0)
    ///     .build();
    ///
    /// let events = row.into_events().unwrap();
    /// assert_eq!(events.len(), 3); // doses at 24, 48, and 0
    ///
    /// let times: Vec<f64> = events.iter().map(|e| e.time()).collect();
    /// assert_eq!(times, vec![24.0, 48.0, 0.0]);
    /// ```
    pub fn into_events(self) -> Result<Vec<Event>, DataError> {
        let mut events: Vec<Event> = Vec::new();

        match self.evid {
            0 => {
                // Observation event
                events.push(Event::Observation(Observation::new(
                    self.time,
                    self.out,
                    self.outeq
                        .ok_or_else(|| DataError::MissingObservationOuteq {
                            id: self.id.clone(),
                            time: self.time,
                        })?, // Keep 1-indexed as provided by Pmetrics
                    self.get_errorpoly(),
                    0, // occasion set later
                    self.cens.unwrap_or(Censor::None),
                )));
            }
            1 | 4 => {
                // Dosing event (1) or reset with dose (4)

                let input = self.input.ok_or_else(|| DataError::MissingBolusInput {
                    id: self.id.clone(),
                    time: self.time,
                })?; // Keep 1-indexed as provided by Pmetrics

                let event = if self.dur.unwrap_or(0.0) > 0.0 {
                    // Infusion
                    Event::Infusion(Infusion::new(
                        self.time,
                        self.dose.ok_or_else(|| DataError::MissingInfusionDose {
                            id: self.id.clone(),
                            time: self.time,
                        })?,
                        input,
                        self.dur.ok_or_else(|| DataError::MissingInfusionDur {
                            id: self.id.clone(),
                            time: self.time,
                        })?,
                        0,
                    ))
                } else {
                    // Bolus
                    Event::Bolus(Bolus::new(
                        self.time,
                        self.dose.ok_or_else(|| DataError::MissingBolusDose {
                            id: self.id.clone(),
                            time: self.time,
                        })?,
                        input,
                        0,
                    ))
                };

                // Handle ADDL/II expansion
                if let (Some(addl), Some(ii)) = (self.addl, self.ii) {
                    if addl != 0 && ii > 0.0 {
                        let mut ev = event.clone();
                        let interval = ii.abs();
                        let repetitions = addl.abs();
                        let direction = addl.signum() as f64;

                        for _ in 0..repetitions {
                            ev.inc_time(direction * interval);
                            events.push(ev.clone());
                        }
                    }
                }

                events.push(event);
            }
            _ => {
                return Err(DataError::UnknownEvid {
                    evid: self.evid as isize,
                    id: self.id.clone(),
                    time: self.time,
                });
            }
        }

        Ok(events)
    }

    /// Get the covariate values for this row
    ///
    /// Returns a reference to the HashMap of covariate name → value pairs.
    pub fn covariates(&self) -> &HashMap<String, f64> {
        &self.covariates
    }

    /// Check if this row represents a new occasion (EVID=4)
    pub fn is_occasion_reset(&self) -> bool {
        self.evid == 4
    }

    /// Get the subject ID
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Get the event time
    pub fn time(&self) -> f64 {
        self.time
    }
}

/// Builder for constructing DataRow with a fluent API
///
/// # Example
///
/// ```rust
/// use pharmsol::data::parser::DataRow;
/// use pharmsol::data::Censor;
///
/// let row = DataRow::builder("patient_001", 1.5)
///     .evid(0)
///     .out(25.5)
///     .outeq(1)
///     .cens(Censor::None)
///     .covariate("weight", 70.0)
///     .covariate("age", 45.0)
///     .build();
/// ```
#[derive(Debug, Clone)]
pub struct DataRowBuilder {
    row: DataRow,
}

impl DataRowBuilder {
    /// Create a new builder with required fields
    ///
    /// # Arguments
    ///
    /// * `id` - Subject identifier
    /// * `time` - Event time
    pub fn new(id: impl Into<String>, time: f64) -> Self {
        Self {
            row: DataRow {
                id: id.into(),
                time,
                evid: 0, // Default to observation
                ..Default::default()
            },
        }
    }

    /// Set the event type
    ///
    /// # Arguments
    ///
    /// * `evid` - Event ID: 0=observation, 1=dose, 4=reset/new occasion
    pub fn evid(mut self, evid: i32) -> Self {
        self.row.evid = evid;
        self
    }

    /// Set the dose amount
    ///
    /// Required for EVID=1 (dosing events).
    pub fn dose(mut self, dose: f64) -> Self {
        self.row.dose = Some(dose);
        self
    }

    /// Set the infusion duration
    ///
    /// If > 0, the dose is treated as an infusion rather than a bolus.
    pub fn dur(mut self, dur: f64) -> Self {
        self.row.dur = Some(dur);
        self
    }

    /// Set the additional doses count
    ///
    /// # Arguments
    ///
    /// * `addl` - Number of additional doses
    ///   - Positive: doses placed after the base time
    ///   - Negative: doses placed before the base time
    pub fn addl(mut self, addl: i64) -> Self {
        self.row.addl = Some(addl);
        self
    }

    /// Set the interdose interval
    ///
    /// Used with ADDL to specify time between additional doses.
    pub fn ii(mut self, ii: f64) -> Self {
        self.row.ii = Some(ii);
        self
    }

    /// Set the input compartment (1-indexed)
    ///
    /// Required for EVID=1 (dosing events).
    /// Kept as 1-indexed; user must size state arrays accordingly.
    pub fn input(mut self, input: usize) -> Self {
        self.row.input = Some(input);
        self
    }

    /// Set the observed value
    ///
    /// Used for EVID=0 (observation events).
    pub fn out(mut self, out: f64) -> Self {
        self.row.out = Some(out);
        self
    }

    /// Set the output equation (1-indexed)
    ///
    /// Required for EVID=0 (observation events).
    /// Will be converted to 0-indexed internally.
    pub fn outeq(mut self, outeq: usize) -> Self {
        self.row.outeq = Some(outeq);
        self
    }

    /// Set the censoring type
    pub fn cens(mut self, cens: Censor) -> Self {
        self.row.cens = Some(cens);
        self
    }

    /// Set error polynomial coefficients
    ///
    /// The error polynomial models observation error as:
    /// SD = c0 + c1*Y + c2*Y² + c3*Y³
    pub fn error_poly(mut self, c0: f64, c1: f64, c2: f64, c3: f64) -> Self {
        self.row.c0 = Some(c0);
        self.row.c1 = Some(c1);
        self.row.c2 = Some(c2);
        self.row.c3 = Some(c3);
        self
    }

    /// Add a covariate value
    ///
    /// Can be called multiple times to add multiple covariates.
    ///
    /// # Arguments
    ///
    /// * `name` - Covariate name
    /// * `value` - Covariate value at this time point
    pub fn covariate(mut self, name: impl Into<String>, value: f64) -> Self {
        self.row.covariates.insert(name.into(), value);
        self
    }

    /// Build the DataRow
    pub fn build(self) -> DataRow {
        self.row
    }
}

/// Build a [Data] object from an iterator of [DataRow]s
///
/// This function handles all the complex assembly logic:
/// - Groups rows by subject ID
/// - Splits into occasions at EVID=4 boundaries
/// - Converts rows to events via [`DataRow::into_events()`]
/// - Builds covariates from row covariate data
///
/// # Example
///
/// ```rust
/// use pharmsol::data::parser::{DataRow, build_data};
///
/// let rows = vec![
///     // Subject 1, Occasion 0
///     DataRow::builder("pt1", 0.0)
///         .evid(1).dose(100.0).input(1).build(),
///     DataRow::builder("pt1", 1.0)
///         .evid(0).out(50.0).outeq(1).build(),
///     // Subject 1, Occasion 1 (EVID=4 starts new occasion)
///     DataRow::builder("pt1", 24.0)
///         .evid(4).dose(100.0).input(1).build(),
///     DataRow::builder("pt1", 25.0)
///         .evid(0).out(48.0).outeq(1).build(),
///     // Subject 2
///     DataRow::builder("pt2", 0.0)
///         .evid(1).dose(50.0).input(1).build(),
/// ];
///
/// let data = build_data(rows).unwrap();
/// assert_eq!(data.subjects().len(), 2);
/// ```
pub fn build_data(rows: impl IntoIterator<Item = DataRow>) -> Result<Data, DataError> {
    // Group rows by subject ID
    let mut rows_map: std::collections::HashMap<String, Vec<DataRow>> =
        std::collections::HashMap::new();
    for row in rows {
        rows_map.entry(row.id.clone()).or_default().push(row);
    }

    let mut subjects: Vec<Subject> = Vec::new();

    for (id, rows) in rows_map {
        // Split rows into occasion blocks at EVID=4 boundaries
        let split_indices: Vec<usize> = rows
            .iter()
            .enumerate()
            .filter_map(|(i, row)| if row.evid == 4 { Some(i) } else { None })
            .collect();

        let mut block_rows_vec: Vec<&[DataRow]> = Vec::new();
        let mut start = 0;
        for &split_index in &split_indices {
            if start < split_index {
                block_rows_vec.push(&rows[start..split_index]);
            }
            start = split_index;
        }
        if start < rows.len() {
            block_rows_vec.push(&rows[start..]);
        }

        // Build occasions
        let mut occasions: Vec<Occasion> = Vec::new();
        for (block_index, block) in block_rows_vec.iter().enumerate() {
            let mut events: Vec<Event> = Vec::new();

            // Collect covariate observations for this block
            let mut observed_covariates: std::collections::HashMap<
                String,
                Vec<(f64, Option<f64>)>,
            > = std::collections::HashMap::new();

            for row in *block {
                // Parse events
                let row_events = row.clone().into_events()?;
                events.extend(row_events);

                // Collect covariates
                for (name, value) in &row.covariates {
                    observed_covariates
                        .entry(name.clone())
                        .or_default()
                        .push((row.time, Some(*value)));
                }
            }

            // Set occasion index on all events
            events.iter_mut().for_each(|e| e.set_occasion(block_index));

            // Build covariates
            let covariates = Covariates::from_pmetrics_observations(&observed_covariates);

            // Create occasion
            let mut occasion = Occasion::new(block_index);
            occasion.events = events;
            occasion.covariates = covariates;
            occasion.sort();
            occasions.push(occasion);
        }

        subjects.push(Subject::new(id, occasions));
    }

    // Sort subjects alphabetically by ID for consistent ordering
    subjects.sort_by(|a, b| a.id().cmp(b.id()));

    Ok(Data::new(subjects))
}

/// Custom error type for the module
#[allow(private_interfaces)]
#[derive(Error, Debug, Clone)]
pub enum DataError {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_observation_row() {
        let row = DataRow::builder("pt1", 1.0)
            .evid(0)
            .out(25.5)
            .outeq(1)
            .build();

        let events = row.into_events().unwrap();
        assert_eq!(events.len(), 1);

        match &events[0] {
            Event::Observation(obs) => {
                assert_eq!(obs.time(), 1.0);
                assert_eq!(obs.value(), Some(25.5));
                assert_eq!(obs.outeq(), 1); // Kept as 1-indexed
            }
            _ => panic!("Expected observation event"),
        }
    }

    #[test]
    fn test_bolus_row() {
        let row = DataRow::builder("pt1", 0.0)
            .evid(1)
            .dose(100.0)
            .input(1)
            .build();

        let events = row.into_events().unwrap();
        assert_eq!(events.len(), 1);

        match &events[0] {
            Event::Bolus(bolus) => {
                assert_eq!(bolus.time(), 0.0);
                assert_eq!(bolus.amount(), 100.0);
                assert_eq!(bolus.input(), 1); // Kept as 1-indexed
            }
            _ => panic!("Expected bolus event"),
        }
    }

    #[test]
    fn test_infusion_row() {
        let row = DataRow::builder("pt1", 0.0)
            .evid(1)
            .dose(100.0)
            .dur(2.0)
            .input(1)
            .build();

        let events = row.into_events().unwrap();
        assert_eq!(events.len(), 1);

        match &events[0] {
            Event::Infusion(inf) => {
                assert_eq!(inf.time(), 0.0);
                assert_eq!(inf.amount(), 100.0);
                assert_eq!(inf.duration(), 2.0);
                assert_eq!(inf.input(), 1); // Kept as 1-indexed
            }
            _ => panic!("Expected infusion event"),
        }
    }

    #[test]
    fn test_positive_addl() {
        let row = DataRow::builder("pt1", 0.0)
            .evid(1)
            .dose(100.0)
            .input(1)
            .addl(3)
            .ii(12.0)
            .build();

        let events = row.into_events().unwrap();
        assert_eq!(events.len(), 4); // Original + 3 additional

        let times: Vec<f64> = events.iter().map(|e| e.time()).collect();
        // Additional doses come first, then original
        assert_eq!(times, vec![12.0, 24.0, 36.0, 0.0]);
    }

    #[test]
    fn test_negative_addl() {
        let row = DataRow::builder("pt1", 0.0)
            .evid(1)
            .dose(100.0)
            .input(1)
            .addl(-3)
            .ii(12.0)
            .build();

        let events = row.into_events().unwrap();
        assert_eq!(events.len(), 4); // Original + 3 additional

        let times: Vec<f64> = events.iter().map(|e| e.time()).collect();
        // Negative ADDL: doses go backward in time
        assert_eq!(times, vec![-12.0, -24.0, -36.0, 0.0]);
    }

    #[test]
    fn test_large_negative_addl() {
        // Match the pharmsol pmetrics test case
        let row = DataRow::builder("pt1", 0.0)
            .evid(1)
            .dose(100.0)
            .input(1)
            .addl(-10)
            .ii(12.0)
            .build();

        let events = row.into_events().unwrap();
        assert_eq!(events.len(), 11); // Original + 10 additional

        let times: Vec<f64> = events.iter().map(|e| e.time()).collect();
        assert_eq!(
            times,
            vec![-12.0, -24.0, -36.0, -48.0, -60.0, -72.0, -84.0, -96.0, -108.0, -120.0, 0.0]
        );
    }

    #[test]
    fn test_infusion_with_addl() {
        let row = DataRow::builder("pt1", 0.0)
            .evid(1)
            .dose(100.0)
            .dur(1.0)
            .input(1)
            .addl(2)
            .ii(24.0)
            .build();

        let events = row.into_events().unwrap();
        assert_eq!(events.len(), 3);

        // All events should be infusions
        for event in &events {
            match event {
                Event::Infusion(inf) => {
                    assert_eq!(inf.amount(), 100.0);
                    assert_eq!(inf.duration(), 1.0);
                }
                _ => panic!("Expected infusion event"),
            }
        }
    }

    #[test]
    fn test_covariates() {
        let row = DataRow::builder("pt1", 0.0)
            .evid(0)
            .out(25.0)
            .outeq(1)
            .covariate("weight", 70.0)
            .covariate("age", 45.0)
            .build();

        assert_eq!(row.covariates().len(), 2);
        assert_eq!(row.covariates().get("weight"), Some(&70.0));
        assert_eq!(row.covariates().get("age"), Some(&45.0));
    }

    #[test]
    fn test_error_poly() {
        let row = DataRow::builder("pt1", 1.0)
            .evid(0)
            .out(25.0)
            .outeq(1)
            .error_poly(0.1, 0.2, 0.0, 0.0)
            .build();

        let events = row.into_events().unwrap();
        match &events[0] {
            Event::Observation(obs) => {
                let ep = obs.errorpoly().unwrap();
                assert_eq!(ep.coefficients(), (0.1, 0.2, 0.0, 0.0));
            }
            _ => panic!("Expected observation"),
        }
    }

    #[test]
    fn test_censoring() {
        let row = DataRow::builder("pt1", 1.0)
            .evid(0)
            .out(0.5)
            .outeq(1)
            .cens(Censor::BLOQ)
            .build();

        let events = row.into_events().unwrap();
        match &events[0] {
            Event::Observation(obs) => {
                assert!(obs.censored());
                assert_eq!(obs.censoring(), Censor::BLOQ);
            }
            _ => panic!("Expected observation"),
        }
    }

    #[test]
    fn test_missing_outeq_error() {
        let row = DataRow::builder("pt1", 1.0)
            .evid(0)
            .out(25.0)
            // Missing outeq
            .build();

        let result = row.into_events();
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            DataError::MissingObservationOuteq { .. }
        ));
    }

    #[test]
    fn test_missing_dose_error() {
        let row = DataRow::builder("pt1", 0.0)
            .evid(1)
            .input(1)
            // Missing dose
            .build();

        let result = row.into_events();
        assert!(result.is_err());
    }

    #[test]
    fn test_missing_input_error() {
        let row = DataRow::builder("pt1", 0.0)
            .evid(1)
            .dose(100.0)
            // Missing input
            .build();

        let result = row.into_events();
        assert!(result.is_err());
    }

    #[test]
    fn test_unknown_evid_error() {
        let row = DataRow::builder("pt1", 0.0).evid(99).build();

        let result = row.into_events();
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            DataError::UnknownEvid { evid: 99, .. }
        ));
    }

    #[test]
    fn test_addl_zero_has_no_effect() {
        let row = DataRow::builder("pt1", 0.0)
            .evid(1)
            .dose(100.0)
            .input(1)
            .addl(0)
            .ii(12.0)
            .build();

        let events = row.into_events().unwrap();
        assert_eq!(events.len(), 1); // Only original dose
    }

    #[test]
    fn test_addl_without_ii_has_no_effect() {
        let row = DataRow::builder("pt1", 0.0)
            .evid(1)
            .dose(100.0)
            .input(1)
            .addl(5)
            // Missing ii
            .build();

        let events = row.into_events().unwrap();
        assert_eq!(events.len(), 1); // Only original dose
    }

    #[test]
    fn test_evid_4_reset() {
        let row = DataRow::builder("pt1", 24.0)
            .evid(4)
            .dose(100.0)
            .input(1)
            .build();

        assert!(row.is_occasion_reset());
        let events = row.into_events().unwrap();
        assert_eq!(events.len(), 1);
    }
}
