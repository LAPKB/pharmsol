use crate::{
    data::*,
    simulator::{Fa, Lag},
    PharmsolError,
};
use csv::WriterBuilder;
use serde::{Deserialize, Serialize};
use std::fmt;

/// The main data container for pharmacokinetic/pharmacodynamic data
///
/// [Data] is a collection of [Subject] instances, which themselves contain [Occasion] instances with [Event]s.
/// This structure represents the complete dataset for pharmacometric analysis.
///
/// # Examples
///
/// ```
/// use pharmsol::*;
///
/// // Create subjects
/// let subject1 = Subject::builder("patient_001")
///     .bolus(0.0, 100.0, 0)
///     .observation(1.0, 5.0, 0)
///     .build();
///     
/// let subject2 = Subject::builder("patient_002")
///     .bolus(0.0, 120.0, 0)
///     .observation(1.0, 6.0, 0)
///     .build();
///     
/// // Create dataset with multiple subjects
/// let mut data = Data::new(vec![subject1]);
/// data.add_subject(subject2);
///
/// // Filter data
/// let filtered = data.filter_include(&["patient_001".to_string()]);
/// ```
#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct Data {
    subjects: Vec<Subject>,
}
impl Data {
    /// Constructs a new [Data] object from a vector of [Subject]s
    ///
    /// It is recommended to construct subjects using the [SubjectBuilder] to ensure proper data formatting.
    ///
    /// # Arguments
    ///
    /// * `subjects` - Vector of [Subject]s to include in the dataset
    pub fn new(subjects: Vec<Subject>) -> Self {
        Data { subjects }
    }

    /// Get a vector of references to all subjects in the dataset
    ///
    /// # Returns
    ///
    /// Vector of references to all subjects
    pub fn get_subjects(&self) -> Vec<&Subject> {
        self.subjects.iter().collect()
    }

    /// Add a subject to the dataset
    ///
    /// # Arguments
    ///
    /// * `subject` - Subject to add to the dataset
    pub fn add_subject(&mut self, subject: Subject) {
        self.subjects.push(subject);
    }

    /// Get a specific subject by ID
    ///
    /// # Arguments
    ///
    /// * `id` - The ID of the subject to retrieve
    ///
    /// # Returns
    ///
    /// An `Option` containing a reference to the subject if found, or `None` if not found
    pub fn get_subject(&self, id: &str) -> Option<&Subject> {
        self.subjects.iter().find(|subject| subject.id() == id)
    }

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

        for subject in self.get_subjects() {
            for occasion in subject.occasions() {
                for event in occasion.get_events(None, false) {
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

    /// Filter the dataset to include only subjects with specific IDs
    ///
    /// # Arguments
    ///
    /// * `include` - Vector of subject IDs to include
    ///
    /// # Returns
    ///
    /// A new `Data` object containing only the specified subjects
    pub fn filter_include(&self, include: &[String]) -> Data {
        let subjects = self
            .subjects
            .iter()
            .filter(|subject| include.iter().any(|id| id == subject.id()))
            .cloned()
            .collect();
        Data::new(subjects)
    }

    /// Filter the dataset to exclude subjects with specific IDs
    ///
    /// # Arguments
    ///
    /// * `exclude` - Vector of subject IDs to exclude
    ///
    /// # Returns
    ///
    /// A new `Data` object with the specified subjects excluded
    pub fn filter_exclude(&self, exclude: Vec<String>) -> Data {
        let subjects = self
            .subjects
            .iter()
            .filter(|subject| !exclude.iter().any(|id| id == subject.id()))
            .cloned()
            .collect();
        Data::new(subjects)
    }

    /// Expand the dataset by adding observations at regular time intervals
    ///
    /// This is useful for creating a dense grid of time points for simulations.
    ///
    /// # Arguments
    ///
    /// * `idelta` - Time interval between added observations
    /// * `tad` - Additional time to add after the last observation
    ///
    /// # Returns
    ///
    /// A new `Data` object with expanded observations
    pub fn expand(&self, idelta: f64, tad: f64) -> Data {
        if idelta <= 0.0 {
            return self.clone();
        }
        // Determine the last time of the last observation, or Infusion + Duration
        let mut last_time = self
            .subjects
            .iter()
            .map(|subject| {
                subject
                    .occasions
                    .iter()
                    .map(|occasion| {
                        occasion
                            .events
                            .iter()
                            .filter_map(|event| match event {
                                Event::Observation(observation) => Some(observation.time()),
                                Event::Infusion(infusion) => {
                                    Some(infusion.time() + infusion.duration())
                                }
                                _ => None,
                            })
                            .max_by(|a, b| a.partial_cmp(b).unwrap())
                            .unwrap_or(0.0)
                    })
                    .max_by(|a, b| a.partial_cmp(b).unwrap())
                    .unwrap_or(0.0)
            })
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(0.0);
        last_time += tad;

        let mut outeq_values: Vec<usize> = Vec::new();
        for subject in &self.subjects {
            for occasion in &subject.occasions {
                for event in occasion.get_events(None, true) {
                    if let Event::Observation(obs) = event {
                        outeq_values.push(obs.outeq());
                    }
                }
            }
        }
        outeq_values.sort();
        outeq_values.dedup();

        // Create a new data structure with added observations at intervals of idelta
        let mut new_subjects: Vec<Subject> = Vec::new();
        for subject in &self.subjects {
            let mut new_occasions: Vec<Occasion> = Vec::new();
            for occasion in &subject.occasions {
                let old_events = occasion.get_events(None, true);
                let mut new_events: Vec<Event> = Vec::new();
                let mut time = 0.0;
                while time < last_time {
                    for outeq in &outeq_values {
                        let obs = Observation::new(time, None, *outeq, None);
                        new_events.push(Event::Observation(obs));
                    }

                    time += idelta;
                    time = (time * 1e6).round() / 1e6;
                }

                new_events.extend(old_events);
                new_occasions.push(Occasion::new(
                    new_events,
                    occasion.covariates.clone(),
                    occasion.index,
                ));
                new_occasions
                    .iter_mut()
                    .for_each(|occasion| occasion.sort());
            }
            // Add the new occasions to the new subject
            new_subjects.push(Subject::new(subject.id.clone(), new_occasions));
        }

        Data::new(new_subjects)
    }

    /// Get the number of subjects in the dataset
    pub(crate) fn len(&self) -> usize {
        self.subjects.len()
    }

    // /// Returns the number of subjects in the data
    // fn nsubjects(&self) -> usize {
    //     self.subjects.len()
    // }

    // fn nobs(&self) -> usize {
    //     // Count the number of the event type Observation in the data
    //     self.subjects
    //         .iter()
    //         .map(|subject| {
    //             subject
    //                 .occasions
    //                 .iter()
    //                 .map(|occasion| {
    //                     occasion
    //                         .events
    //                         .iter()
    //                         .filter(|event| matches!(event, Event::Observation(_)))
    //                         .count()
    //                 })
    //                 .sum::<usize>()
    //         })
    //         .sum()
    // }
}

impl IntoIterator for Data {
    type Item = Subject;
    type IntoIter = std::vec::IntoIter<Subject>;

    fn into_iter(self) -> Self::IntoIter {
        self.subjects.into_iter()
    }
}

/// A subject in a pharmacometric dataset
///
/// A [Subject] represents a single individual with one or more occasions of data,
/// each containing events (doses, observations) and covariates.
#[derive(serde::Serialize, Debug, Deserialize, Clone)]
pub struct Subject {
    id: String,
    occasions: Vec<Occasion>,
}
impl Subject {
    /// Create a new subject with the given ID and occasions
    ///
    /// # Arguments
    ///
    /// * `id` - The subject identifier
    /// * `occasions` - Vector of occasions for this subject
    pub(crate) fn new(id: String, occasions: Vec<Occasion>) -> Self {
        let mut subject = Subject { id, occasions };
        for occasion in subject.occasions.iter_mut() {
            occasion.sort();
        }
        subject
    }

    /// Get a vector of references to all occasions for this subject
    ///
    /// # Returns
    ///
    /// Vector of references to all occasions
    pub fn occasions(&self) -> Vec<&Occasion> {
        self.occasions.iter().collect()
    }

    /// Get the ID of the subject
    ///
    /// # Returns
    ///
    /// The subject's identifier
    pub fn id(&self) -> &String {
        &self.id
    }

    /// Create a new subject from one or more occasions
    ///
    /// This is useful when you want to create a subject from specific occasions
    /// rather than building it completely from scratch.
    ///
    /// # Arguments
    ///
    /// * `id` - The subject identifier
    /// * `occasions` - Vector of occasions to include in this subject
    ///
    /// # Returns
    ///
    /// A new subject containing the specified occasions
    pub fn from_occasions(id: String, occasions: Vec<Occasion>) -> Self {
        Subject { id, occasions }
    }

    /// Iterate over a mutable reference to the occasions
    pub fn occasions_mut(&mut self) -> &mut Vec<Occasion> {
        &mut self.occasions
    }

    /// Get a mutable iterator to the occasions
    pub fn occasions_iter_mut(&mut self) -> std::slice::IterMut<Occasion> {
        self.occasions.iter_mut()
    }

    pub fn get_output_equations(&self) -> Vec<usize> {
        // Collect all unique outeq values in order of occurrence
        let outeq_values: Vec<usize> = self
            .occasions
            .iter()
            .flat_map(|occasion| {
                occasion.events.iter().filter_map(|event| match event {
                    Event::Observation(obs) => Some(obs.outeq()),
                    _ => None,
                })
            })
            .collect();
        outeq_values
    }
}

/// An occasion within a subject's dataset
///
/// An [Occasion] represents a distinct period of data collection for a subject,
/// such as a hospital visit or dosing regimen. It contains events (doses, observations)
/// and time-varying covariates.
#[derive(serde::Serialize, Debug, Deserialize, Clone)]
pub struct Occasion {
    events: Vec<Event>,
    covariates: Covariates,
    index: usize,
}

impl Occasion {
    /// Create a new occasion
    ///
    /// # Arguments
    ///
    /// * `events` - Vector of events for this occasion
    /// * `covariates` - Covariates for this occasion
    /// * `index` - The occasion index (0-based)
    pub(crate) fn new(events: Vec<Event>, covariates: Covariates, index: usize) -> Self {
        Occasion {
            events,
            covariates,
            index,
        }
    }

    /// Get a vector of references to all events in this occasion
    ///
    /// # Returns
    ///
    /// Vector of references to all events
    pub fn events(&self) -> Vec<&Event> {
        self.events.iter().collect()
    }

    /// Get the index of the occasion
    ///
    /// # Returns
    ///
    /// The occasion index (0-based)
    pub fn index(&self) -> usize {
        self.index
    }

    /// Add a covariate to this occasion
    ///
    /// # Arguments
    ///
    /// * `name` - Name of the covariate
    /// * `covariate` - The covariate to add
    pub(crate) fn add_covariate(&mut self, name: String, covariate: Covariate) {
        self.covariates.add_covariate(name, covariate);
    }

    fn add_lagtime(&mut self, reorder: Option<(&Fa, &Lag, &Vec<f64>, &Covariates)>) {
        if let Some((_, fn_lag, spp, covariates)) = reorder {
            let spp = nalgebra::DVector::from_vec(spp.to_vec());
            for event in self.events.iter_mut() {
                let time = event.time();
                if let Event::Bolus(bolus) = event {
                    let lagtime = fn_lag(&spp, time, covariates);
                    if let Some(l) = lagtime.get(&bolus.input()) {
                        *bolus.mut_time() += l;
                    }
                }
            }
        }
        self.sort();
    }

    fn add_bioavailability(&mut self, reorder: Option<(&Fa, &Lag, &Vec<f64>, &Covariates)>) {
        // If lagtime is empty, return early
        if let Some((fn_fa, _, spp, covariates)) = reorder {
            let spp = nalgebra::DVector::from_vec(spp.to_vec());
            for event in self.events.iter_mut() {
                let time = event.time();
                if let Event::Bolus(bolus) = event {
                    let fa = fn_fa(&spp, time, covariates);
                    if let Some(f) = fa.get(&bolus.input()) {
                        bolus.set_amount(bolus.amount() * f);
                    }
                }
            }
        }
    }

    /// Sort events by time, then by [Event] type so that [Bolus] and [Infusion] come before [Observation]
    pub(crate) fn sort(&mut self) {
        self.events.sort_by(|a, b| {
            // First, compare times using partial_cmp, then compare types if times are equal.
            let time_a = match a {
                Event::Bolus(bolus) => bolus.time(),
                Event::Infusion(infusion) => infusion.time(),
                Event::Observation(observation) => observation.time(),
            };
            let time_b = match b {
                Event::Bolus(bolus) => bolus.time(),
                Event::Infusion(infusion) => infusion.time(),
                Event::Observation(observation) => observation.time(),
            };

            match time_a.partial_cmp(&time_b) {
                Some(std::cmp::Ordering::Equal) => {
                    // If times are equal, sort by event type.
                    let type_order_a = match a {
                        Event::Bolus(_) => 1,
                        Event::Infusion(_) => 2,
                        Event::Observation(_) => 3,
                    };
                    let type_order_b = match b {
                        Event::Bolus(_) => 1,
                        Event::Infusion(_) => 2,
                        Event::Observation(_) => 3,
                    };
                    type_order_a.cmp(&type_order_b)
                }
                other => other.unwrap_or(std::cmp::Ordering::Equal),
            }
        });
    }
    // TODO: This clones the occasion, which is not ideal

    /// Get events with modifications for lag time and bioavailability
    ///
    /// # Arguments
    ///
    /// * `lagtime` - Optional map of compartment-specific lag times
    /// * `bioavailability` - Optional map of compartment-specific bioavailability factors
    /// * `ignore` - Whether to exclude events marked as "ignore"
    ///
    /// # Returns
    ///
    /// Vector of events, potentially filtered and with times adjusted for lag and bioavailability
    pub fn get_events(
        &self,
        reorder: Option<(&Fa, &Lag, &Vec<f64>, &Covariates)>,
        ignore: bool,
    ) -> Vec<Event> {
        let mut occ = self.clone();

        occ.add_lagtime(reorder);
        occ.add_bioavailability(reorder);

        // Filter out events that are marked as ignore
        if ignore {
            occ.events.iter().cloned().collect()
        } else {
            occ.events.clone()
        }
    }

    /// Get the covariates for this occasion
    ///
    /// # Returns
    ///
    /// Reference to the occasion's covariates, if any
    pub fn covariates(&self) -> &Covariates {
        &self.covariates
    }

    /// Add an event to the [Occasion]
    ///
    /// Note that this will sort the events automatically, ensuring events are sorted by time, then by [Event] type so that [Bolus] and [Infusion] come before [Observation]
    pub fn add_event(&mut self, event: Event) {
        self.events.push(event);
        self.sort();
    }

    /// Get the last event in this occasion
    ///
    /// # Returns
    ///
    /// Reference to the last event, if any
    pub(crate) fn last_event(&self) -> Option<&Event> {
        self.events.last()
    }

    /// Get a mutable reference to the events
    pub fn events_mut(&mut self) -> &mut Vec<Event> {
        &mut self.events
    }

    /// Get a mutable iterator to the events
    pub fn events_iter_mut(&mut self) -> std::slice::IterMut<Event> {
        self.events.iter_mut()
    }

    pub(crate) fn initial_time(&self) -> f64 {
        //TODO this can be pre-computed when the struct is initially created
        self.events
            .iter()
            .filter_map(|event| match event {
                Event::Observation(observation) => Some(observation.time()),
                Event::Bolus(bolus) => Some(bolus.time()),
                Event::Infusion(infusion) => Some(infusion.time()),
            })
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(0.0)
    }

    pub(crate) fn infusions_ref(&self) -> Vec<&Infusion> {
        //TODO this can be pre-computed when the struct is initially created
        self.events
            .iter()
            .filter_map(|event| match event {
                Event::Infusion(infusion) => Some(infusion),
                _ => None,
            })
            .collect()
    }
}

impl fmt::Display for Data {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "Data Overview: {} subjects", self.subjects.len())?;
        for subject in &self.subjects {
            writeln!(f, "{}", subject)?;
        }
        Ok(())
    }
}
impl fmt::Display for Subject {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "Subject ID: {}", self.id)?;
        for occasion in &self.occasions {
            writeln!(f, "{}", occasion)?;
        }
        Ok(())
    }
}
impl fmt::Display for Occasion {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "Occasion {}:", self.index)?;
        for event in &self.events {
            writeln!(f, "  {}", event)?;
        }

        writeln!(f, "  Covariates:\n{}", self.covariates)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prelude::*;

    fn create_sample_data() -> Data {
        let subject1 = Subject::builder("subject1")
            .observation(1.0, Some(10.0), 1, None)
            .bolus(2.0, 50.0, 1)
            .infusion(3.0, 100.0, 1, 1.0)
            .covariate("age", 0.0, 30.0)
            .covariate("weight", 0.0, 70.0)
            .reset()
            .observation(4.0, Some(20.0), 2, None)
            .bolus(5.0, 60.0, 2)
            .infusion(6.0, 120.0, 2, 2.0)
            .covariate("age", 0.0, 31.0)
            .covariate("weight", 0.0, 75.0)
            .build();

        let subject2 = Subject::builder("subject2")
            .observation(1.5, Some(15.0), 1, None)
            .bolus(2.5, 55.0, 1)
            .infusion(3.5, 110.0, 1, 1.5)
            .covariate("age", 0.0, 25.0)
            .covariate("weight", 0.0, 65.0)
            .reset()
            .observation(4.5, Some(25.0), 2, None)
            .bolus(5.5, 65.0, 2)
            .infusion(6.5, 130.0, 2, 2.5)
            .covariate("age", 0.0, 26.0)
            .covariate("weight", 0.0, 68.0)
            .build();

        Data::new(vec![subject1, subject2])
    }

    #[test]
    fn test_new_data() {
        let data = create_sample_data();
        assert_eq!(data.len(), 2);
    }

    #[test]
    fn test_get_subjects() {
        let data = create_sample_data();
        let subjects = data.get_subjects();
        assert_eq!(subjects.len(), 2);
        assert_eq!(subjects[0].id(), "subject1");
        assert_eq!(subjects[1].id(), "subject2");
    }

    #[test]
    fn test_add_subject() {
        let mut data = create_sample_data();
        let new_subject = Subject::builder("subject3")
            .observation(1.0, Some(10.0), 1, None)
            .bolus(2.0, 50.0, 1)
            .infusion(3.0, 100.0, 1, 1.0)
            .covariate("age", 0.0, 30.0)
            .covariate("weight", 0.0, 70.0)
            .build();
        data.add_subject(new_subject);
        assert_eq!(data.len(), 3);
        assert_eq!(data.get_subjects()[2].id(), "subject3");
    }

    #[test]
    fn test_filter_include() {
        let data = create_sample_data();
        let include = vec!["subject1".to_string()];
        let filtered_data = data.filter_include(&include);
        assert_eq!(filtered_data.get_subjects().len(), 1);
        assert_eq!(filtered_data.get_subjects()[0].id(), "subject1");
    }

    #[test]
    fn test_filter_exclude() {
        let data = create_sample_data();
        let filtered_data = data.filter_exclude(vec!["subject1".to_string()]);
        assert_eq!(filtered_data.len(), 1);
        assert_eq!(filtered_data.get_subjects()[0].id(), "subject2");
    }

    #[test]
    fn test_occasion_sort() {
        let mut occasion = Occasion::new(
            vec![
                Event::Observation(Observation::new(2.0, Some(1.0), 1, None)),
                Event::Bolus(Bolus::new(1.0, 100.0, 1)),
            ],
            Covariates::new(),
            1,
        );
        occasion.sort();
        let events = occasion.get_events(None, false);
        match &events[0] {
            Event::Bolus(b) => assert_eq!(b.time(), 1.0),
            _ => panic!("First event should be a Bolus"),
        }
        match &events[1] {
            Event::Observation(o) => assert_eq!(o.time(), 2.0),
            _ => panic!("Second event should be an Observation"),
        }
    }
}
