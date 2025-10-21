use crate::{
    data::*,
    simulator::{Fa, Lag},
    Censor,
};
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
    pub fn subjects(&self) -> Vec<&Subject> {
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

    /// Get a mutable reference to a specific subject by ID
    ///
    /// # Arguments
    ///
    /// * `id` - The ID of the subject to retrieve
    ///
    /// # Returns
    ///
    /// An `Option` containing a reference to the subject if found, or `None` if not found
    pub fn get_subject_mut(&mut self, id: &str) -> Option<&mut Subject> {
        self.subjects.iter_mut().find(|subject| subject.id() == id)
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
    /// Observations are only added if they don't already exist at that time/outeq combination.
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

        // Determine the last time across all subjects and occasions
        let last_time = self
            .subjects
            .iter()
            .flat_map(|subject| &subject.occasions)
            .flat_map(|occasion| &occasion.events)
            .filter_map(|event| match event {
                Event::Observation(observation) => Some(observation.time()),
                Event::Infusion(infusion) => Some(infusion.time() + infusion.duration()),
                _ => None,
            })
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(0.0)
            + tad;

        // Collect unique output equations more efficiently
        let outeq_values = self.get_output_equations();

        // Create new data structure with expanded observations
        let new_subjects = self
            .subjects
            .iter()
            .map(|subject| {
                let new_occasions = subject
                    .occasions
                    .iter()
                    .map(|occasion| {
                        let old_events = occasion.process_events(None, true);

                        // Create a set of existing (time, outeq) pairs for fast lookup
                        let existing_obs: std::collections::HashSet<(u64, usize)> = old_events
                            .iter()
                            .filter_map(|event| match event {
                                Event::Observation(obs) => {
                                    // Convert to microseconds for consistent comparison
                                    let time_key = (obs.time() * 1e6).round() as u64;
                                    Some((time_key, obs.outeq()))
                                }
                                _ => None,
                            })
                            .collect();

                        // Generate new observation times
                        let mut new_events = Vec::new();
                        let mut time = 0.0;
                        while time < last_time {
                            let time_key = (time * 1e6).round() as u64;

                            for &outeq in &outeq_values {
                                // Only add if this (time, outeq) combination doesn't exist
                                if !existing_obs.contains(&(time_key, outeq)) {
                                    let obs = Observation::new(
                                        time,
                                        None,
                                        outeq,
                                        None,
                                        occasion.index,
                                        Censor::None,
                                    );
                                    new_events.push(Event::Observation(obs));
                                }
                            }

                            time += idelta;
                            time = (time * 1e6).round() / 1e6;
                        }

                        // Add original events
                        new_events.extend(old_events);

                        // Create new occasion and sort events
                        let mut new_occasion = Occasion::new(occasion.index);
                        new_occasion.events = new_events;
                        new_occasion.covariates = occasion.covariates.clone();

                        new_occasion.sort();
                        new_occasion
                    })
                    .collect();

                Subject::new(subject.id.clone(), new_occasions)
            })
            .collect();

        Data::new(new_subjects)
    }

    /// Get an iterator over all subjects
    ///
    /// # Returns
    ///
    /// An iterator yielding references to subjects
    pub fn iter(&'_ self) -> std::slice::Iter<'_, Subject> {
        self.subjects.iter()
    }

    /// Get a mutable iterator over all subjects
    ///
    /// # Returns
    ///
    /// A mutable iterator yielding references to subjects
    pub fn iter_mut(&'_ mut self) -> std::slice::IterMut<'_, Subject> {
        self.subjects.iter_mut()
    }

    /// Get the number of subjects in the dataset
    ///
    /// # Returns
    ///
    /// The number of subjects
    pub fn len(&self) -> usize {
        self.subjects.len()
    }

    /// Check if the dataset is empty
    ///
    /// # Returns
    ///
    /// `true` if there are no subjects, `false` otherwise
    pub fn is_empty(&self) -> bool {
        self.subjects.is_empty()
    }

    /// Get a vector of all unique output equations (outeq) across all subjects
    pub fn get_output_equations(&self) -> Vec<usize> {
        // Collect all unique outeq values in order of occurrence
        let mut outeq_values: Vec<usize> = self
            .subjects
            .iter()
            .flat_map(|subject| subject.get_output_equations())
            .collect();
        outeq_values.sort_unstable();
        outeq_values.dedup();
        outeq_values
    }
}

impl IntoIterator for Data {
    type Item = Subject;
    type IntoIter = std::vec::IntoIter<Subject>;
    /// Consumes the data and yields owned subjects
    fn into_iter(self) -> Self::IntoIter {
        self.subjects.into_iter()
    }
}

impl<'a> IntoIterator for &'a Data {
    type Item = &'a Subject;
    type IntoIter = std::slice::Iter<'a, Subject>;
    /// Iterate immutably over all subjects in the dataset
    fn into_iter(self) -> Self::IntoIter {
        self.subjects.iter()
    }
}
impl<'a> IntoIterator for &'a mut Data {
    type Item = &'a mut Subject;
    type IntoIter = std::slice::IterMut<'a, Subject>;
    /// Iterate mutably over all subjects in the dataset
    fn into_iter(self) -> Self::IntoIter {
        self.subjects.iter_mut()
    }
}

impl Into<Data> for Vec<Subject> {
    /// Convert a vector of subjects into a Data object
    fn into(self) -> Data {
        Data::new(self)
    }
}

impl Into<Data> for Subject {
    /// Convert a subject into a Data object
    fn into(self) -> Data {
        Data::new(vec![self])
    }
}

/// A subject in a pharmacometric dataset
///
/// A [Subject] represents a single individual with one or more occasions of data,
/// each containing events (doses, observations) and covariates.
#[derive(Serialize, Debug, Deserialize, Clone)]
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
    pub fn occasions_iter_mut(&'_ mut self) -> std::slice::IterMut<'_, Occasion> {
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

    /// Get a mutable reference to an occasion by index
    ///
    /// # Arguments
    ///
    /// * `index` - The index of the occasion to retrieve
    ///
    /// # Returns
    ///
    /// An `Option` containing a mutable reference to the occasion if found, or `None` if not found
    pub fn get_occasion_mut(&mut self, index: usize) -> Option<&mut Occasion> {
        self.occasions.iter_mut().find(|occ| occ.index() == index)
    }

    /// Get an iterator over all occasions
    ///
    /// # Returns
    ///
    /// An iterator yielding references to occasions
    pub fn iter(&'_ self) -> std::slice::Iter<'_, Occasion> {
        self.occasions.iter()
    }

    /// Get a mutable iterator over all occasions
    ///
    /// # Returns
    ///
    /// A mutable iterator yielding references to occasions
    pub fn iter_mut(&'_ mut self) -> std::slice::IterMut<'_, Occasion> {
        self.occasions.iter_mut()
    }

    /// Get the number of occasions for this subject
    ///
    /// # Returns
    ///
    /// The number of occasions
    pub fn len(&self) -> usize {
        self.occasions.len()
    }

    /// Check if the subject has any occasions
    ///
    /// # Returns
    ///
    /// `true` if there are no occasions, `false` otherwise
    pub fn is_empty(&self) -> bool {
        self.occasions.is_empty()
    }
}

impl IntoIterator for Subject {
    type Item = Occasion;
    type IntoIter = std::vec::IntoIter<Occasion>;
    /// Consumes the subject and yields owned occasions
    fn into_iter(self) -> Self::IntoIter {
        self.occasions.into_iter()
    }
}
impl<'a> IntoIterator for &'a Subject {
    type Item = &'a Occasion;
    type IntoIter = std::slice::Iter<'a, Occasion>;
    /// Iterate immutably over all occasions in the subject
    fn into_iter(self) -> Self::IntoIter {
        self.occasions.iter()
    }
}
impl<'a> IntoIterator for &'a mut Subject {
    type Item = &'a mut Occasion;
    type IntoIter = std::slice::IterMut<'a, Occasion>;
    /// Iterate mutably over all occasions in the subject
    fn into_iter(self) -> Self::IntoIter {
        self.occasions.iter_mut()
    }
}

/// An occasion within a subject's dataset
///
/// An [Occasion] represents a distinct period of data collection for a subject,
/// such as a hospital visit or dosing regimen. It contains events (doses, observations)
/// and time-varying covariates.
#[derive(Serialize, Debug, Deserialize, Clone)]
pub struct Occasion {
    pub(crate) events: Vec<Event>,
    pub(crate) covariates: Covariates,
    pub(crate) index: usize,
}

impl Occasion {
    /// Create a new occasion
    ///
    /// # Arguments
    ///
    /// * `events` - Vector of events for this occasion
    /// * `covariates` - Covariates for this occasion
    /// * `index` - The occasion index (0-based)
    pub(crate) fn new(index: usize) -> Self {
        Occasion {
            events: Vec::new(),
            covariates: Covariates::new(),
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
    pub fn add_covariate(&mut self, name: String, covariate: Covariate) {
        self.covariates.add_covariate(name, covariate);
    }

    /// Set covariates for this occasion
    pub(crate) fn set_covariates(&mut self, covariates: Covariates) {
        self.covariates = covariates;
    }

    fn add_lagtime(&mut self, reorder: Option<(&Fa, &Lag, &Vec<f64>, &Covariates)>) {
        if let Some((_, fn_lag, spp, covariates)) = reorder {
            let spp = nalgebra::DVector::from_vec(spp.to_vec());
            for event in self.events.iter_mut() {
                let time = event.time();
                if let Event::Bolus(bolus) = event {
                    let lagtime = fn_lag(&spp.clone().into(), time, covariates);
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
                    let fa = fn_fa(&spp.clone().into(), time, covariates);
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
            // Helper function to get event type order
            #[inline]
            fn event_type_order(event: &Event) -> u8 {
                match event {
                    Event::Bolus(_) => 1,
                    Event::Infusion(_) => 2,
                    Event::Observation(_) => 3,
                }
            }

            // Compare times first using the existing time() method
            let time_cmp = a.time().partial_cmp(&b.time());

            match time_cmp {
                Some(std::cmp::Ordering::Equal) => {
                    // If times are equal, sort by event type
                    event_type_order(a).cmp(&event_type_order(b))
                }
                Some(ordering) => ordering,
                None => std::cmp::Ordering::Equal, // Handle NaN cases
            }
        });
    }

    /// Process the events with modifications for lag time, bioavailability and input remapping.
    ///
    /// # Arguments
    ///
    /// * `reorder` - Optional tuple containing references to (Fa, Lag, support point, covariates) for adjustments
    /// * `ignore` - If true, filter out events marked as ignore
    /// * `mappings` - Optional reference to an [equation::Mapper] for input remapping
    ///
    /// # Returns
    ///
    /// Vector of events, potentially filtered and with times adjusted for lag and bioavailability
    pub(crate) fn process_events(
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

    /// Get a reference to the  covariates for this occasion
    ///
    /// # Returns
    ///
    /// Reference to the occasion's covariates, if any
    pub fn covariates(&self) -> &Covariates {
        &self.covariates
    }

    /// Get a mutable refernce to the covariates for this occasion
    ///
    /// # Returns
    ///
    /// Reference to the occasion's covariates, if any
    pub fn covariates_mut(&mut self) -> &mut Covariates {
        &mut self.covariates
    }

    /// Add an event to the [Occasion]
    ///
    /// Note that this will sort the events automatically, ensuring events are sorted by time, then by [Event] type so that [Bolus] and [Infusion] come before [Observation]
    pub(crate) fn add_event(&mut self, event: Event) {
        self.events.push(event);
        self.sort();
    }

    /// Add an [Observation] event to the [Occasion]
    pub fn add_observation(
        &mut self,
        time: f64,
        value: f64,
        outeq: usize,
        errorpoly: Option<ErrorPoly>,
        censored: Censor,
    ) {
        let observation =
            Observation::new(time, Some(value), outeq, errorpoly, self.index, censored);
        self.add_event(Event::Observation(observation));
    }

    /// Add a missing [Observation] event to the [Occasion]
    pub fn add_missing_observation(&mut self, time: f64, outeq: usize) {
        let observation = Observation::new(time, None, outeq, None, self.index, Censor::None);
        self.add_event(Event::Observation(observation));
    }

    /// Add a missing [Observation] with a custom [ErrorPoly] to the [Occasion]
    ///
    /// This is useful if you want a different weight for the observation
    pub fn add_observation_with_error(
        &mut self,
        time: f64,
        value: f64,
        outeq: usize,
        errorpoly: ErrorPoly,
        censored: Censor,
    ) {
        let observation = Observation::new(
            time,
            Some(value),
            outeq,
            Some(errorpoly),
            self.index,
            censored,
        );
        self.add_event(Event::Observation(observation));
    }

    /// Add a [Bolus] event to the [Occasion]
    pub fn add_bolus(&mut self, time: f64, amount: f64, input: usize) {
        let bolus = Bolus::new(time, amount, input, self.index);
        self.add_event(Event::Bolus(bolus));
    }

    /// Add an [Infusion] event to the [Occasion]
    pub fn add_infusion(&mut self, time: f64, amount: f64, input: usize, duration: f64) {
        let infusion = Infusion::new(time, amount, input, duration, self.index);
        self.add_event(Event::Infusion(infusion));
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
    pub fn events_iter_mut(&'_ mut self) -> std::slice::IterMut<'_, Event> {
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

    /// Get an iterator over all events
    ///
    /// # Returns
    ///
    /// An iterator yielding references to events
    pub fn iter(&'_ self) -> std::slice::Iter<'_, Event> {
        self.events.iter()
    }

    /// Get a mutable iterator over all events
    ///
    /// # Returns
    ///
    /// A mutable iterator yielding references to events
    pub fn iter_mut(&'_ mut self) -> std::slice::IterMut<'_, Event> {
        self.events.iter_mut()
    }

    /// Get the number of events in this occasion
    ///
    /// # Returns
    ///
    /// The number of events
    pub fn len(&self) -> usize {
        self.events.len()
    }

    /// Check if the occasion has any events
    ///
    /// # Returns
    ///
    /// `true` if there are no events, `false` otherwise
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }
}

impl IntoIterator for Occasion {
    type Item = Event;
    type IntoIter = std::vec::IntoIter<Event>;
    /// Consumes the occasion and yields owned events
    fn into_iter(self) -> Self::IntoIter {
        self.events.into_iter()
    }
}
impl<'a> IntoIterator for &'a Occasion {
    type Item = &'a Event;
    type IntoIter = std::slice::Iter<'a, Event>;
    /// Iterate immutably over all events in the occasion
    fn into_iter(self) -> Self::IntoIter {
        self.events.iter()
    }
}
impl<'a> IntoIterator for &'a mut Occasion {
    type Item = &'a mut Event;
    type IntoIter = std::slice::IterMut<'a, Event>;
    /// Iterate mutably over all events in the occasion
    fn into_iter(self) -> Self::IntoIter {
        self.events.iter_mut()
    }
}

// For Event, IntoIterator yields a single reference to self (for & and &mut)
impl<'a> IntoIterator for &'a Event {
    type Item = &'a Event;
    type IntoIter = std::option::IntoIter<&'a Event>;
    /// Yields a single reference to the event
    fn into_iter(self) -> Self::IntoIter {
        Some(self).into_iter()
    }
}
impl<'a> IntoIterator for &'a mut Event {
    type Item = &'a mut Event;
    type IntoIter = std::option::IntoIter<&'a mut Event>;
    /// Yields a single mutable reference to the event
    fn into_iter(self) -> Self::IntoIter {
        Some(self).into_iter()
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
            .observation(1.0, 10.0, 1)
            .bolus(2.0, 50.0, 1)
            .infusion(3.0, 100.0, 1, 1.0)
            .covariate("age", 0.0, 30.0)
            .covariate("weight", 0.0, 70.0)
            .reset()
            .observation(4.0, 20.0, 2)
            .bolus(5.0, 60.0, 2)
            .infusion(6.0, 120.0, 2, 2.0)
            .covariate("age", 0.0, 31.0)
            .covariate("weight", 0.0, 75.0)
            .build();

        let subject2 = Subject::builder("subject2")
            .observation(1.5, 15.0, 1)
            .bolus(2.5, 55.0, 1)
            .infusion(3.5, 110.0, 1, 1.5)
            .covariate("age", 0.0, 25.0)
            .covariate("weight", 0.0, 65.0)
            .reset()
            .observation(4.5, 25.0, 2)
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
        let subjects = data.subjects();
        assert_eq!(subjects.len(), 2);
        assert_eq!(subjects[0].id(), "subject1");
        assert_eq!(subjects[1].id(), "subject2");
    }

    #[test]
    fn test_add_subject() {
        let mut data = create_sample_data();
        let new_subject = Subject::builder("subject3")
            .observation(1.0, 10.0, 1)
            .bolus(2.0, 50.0, 1)
            .infusion(3.0, 100.0, 1, 1.0)
            .covariate("age", 0.0, 30.0)
            .covariate("weight", 0.0, 70.0)
            .build();
        data.add_subject(new_subject);
        assert_eq!(data.len(), 3);
        assert_eq!(data.subjects()[2].id(), "subject3");
    }

    #[test]
    fn test_filter_include() {
        let data = create_sample_data();
        let include = vec!["subject1".to_string()];
        let filtered_data = data.filter_include(&include);
        assert_eq!(filtered_data.subjects().len(), 1);
        assert_eq!(filtered_data.subjects()[0].id(), "subject1");
    }

    #[test]
    fn test_filter_exclude() {
        let data = create_sample_data();
        let filtered_data = data.filter_exclude(vec!["subject1".to_string()]);
        assert_eq!(filtered_data.len(), 1);
        assert_eq!(filtered_data.subjects()[0].id(), "subject2");
    }

    #[test]
    fn test_occasion_sort() {
        let mut occasion = Occasion::new(0);
        occasion.add_observation(2.0, 1.0, 1, None, Censor::None);
        occasion.add_bolus(1.0, 100.0, 1);
        occasion.sort();
        let events = occasion.process_events(None, false);
        match &events[0] {
            Event::Bolus(b) => assert_eq!(b.time(), 1.0),
            _ => panic!("First event should be a Bolus"),
        }
        match &events[1] {
            Event::Observation(o) => assert_eq!(o.time(), 2.0),
            _ => panic!("Second event should be an Observation"),
        }
    }

    #[test]
    fn test_data_iterators() {
        let data = create_sample_data();
        let mut count = 0;
        for subject in data.iter() {
            assert!(subject.id().starts_with("subject"));
            count += 1;
        }
        assert_eq!(count, 2);

        let mut data = create_sample_data();
        for subject in data.iter_mut() {
            subject.occasions_mut().push(Occasion::new(2));
        }
        assert_eq!(data.subjects()[0].occasions().len(), 3);
    }

    #[test]
    fn test_subject_iterators() {
        let subject = Subject::builder("test")
            .observation(1.0, 10.0, 1)
            .bolus(2.0, 50.0, 1)
            .reset()
            .observation(3.0, 20.0, 1)
            .build();

        let mut count = 0;
        for occasion in subject.iter() {
            assert!(occasion.index() < 2);
            count += 1;
        }
        assert_eq!(count, 2);

        let mut subject = subject;
        for occasion in subject.iter_mut() {
            occasion.add_observation(12.0, 100.0, 0, None, Censor::None);
        }
        assert_eq!(subject.occasions()[0].events().len(), 3);
    }

    #[test]
    fn test_occasion_iterators() {
        let mut occasion = Occasion::new(0);
        occasion.add_observation(1.0, 10.0, 1, None, Censor::None);
        occasion.add_bolus(2.0, 50.0, 1);
        occasion.sort();

        let mut count = 0;
        for event in occasion.iter() {
            match event {
                Event::Observation(_) => count += 1,
                Event::Bolus(_) => count += 2,
                _ => panic!("Unexpected event type"),
            }
        }
        assert_eq!(count, 3);

        let mut occasion = occasion;
        for event in occasion.iter_mut() {
            event.inc_time(1.0);
        }
        assert_eq!(occasion.events()[0].time(), 2.0);
    }

    #[test]
    fn test_data_intoiterator_refs() {
        let data = create_sample_data();
        let mut count = 0;
        for subject in &data {
            assert!(subject.id().starts_with("subject"));
            count += 1;
        }
        assert_eq!(count, 2);
        let mut data = create_sample_data();
        for subject in &mut data {
            subject.occasions_mut().push(Occasion::new(2));
        }
        assert_eq!(data.subjects()[0].occasions().len(), 3);
    }
    #[test]
    fn test_subject_intoiterator_all_forms() {
        let data = create_sample_data();
        let subject = data.get_subject("subject1").unwrap().clone();

        // Test owned iterator - consumes the subject
        let mut occasion_count = 0;
        let mut total_events = 0;
        for occasion in subject.clone() {
            assert_eq!(occasion.index(), occasion_count);
            total_events += occasion.events().len();
            occasion_count += 1;
        }
        assert_eq!(occasion_count, 2); // subject1 has 2 occasions
        assert_eq!(total_events, 6); // 3 events per occasion (obs + bolus + infusion)

        // Test immutable reference iterator
        let mut covariate_ages = Vec::new();
        for occasion in &subject {
            if let Some(age_cov) = occasion.covariates().get_covariate("age") {
                if let Ok(age_value) = age_cov.interpolate(0.1) {
                    covariate_ages.push(age_value);
                }
            }
        }
        assert_eq!(covariate_ages, vec![30.0, 31.0]); // Ages from sample data

        // Test mutable reference iterator - add missing observations
        let mut subject_mut = subject;
        for occasion in &mut subject_mut {
            occasion.add_missing_observation(10.0, 1); // Add observation at t=10 for outeq 1
        }

        // Verify we added observations to both occasions
        assert_eq!(subject_mut.occasions()[0].events().len(), 4);
        assert_eq!(subject_mut.occasions()[1].events().len(), 4);
    }
    #[test]
    fn test_occasion_intoiterator_all_forms() {
        let data = create_sample_data();
        let subject = data.get_subject("subject2").unwrap();
        let occasion = subject.occasions()[0].clone(); // Clone to get owned occasion

        // Test owned iterator - verify event types and ordering
        let mut event_types = Vec::new();
        let mut event_times = Vec::new();
        for event in occasion.clone() {
            event_times.push(event.time());
            match event {
                Event::Observation(_) => event_types.push("obs"),
                Event::Bolus(_) => event_types.push("bolus"),
                Event::Infusion(_) => event_types.push("infusion"),
            }
        }
        // Should be sorted by time, then by event type priority
        assert_eq!(event_times, vec![1.5, 2.5, 3.5]);
        assert_eq!(event_types, vec!["obs", "bolus", "infusion"]);

        // Test immutable reference iterator - calculate total dose
        let mut total_dose = 0.0;
        for event in &occasion {
            match event {
                Event::Bolus(bolus) => total_dose += bolus.amount(),
                Event::Infusion(infusion) => total_dose += infusion.amount(),
                _ => {}
            }
        }
        assert_eq!(total_dose, 165.0); // 55.0 (bolus) + 110.0 (infusion)

        // Test mutable reference iterator - shift all event times by 1 hour
        let mut occasion_mut = occasion;
        for event in &mut occasion_mut {
            event.inc_time(1.0);
        }

        // Verify all times were shifted
        let shifted_times: Vec<f64> = occasion_mut.events().iter().map(|e| e.time()).collect();
        assert_eq!(shifted_times, vec![2.5, 3.5, 4.5]);
    }
    #[test]
    fn test_event_intoiterator_refs() {
        let data = create_sample_data();
        let subject = data.get_subject("subject1").unwrap();
        let occasion = &subject.occasions()[0];

        // Get a bolus event from sample data and test immutable reference iterator
        if let Some(bolus_event) = occasion
            .events()
            .iter()
            .find(|e| matches!(e, Event::Bolus(_)))
        {
            let mut event_count = 0;
            for event in *bolus_event {
                assert_eq!(event.time(), 2.0); // Bolus time from sample data
                assert!(matches!(event, Event::Bolus(_)));
                if let Event::Bolus(bolus) = event {
                    assert_eq!(bolus.amount(), 50.0); // Amount from sample data
                    assert_eq!(bolus.input(), 1); // Input compartment 1
                }
                event_count += 1;
            }
            assert_eq!(event_count, 1);
        }

        // Test mutable reference iterator on a new event
        let mut infusion_event = Event::Infusion(Infusion::new(5.0, 200.0, 1, 2.0, 2));
        let original_time = infusion_event.time();

        for event in &mut infusion_event {
            event.inc_time(3.0); // Increase time by 3 hours
        }

        assert_eq!(infusion_event.time(), original_time + 3.0);

        // Test with observation event from sample data
        if let Some(obs_event) = occasion
            .events()
            .iter()
            .find(|e| matches!(e, Event::Observation(_)))
        {
            for event in *obs_event {
                assert_eq!(event.time(), 1.0); // Observation time from sample data
                if let Event::Observation(observation) = event {
                    assert_eq!(observation.value(), Some(10.0)); // Value from sample data
                    assert_eq!(observation.outeq(), 1); // Output equation 1
                }
            }
        }
    }

    #[test]
    fn test_subject_builder_and_data_modification() {
        // Create a subject with one bolus and three observations using the builder
        let mut subject = Subject::builder("test_subject")
            .bolus(0.0, 100.0, 1)
            .missing_observation(0.0, 1)
            .missing_observation(1.0, 1)
            .missing_observation(3.0, 1)
            .build();

        // Verify initial setup
        assert_eq!(subject.id(), "test_subject");
        assert_eq!(subject.occasions().len(), 1);

        let occasions = subject.occasions();
        let occasion = &occasions.first().unwrap();
        assert_eq!(occasion.events().len(), 4); // 1 bolus + 3 observations

        // Find the bolus and verify initial dose
        if let Some(Event::Bolus(bolus)) = occasion
            .events()
            .iter()
            .find(|e| matches!(e, Event::Bolus(_)))
        {
            assert_eq!(bolus.amount(), 100.0);
            assert_eq!(bolus.time(), 0.0);
            assert_eq!(bolus.input(), 1);
        } else {
            panic!("Bolus event not found");
        }

        // Count observations with None values
        let none_obs_count = occasion
            .events()
            .iter()
            .filter(|e| matches!(e, Event::Observation(obs) if obs.value().is_none()))
            .count();
        assert_eq!(none_obs_count, 3);

        // Edit the data: double the dose
        let occasion_mut = subject.get_occasion_mut(0).unwrap();
        for event in occasion_mut.events_iter_mut() {
            if let Event::Bolus(bolus) = event {
                let dose = bolus.mut_amount();
                *dose *= 2.0; // Double the dose
            }
        }

        // Add an observation at time 12 with value None
        occasion_mut.add_missing_observation(12.0, 1);

        // Verify the modifications
        let occasion = &subject.occasions()[0];
        assert_eq!(occasion.events().len(), 5); // 1 bolus + 4 observations

        // Verify doubled dose
        if let Some(Event::Bolus(bolus)) = occasion
            .events()
            .iter()
            .find(|e| matches!(e, Event::Bolus(_)))
        {
            assert_eq!(bolus.amount(), 200.0); // Should be doubled
        } else {
            panic!("Bolus event not found after modification");
        }

        // Verify the new observation at time 12
        if let Some(Event::Observation(obs)) = occasion
            .events()
            .iter()
            .find(|e| matches!(e, Event::Observation(obs) if obs.time() == 12.0))
        {
            assert_eq!(obs.time(), 12.0);
            assert_eq!(obs.value(), None);
            assert_eq!(obs.outeq(), 1);
        } else {
            panic!("Observation at time 12 not found");
        }

        // Verify all observations still have None values
        let none_obs_count = occasion
            .events()
            .iter()
            .filter(|e| matches!(e, Event::Observation(obs) if obs.value().is_none()))
            .count();
        assert_eq!(none_obs_count, 4); // Should now be 4 observations with None values
    }
}
