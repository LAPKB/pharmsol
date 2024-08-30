use crate::data::*;
use csv::WriterBuilder;
use serde::Deserialize;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::{collections::HashMap, fmt};

/// [Data] is a collection of [Subject]s, which are collections of [Occasion]s, which are collections of [Event]s
///
/// This is the main data structure used to store the data, and is used to pass data to the model
/// [Data] implements the [DataTrait], which provides methods to access the data
#[derive(Debug, Clone, Default)]
pub struct Data {
    subjects: Vec<Subject>,
}
impl Data {
    /// Constructs a new [Data] object from a vector of [Subject]s
    ///
    /// It is recommended that the subjects are constructed using the [builder::SubjectBuilder] to ensure that the data is correctly formatted
    pub fn new(subjects: Vec<Subject>) -> Self {
        Data { subjects }
    }
    /// Get a vector of references to all [Subject]s in the [Data]
    pub fn get_subjects(&self) -> Vec<&Subject> {
        self.subjects.iter().collect()
    }
    /// Add a [Subject] to the [Data]
    pub fn add_subject(&mut self, subject: Subject) {
        self.subjects.push(subject);
    }
    pub fn write_pmetrics(&self, file: &std::fs::File) {
        let mut writer = WriterBuilder::new().has_headers(true).from_writer(file);

        writer
            .write_record(&[
                "ID", "EVID", "TIME", "DUR", "DOSE", "ADDL", "II", "INPUT", "OUT", "OUTEQ", "C0",
                "C1", "C2", "C3",
            ])
            .unwrap();
        for subject in self.get_subjects() {
            for occasion in subject.occasions() {
                for event in occasion.get_events(None, None, false) {
                    match event {
                        Event::Observation(obs) => {
                            // Write each field individually
                            writer
                                .write_record(&[
                                    &subject.id(),
                                    &"0".to_string(),
                                    &obs.time().to_string(),
                                    &".".to_string(),
                                    &".".to_string(),
                                    &".".to_string(),
                                    &".".to_string(),
                                    &".".to_string(),
                                    &obs.value().to_string(),
                                    &(obs.outeq() + 1).to_string(),
                                    &".".to_string(),
                                    &".".to_string(),
                                    &".".to_string(),
                                    &".".to_string(),
                                ])
                                .unwrap();
                        }
                        Event::Infusion(inf) => {
                            writer
                                .write_record(&[
                                    &subject.id(),
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
                                .unwrap();
                        }
                        Event::Bolus(bol) => {
                            writer
                                .write_record(&[
                                    &subject.id(),
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
                                .unwrap();
                        }
                    }
                }
            }
        }
    }
    /// Filter the [Data] to include only the [Subject]s with IDs in the include vector
    pub fn filter_include(&self, include: &Vec<String>) -> Data {
        let subjects = self
            .subjects
            .iter()
            .filter(|subject| include.iter().any(|id| id == subject.id()))
            .cloned()
            .collect();
        Data::new(subjects)
    }
    /// Filter the [Data] to exclude the [Subject]s with IDs in the exclude vector
    pub fn filter_exclude(&self, exclude: Vec<String>) -> Data {
        let subjects = self
            .subjects
            .iter()
            .filter(|subject| !exclude.iter().any(|id| id == subject.id()))
            .cloned()
            .collect();
        Data::new(subjects)
    }
    /// Expand the data by adding observations at intervals of idelta
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

        // Create a new data structure with added observations at intervals of idelta
        let mut new_subjects: Vec<Subject> = Vec::new();
        for subject in &self.subjects {
            let mut new_occasions: Vec<Occasion> = Vec::new();
            for occasion in &subject.occasions {
                let old_events = occasion.get_events(None, None, true);
                let mut new_events: Vec<Event> = Vec::new();
                let mut time = 0.0;
                while time < last_time {
                    let obs = Observation::new(time, -99.0, 0, None, false);

                    new_events.push(Event::Observation(obs));

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

/// [Subject] is a collection of blocks for one individual
#[derive(Debug, Deserialize, Clone)]
pub struct Subject {
    id: String,
    occasions: Vec<Occasion>,
}
impl Subject {
    pub(crate) fn new(id: String, occasions: Vec<Occasion>) -> Self {
        Subject { id, occasions }
    }
    pub fn occasions(&self) -> Vec<&Occasion> {
        self.occasions.iter().collect()
    }
    pub fn id(&self) -> &String {
        &self.id
    }

    // Hasher for subject
    pub fn hash(&self) -> u64 {
        let mut hasher = DefaultHasher::new();

        // Hash the subject ID
        self.id().hash(&mut hasher);

        // Hash each occasion
        for occasion in &self.occasions() {
            occasion.index().hash(&mut hasher);
            for event in &occasion.events {
                match event {
                    Event::Observation(observation) => {
                        observation.time().to_bits().hash(&mut hasher);
                        observation.value().to_bits().hash(&mut hasher);
                        observation.outeq().hash(&mut hasher);
                    }
                    Event::Infusion(infusion) => {
                        infusion.time().to_bits().hash(&mut hasher);
                        infusion.duration().to_bits().hash(&mut hasher);
                        infusion.amount().to_bits().hash(&mut hasher);
                    }
                    Event::Bolus(bolus) => {
                        bolus.time().to_bits().hash(&mut hasher);
                        bolus.amount().to_bits().hash(&mut hasher);
                    }
                }
            }
        }

        hasher.finish()
    }
}

/// An [Occasion] is a collection of events, for a given [Subject], that are from a specific occasion
#[derive(Debug, Deserialize, Clone)]
pub struct Occasion {
    events: Vec<Event>,
    covariates: Covariates,
    index: usize,
}
impl Occasion {
    // Constructor
    pub(crate) fn new(events: Vec<Event>, covariates: Covariates, index: usize) -> Self {
        Occasion {
            events,
            covariates,
            index,
        }
    }

    pub fn events(&self) -> Vec<&Event> {
        self.events.iter().collect()
    }

    /// Get the index of the occasion
    pub fn index(&self) -> usize {
        self.index
    }

    pub(crate) fn add_covariate(&mut self, name: String, covariate: Covariate) {
        self.covariates.add_covariate(name, covariate);
    }

    fn add_lagtime(&mut self, lagtime: Option<&HashMap<usize, f64>>) {
        if let Some(lag) = lagtime {
            for event in self.events.iter_mut() {
                if let Event::Bolus(bolus) = event {
                    if let Some(l) = lag.get(&bolus.input()) {
                        *bolus.mut_time() += l;
                    }
                }
            }
        }
        self.sort();
    }

    fn add_bioavailability(&mut self, bioavailability: Option<&HashMap<usize, f64>>) {
        // If lagtime is empty, return early
        if let Some(fmap) = bioavailability {
            for event in self.events.iter_mut() {
                if let Event::Bolus(bolus) = event {
                    if let Some(f) = fmap.get(&bolus.input()) {
                        // bolus.time *= f;
                        *bolus.mut_time() *= f;
                    }
                }
            }
        }
        self.sort();
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

    pub fn get_events(
        &self,
        lagtime: Option<&HashMap<usize, f64>>,
        bioavailability: Option<&HashMap<usize, f64>>,
        ignore: bool,
    ) -> Vec<Event> {
        let mut occ = self.clone();
        occ.add_bioavailability(bioavailability);
        occ.add_lagtime(lagtime);

        // Filter out events that are marked as ignore
        if ignore {
            occ.events
                .iter()
                .filter(|event| match event {
                    Event::Observation(observation) => !observation.ignore(),
                    _ => true,
                })
                .cloned()
                .collect()
        } else {
            occ.events.clone()
        }
    }

    pub fn get_covariates(&self) -> Option<&Covariates> {
        Some(&self.covariates)
    }

    /// This function is not to be used directly as it does not guarantee that the events are sorted
    pub(crate) fn add_event(&mut self, event: Event) {
        self.events.push(event);
    }

    pub(crate) fn last_event(&self) -> Option<&Event> {
        self.events.last()
    }

    // fn get_infusions_vec(&self) -> Vec<Infusion> {
    //     self.events
    //         .iter()
    //         .filter_map(|event| match event {
    //             Event::Infusion(infusion) => Some(infusion.clone()),
    //             _ => None,
    //         })
    //         .collect()
    // }
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
        let subjects = data.get_subjects();
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
    fn test_subject_hash() {
        let subject = Subject::builder("subject1")
            .observation(1.0, 10.0, 1)
            .bolus(2.0, 50.0, 1)
            .infusion(3.0, 100.0, 1, 1.0)
            .covariate("age", 0.0, 30.0)
            .covariate("weight", 0.0, 70.0)
            .build();
        let hash = subject.hash();
        // Just check that hash is computed without errors
        assert!(hash > 0);
    }

    #[test]
    fn test_occasion_sort() {
        let mut occasion = Occasion::new(
            vec![
                Event::Observation(Observation::new(2.0, 1.0, 1, None, false)),
                Event::Bolus(Bolus::new(1.0, 100.0, 1)),
            ],
            Covariates::new(),
            1,
        );
        occasion.sort();
        let events = occasion.get_events(None, None, false);
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
    fn test_event_get_events_with_ignore() {
        let occasion = Occasion::new(
            vec![
                Event::Observation(Observation::new(1.0, 1.0, 1, None, false)),
                Event::Observation(Observation::new(2.0, 2.0, 2, None, true)),
            ],
            Covariates::new(),
            1,
        );
        let events = occasion.get_events(None, None, true);
        assert_eq!(events.len(), 1);
        match &events[0] {
            Event::Observation(o) => assert_eq!(o.time(), 1.0),
            _ => panic!("Event should be an Observation"),
        }
    }
}
