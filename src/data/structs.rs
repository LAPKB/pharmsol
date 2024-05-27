use crate::data::*;
use serde::Deserialize;
use std::{collections::HashMap, fmt};

/// [Data] is a collection of [Subject]s, which are collections of [Occasion]s, which are collections of [Event]s
///
/// This is the main data structure used to store the data, and is used to pass data to the model
/// [Data] implements the [DataTrait], which provides methods to access the data
#[derive(Debug, Clone)]
pub struct Data {
    subjects: Vec<Subject>,
}
impl Data {
    pub(crate) fn new(subjects: Vec<Subject>) -> Self {
        Data { subjects }
    }
    pub fn get_subjects(&self) -> Vec<&Subject> {
        self.subjects.iter().collect()
    }
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
