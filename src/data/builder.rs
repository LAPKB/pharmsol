use std::collections::HashMap;

use crate::data::*;

/// Extension trait for creating [Subject] instances using the builder pattern
pub trait SubjectBuilderExt {
    /// Create a new SubjectBuilder with the specified ID
    ///
    /// # Arguments
    ///
    /// * `id` - The subject identifier
    ///
    /// # Example
    ///
    /// ```rust
    /// use pharmsol::*;
    ///
    /// let subject = Subject::builder("patient_001")
    ///     .bolus(0.0, 100.0, 0)
    ///     .observation(1.0, 10.5, 0)
    ///     .build();
    /// ```
    fn builder(id: impl Into<String>) -> SubjectBuilder;
}
impl SubjectBuilderExt for Subject {
    fn builder(id: impl Into<String>) -> SubjectBuilder {
        let occasion = Occasion::new(Vec::new(), Covariates::new(), 0);

        SubjectBuilder {
            id: id.into(),
            occasions: Vec::new(),
            current_occasion: occasion,
            current_covariates: Vec::new(),
            current_segment: HashMap::new(),
        }
    }
}

/// Builder for creating [Subject] instances with a fluent API
///
/// The [SubjectBuilder] allows for constructing complex subject data with a
/// chainable, readable syntax. Events like doses and observations can be
/// added sequentially, and the builder handles organizing them into occasions.
#[derive(Debug, Clone)]
pub struct SubjectBuilder {
    id: String,
    occasions: Vec<Occasion>,
    current_occasion: Occasion,
    current_covariates: Vec<(String, CovariateSegment)>,
    current_segment: HashMap<String, (f64, f64)>,
}

impl SubjectBuilder {
    /// Add an event to the current occasion
    ///
    /// # Arguments
    ///
    /// * `event` - The event to add
    pub fn event(mut self, event: Event) -> Self {
        self.current_occasion.add_event(event);
        self
    }

    /// Add a bolus dosing event
    ///
    /// # Arguments
    ///
    /// * `time` - Time of the bolus dose
    /// * `amount` - Amount of drug administered
    /// * `input` - The compartment number (zero-indexed) receiving the dose
    pub fn bolus(self, time: f64, amount: f64, input: usize) -> Self {
        let bolus = Bolus::new(time, amount, input);
        let event = Event::Bolus(bolus);
        self.event(event)
    }

    /// Add an infusion event
    ///
    /// # Arguments
    ///
    /// * `time` - Start time of the infusion
    /// * `amount` - Total amount of drug to be administered
    /// * `input` - The compartment number (zero-indexed) receiving the dose
    /// * `duration` - Duration of the infusion in time units
    pub fn infusion(self, time: f64, amount: f64, input: usize, duration: f64) -> Self {
        let infusion = Infusion::new(time, amount, input, duration);
        let event = Event::Infusion(infusion);
        self.event(event)
    }

    /// Add an observation event
    ///
    /// # Arguments
    ///
    /// * `time` - Time of the observation
    /// * `value` - Observed value (e.g., drug concentration)
    /// * `outeq` - Output equation number (zero-indexed) corresponding to this observation
    pub fn observation(self, time: f64, value: f64, outeq: usize) -> Self {
        let observation = Observation::new(time, value, outeq, None, false);
        let event = Event::Observation(observation);
        self.event(event)
    }

    /// Add an observation with error model parameters and ignore flag
    ///
    /// # Arguments
    ///
    /// * `time` - Time of the observation
    /// * `value` - Observed value (e.g., drug concentration)
    /// * `outeq` - Output equation number (zero-indexed) corresponding to this observation
    /// * `errorpoly` - Error polynomial coefficients (c0, c1, c2, c3)
    /// * `ignore` - Whether to ignore this observation in calculations
    pub fn observation_with_error(
        self,
        time: f64,
        value: f64,
        outeq: usize,
        errorpoly: Option<ErrorPoly>,
        ignore: bool,
    ) -> Self {
        let observation = Observation::new(time, value, outeq, errorpoly, ignore);
        let event = Event::Observation(observation);
        self.event(event)
    }

    /// Repeat the last event `n` times, separated by some interval `delta`
    ///
    /// # Arguments
    ///
    /// * `n` - Number of repetitions
    /// * `delta` - Time increment between repetitions
    ///
    /// # Example
    ///
    /// ```rust
    /// use pharmsol::*;
    ///
    ///
    /// let subject = Subject::builder("patient_001")
    ///     .bolus(0.0, 100.0, 0)  // First dose at time 0
    ///     .repeat(3, 24.0)       // Repeat the dose at times 24, 48, and 72
    ///     .build();
    /// ```
    pub fn repeat(mut self, n: usize, delta: f64) -> Self {
        let last_event = match self.current_occasion.last_event() {
            Some(event) => event.clone(),
            None => panic!("There is no event to repeat"),
        };
        for i in 1..=n {
            self = match last_event.clone() {
                Event::Bolus(bolus) => self.bolus(
                    bolus.time() + delta * i as f64,
                    bolus.amount(),
                    bolus.input(),
                ),
                Event::Infusion(infusion) => self.infusion(
                    infusion.time() + delta * i as f64,
                    infusion.amount(),
                    infusion.input(),
                    infusion.duration(),
                ),
                Event::Observation(observation) => self.observation_with_error(
                    observation.time() + delta * i as f64,
                    observation.value(),
                    observation.outeq(),
                    observation.errorpoly(),
                    observation.ignore(),
                ),
            };
        }
        self
    }

    /// Complete the current occasion and start a new one
    ///
    /// This finalizes the current occasion, adds it to the subject,
    /// and creates a new occasion for subsequent events.
    /// This is useful if a patient has new observations at some other occasion.
    /// Note that all states are reset!
    pub fn reset(mut self) -> Self {
        let block_index = self.current_occasion.index() + 1;
        self.current_occasion.sort();
        self.add_covariates();
        self.occasions.push(self.current_occasion);
        let occasion = Occasion::new(Vec::new(), Covariates::new(), block_index);
        self.current_occasion = occasion;
        self
    }

    /// Add a covariate value at a specific time
    ///
    /// Multiple calls for the same covariate at different times will create
    /// linear interpolation between the time points.
    ///
    /// # Arguments
    ///
    /// * `name` - Name of the covariate
    /// * `time` - Time point for this covariate value
    /// * `value` - Value of the covariate at this time
    ///
    /// # Example
    ///
    /// ```rust
    /// use pharmsol::*;
    ///
    /// let subject = Subject::builder("patient_001")
    ///     .covariate("weight", 0.0, 70.0)   // Weight at baseline
    ///     .covariate("weight", 30.0, 68.5)  // Weight at day 30
    ///     .build();
    /// ```
    pub fn covariate(mut self, name: impl Into<String>, time: f64, value: f64) -> Self {
        let name = name.into();
        if let Some((p_time, p_value)) = self.current_segment.get(&name) {
            let slope = (value - p_value) / (time - p_time);
            let intercept = p_value - slope * p_time;
            let segment =
                CovariateSegment::new(*p_time, time, Interpolation::Linear { slope, intercept });
            self.current_covariates.push((name.clone(), segment));
            self.current_segment.remove(&name);
            self.current_segment.insert(name.clone(), (time, value));
        } else {
            self.current_segment.insert(name.clone(), (time, value));
        }
        self
    }

    fn add_covariates(&mut self) {
        for (name, (time, val)) in &self.current_segment {
            let segment = CovariateSegment::new(
                *time,
                f64::INFINITY,
                Interpolation::CarryForward { value: *val },
            );
            self.current_covariates.push((name.clone(), segment));
        }
        self.current_segment.clear();
        // collect all the current covariates with the same name together
        let mut covariates: Vec<(String, Vec<CovariateSegment>)> = Vec::new();
        for (name, segment) in self.current_covariates.clone() {
            if let Some((_, segments)) = covariates.iter_mut().find(|(n, _)| n == &name) {
                segments.push(segment);
            } else {
                covariates.push((name, vec![segment]));
            }
        }

        // create the covariate object and add it to the current occasion
        for (name, segments) in covariates {
            let covariate = Covariate::new(name.clone(), segments);
            self.current_occasion.add_covariate(name, covariate);
        }
        self.current_covariates.clear();
    }

    /// Finalize and build the Subject
    ///
    /// This completes the current occasion and returns a new Subject with all
    /// the accumulated data.
    pub fn build(mut self) -> Subject {
        self = self.reset();
        Subject::new(self.id, self.occasions)
    }
}

#[cfg(test)]
mod tests {
    use crate::prelude::*;

    #[test]
    fn test_subject_builder() {
        let subject = Subject::builder("s1")
            .observation(3.0, 100.0, 0)
            .repeat(2, 0.5)
            .bolus(1.0, 100.0, 0)
            .infusion(0.0, 100.0, 0, 1.0)
            .repeat(3, 0.5)
            .covariate("c1", 0.0, 5.0)
            .covariate("c1", 5.0, 10.0)
            .covariate("c2", 0.0, 10.0)
            .reset()
            .observation(10.0, 100.0, 0)
            .bolus(7.0, 100.0, 0)
            .repeat(4, 1.0)
            .covariate("c1", 0.0, 5.0)
            .covariate("c1", 5.0, 10.0)
            .covariate("c2", 0.0, 10.0)
            .build();
        println!("{}", subject);
        assert_eq!(subject.id(), "s1");
        assert_eq!(subject.occasions().len(), 2);
    }

    #[test]
    fn test_complex_subject_builder() {
        let subject = Subject::builder("patient_002")
            .bolus(0.0, 50.0, 0)
            .observation(1.0, 45.3, 0)
            .observation(2.0, 40.1, 0)
            .observation_with_error(
                3.0,
                36.5,
                0,
                Some(ErrorPoly::new(0.1, 0.05, 0.0, 0.0)),
                false,
            )
            .bolus(4.0, 50.0, 0)
            .repeat(1, 12.0) // Repeat bolus at 16.0
            .reset()
            .bolus(24.0, 50.0, 0)
            .observation(25.0, 48.2, 0)
            .observation(26.0, 43.7, 0)
            .build();

        assert_eq!(subject.id(), "patient_002");
        assert_eq!(subject.occasions().len(), 2);

        let first_occasion = &subject.occasions()[0];
        assert_eq!(first_occasion.events().len(), 6); // 1 bolus + 3 observations + 1 bolus + 1 repeat

        let second_occasion = &subject.occasions()[1];
        assert_eq!(second_occasion.events().len(), 3); // 1 bolus + 2 observations
    }

    #[test]
    fn test_infusion_and_repetition() {
        let subject = Subject::builder("patient_003")
            .infusion(0.0, 100.0, 0, 2.0)
            .repeat(3, 6.0) // Repeat infusion at 6.0, 12.0, and 18.0
            .observation(1.0, 80.0, 0)
            .observation(7.0, 85.0, 0)
            .observation(13.0, 82.0, 0)
            .observation(19.0, 79.0, 0)
            .build();

        assert_eq!(subject.id(), "patient_003");
        assert_eq!(subject.occasions().len(), 1);

        // Check the correct number of events
        let events = subject.occasions()[0].events();
        assert_eq!(events.len(), 8); // 4 infusions + 4 observations

        // Count infusions
        let infusion_count = events
            .iter()
            .filter(|e| matches!(e, Event::Infusion(_)))
            .count();
        assert_eq!(infusion_count, 4);

        // Count observations
        let observation_count = events
            .iter()
            .filter(|e| matches!(e, Event::Observation(_)))
            .count();
        assert_eq!(observation_count, 4);
    }
}
