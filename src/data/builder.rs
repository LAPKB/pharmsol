//! Builder API for constructing [`Subject`] schedules in Rust.
//!
//! Use `Subject::builder(...)` when you want to describe a subject directly in
//! code with a schedule-oriented API. This is the preferred high-level
//! path for hand-written datasets.
//!
//! Builder methods accept public input and output labels. Prefer stable strings
//! such as `"depot"`, `"iv"`, and `"cp"`. Numeric values are accepted, but
//! they remain public labels rather than automatically becoming dense internal
//! indices.

use crate::{data::*, Censor};

/// Extension trait that enables `Subject::builder(...)`.
///
/// Most users do not need to import [`SubjectBuilder`] directly. Import this
/// trait from the crate root or [`crate::prelude`] and then start with
/// `Subject::builder("id")`.
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
    ///     .bolus(0.0, 100.0, "depot")
    ///     .observation(1.0, 10.5, "cp")
    ///     .build();
    /// ```
    fn builder(id: impl Into<String>) -> SubjectBuilder;
}
impl SubjectBuilderExt for Subject {
    fn builder(id: impl Into<String>) -> SubjectBuilder {
        let occasion = Occasion::new(0);

        SubjectBuilder {
            id: id.into(),
            occasions: Vec::new(),
            current_occasion: occasion,
            covariates: Covariates::new(),
            last_added_event: None,
        }
    }
}

/// Builder for creating [`Subject`] values with a fluent API.
///
/// Use [`SubjectBuilder`] when you want to author common dose and observation
/// schedules directly in Rust without constructing low-level event values by
/// hand.
///
/// A builder instance accumulates events inside the current [`Occasion`].
/// [`SubjectBuilder::repeat`] duplicates the most recently added event at later
/// times, and [`SubjectBuilder::reset`] closes the current occasion and starts a
/// new one with fresh occasion-local state.
///
/// Input and output arguments are public labels. Prefer stable model-facing
/// names such as `"depot"`, `"iv"`, and `"cp"`.
///
/// # Example
///
/// ```rust
/// use pharmsol::*;
///
/// let subject = Subject::builder("patient_001")
///     .bolus(0.0, 100.0, "depot")
///     .repeat(1, 24.0)
///     .observation(1.0, 12.3, "cp")
///     .missing_observation(25.0, "cp")
///     .reset()
///     .bolus(0.0, 80.0, "depot")
///     .observation(1.0, 10.1, "cp")
///     .build();
///
/// assert_eq!(subject.occasions().len(), 2);
/// ```
#[derive(Debug, Clone)]
pub struct SubjectBuilder {
    id: String,
    occasions: Vec<Occasion>,
    current_occasion: Occasion,
    covariates: Covariates,
    last_added_event: Option<Event>,
}

impl SubjectBuilder {
    /// Add a fully constructed event to the current occasion.
    ///
    /// Use this when you want to mix builder convenience methods with direct
    /// [`Event`] values.
    pub fn event(mut self, event: Event) -> Self {
        self.last_added_event = Some(event.clone());
        self.current_occasion.add_event(event);
        self
    }

    /// Add an instantaneous dose.
    ///
    /// # Arguments
    ///
    /// * `time` - Time of the bolus dose
    /// * `amount` - Amount of drug administered
    /// * `input` - Public input label receiving the dose
    ///
    /// Prefer stable route names such as `"depot"` or `"iv"` when the model
    /// declares named routes.
    pub fn bolus(self, time: f64, amount: f64, input: impl ToString) -> Self {
        let bolus = Bolus::new(time, amount, input, self.current_occasion.index());
        let event = Event::Bolus(bolus);
        self.event(event)
    }

    /// Add a continuous dose over a duration.
    ///
    /// # Arguments
    ///
    /// * `time` - Start time of the infusion
    /// * `amount` - Total amount of drug to be administered
    /// * `input` - Public input label receiving the dose
    /// * `duration` - Duration of the infusion in time units
    pub fn infusion(self, time: f64, amount: f64, input: impl ToString, duration: f64) -> Self {
        let infusion = Infusion::new(time, amount, input, duration, self.current_occasion.index());
        let event = Event::Infusion(infusion);
        self.event(event)
    }

    /// Add an observed value at a given time.
    ///
    /// # Arguments
    ///
    /// * `time` - Time of the observation
    /// * `value` - Observed value (e.g., drug concentration)
    /// * `outeq` - Public output label for this observation
    pub fn observation(self, time: f64, value: f64, outeq: impl ToString) -> Self {
        let observation = Observation::new(
            time,
            Some(value),
            outeq,
            None,
            self.current_occasion.index(),
            Censor::None,
        );
        let event = Event::Observation(observation);
        self.event(event)
    }

    /// Add an observed value with explicit censoring information.
    ///
    /// # Arguments
    ///
    /// * `time` - Time of the observation
    /// * `value` - Observed value (e.g., drug concentration)
    /// * `outeq` - Public output label for this observation
    /// * `censoring` - Censoring status for the observation value
    pub fn censored_observation(
        self,
        time: f64,
        value: f64,
        outeq: impl ToString,
        censoring: Censor,
    ) -> Self {
        let observation = Observation::new(
            time,
            Some(value),
            outeq,
            None,
            self.current_occasion.index(),
            censoring,
        );
        let event = Event::Observation(observation);
        self.event(event)
    }

    /// Add a prediction-only observation slot.
    ///
    /// # Arguments
    ///
    /// * `time` - Time of the observation
    /// * `outeq` - Public output label for this observation
    ///
    /// Use this when you want a prediction at a time point but do not have an
    /// observed value.
    pub fn missing_observation(self, time: f64, outeq: impl ToString) -> Self {
        let observation = Observation::new(
            time,
            None,
            outeq,
            None,
            self.current_occasion.index(),
            Censor::None,
        );
        let event = Event::Observation(observation);
        self.event(event)
    }

    /// Add an observed value with an explicit assay error polynomial.
    ///
    /// # Arguments
    ///
    /// * `time` - Time of the observation
    /// * `value` - Observed value (e.g., drug concentration)
    /// * `outeq` - Public output label for this observation
    /// * `errorpoly` - Error polynomial coefficients (c0, c1, c2, c3)
    /// * `censored` - Censoring status for the observation value
    pub fn observation_with_error(
        self,
        time: f64,
        value: f64,
        outeq: impl ToString,
        errorpoly: ErrorPoly,
        censored: Censor,
    ) -> Self {
        let observation = Observation::new(
            time,
            Some(value),
            outeq,
            Some(errorpoly),
            self.current_occasion.index(),
            censored,
        );
        let event = Event::Observation(observation);
        self.event(event)
    }

    /// Repeat the last event `n` times, separated by `delta`.
    ///
    /// The repeated events keep the same label, value, censoring state, and
    /// error polynomial as the original event. Only the event time changes.
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
    /// let subject = Subject::builder("patient_001")
    ///     .bolus(0.0, 100.0, "depot") // First dose at time 0
    ///     .repeat(3, 24.0)       // Repeat the dose at times 24, 48, and 72
    ///     .build();
    /// ```
    pub fn repeat(mut self, n: usize, delta: f64) -> Self {
        let last_event = match &self.last_added_event {
            Some(event) => event.clone(),
            None => {
                return self; // No event to repeat
            }
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
                Event::Observation(observation) => {
                    if observation.value().is_some() {
                        if observation.errorpoly().is_some() {
                            self.observation_with_error(
                                observation.time() + delta * i as f64,
                                observation.value().unwrap(),
                                observation.outeq(),
                                observation.errorpoly().unwrap(),
                                observation.censoring(),
                            )
                        } else if observation.censored() {
                            self.censored_observation(
                                observation.time() + delta * i as f64,
                                observation.value().unwrap(),
                                observation.outeq(),
                                observation.censoring(),
                            )
                        } else {
                            self.observation(
                                observation.time() + delta * i as f64,
                                observation.value().unwrap(),
                                observation.outeq(),
                            )
                        }
                    } else {
                        self.missing_observation(
                            observation.time() + delta * i as f64,
                            observation.outeq(),
                        )
                    }
                }
            };
        }
        self
    }

    /// Complete the current occasion and start a new one.
    ///
    /// This finalizes the current occasion, adds it to the subject,
    /// and creates a new occasion for subsequent events.
    /// Use this when the subject should begin a new occasion with reset state.
    ///
    /// Covariates collected since the previous reset are attached to the
    /// finished occasion. The new occasion starts empty and its state is reset.
    pub fn reset(mut self) -> Self {
        let block_index = self.current_occasion.index() + 1;
        self.current_occasion.sort();

        self.current_occasion.set_covariates(self.covariates);
        self.occasions.push(self.current_occasion);
        let occasion = Occasion::new(block_index);
        self.current_occasion = occasion;
        self.covariates = Covariates::new();
        self.last_added_event = None;
        self
    }

    /// Add a covariate value at a specific time.
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
    pub fn covariate(mut self, name: &str, time: f64, value: f64) -> Self {
        self.covariates.add_observation(name, time, value);
        self
    }

    /// Finalize and build the [`Subject`].
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
    use crate::{prelude::*, Censor};

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
            .observation(2.0, 0.1, 0)
            .observation_with_error(
                3.0,
                36.5,
                0,
                ErrorPoly::new(0.1, 0.05, 0.0, 0.0),
                Censor::None,
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

    #[test]
    fn test_repeat_with_multiple_outeqs() {
        // Test the fix for repeat() after observation() with multiple output equations
        // This reproduces the issue from v019.0 where repeat() was not correctly
        // repeating the last added observation when events were sorted
        let subject = Subject::builder("test_repeat")
            .bolus(0.0, 500.0, 0)
            .observation(0.0, 0.0, 0)
            .repeat(10, 0.1)
            .observation(0.0, 0.0, 1)
            .repeat(10, 0.1)
            .build();

        assert_eq!(subject.id(), "test_repeat");
        assert_eq!(subject.occasions().len(), 1);

        let occasion = &subject.occasions()[0];
        let events = occasion.events();

        // Should have 1 bolus + 11 observations for outeq=0 + 11 observations for outeq=1 = 23 events
        assert_eq!(events.len(), 23);

        // Count observations by outeq and collect times
        let mut outeq_0_count = 0;
        let mut outeq_1_count = 0;
        let mut times_outeq_0 = Vec::new();
        let mut times_outeq_1 = Vec::new();

        for event in events {
            if let Event::Observation(obs) = event {
                if obs.outeq() == 0 {
                    outeq_0_count += 1;
                    times_outeq_0.push(obs.time());
                } else if obs.outeq() == 1 {
                    outeq_1_count += 1;
                    times_outeq_1.push(obs.time());
                }
            }
        }

        // Should have 11 observations for each outeq
        assert_eq!(outeq_0_count, 11, "Expected 11 observations for outeq=0");
        assert_eq!(outeq_1_count, 11, "Expected 11 observations for outeq=1");

        // Verify that observations appear at the same times for both outeqs
        times_outeq_0.sort_by(|a, b| a.partial_cmp(b).unwrap());
        times_outeq_1.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // Both should have observations at times 0.0, 0.1, 0.2, ..., 1.0
        assert_eq!(times_outeq_0.len(), 11);
        assert_eq!(times_outeq_1.len(), 11);

        for (t0, t1) in times_outeq_0.iter().zip(times_outeq_1.iter()) {
            assert!(
                (t0 - t1).abs() < 1e-10,
                "Times should match for both outeqs"
            );
        }
    }
}
