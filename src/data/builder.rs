use crate::data::*;

pub trait SubjectBuilderExt {
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
        }
    }
}

pub struct SubjectBuilder {
    id: String,
    occasions: Vec<Occasion>,
    current_occasion: Occasion,
    current_covariates: Vec<(String, CovariateSegment)>,
}

impl SubjectBuilder {
    pub fn event(mut self, event: Event) -> Self {
        self.current_occasion.add_event(event);
        self
    }

    pub fn bolus(self, time: f64, amount: f64, input: usize) -> Self {
        let bolus = Bolus::new(time, amount, input);
        let event = Event::Bolus(bolus);
        self.event(event)
    }

    pub fn infusion(self, time: f64, amount: f64, input: usize, duration: f64) -> Self {
        let infusion = Infusion::new(time, amount, input, duration);
        let event = Event::Infusion(infusion);
        self.event(event)
    }

    pub fn observation(
        self,
        time: f64,
        value: f64,
        outeq: usize,
        errorpoly: Option<(f64, f64, f64, f64)>,
        ignore: bool,
    ) -> Self {
        let observation = Observation::new(time, value, outeq, errorpoly, ignore);
        let event = Event::Observation(observation);
        self.event(event)
    }

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
                Event::Observation(observation) => self.observation(
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

    pub fn reset(mut self) -> Self {
        let block_index = self.current_occasion.index() + 1;
        self.current_occasion.sort();
        self.add_covariates();
        self.occasions.push(self.current_occasion);
        let occasion = Occasion::new(Vec::new(), Covariates::new(), block_index);
        self.current_occasion = occasion;
        self
    }

    pub fn covariate(
        mut self,
        name: &str,
        from: f64,
        to: f64,
        method: InterpolationMethod,
    ) -> Self {
        let segment = CovariateSegment::new(from, to, method);
        self.current_covariates.push((name.to_owned(), segment));
        self
    }

    fn add_covariates(&mut self) {
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
            .observation(3.0, 100.0, 0, None, false)
            .repeat(2, 0.5)
            .bolus(1.0, 100.0, 0)
            .infusion(0.0, 100.0, 0, 1.0)
            .repeat(3, 0.5)
            .covariate(
                "c1",
                0.0,
                5.0,
                Linear {
                    slope: 1.0,
                    intercept: 0.0,
                },
            )
            .covariate("c2", 5.0, f64::INFINITY, CarryForward { value: 5.0 })
            .reset()
            .observation(10.0, 100.0, 0, None, false)
            .bolus(7.0, 100.0, 0)
            .repeat(4, 1.0)
            .covariate(
                "c3",
                0.0,
                5.0,
                Linear {
                    slope: 1.0,
                    intercept: 0.0,
                },
            )
            .covariate("c4", 5.0, f64::INFINITY, CarryForward { value: 5.0 })
            .build();
        println!("{}", subject);
        assert_eq!(subject.id(), "s1");
        assert_eq!(subject.occasions().len(), 2);
    }
}
