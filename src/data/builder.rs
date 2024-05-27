use rand::{distributions::Alphanumeric, Rng};

use crate::data::*;

pub trait SubjectBuilderExt {
    fn builder() -> SubjectBuilder;
}
impl SubjectBuilderExt for Subject {
    fn builder() -> SubjectBuilder {
        let rndu8: Vec<u8> = rand::thread_rng()
            .sample_iter(&Alphanumeric)
            .take(5)
            .collect();
        let id = String::from_utf8(rndu8).unwrap();
        let occasion = Occasion::new(Vec::new(), Covariates::new(), 0);

        SubjectBuilder {
            id: id,
            occasions: Vec::new(),
            current_occasion: occasion,
        }
    }
}

pub struct SubjectBuilder {
    id: String,
    occasions: Vec<Occasion>,
    current_occasion: Occasion,
}

impl SubjectBuilder {
    pub fn id(mut self, id: String) -> Self {
        self.id = id;
        self
    }

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
        amount: f64,
        input: usize,
        error_poly: Option<(f64, f64, f64, f64)>,
        ignore: bool,
    ) -> Self {
        let observation = Observation::new(time, amount, input, error_poly, ignore);
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
                    observation.errorpoly().clone(),
                    observation.ignore(),
                ),
            };
        }
        self
    }

    pub fn reset(mut self) -> Self {
        let block_index = self.current_occasion.block_index() + 1;
        self.current_occasion.sort();
        self.occasions.push(self.current_occasion);
        let occasion = Occasion::new(Vec::new(), Covariates::new(), block_index);
        self.current_occasion = occasion;
        self
    }

    pub fn build(mut self) -> Subject {
        self = self.reset();
        Subject::new(self.id, self.occasions)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_subject_builder() {
        let subject = Subject::builder()
            .id("test".to_string())
            .observation(3.0, 100.0, 0, None, false)
            .repeat(2, 0.5)
            .bolus(1.0, 100.0, 0)
            .infusion(0.0, 100.0, 0, 1.0)
            .repeat(3, 0.3)
            .reset()
            .observation(10.0, 100.0, 0, None, false)
            .bolus(7.0, 100.0, 0)
            .repeat(3, 1.0)
            .build();
        println!("{}", subject);
        assert_eq!(subject.id(), "test");
        assert_eq!(subject.occasions().len(), 2);
    }
}
