pub mod one_compartment_models;
pub mod three_compartment_models;
pub mod two_compartment_models;

pub use one_compartment_models::*;
pub use three_compartment_models::*;
pub use two_compartment_models::*;

use crate::{
    data::Covariates,
    simulator::{model::Model, *},
    Equation, Observation, Occasion, Subject,
};
use cached::proc_macro::cached;
use cached::UnboundCache;

use super::State;

/// Model equation using analytical solutions.
///
/// This implementation uses closed-form analytical solutions for the model
/// equations rather than numerical integration.
#[derive(Clone, Debug)]
pub struct Analytical {
    eq: AnalyticalEq,
    seq_eq: SecEq,
    lag: Lag,
    fa: Fa,
    init: Init,
    out: Out,
    neqs: Neqs,
}

pub struct AnalyticalModel<'a> {
    equation: &'a Analytical,
    subject: &'a Subject,
    support_point: Vec<f64>,
    state: V,
}

impl Analytical {
    /// Create a new Analytical equation model.
    ///
    /// # Parameters
    /// - `eq`: The analytical equation function
    /// - `seq_eq`: The secondary equation function
    /// - `lag`: The lag time function
    /// - `fa`: The fraction absorbed function
    /// - `init`: The initial state function
    /// - `out`: The output equation function
    /// - `neqs`: The number of states and output equations
    pub fn new(
        eq: AnalyticalEq,
        seq_eq: SecEq,
        lag: Lag,
        fa: Fa,
        init: Init,
        out: Out,
        neqs: Neqs,
    ) -> Self {
        Self {
            eq,
            seq_eq,
            lag,
            fa,
            init,
            out,
            neqs,
        }
    }
}

impl State for V {
    fn add_bolus(&mut self, input: usize, amount: f64) {
        self[input] += amount;
    }
}

impl<'a> Equation<'a> for Analytical {
    type S = V;
    type P = SubjectPredictions;
    type Mod = AnalyticalModel<'a>;
    #[inline(always)]
    fn get_nstates(&self) -> usize {
        self.neqs.0
    }

    #[inline(always)]
    fn get_nouteqs(&self) -> usize {
        self.neqs.1
    }
    #[inline(always)]
    fn initialize_model(&'a self, subject: &'a Subject, spp: Vec<f64>) -> Self::Mod {
        AnalyticalModel::new(self, subject, spp)
    }
}
impl<'a> Model<'a> for AnalyticalModel<'a> {
    type Eq = Analytical;

    #[inline(always)]
    fn new(equation: &'a Self::Eq, subject: &'a Subject, spp: Vec<f64>) -> Self {
        Self {
            equation,
            subject,
            support_point: spp,
            state: V::zeros(equation.get_nstates()),
        }
    }
    #[inline(always)]
    fn add_bolus(&mut self, input: usize, amount: f64) {
        self.state.add_bolus(input, amount);
    }

    #[inline(always)]
    fn equation(&self) -> &Self::Eq {
        self.equation
    }

    #[inline(always)]
    fn subject(&self) -> &Subject {
        self.subject
    }

    #[inline(always)]
    fn get_lag(&self) -> Option<HashMap<usize, f64>> {
        Some((self.equation.lag)(&V::from_vec(
            self.support_point.clone(),
        )))
    }

    #[inline(always)]
    fn get_fa(&self) -> Option<HashMap<usize, f64>> {
        Some((self.equation.fa)(&V::from_vec(self.support_point.clone())))
    }
    #[inline(always)]
    fn solve(&mut self, covariates: &Covariates, infusions: Vec<&Infusion>, ti: f64, tf: f64) {
        if ti == tf {
            return;
        }
        let mut support_point = V::from_vec(self.support_point.to_owned());
        let mut rateiv = V::from_vec(vec![0.0, 0.0, 0.0]);
        //TODO: This should be pre-calculated
        for infusion in infusions {
            if tf >= infusion.time() && tf <= infusion.duration() + infusion.time() {
                rateiv[infusion.input()] += infusion.amount() / infusion.duration();
            }
        }
        (self.equation.seq_eq)(&mut support_point, tf, covariates);
        self.state = (self.equation.eq)(&self.state, &support_point, tf - ti, rateiv, covariates);
    }
    #[inline(always)]
    fn process_observation(
        &mut self,
        observation: &Observation,
        error_model: Option<&ErrorModel>,
        _time: f64,
        covariates: &Covariates,
        likelihood: &mut Vec<f64>,
        output: &mut <Self::Eq as Equation>::P,
    ) {
        let mut y = V::zeros(self.equation.get_nouteqs());
        let out = &self.equation.out;
        (out)(
            &self.state,
            &V::from_vec(self.support_point.clone()),
            observation.time(),
            covariates,
            &mut y,
        );
        let pred = y[observation.outeq()];
        let pred = observation.to_obs_pred(pred, self.state.as_slice().to_vec());
        if let Some(error_model) = error_model {
            likelihood.push(pred.likelihood(error_model));
        }
        output.add_prediction(pred);
    }
    #[inline(always)]
    fn initial_state(&mut self, occasion: &Occasion) {
        let init = &self.equation.init;
        let mut x = V::zeros(self.equation.get_nstates());
        let covariates = occasion.get_covariates().unwrap();
        if occasion.index() == 0 {
            (init)(
                &V::from_vec(self.support_point.to_vec()),
                0.0,
                covariates,
                &mut x,
            );
        }
        self.state = x;
    }
    fn estimate_likelihood(self, error_model: &ErrorModel, cache: bool) -> f64 {
        _estimate_likelihood(self, error_model, cache)
    }
}

fn spphash(spp: &[f64]) -> u64 {
    spp.iter().fold(0, |acc, x| acc + x.to_bits())
}
#[inline(always)]
#[cached(
    ty = "UnboundCache<String, SubjectPredictions>",
    create = "{ UnboundCache::with_capacity(100_000) }",
    convert = r#"{ format!("{}{}", model.subject.id(), spphash(&model.support_point)) }"#
)]
fn _subject_predictions(model: AnalyticalModel) -> SubjectPredictions {
    model.simulate_subject(None).0
}

fn _estimate_likelihood(model: AnalyticalModel, error_model: &ErrorModel, cache: bool) -> f64 {
    let ypred = if cache {
        _subject_predictions(model)
    } else {
        _subject_predictions_no_cache(model)
    };
    ypred.likelihood(error_model)
}
