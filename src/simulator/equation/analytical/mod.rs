pub mod one_compartment_models;
pub mod three_compartment_models;
pub mod two_compartment_models;

pub use one_compartment_models::*;
pub use three_compartment_models::*;
pub use two_compartment_models::*;

use crate::simulator::model::Model;
use crate::{data::Covariates, simulator::*, Equation, Observation, Subject};
use cached::proc_macro::cached;
use cached::UnboundCache;

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
    state: V,
}

pub struct AnalyticalModel<'a> {
    equation: &'a Analytical,
    data: &'a Subject,
    state: V,
}
impl<'a> Equation<'a> for Analytical {
    type S = V;
    type P = SubjectPredictions;
    type Mod = AnalyticalModel<'a>;
    fn get_nstates(&self) -> usize {
        self.neqs.0
    }
    fn get_nouteqs(&self) -> usize {
        self.neqs.1
    }
    fn initialize_model(&'a self, subject: &'a Subject, spp: &[f64]) -> Self::Mod {
        AnalyticalModel::new(self, subject, spp)
    }
}

impl<'a> Model<'a> for AnalyticalModel<'a> {
    type Eq = Analytical;

    fn new(equation: &'a Self::Eq, data: &'a Subject, _spp: &[f64]) -> Self {
        let state = V::zeros(equation.get_nstates());
        Self {
            equation,
            data,
            state,
        }
    }

    fn equation(&self) -> &Self::Eq {
        self.equation
    }

    fn subject(&self) -> &Subject {
        self.data
    }

    fn state(&self) -> &<Self::Eq as Equation>::S {
        &self.state
    }
    #[inline(always)]
    fn get_lag(&self, spp: &[f64]) -> Option<HashMap<usize, f64>> {
        Some((self.equation.lag)(&V::from_vec(spp.to_owned())))
    }

    #[inline(always)]
    fn get_fa(&self, spp: &[f64]) -> Option<HashMap<usize, f64>> {
        Some((self.equation.fa)(&V::from_vec(spp.to_owned())))
    }

    fn add_bolus(&mut self, input: usize, amount: f64) {
        self.state[input] += amount;
    }

    #[inline(always)]
    fn solve(
        &mut self,
        support_point: &Vec<f64>,
        covariates: &Covariates,
        infusions: &Vec<Infusion>,
        ti: f64,
        tf: f64,
    ) {
        if ti == tf {
            return;
        }
        let mut support_point = V::from_vec(support_point.to_owned());
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
        support_point: &Vec<f64>,
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
            &V::from_vec(support_point.clone()),
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
    fn initial_state(&mut self, spp: &Vec<f64>, covariates: &Covariates, occasion_index: usize) {
        let init = &self.equation.init;
        let mut x = V::zeros(self.equation.get_nstates());
        if occasion_index == 0 {
            (init)(&V::from_vec(spp.to_vec()), 0.0, covariates, &mut x);
        }
        self.state = x;
    }
    fn estimate_likelihood(
        &mut self,
        support_point: &Vec<f64>,
        error_model: &ErrorModel,
        cache: bool,
    ) -> f64 {
        _estimate_likelihood(self, support_point, error_model, cache)
    }
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
            state: V::zeros(neqs.0),
        }
    }
}

fn spphash(spp: &[f64]) -> u64 {
    spp.iter().fold(0, |acc, x| acc + x.to_bits())
}
#[inline(always)]
#[cached(
    ty = "UnboundCache<String, SubjectPredictions>",
    create = "{ UnboundCache::with_capacity(100_000) }",
    convert = r#"{ format!("{}{}", model.data.id(), spphash(support_point)) }"#
)]
fn _subject_predictions(
    model: &mut AnalyticalModel,

    support_point: &Vec<f64>,
) -> SubjectPredictions {
    model.simulate_subject(support_point, None).0
}

fn _estimate_likelihood(
    model: &mut AnalyticalModel,
    support_point: &Vec<f64>,
    error_model: &ErrorModel,
    cache: bool,
) -> f64 {
    let ypred = if cache {
        _subject_predictions(model, support_point)
    } else {
        _subject_predictions_no_cache(model, support_point)
    };
    ypred.likelihood(error_model)
}
