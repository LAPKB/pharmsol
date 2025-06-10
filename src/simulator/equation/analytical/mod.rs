pub mod one_compartment_models;
pub mod three_compartment_models;
pub mod two_compartment_models;

pub use one_compartment_models::*;
pub use three_compartment_models::*;
pub use two_compartment_models::*;

use crate::PharmsolError;
use crate::{
    data::Covariates, simulator::*, Equation, EquationPriv, EquationTypes, Observation, Subject,
};
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

impl EquationTypes for Analytical {
    type S = V;
    type P = SubjectPredictions;
}

impl EquationPriv for Analytical {
    // #[inline(always)]
    // fn get_init(&self) -> &Init {
    //     &self.init
    // }

    // #[inline(always)]
    // fn get_out(&self) -> &Out {
    //     &self.out
    // }

    #[inline(always)]
    fn get_lag(&self, spp: &[f64]) -> Option<HashMap<usize, f64>> {
        Some((self.lag)(&V::from_vec(spp.to_owned())))
    }

    #[inline(always)]
    fn get_fa(&self, spp: &[f64]) -> Option<HashMap<usize, f64>> {
        Some((self.fa)(&V::from_vec(spp.to_owned())))
    }

    #[inline(always)]
    fn get_nstates(&self) -> usize {
        self.neqs.0
    }

    #[inline(always)]
    fn get_nouteqs(&self) -> usize {
        self.neqs.1
    }
    #[inline(always)]
    fn solve(
        &self,
        x: &mut Self::S,
        support_point: &Vec<f64>,
        covariates: &Covariates,
        infusions: &Vec<Infusion>,
        ti: f64,
        tf: f64,
    ) -> Result<(), PharmsolError> {
        if ti == tf {
            return Ok(());
        }
        let mut support_point = V::from_vec(support_point.to_owned());
        let mut rateiv = V::from_vec(vec![0.0, 0.0, 0.0]);
        //TODO: This should be pre-calculated
        for infusion in infusions {
            if tf >= infusion.time() && tf <= infusion.duration() + infusion.time() {
                rateiv[infusion.input()] += infusion.amount() / infusion.duration();
            }
        }
        (self.seq_eq)(&mut support_point, tf, covariates);
        *x = (self.eq)(x, &support_point, tf - ti, rateiv, covariates);
        Ok(())
    }
    #[inline(always)]
    fn process_observation(
        &self,
        support_point: &Vec<f64>,
        observation: &Observation,
        error_models: Option<&ErrorModels>,
        _time: f64,
        covariates: &Covariates,
        x: &mut Self::S,
        likelihood: &mut Vec<f64>,
        output: &mut Self::P,
    ) -> Result<(), PharmsolError> {
        let mut y = V::zeros(self.get_nouteqs());
        let out = &self.out;
        (out)(
            x,
            &V::from_vec(support_point.clone()),
            observation.time(),
            covariates,
            &mut y,
        );
        let pred = y[observation.outeq()];
        let pred = observation.to_prediction(pred, x.as_slice().to_vec());
        if let Some(error_models) = error_models {
            likelihood.push(pred.likelihood(error_models)?);
        }
        output.add_prediction(pred);
        Ok(())
    }
    #[inline(always)]
    fn initial_state(&self, spp: &Vec<f64>, covariates: &Covariates, occasion_index: usize) -> V {
        let init = &self.init;
        let mut x = V::zeros(self.get_nstates());
        if occasion_index == 0 {
            (init)(&V::from_vec(spp.to_vec()), 0.0, covariates, &mut x);
        }
        x
    }
}

impl Equation for Analytical {
    fn estimate_likelihood(
        &self,
        subject: &Subject,
        support_point: &Vec<f64>,
        error_models: &ErrorModels,
        cache: bool,
    ) -> Result<f64, PharmsolError> {
        _estimate_likelihood(self, subject, support_point, error_models, cache)
    }
}
fn spphash(spp: &[f64]) -> u64 {
    spp.iter().fold(0, |acc, x| acc + x.to_bits())
}

#[inline(always)]
#[cached(
    ty = "UnboundCache<String, SubjectPredictions>",
    create = "{ UnboundCache::with_capacity(100_000) }",
    convert = r#"{ format!("{}{}", subject.id(), spphash(support_point)) }"#,
    result = "true"
)]
fn _subject_predictions(
    ode: &Analytical,
    subject: &Subject,
    support_point: &Vec<f64>,
) -> Result<SubjectPredictions, PharmsolError> {
    Ok(ode.simulate_subject(subject, support_point, None)?.0)
}

fn _estimate_likelihood(
    ode: &Analytical,
    subject: &Subject,
    support_point: &Vec<f64>,
    error_models: &ErrorModels,
    cache: bool,
) -> Result<f64, PharmsolError> {
    let ypred = if cache {
        _subject_predictions(ode, subject, support_point)
    } else {
        _subject_predictions_no_cache(ode, subject, support_point)
    }?;
    ypred.likelihood(error_models)
}
