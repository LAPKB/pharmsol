//! Models with closed-form solutions.
//!
//! [`Analytical`] pairs an [`AnalyticalModel`] with a prediction cache. Between
//! events the state is advanced in closed form, split at every infusion start
//! and stop so each step sees constant rates.

pub mod one_compartment_cl_models;
pub mod one_compartment_models;
pub mod three_compartment_cl_models;
pub mod three_compartment_models;
pub mod two_compartment_cl_models;
pub mod two_compartment_models;

mod native;

pub use native::NativeAnalytical;
pub use one_compartment_cl_models::*;
pub use one_compartment_models::*;
pub use three_compartment_cl_models::*;
pub use three_compartment_models::*;
pub use two_compartment_cl_models::*;
pub use two_compartment_models::*;

use std::sync::Arc;

use diffsol::{NalgebraContext, Vector, VectorHost};
use pharmsol_dsl::AnalyticalKernel;

use super::model::{
    fill_active_rates, prepare_occasion, with_subject_context, InfusionDose, Model, Step,
};
use super::{salted_parameters_hash, Cache, Equation, ValidatedModelMetadata};
use crate::data::error_model::AssayErrorModels;
use crate::prelude::simulator::SubjectPredictions;
use crate::simulator::cache::{PredictionCache, PredictionKey, DEFAULT_CACHE_SIZE};
use crate::simulator::{AnalyticalEq, T, V};
use crate::{data::Covariates, Occasion, PharmsolError, Subject};

/// Model functions for an [`Analytical`] equation.
pub trait AnalyticalModel: Model {
    /// Advance `x` by `dt`, ending at `time`, with constant infusion `rates`.
    ///
    /// `p` is reset to the support point at each event and carries over
    /// between the steps of one event interval, so secondary equations may
    /// update it in place.
    #[allow(clippy::too_many_arguments)]
    fn step(
        &self,
        ws: &mut Self::Workspace,
        time: f64,
        dt: f64,
        x: &mut V,
        p: &mut V,
        cov: &Covariates,
        rates: &V,
    ) -> Result<(), PharmsolError>;
}

/// An [`AnalyticalModel`] with a prediction cache.
///
/// Caching is on by default; see [`Cache`]. Clones share the cache.
#[derive(Clone, Debug)]
pub struct Analytical<M = NativeAnalytical> {
    model: M,
    cache: Option<PredictionCache>,
}

impl<M: AnalyticalModel> Analytical<M> {
    /// Wrap a model with the default cache.
    pub fn from_model(model: M) -> Self {
        Self {
            model,
            cache: Some(PredictionCache::new(DEFAULT_CACHE_SIZE)),
        }
    }

    /// The model functions.
    pub fn model(&self) -> &M {
        &self.model
    }

    #[cfg(all(test, feature = "dsl"))]
    pub(crate) fn prediction_cache(&self) -> Option<&PredictionCache> {
        self.cache.as_ref()
    }

    pub fn parameter_index(&self, name: &str) -> Option<usize> {
        self.model.metadata()?.parameter_index(name)
    }

    pub fn covariate_index(&self, name: &str) -> Option<usize> {
        self.model.metadata()?.covariate_index(name)
    }

    pub fn state_index(&self, name: &str) -> Option<usize> {
        self.model.metadata()?.state_index(name)
    }

    /// Predictions for a subject, from the cache when possible.
    fn predictions(
        &self,
        subject: &Subject,
        parameters: &[f64],
    ) -> Result<Arc<SubjectPredictions>, PharmsolError> {
        let simulate = || self.simulate(subject, parameters).map(Arc::new);
        match &self.cache {
            Some(cache) => {
                cache.get_or_try_insert(self.prediction_key(subject, parameters), simulate)
            }
            None => simulate(),
        }
    }

    pub(crate) fn prediction_key(&self, subject: &Subject, parameters: &[f64]) -> PredictionKey {
        (subject.hash(), salted_parameters_hash(parameters, 0))
    }

    fn simulate(
        &self,
        subject: &Subject,
        parameters: &[f64],
    ) -> Result<SubjectPredictions, PharmsolError> {
        let in_context = |error| with_subject_context(&self.model, error, subject.id(), parameters);
        self.model
            .check_parameters(parameters)
            .map_err(in_context)?;

        let p = V::from_vec(parameters.to_vec(), NalgebraContext::new());
        let mut scratch = Scratch::new(&self.model, parameters.len());
        let mut output = SubjectPredictions::default();
        for occasion in subject.occasions() {
            self.simulate_occasion(occasion, &p, &mut scratch, &mut output)
                .map_err(in_context)?;
        }
        Ok(output)
    }

    fn simulate_occasion(
        &self,
        occasion: &Occasion,
        p: &V,
        scratch: &mut Scratch,
        output: &mut SubjectPredictions,
    ) -> Result<(), PharmsolError> {
        let covariates = occasion.covariates();
        let mut ws = self.model.workspace();
        let schedule = prepare_occasion(&self.model, &mut ws, occasion, p)?;
        let mut x = V::zeros(self.model.nstates(), NalgebraContext::new());
        self.model.init(&mut ws, 0.0, p, covariates, &mut x)?;

        for (index, step) in schedule.steps.iter().enumerate() {
            match *step {
                Step::Bolus {
                    time,
                    input,
                    amount,
                } => {
                    self.model
                        .bolus(&mut ws, time, &mut x, p, covariates, input, amount)?;
                }
                // Infusions enter through the rates of each step.
                Step::Infusion(_) => {}
                Step::Observation { observation, outeq } => {
                    fill_active_rates(&schedule.infusions, observation.time(), &mut scratch.rates);
                    scratch.y.fill(0.0);
                    self.model.outputs(
                        &mut ws,
                        observation.time(),
                        &x,
                        p,
                        covariates,
                        &scratch.rates,
                        &mut scratch.y,
                    )?;
                    output.add_prediction(
                        observation.to_prediction(scratch.y[outeq], x.as_slice().to_vec()),
                    );
                }
            }

            if let Some(next) = schedule.steps.get(index + 1) {
                self.advance(
                    &mut ws,
                    &mut x,
                    p,
                    covariates,
                    &schedule.infusions,
                    step.time(),
                    next.time(),
                    scratch,
                )?;
            }
        }
        Ok(())
    }

    /// Advance `x` from `start` to `end`, splitting at infusion boundaries.
    #[allow(clippy::too_many_arguments)]
    fn advance(
        &self,
        ws: &mut M::Workspace,
        x: &mut V,
        p: &V,
        covariates: &Covariates,
        infusions: &[InfusionDose],
        start: f64,
        end: f64,
        scratch: &mut Scratch,
    ) -> Result<(), PharmsolError> {
        if start == end {
            return Ok(());
        }

        let mut breakpoints = vec![start, end];
        for infusion in infusions {
            let begin = infusion.time;
            let finish = infusion.end();
            if begin > start && begin < end {
                breakpoints.push(begin);
            }
            if finish > start && finish < end {
                breakpoints.push(finish);
            }
        }
        breakpoints.sort_by(|a, b| a.total_cmp(b));
        breakpoints.dedup_by(|a, b| (*a - *b).abs() < 1e-12);

        scratch.p.copy_from(p);
        for segment in breakpoints.windows(2) {
            let (from, to) = (segment[0], segment[1]);
            fill_segment_rates(infusions, from, to, &mut scratch.rates);
            self.model.step(
                ws,
                to,
                to - from,
                x,
                &mut scratch.p,
                covariates,
                &scratch.rates,
            )?;
        }
        Ok(())
    }
}

/// Fill `rates` with the infusions that run for the whole segment `[from, to]`.
fn fill_segment_rates(infusions: &[InfusionDose], from: f64, to: f64, rates: &mut V) {
    let rates = rates.as_mut_slice();
    rates.fill(0.0);
    for infusion in infusions {
        if from >= infusion.time && to <= infusion.end() {
            rates[infusion.input] += infusion.rate();
        }
    }
}

/// Buffers reused across a subject's occasions.
struct Scratch {
    /// Structural parameters of the current step.
    p: V,
    /// Outputs at an observation.
    y: V,
    /// Infusion rates of the current step or observation.
    rates: V,
}

impl Scratch {
    fn new(model: &impl Model, nparams: usize) -> Self {
        let ctx = NalgebraContext::new();
        Self {
            p: V::zeros(nparams, ctx),
            y: V::zeros(model.noutputs(), ctx),
            rates: V::zeros(model.ninputs(), ctx),
        }
    }
}

impl<M: AnalyticalModel> Cache for Analytical<M> {
    fn with_cache_capacity(mut self, size: usize) -> Self {
        self.cache = Some(PredictionCache::new(size));
        self
    }

    fn enable_cache(mut self) -> Self {
        self.cache = Some(PredictionCache::new(DEFAULT_CACHE_SIZE));
        self
    }

    fn clear_cache(&self) {
        if let Some(cache) = &self.cache {
            cache.invalidate_all();
        }
    }

    fn disable_cache(mut self) -> Self {
        self.cache = None;
        self
    }
}

impl<M: AnalyticalModel> Equation for Analytical<M> {
    type P = SubjectPredictions;

    fn metadata(&self) -> Option<&ValidatedModelMetadata> {
        self.model.metadata()
    }

    fn nstates(&self) -> usize {
        self.model.nstates()
    }

    fn nouteqs(&self) -> usize {
        self.model.noutputs()
    }

    fn estimate_predictions(
        &self,
        subject: &Subject,
        parameters: &[f64],
    ) -> Result<SubjectPredictions, PharmsolError> {
        self.predictions(subject, parameters)
            .map(Arc::unwrap_or_clone)
    }

    fn estimate_log_likelihood(
        &self,
        subject: &Subject,
        parameters: &[f64],
        error_models: &AssayErrorModels,
    ) -> Result<f64, PharmsolError> {
        self.validate_error_models(error_models)?;
        self.predictions(subject, parameters)?
            .log_likelihood(error_models)
    }
}

/// The closed-form solution behind a built-in analytical kernel.
pub fn kernel_function(kernel: AnalyticalKernel) -> AnalyticalEq {
    match kernel {
        AnalyticalKernel::OneCompartment => one_compartment,
        AnalyticalKernel::OneCompartmentCl => one_compartment_cl,
        AnalyticalKernel::OneCompartmentClWithAbsorption => one_compartment_cl_with_absorption,
        AnalyticalKernel::OneCompartmentWithAbsorption => one_compartment_with_absorption,
        AnalyticalKernel::TwoCompartments => two_compartments,
        AnalyticalKernel::TwoCompartmentsCl => two_compartments_cl,
        AnalyticalKernel::TwoCompartmentsClWithAbsorption => two_compartments_cl_with_absorption,
        AnalyticalKernel::TwoCompartmentsWithAbsorption => two_compartments_with_absorption,
        AnalyticalKernel::ThreeCompartments => three_compartments,
        AnalyticalKernel::ThreeCompartmentsCl => three_compartments_cl,
        AnalyticalKernel::ThreeCompartmentsClWithAbsorption => {
            three_compartments_cl_with_absorption
        }
        AnalyticalKernel::ThreeCompartmentsWithAbsorption => three_compartments_with_absorption,
    }
}

#[inline(always)]
pub(crate) fn compact_public_vector(vector: &V) -> V {
    V::from_vec(
        vector.as_slice().get(1..).unwrap_or(&[]).to_vec(),
        NalgebraContext::new(),
    )
}

#[inline(always)]
pub(crate) fn pad_public_vector(vector: &V) -> V {
    let mut padded = Vec::with_capacity(vector.len() + 1);
    padded.push(0.0);
    padded.extend(vector.as_slice().iter().copied());
    V::from_vec(padded, NalgebraContext::new())
}

#[inline(always)]
pub(crate) fn wrap_pmetrics_analytical(
    x: &V,
    p: &V,
    t: T,
    rateiv: &V,
    cov: &Covariates,
    native: AnalyticalEq,
) -> V {
    let compact_x = compact_public_vector(x);
    let compact_rateiv = compact_public_vector(rateiv);
    let compact_output = native(&compact_x, p, t, &compact_rateiv, cov);
    pad_public_vector(&compact_output)
}

#[cfg(test)]
pub(crate) mod tests;
