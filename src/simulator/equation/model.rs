//! Model functions shared by every equation family.
//!
//! An equation family ([`ODE`](super::ODE), [`Analytical`](super::Analytical),
//! [`SDE`](super::SDE)) decides *how* a model is advanced through time. A
//! [`Model`] supplies *what* is evaluated along the way: initial state, doses,
//! lag and bioavailability, and outputs. Each family adds its own trait on top
//! ([`OdeModel`](super::OdeModel), [`AnalyticalModel`](super::AnalyticalModel),
//! [`SdeModel`](super::SdeModel)).
//!
//! Handwritten closure models and DSL-compiled models implement the same
//! traits, so both run through the same family driver.

use std::fmt::Debug;

use diffsol::VectorHost;
use thiserror::Error;

use crate::{
    data::event::SameTimeOrder,
    data::{Covariates, Event, InputLabel, Observation, Occasion, OutputLabel},
    simulator::V,
    PharmsolError,
};

use super::metadata::{ModelMetadataError, RouteKind, ValidatedModelMetadata};

/// Model functions used by every equation family.
///
/// Every method that evaluates the model takes a mutable [`Workspace`]
/// (scratch buffers owned by the caller for one occasion) and returns a
/// `Result`, so compiled models can report failures instead of panicking.
/// Vectors are the crate's dense [`V`] type: `p` is the support point,
/// `x` the state, `rates` the active infusion rate per input.
///
/// [`Workspace`]: Model::Workspace
pub trait Model: Clone + Debug + Send + Sync + 'static {
    /// Scratch space for one occasion.
    type Workspace;

    /// Allocate a fresh workspace.
    fn workspace(&self) -> Self::Workspace;

    /// Number of state variables.
    fn nstates(&self) -> usize;

    /// Number of dense input slots (bolus and infusion routes).
    fn ninputs(&self) -> usize;

    /// Number of outputs.
    fn noutputs(&self) -> usize;

    /// Validated metadata, if the model declares names.
    fn metadata(&self) -> Option<&ValidatedModelMetadata>;

    /// Reject a support point the model cannot use. The default accepts any.
    fn check_parameters(&self, _parameters: &[f64]) -> Result<(), PharmsolError> {
        Ok(())
    }

    /// Lag time for a bolus on `input`, evaluated at the scheduled dose time.
    fn lag(
        &self,
        ws: &mut Self::Workspace,
        input: usize,
        time: f64,
        p: &V,
        cov: &Covariates,
    ) -> Result<f64, PharmsolError>;

    /// Bioavailability for a bolus on `input`, evaluated at the lagged dose time.
    fn bioavailability(
        &self,
        ws: &mut Self::Workspace,
        input: usize,
        time: f64,
        p: &V,
        cov: &Covariates,
    ) -> Result<f64, PharmsolError>;

    /// Write the initial state into `x`, which is zeroed on entry.
    fn init(
        &self,
        ws: &mut Self::Workspace,
        time: f64,
        p: &V,
        cov: &Covariates,
        x: &mut V,
    ) -> Result<(), PharmsolError>;

    /// Apply a bolus of `amount` on `input` to the state `x`.
    #[allow(clippy::too_many_arguments)]
    fn bolus(
        &self,
        ws: &mut Self::Workspace,
        time: f64,
        x: &mut V,
        p: &V,
        cov: &Covariates,
        input: usize,
        amount: f64,
    ) -> Result<(), PharmsolError>;

    /// Write the outputs for state `x` into `y`, which is zeroed on entry.
    #[allow(clippy::too_many_arguments)]
    fn outputs(
        &self,
        ws: &mut Self::Workspace,
        time: f64,
        x: &V,
        p: &V,
        cov: &Covariates,
        rates: &V,
        y: &mut V,
    ) -> Result<(), PharmsolError>;
}

/// Resolve a dose label to a dense input index.
///
/// With metadata, the label must name a route of the dose's kind (or a route
/// that accepts either kind). Without metadata, the label must be a dense index.
pub(crate) fn resolve_input<M: Model>(
    model: &M,
    label: &InputLabel,
    kind: RouteKind,
) -> Result<usize, PharmsolError> {
    let input = match model.metadata() {
        Some(metadata) => match metadata.route_for_label(label.as_str(), kind) {
            Some(route) => route.input_index(),
            None => {
                let other_kind = match kind {
                    RouteKind::Bolus => RouteKind::Infusion,
                    RouteKind::Infusion => RouteKind::Bolus,
                };
                return Err(match metadata.route_for_label(label.as_str(), other_kind) {
                    Some(other) => PharmsolError::UnsupportedInputRouteKind {
                        input: other.input_index(),
                        kind,
                    },
                    None => {
                        PharmsolError::unknown_input_label(label.as_str(), &metadata.route_labels())
                    }
                });
            }
        },
        None => label
            .index()
            .ok_or_else(|| PharmsolError::unknown_input_label(label.as_str(), &[]))?,
    };

    if input >= model.ninputs() {
        return Err(PharmsolError::InputOutOfRange {
            input,
            ndrugs: model.ninputs(),
        });
    }
    Ok(input)
}

/// Resolve an observation label to a dense output index.
pub(crate) fn resolve_output<M: Model>(
    model: &M,
    label: &OutputLabel,
) -> Result<usize, PharmsolError> {
    let outeq = match model.metadata() {
        Some(metadata) => metadata.output_for_label(label.as_str()).ok_or_else(|| {
            PharmsolError::unknown_output_label(label.as_str(), &metadata.output_labels())
        })?,
        None => label
            .index()
            .ok_or_else(|| PharmsolError::unknown_output_label(label.as_str(), &[]))?,
    };

    if outeq >= model.noutputs() {
        return Err(PharmsolError::OuteqOutOfRange {
            outeq,
            nout: model.noutputs(),
        });
    }
    Ok(outeq)
}

/// An infusion with its input resolved to a dense index.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct InfusionDose {
    pub(crate) time: f64,
    pub(crate) duration: f64,
    pub(crate) amount: f64,
    pub(crate) input: usize,
}

impl InfusionDose {
    pub(crate) fn end(&self) -> f64 {
        self.time + self.duration
    }

    pub(crate) fn rate(&self) -> f64 {
        self.amount / self.duration
    }
}

/// One event of an occasion with its labels resolved, ready to simulate.
#[derive(Debug, Clone, Copy)]
pub(crate) enum Step<'a> {
    Bolus {
        time: f64,
        input: usize,
        amount: f64,
    },
    Infusion(InfusionDose),
    Observation {
        observation: &'a Observation,
        outeq: usize,
    },
}

impl Step<'_> {
    pub(crate) fn time(&self) -> f64 {
        match self {
            Step::Bolus { time, .. } => *time,
            Step::Infusion(infusion) => infusion.time,
            Step::Observation { observation, .. } => observation.time(),
        }
    }

    fn same_time_order(&self) -> SameTimeOrder {
        match self {
            Step::Observation { .. } => SameTimeOrder::Observation,
            Step::Bolus { .. } => SameTimeOrder::Bolus,
            Step::Infusion(_) => SameTimeOrder::Infusion,
        }
    }
}

/// The steps of one occasion, plus its infusions for rate lookups.
pub(crate) struct Schedule<'a> {
    pub(crate) steps: Vec<Step<'a>>,
    pub(crate) infusions: Vec<InfusionDose>,
}

/// Turn an occasion's events into the schedule a family driver runs.
///
/// Resolves every label to a dense index, delays boluses by their lag,
/// scales them by bioavailability, and sorts the result by time (observations
/// before doses at equal times).
pub(crate) fn prepare_occasion<'a, M: Model>(
    model: &M,
    ws: &mut M::Workspace,
    occasion: &'a Occasion,
    p: &V,
) -> Result<Schedule<'a>, PharmsolError> {
    let covariates = occasion.covariates();
    let mut steps = Vec::with_capacity(occasion.events().len());
    let mut infusions = Vec::new();

    for event in occasion.events() {
        let step = match event {
            Event::Bolus(bolus) => {
                let input = resolve_input(model, bolus.input(), RouteKind::Bolus)?;
                let time = bolus.time() + model.lag(ws, input, bolus.time(), p, covariates)?;
                let fa = model.bioavailability(ws, input, time, p, covariates)?;
                Step::Bolus {
                    time,
                    input,
                    amount: bolus.amount() * fa,
                }
            }
            Event::Infusion(infusion) => {
                let dose = InfusionDose {
                    time: infusion.time(),
                    duration: infusion.duration(),
                    amount: infusion.amount(),
                    input: resolve_input(model, infusion.input(), RouteKind::Infusion)?,
                };
                infusions.push(dose);
                Step::Infusion(dose)
            }
            Event::Observation(observation) => Step::Observation {
                observation,
                outeq: resolve_output(model, observation.outeq())?,
            },
        };
        steps.push(step);
    }

    for step in &steps {
        check_step(step)?;
    }
    steps.sort_by(|a, b| {
        a.time()
            .total_cmp(&b.time())
            .then(a.same_time_order().cmp(&b.same_time_order()))
    });
    Ok(Schedule { steps, infusions })
}

/// Reject a dose no family can simulate: a non-finite time or amount, or
/// an infusion without a positive, finite duration.
fn check_step(step: &Step) -> Result<(), PharmsolError> {
    let invalid = |detail: String| {
        Err(PharmsolError::OtherError(format!(
            "invalid event schedule: {detail}"
        )))
    };
    let time = step.time();
    if !time.is_finite() {
        return invalid(format!("event time {time:?} is not finite"));
    }
    match *step {
        Step::Bolus { amount, .. } if !amount.is_finite() => invalid(format!(
            "bolus at t = {time} has non-finite amount {amount:?}"
        )),
        Step::Infusion(infusion) if !infusion.amount.is_finite() => invalid(format!(
            "infusion at t = {time} has non-finite amount {:?}",
            infusion.amount
        )),
        Step::Infusion(infusion) if !(infusion.duration > 0.0 && infusion.duration.is_finite()) => {
            invalid(format!(
                "infusion at t = {time} must have a positive, finite duration, got {:?}",
                infusion.duration
            ))
        }
        _ => Ok(()),
    }
}

/// Fill `rates` with the total rate of every infusion running at `time`.
///
/// An infusion counts as running on its closed interval `[start, end]`.
pub(crate) fn fill_active_rates(infusions: &[InfusionDose], time: f64, rates: &mut V) {
    let rates = rates.as_mut_slice();
    rates.fill(0.0);
    for infusion in infusions {
        if time >= infusion.time && time <= infusion.end() {
            rates[infusion.input] += infusion.rate();
        }
    }
}

/// Attach the subject and support point to an error raised while simulating.
pub(crate) fn with_subject_context<M: Model>(
    model: &M,
    error: PharmsolError,
    subject_id: &str,
    parameters: &[f64],
) -> PharmsolError {
    let names = model
        .metadata()
        .map(|metadata| metadata.parameter_names())
        .unwrap_or_default();
    error.with_subject_context(subject_id, parameters, &names)
}

/// Reject fewer parameters than a handwritten model's metadata declares;
/// its functions would index past the end of `p`.
pub(crate) fn check_declared_parameters(
    metadata: Option<&ValidatedModelMetadata>,
    parameters: &[f64],
) -> Result<(), PharmsolError> {
    let Some(metadata) = metadata else {
        return Ok(());
    };
    let declared = metadata.parameters().len();
    if parameters.len() < declared {
        return Err(PharmsolError::OtherError(format!(
            "model `{}` declares {declared} parameter(s), got {}",
            metadata.name(),
            parameters.len()
        )));
    }
    Ok(())
}

/// Error from attaching metadata to a handwritten model.
#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum AttachMetadataError {
    #[error(transparent)]
    Validation(#[from] ModelMetadataError),
    #[error("metadata declares {declared} states but the model has {expected}")]
    StateCountMismatch { expected: usize, declared: usize },
    #[error("metadata declares {declared} route inputs but the model has {expected}")]
    RouteCountMismatch { expected: usize, declared: usize },
    #[error("metadata declares {declared} outputs but the model has {expected}")]
    OutputCountMismatch { expected: usize, declared: usize },
}

/// Check that metadata matches a handwritten model's dimensions.
pub(crate) fn check_dimensions(
    metadata: &ValidatedModelMetadata,
    nstates: usize,
    ninputs: usize,
    noutputs: usize,
) -> Result<(), AttachMetadataError> {
    let declared = metadata.states().len();
    if declared != nstates {
        return Err(AttachMetadataError::StateCountMismatch {
            expected: nstates,
            declared,
        });
    }

    let declared = metadata.route_input_count();
    if declared != ninputs {
        return Err(AttachMetadataError::RouteCountMismatch {
            expected: ninputs,
            declared,
        });
    }

    let declared = metadata.outputs().len();
    if declared != noutputs {
        return Err(AttachMetadataError::OutputCountMismatch {
            expected: noutputs,
            declared,
        });
    }

    Ok(())
}
