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
    data::{Covariates, Event, Infusion, InputLabel, Occasion, OutputLabel},
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

/// Turn an occasion's events into the schedule a family driver runs.
///
/// Resolves every label to a dense index, delays boluses by their lag,
/// scales them by bioavailability, and sorts the result by time (observations
/// before doses at equal times).
pub(crate) fn prepare_occasion<M: Model>(
    model: &M,
    ws: &mut M::Workspace,
    occasion: &Occasion,
    p: &V,
) -> Result<Vec<Event>, PharmsolError> {
    let covariates = occasion.covariates();
    let mut events = occasion.events().to_vec();

    for event in events.iter_mut() {
        match event {
            Event::Bolus(bolus) => {
                let input = resolve_input(model, bolus.input(), RouteKind::Bolus)?;
                bolus.set_input(input);

                let lag = model.lag(ws, input, bolus.time(), p, covariates)?;
                if lag != 0.0 {
                    *bolus.mut_time() += lag;
                }
                let fa = model.bioavailability(ws, input, bolus.time(), p, covariates)?;
                if fa != 1.0 {
                    bolus.set_amount(bolus.amount() * fa);
                }
            }
            Event::Infusion(infusion) => {
                let input = resolve_input(model, infusion.input(), RouteKind::Infusion)?;
                infusion.set_input(input);
            }
            Event::Observation(observation) => {
                let outeq = resolve_output(model, observation.outeq())?;
                observation.set_outeq_slot(outeq);
            }
        }
    }

    events.sort_by(Event::cmp_time_then_type);
    Ok(events)
}

/// The infusions in a prepared schedule.
pub(crate) fn infusions(events: &[Event]) -> Vec<Infusion> {
    events
        .iter()
        .filter_map(|event| match event {
            Event::Infusion(infusion) => Some(infusion.clone()),
            _ => None,
        })
        .collect()
}

/// Fill `rates` with the total rate of every infusion running at `time`.
///
/// An infusion counts as running on its closed interval `[start, end]`.
pub(crate) fn fill_active_rates(infusions: &[Infusion], time: f64, rates: &mut V) {
    let rates = rates.as_mut_slice();
    rates.fill(0.0);
    for infusion in infusions {
        let input = infusion
            .input_index()
            .expect("prepared infusions carry a dense input index");
        if time >= infusion.time() && time <= infusion.time() + infusion.duration() {
            rates[input] += infusion.amount() / infusion.duration();
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
