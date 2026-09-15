//! Resolved events: public labels paired with their dense runtime slots.
//!
//! An [`InputLabel`] or [`OutputLabel`] is the *identity* of a route or an
//! output. The dense `usize` slot it maps to is an implementation detail of the
//! execution layer's flat vectors.
//!
//! Resolution happens exactly once per occasion per simulation. Instead of
//! overwriting the label with its slot - which destroyed the identity and cost
//! a [`String`] allocation per event - the slot is carried *beside* the event in
//! the types below. The original event is borrowed, so no label is cloned and
//! the public label survives all the way into [`Prediction`].
//!
//! [`Prediction`]: crate::simulator::likelihood::Prediction

use std::cmp::Ordering;

use crate::data::event::{Bolus, Event, Infusion, InputLabel, Observation, OutputLabel};
use crate::data::structs::Occasion;
use crate::simulator::{Fa, Lag, V};
use crate::{Covariates, PharmsolError};

/// Maps public data labels onto the dense slots used by one execution backend.
///
/// Implemented by the handwritten equation families through
/// [`crate::simulator::equation::EquationPriv`] and by the compiled DSL
/// backend. Bolus and infusion routes are resolved separately because a label
/// may be declared for both kinds and the two are distinct ordinal spaces.
pub(crate) trait LabelResolver {
    /// Resolve a bolus route label to its dense input slot.
    fn resolve_bolus_input(&self, label: &InputLabel) -> Result<usize, PharmsolError>;

    /// Resolve an infusion route label to its dense input slot.
    fn resolve_infusion_input(&self, label: &InputLabel) -> Result<usize, PharmsolError>;

    /// Resolve an output label to its dense output slot.
    fn resolve_output(&self, label: &OutputLabel) -> Result<usize, PharmsolError>;
}

/// A bolus paired with its resolved dense input slot.
///
/// `time` and `amount` are copied out of the borrowed [`Bolus`] so lag time and
/// bioavailability can adjust them without mutating - or cloning - the subject
/// data.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ResolvedBolus<'a> {
    bolus: &'a Bolus,
    input: usize,
    time: f64,
    amount: f64,
}

impl<'a> ResolvedBolus<'a> {
    /// Pair a bolus with an already-resolved dense input slot.
    pub(crate) fn new(bolus: &'a Bolus, input: usize) -> Self {
        Self {
            bolus,
            input,
            time: bolus.time(),
            amount: bolus.amount(),
        }
    }

    /// The dense input slot this bolus was resolved to.
    pub(crate) fn input_slot(&self) -> usize {
        self.input
    }

    /// The original public route label.
    pub(crate) fn label(&self) -> &'a InputLabel {
        self.bolus.input()
    }

    /// The dose time, including any applied lag time.
    pub(crate) fn time(&self) -> f64 {
        self.time
    }

    /// The dose amount, including any applied bioavailability factor.
    pub(crate) fn amount(&self) -> f64 {
        self.amount
    }

    /// Shift the dose time, e.g. by a lag time.
    pub(crate) fn shift_time(&mut self, delta: f64) {
        self.time += delta;
    }

    /// Scale the dose amount, e.g. by a bioavailability factor.
    pub(crate) fn scale_amount(&mut self, factor: f64) {
        self.amount *= factor;
    }
}

/// An infusion paired with its resolved dense input slot.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ResolvedInfusion<'a> {
    infusion: &'a Infusion,
    input: usize,
}

impl<'a> ResolvedInfusion<'a> {
    /// Pair an infusion with an already-resolved dense input slot.
    pub(crate) fn new(infusion: &'a Infusion, input: usize) -> Self {
        Self { infusion, input }
    }

    /// The dense input slot this infusion was resolved to.
    pub(crate) fn input_slot(&self) -> usize {
        self.input
    }

    /// The original public route label.
    pub(crate) fn label(&self) -> &'a InputLabel {
        self.infusion.input()
    }

    /// Infusion start time.
    pub(crate) fn time(&self) -> f64 {
        self.infusion.time()
    }

    /// Total amount delivered over the infusion.
    pub(crate) fn amount(&self) -> f64 {
        self.infusion.amount()
    }

    /// Infusion duration.
    pub(crate) fn duration(&self) -> f64 {
        self.infusion.duration()
    }
}

/// An observation paired with its resolved dense output slot.
///
/// The [`Observation`] is borrowed, so its [`OutputLabel`] is untouched. The
/// slot is only used to index the dense output vector and the dense error
/// models.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ResolvedObservation<'a> {
    observation: &'a Observation,
    outeq: usize,
}

impl<'a> ResolvedObservation<'a> {
    /// Pair an observation with an already-resolved dense output slot.
    pub(crate) fn new(observation: &'a Observation, outeq: usize) -> Self {
        Self { observation, outeq }
    }

    /// The dense output slot this observation was resolved to.
    pub(crate) fn outeq_slot(&self) -> usize {
        self.outeq
    }

    /// The borrowed observation, with its original public output label.
    pub(crate) fn observation(&self) -> &'a Observation {
        self.observation
    }

    /// Observation time.
    pub(crate) fn time(&self) -> f64 {
        self.observation.time()
    }

    /// Build a [`Prediction`] that carries both the public label and the dense
    /// slot.
    ///
    /// [`Prediction`]: crate::simulator::likelihood::Prediction
    pub(crate) fn to_prediction(
        self,
        pred: f64,
        state: Vec<f64>,
    ) -> crate::simulator::likelihood::Prediction {
        self.observation().to_prediction_at(self.outeq, pred, state)
    }
}

/// One event of an occasion with its label already resolved to a dense slot.
#[derive(Debug, Clone, Copy)]
pub(crate) enum ResolvedEvent<'a> {
    Bolus(ResolvedBolus<'a>),
    Infusion(ResolvedInfusion<'a>),
    Observation(ResolvedObservation<'a>),
}

impl ResolvedEvent<'_> {
    /// Time of the event, including any adjustment applied to a bolus.
    pub(crate) fn time(&self) -> f64 {
        match self {
            ResolvedEvent::Bolus(bolus) => bolus.time(),
            ResolvedEvent::Infusion(infusion) => infusion.time(),
            ResolvedEvent::Observation(observation) => observation.time(),
        }
    }

    /// Compare events by time, with observations before doses at equal times.
    ///
    /// Mirrors [`Event::cmp_time_then_type`] so a resolved schedule keeps the
    /// same ordering guarantees as an unresolved one.
    pub(crate) fn cmp_time_then_type(&self, other: &Self) -> Ordering {
        fn rank(event: &ResolvedEvent<'_>) -> u8 {
            match event {
                ResolvedEvent::Observation(_) => 0,
                ResolvedEvent::Bolus(_) => 1,
                ResolvedEvent::Infusion(_) => 2,
            }
        }

        self.time()
            .total_cmp(&other.time())
            .then_with(|| rank(self).cmp(&rank(other)))
    }
}

/// Resolve every event of an occasion exactly once and apply the model's lag
/// time and bioavailability to the resulting dense schedule.
///
/// This is the single place where a public label becomes a dense slot on the
/// simulation path. Nothing downstream re-resolves, and nothing downstream sees
/// a slot where a label belongs.
pub(crate) fn resolve_occasion<'a, R: LabelResolver + ?Sized>(
    occasion: &'a Occasion,
    resolver: &R,
    reorder: Option<(&Fa, &Lag, &[f64], &Covariates)>,
) -> Result<Vec<ResolvedEvent<'a>>, PharmsolError> {
    let mut events = Vec::with_capacity(occasion.events().len());
    for event in occasion.iter() {
        events.push(match event {
            Event::Bolus(bolus) => {
                let input = resolver.resolve_bolus_input(bolus.input())?;
                ResolvedEvent::Bolus(ResolvedBolus::new(bolus, input))
            }
            Event::Infusion(infusion) => {
                let input = resolver.resolve_infusion_input(infusion.input())?;
                ResolvedEvent::Infusion(ResolvedInfusion::new(infusion, input))
            }
            Event::Observation(observation) => {
                let outeq = resolver.resolve_output(observation.outeq())?;
                ResolvedEvent::Observation(ResolvedObservation::new(observation, outeq))
            }
        });
    }

    apply_dose_adjustments(&mut events, reorder);
    Ok(events)
}

/// Apply lag time and bioavailability to the boluses of a resolved schedule.
///
/// Both model functions are keyed by the dense input slot, which is why this
/// runs after resolution. With no model context the schedule is already sorted
/// by construction and nothing is adjusted.
fn apply_dose_adjustments(
    events: &mut [ResolvedEvent<'_>],
    reorder: Option<(&Fa, &Lag, &[f64], &Covariates)>,
) {
    let Some((fn_fa, fn_lag, parameters, covariates)) = reorder else {
        return;
    };

    // Build the parameter vector once and reuse it for every dose.
    let parameters: V = nalgebra::DVector::from_vec(parameters.to_vec()).into();
    let mut shifted = false;

    for event in events.iter_mut() {
        // Lag time delays boluses only; infusions are never lagged.
        let ResolvedEvent::Bolus(bolus) = event else {
            continue;
        };
        let lagtime = fn_lag(&parameters, bolus.time(), covariates);
        if let Some(&l) = lagtime.get(&bolus.input_slot()) {
            if l != 0.0 {
                bolus.shift_time(l);
                shifted = true;
            }
        }
    }

    // Re-sort only when a lag actually moved an event; the events were already
    // sorted at construction time, so an unchanged pass stays sorted.
    if shifted {
        events.sort_by(ResolvedEvent::cmp_time_then_type);
    }

    for event in events.iter_mut() {
        // Bioavailability scales bolus amounts only.
        let ResolvedEvent::Bolus(bolus) = event else {
            continue;
        };
        let fa = fn_fa(&parameters, bolus.time(), covariates);
        if let Some(&f) = fa.get(&bolus.input_slot()) {
            bolus.scale_amount(f);
        }
    }
}
