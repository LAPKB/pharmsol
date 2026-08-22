use crate::{Covariates, Infusion, PharmsolError};
use diffsol::{
    ConstantOp, LinearOp, MatrixCommon, NalgebraContext, NalgebraMat, NonLinearOp,
    NonLinearOpJacobian, OdeEquations, OdeEquationsRef, Op, UnitCallable, Vector,
};
use std::{
    cell::{Cell, RefCell},
    cmp::Ordering,
};
type M = NalgebraMat<f64>;
type V = <M as MatrixCommon>::V;
type C = <M as MatrixCommon>::C;
type T = <M as MatrixCommon>::T;

#[derive(Debug, Clone)]
struct InfusionTrack {
    input: usize,
    event_times: Vec<f64>,
    cumulative_rates: Vec<f64>,
}

impl InfusionTrack {
    fn new(input: usize, mut events: Vec<(f64, f64)>) -> Self {
        events.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));

        let mut event_times = Vec::with_capacity(events.len());
        let mut cumulative_rates = Vec::with_capacity(events.len());
        let mut current_rate = 0.0;

        for (time, delta) in events {
            current_rate += delta;
            event_times.push(time);
            cumulative_rates.push(current_rate);
        }

        Self {
            input,
            event_times,
            cumulative_rates,
        }
    }

    fn rate_at(&self, time: f64, left_continuity_time: Option<f64>) -> f64 {
        if let Some(left_continuity_time) = left_continuity_time {
            if left_continuity_time == time {
                return self.rate_at_left(time);
            }
        }

        self.rate_at_right(time)
    }

    fn rate_at_left(&self, time: f64) -> f64 {
        if self.event_times.is_empty() {
            return 0.0;
        }

        let event_search = self
            .event_times
            .binary_search_by(|probe| probe.partial_cmp(&time).unwrap_or(Ordering::Less));

        match event_search {
            Ok(mut idx) => {
                while idx > 0 && self.event_times[idx - 1] == time {
                    idx -= 1;
                }

                if idx == 0 {
                    0.0
                } else {
                    self.cumulative_rates[idx - 1]
                }
            }
            Err(0) => 0.0,
            Err(idx) => self.cumulative_rates[idx - 1],
        }
    }

    fn rate_at_right(&self, time: f64) -> f64 {
        if self.event_times.is_empty() {
            return 0.0;
        }

        match self
            .event_times
            .binary_search_by(|probe| probe.partial_cmp(&time).unwrap_or(Ordering::Less))
        {
            Ok(mut idx) => {
                while idx + 1 < self.event_times.len()
                    && self.event_times[idx + 1] == self.event_times[idx]
                {
                    idx += 1;
                }
                self.cumulative_rates[idx]
            }
            Err(0) => 0.0,
            Err(idx) => self.cumulative_rates[idx - 1],
        }
    }
}

#[derive(Debug, Clone, Default)]
struct InfusionSchedule {
    tracks: Vec<InfusionTrack>,
    boundary_times: Vec<f64>,
    left_continuity_time: Cell<Option<f64>>,
}

impl InfusionSchedule {
    fn new<'a, I>(ndrugs: usize, infusions: I) -> Result<Self, PharmsolError>
    where
        I: IntoIterator<Item = &'a Infusion>,
    {
        if ndrugs == 0 {
            return Ok(Self {
                tracks: Vec::new(),
                boundary_times: Vec::new(),
                left_continuity_time: Cell::new(None),
            });
        }

        let mut per_input: Vec<Vec<(f64, f64)>> = vec![Vec::new(); ndrugs];
        let mut saw_infusion = false;
        let mut boundary_times = Vec::new();
        for infusion in infusions {
            saw_infusion = true;
            if infusion.duration() <= 0.0 {
                continue;
            }

            let input = infusion
                .input_index()
                .ok_or_else(|| PharmsolError::unknown_input_label(infusion.input(), &[]))?;
            if input >= ndrugs {
                return Err(PharmsolError::InputOutOfRange { input, ndrugs });
            }

            let rate = infusion.amount() / infusion.duration();
            let end = infusion.time() + infusion.duration();

            per_input[input].push((infusion.time(), rate));
            per_input[input].push((end, -rate));
            boundary_times.push(infusion.time());
            boundary_times.push(end);
        }

        boundary_times.sort_by(|a, b| a.total_cmp(b));
        boundary_times.dedup();

        if !saw_infusion {
            return Ok(Self {
                tracks: Vec::new(),
                boundary_times,
                left_continuity_time: Cell::new(None),
            });
        }

        let tracks = per_input
            .into_iter()
            .enumerate()
            .filter_map(|(input, events)| {
                if events.is_empty() {
                    None
                } else {
                    Some(InfusionTrack::new(input, events))
                }
            })
            .collect();

        Ok(Self {
            tracks,
            boundary_times,
            left_continuity_time: Cell::new(None),
        })
    }

    fn set_left_continuity_time(&self, time: Option<f64>) {
        self.left_continuity_time.set(time);
    }

    fn infusion_boundary_times(&self) -> &[f64] {
        &self.boundary_times
    }

    /// Absolute infusion input in an interval that the backend could not
    /// integrate. The event loop uses this to produce a dose-specific failure
    /// rather than silently advancing across the unresolved boundary.
    fn infusion_amount_between(&self, from: f64, to: f64) -> f64 {
        if to <= from {
            return 0.0;
        }
        let duration = to - from;
        self.tracks
            .iter()
            .map(|track| track.rate_at_left(to).abs() * duration)
            .sum()
    }

    fn fill_rate_vector(&self, time: f64, rateiv: &mut V) {
        let left_continuity_time = self.left_continuity_time.get();
        rateiv.fill(0.0);
        for track in &self.tracks {
            let rate = track.rate_at(time, left_continuity_time);
            if rate != 0.0 {
                rateiv[track.input] = rate;
            }
        }
    }
}

pub struct PmRhs<'a, F>
where
    F: Fn(&V, &V, T, &mut V, &V, &V, &Covariates),
{
    nstates: usize,
    nparams: usize,
    infusion_schedule: &'a InfusionSchedule,
    time_origin: &'a Cell<f64>,
    covariates: &'a Covariates,
    p_as_v: &'a V,
    func: &'a F,
    rateiv_buffer: &'a RefCell<V>,
    zero_bolus: &'a V,
}

impl<F> Op for PmRhs<'_, F>
where
    F: Fn(&V, &V, T, &mut V, &V, &V, &Covariates),
{
    type T = T;
    type V = V;
    type M = M;
    type C = C;
    fn nstates(&self) -> usize {
        self.nstates
    }
    fn nout(&self) -> usize {
        self.nstates
    }
    fn nparams(&self) -> usize {
        self.nparams
    }
    fn context(&self) -> &Self::C {
        &NalgebraContext {}
    }
}

pub struct PmMass {
    nstates: usize,
    nout: usize,
    nparams: usize,
}

impl Op for PmMass {
    type T = T;
    type V = V;
    type M = M;
    type C = C;
    fn nstates(&self) -> usize {
        self.nstates
    }
    fn nout(&self) -> usize {
        self.nout
    }
    fn nparams(&self) -> usize {
        self.nparams
    }
    fn context(&self) -> &Self::C {
        &NalgebraContext {}
    }
}

pub struct PmInit<'a> {
    nstates: usize,
    nout: usize,
    nparams: usize,
    init: &'a V,
}

impl Op for PmInit<'_> {
    type T = T;
    type V = V;
    type M = M;
    type C = C;
    fn nstates(&self) -> usize {
        self.nstates
    }
    fn nout(&self) -> usize {
        self.nout
    }
    fn nparams(&self) -> usize {
        self.nparams
    }
    fn context(&self) -> &Self::C {
        &NalgebraContext {}
    }
}

impl ConstantOp for PmInit<'_> {
    fn call_inplace(&self, _t: Self::T, y: &mut Self::V) {
        y.copy_from(self.init);
    }
}

pub struct PmRoot {
    nstates: usize,
    nout: usize,
    nparams: usize,
}

impl Op for PmRoot {
    type T = T;
    type V = V;
    type M = M;
    type C = C;
    fn nstates(&self) -> usize {
        self.nstates
    }
    fn nout(&self) -> usize {
        self.nout
    }
    fn nparams(&self) -> usize {
        self.nparams
    }
    fn context(&self) -> &Self::C {
        &NalgebraContext {}
    }
}

pub struct PmOut {
    nstates: usize,
    nout: usize,
    nparams: usize,
}

impl Op for PmOut {
    type T = T;
    type V = V;
    type M = M;
    type C = C;
    fn nstates(&self) -> usize {
        self.nstates
    }
    fn nout(&self) -> usize {
        self.nout
    }
    fn nparams(&self) -> usize {
        self.nparams
    }
    fn context(&self) -> &Self::C {
        &NalgebraContext {}
    }
}

impl<F> NonLinearOp for PmRhs<'_, F>
where
    F: Fn(&V, &V, T, &mut V, &V, &V, &Covariates),
{
    fn call_inplace(&self, x: &Self::V, t: Self::T, y: &mut Self::V) {
        let absolute_time = self.time_origin.get() + t;
        let mut rateiv_ref = self.rateiv_buffer.borrow_mut();
        self.infusion_schedule
            .fill_rate_vector(absolute_time, &mut rateiv_ref);

        (self.func)(
            x,
            self.p_as_v,
            absolute_time,
            y,
            self.zero_bolus,
            &rateiv_ref,
            self.covariates,
        );
    }
}

impl<F> NonLinearOpJacobian for PmRhs<'_, F>
where
    F: Fn(&V, &V, T, &mut V, &V, &V, &Covariates),
{
    fn jac_mul_inplace(&self, _x: &Self::V, t: Self::T, v: &Self::V, y: &mut Self::V) {
        let absolute_time = self.time_origin.get() + t;
        (self.func)(
            v,
            self.p_as_v,
            absolute_time,
            y,
            self.zero_bolus,
            self.zero_bolus,
            self.covariates,
        );
    }
}

impl LinearOp for PmMass {
    fn gemv_inplace(&self, _x: &Self::V, _t: Self::T, _beta: Self::T, _y: &mut Self::V) {}
}

impl NonLinearOp for PmRoot {
    fn call_inplace(&self, _x: &Self::V, _t: Self::T, _y: &mut Self::V) {}
}

impl NonLinearOp for PmOut {
    fn call_inplace(&self, _x: &Self::V, _t: Self::T, _y: &mut Self::V) {}
}

// Completely revised PMProblem to fix lifetime issues and improve performance
pub(crate) struct PMProblem<'a, F>
where
    F: Fn(&V, &V, T, &mut V, &V, &V, &Covariates) + 'a,
{
    func: F,
    nstates: usize,
    nparams: usize,
    init: V,
    p_as_v: V,
    zero_bolus: V,
    covariates: &'a Covariates,
    infusion_schedule: InfusionSchedule,
    time_origin: Cell<f64>,
    rateiv_buffer: RefCell<V>,
}

impl<'a, F> PMProblem<'a, F>
where
    F: Fn(&V, &V, T, &mut V, &V, &V, &Covariates) + 'a,
{
    /// Convert diffsol's local independent variable to model/data time.
    pub(crate) fn absolute_time(&self, solver_time: f64) -> f64 {
        self.time_origin.get() + solver_time
    }

    /// Return diffsol's current absolute-time origin.
    pub(crate) fn time_origin(&self) -> f64 {
        self.time_origin.get()
    }

    /// Start a new local coordinate at an accepted absolute-time state.
    pub(crate) fn rebase_time_origin(&self, absolute_time: f64) {
        self.time_origin.set(absolute_time);
    }

    pub(crate) fn set_left_continuity_time(&self, time: Option<f64>) {
        self.infusion_schedule.set_left_continuity_time(time);
    }

    pub(crate) fn infusion_boundary_times(&self) -> &[f64] {
        self.infusion_schedule.infusion_boundary_times()
    }

    pub(crate) fn infusion_amount_between(&self, from: f64, to: f64) -> f64 {
        self.infusion_schedule.infusion_amount_between(from, to)
    }

    /// Evaluate the full RHS (including the currently scheduled infusion
    /// rates) at local solver time `t` into `dx`.
    ///
    /// Used at infusion boundaries to refresh the solver's stored derivative
    /// against the post-boundary (right-continuous) RHS, so a solver restart
    /// predicts with the new dynamics instead of the pre-boundary ones.
    pub(crate) fn refresh_state_derivative(&self, t: f64, x: &V, dx: &mut V) {
        let absolute_time = self.absolute_time(t);
        let mut rateiv = self.rateiv_buffer.borrow_mut();
        self.infusion_schedule
            .fill_rate_vector(absolute_time, &mut rateiv);
        (self.func)(
            x,
            &self.p_as_v,
            absolute_time,
            dx,
            &self.zero_bolus,
            &rateiv,
            self.covariates,
        );
    }

    /// Creates a new PMProblem with a pre-converted parameter vector and an
    /// absolute-time origin for diffsol's local clock.
    ///
    /// This avoids an allocation when the caller already has a V representation
    /// while keeping all model callbacks in absolute time.
    #[allow(clippy::too_many_arguments)]
    pub fn with_params_v<'b, I>(
        func: F,
        nstates: usize,
        ndrugs: usize,
        p_as_v: V,
        covariates: &'a Covariates,
        infusions: I,
        init: V,
        time_origin: f64,
    ) -> Result<Self, PharmsolError>
    where
        I: IntoIterator<Item = &'b Infusion>,
    {
        if !time_origin.is_finite() {
            return Err(PharmsolError::OtherError(format!(
                "invalid ODE time origin {time_origin:?}: the resolved event schedule must be finite"
            )));
        }

        let nparams = p_as_v.len();
        let rateiv_buffer = RefCell::new(V::zeros(ndrugs, NalgebraContext::new()));
        let infusion_schedule = InfusionSchedule::new(ndrugs, infusions)?;
        // Pre-allocate zero bolus vector
        let zero_bolus = V::zeros(ndrugs, NalgebraContext::new());

        Ok(Self {
            func,
            nstates,
            nparams,
            init,
            p_as_v,
            zero_bolus,
            covariates,
            infusion_schedule,
            time_origin: Cell::new(time_origin),
            rateiv_buffer,
        })
    }
}

impl<'a, F> Op for PMProblem<'a, F>
where
    F: Fn(&V, &V, T, &mut V, &V, &V, &Covariates) + 'a,
{
    type T = T;
    type V = V;
    type M = M;
    type C = C;
    fn nstates(&self) -> usize {
        self.nstates
    }
    fn nout(&self) -> usize {
        self.nstates
    }
    fn nparams(&self) -> usize {
        self.nparams
    }
    fn context(&self) -> &Self::C {
        &NalgebraContext {}
    }
}

// Implement OdeEquationsRef for PMProblem for any lifetime 'b
impl<'a, 'b, F> OdeEquationsRef<'b> for PMProblem<'a, F>
where
    F: Fn(&V, &V, T, &mut V, &V, &V, &Covariates) + 'a,
{
    type Rhs = PmRhs<'b, F>;
    type Mass = PmMass;
    type Init = PmInit<'b>;
    type Root = PmRoot;
    type Out = PmOut;
    type Reset = UnitCallable<M>;
}

// Implement OdeEquations with correct lifetime handling
impl<'a, F> OdeEquations for PMProblem<'a, F>
where
    F: Fn(&V, &V, T, &mut V, &V, &V, &Covariates) + 'a,
{
    fn rhs(&self) -> PmRhs<'_, F> {
        PmRhs {
            nstates: self.nstates,
            nparams: self.nparams,
            infusion_schedule: &self.infusion_schedule,
            time_origin: &self.time_origin,
            covariates: self.covariates,
            p_as_v: &self.p_as_v,
            func: &self.func,
            rateiv_buffer: &self.rateiv_buffer,
            zero_bolus: &self.zero_bolus,
        }
    }

    fn mass(&self) -> Option<PmMass> {
        None
    }

    fn init(&self) -> PmInit<'_> {
        PmInit {
            nstates: self.nstates,
            nout: self.nstates,
            nparams: self.nparams,
            init: &self.init,
        }
    }

    fn get_params(&self, p: &mut V) {
        if p.len() == self.p_as_v.len() {
            p.copy_from(&self.p_as_v);
        } else {
            *p = self.p_as_v.clone();
        }
    }

    fn root(&self) -> Option<PmRoot> {
        None
    }

    fn out(&self) -> Option<PmOut> {
        None
    }

    fn reset(&self) -> Option<UnitCallable<M>> {
        None
    }

    fn set_params(&mut self, p: &V) {
        if self.p_as_v.len() == p.len() {
            self.p_as_v.copy_from(p);
        } else {
            self.p_as_v = p.clone();
        }
    }
}
