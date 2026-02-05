use crate::{Covariates, Infusion};
use diffsol::{
    ConstantOp, LinearOp, MatrixCommon, NalgebraContext, NalgebraMat, NonLinearOp,
    NonLinearOpJacobian, OdeEquations, OdeEquationsRef, Op, Vector, VectorCommon,
};
use nalgebra::DVector;
use std::{cell::RefCell, cmp::Ordering};
type M = NalgebraMat<f64>;
type V = <M as MatrixCommon>::V;
type C = <M as MatrixCommon>::C;
type T = <M as MatrixCommon>::T;

#[derive(Debug, Clone)]
struct InfusionChannel {
    input: usize,
    event_times: Vec<f64>,
    cumulative_rates: Vec<f64>,
}

impl InfusionChannel {
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

    fn rate_at(&self, time: f64) -> f64 {
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
    channels: Vec<InfusionChannel>,
}

impl InfusionSchedule {
    fn new(nstates: usize, infusions: &[&Infusion]) -> Self {
        if nstates == 0 || infusions.is_empty() {
            return Self {
                channels: Vec::new(),
            };
        }

        // Use nstates + 1 to support both 0-indexed and 1-indexed data
        let buffer_size = nstates + 1;
        let mut per_input: Vec<Vec<(f64, f64)>> = vec![Vec::new(); buffer_size];
        for infusion in infusions {
            if infusion.duration() <= 0.0 {
                continue;
            }

            let input = infusion.input();
            if input >= buffer_size {
                continue;
            }

            let rate = infusion.amount() / infusion.duration();
            per_input[input].push((infusion.time(), rate));
            per_input[input].push((infusion.time() + infusion.duration(), -rate));
        }

        let channels = per_input
            .into_iter()
            .enumerate()
            .filter_map(|(input, events)| {
                if events.is_empty() {
                    None
                } else {
                    Some(InfusionChannel::new(input, events))
                }
            })
            .collect();

        Self { channels }
    }

    fn fill_rate_vector(&self, time: f64, rateiv: &mut V) {
        rateiv.fill(0.0);
        for channel in &self.channels {
            let rate = channel.rate_at(time);
            if rate != 0.0 {
                rateiv[channel.input] = rate;
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
        &NalgebraContext
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
        &NalgebraContext
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
        &NalgebraContext
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
        &NalgebraContext
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
        &NalgebraContext
    }
}

impl<F> NonLinearOp for PmRhs<'_, F>
where
    F: Fn(&V, &V, T, &mut V, &V, &V, &Covariates),
{
    fn call_inplace(&self, x: &Self::V, t: Self::T, y: &mut Self::V) {
        let mut rateiv_ref = self.rateiv_buffer.borrow_mut();
        self.infusion_schedule.fill_rate_vector(t, &mut rateiv_ref);

        (self.func)(
            x,
            self.p_as_v,
            t,
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
        (self.func)(
            v,
            self.p_as_v,
            t,
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
pub struct PMProblem<'a, F>
where
    F: Fn(&V, &V, T, &mut V, &V, &V, &Covariates) + 'a,
{
    func: F,
    nstates: usize,
    nparams: usize,
    init: V,
    p: Vec<f64>,
    p_as_v: V,
    zero_bolus: V,
    covariates: &'a Covariates,
    infusion_schedule: InfusionSchedule,
    rateiv_buffer: RefCell<V>,
}

impl<'a, F> PMProblem<'a, F>
where
    F: Fn(&V, &V, T, &mut V, &V, &V, &Covariates) + 'a,
{
    /// Creates a new PMProblem with a pre-converted parameter vector.
    /// This avoids an allocation when the caller already has a V representation.
    pub fn with_params_v(
        func: F,
        nstates: usize,
        p: Vec<f64>,
        p_as_v: V,
        covariates: &'a Covariates,
        infusions: &[&'a Infusion],
        init: V,
    ) -> Self {
        let nparams = p.len();
        // Use nstates + 1 to support both 0-indexed and 1-indexed data
        let buffer_size = nstates + 1;
        let rateiv_buffer = RefCell::new(V::zeros(buffer_size, NalgebraContext));
        let infusion_schedule = InfusionSchedule::new(nstates, infusions);
        // Pre-allocate zero bolus vector
        let zero_bolus = V::zeros(buffer_size, NalgebraContext);

        Self {
            func,
            nstates,
            nparams,
            init,
            p,
            p_as_v,
            zero_bolus,
            covariates,
            infusion_schedule,
            rateiv_buffer,
        }
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
        &NalgebraContext
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
        // Avoid unnecessary cloning by directly copying values from self.p
        if p.len() == self.p.len() {
            for i in 0..self.p.len() {
                p[i] = self.p[i];
            }
        } else {
            p.copy_from(&DVector::from_vec(self.p.clone()).into());
        }
    }

    fn root(&self) -> Option<PmRoot> {
        None
    }

    fn out(&self) -> Option<PmOut> {
        None
    }

    fn set_params(&mut self, p: &V) {
        if self.p.len() == p.len() {
            for i in 0..p.len() {
                self.p[i] = p[i];
                self.p_as_v[i] = p[i];
            }
        } else {
            self.p = p.inner().iter().cloned().collect();
            self.p_as_v = p.clone();
        }
    }
}
