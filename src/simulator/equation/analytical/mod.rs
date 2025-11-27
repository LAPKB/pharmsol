pub mod one_compartment_models;
pub mod three_compartment_models;
pub mod two_compartment_models;

use std::collections::HashMap;
use std::marker::PhantomData;

use diffsol::{NalgebraContext, Vector, VectorHost};
pub use one_compartment_models::*;
pub use three_compartment_models::*;
pub use two_compartment_models::*;

use crate::PharmsolError;
use crate::{
    data::Covariates, simulator::*, Equation, EquationPriv, EquationTypes, Missing, Observation,
    Provided, Subject,
};
use cached::proc_macro::cached;
use cached::UnboundCache;

/// Model equation using analytical solutions.
///
/// This implementation uses closed-form analytical solutions for the model
/// equations rather than numerical integration.
#[repr(C)]
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
    /// For a more ergonomic API, consider using [`AnalyticalBuilder`] instead.
    ///
    /// # Parameters
    /// - `eq`: The analytical equation function
    /// - `seq_eq`: The secondary equation function
    /// - `lag`: The lag time function
    /// - `fa`: The fraction absorbed function
    /// - `init`: The initial state function
    /// - `out`: The output equation function
    /// - `neqs`: The number of states and output equations (can be a tuple or [`Neqs`])
    pub fn new(
        eq: AnalyticalEq,
        seq_eq: SecEq,
        lag: Lag,
        fa: Fa,
        init: Init,
        out: Out,
        neqs: impl Into<Neqs>,
    ) -> Self {
        Self {
            eq,
            seq_eq,
            lag,
            fa,
            init,
            out,
            neqs: neqs.into(),
        }
    }

    /// Returns a new [`AnalyticalBuilder`] for constructing an Analytical equation.
    ///
    /// # Example
    /// ```ignore
    /// use pharmsol::prelude::*;
    ///
    /// // Minimal builder - only required fields
    /// let analytical = Analytical::builder()
    ///     .eq(one_compartment)
    ///     .seq_eq(|p, _t, _cov| {})
    ///     .out(|x, p, _t, _cov, y| { y[0] = x[0] / p[1]; })
    ///     .nstates(1)
    ///     .nouteqs(1)
    ///     .build();
    ///
    /// // With optional fields
    /// let analytical = Analytical::builder()
    ///     .eq(one_compartment)
    ///     .seq_eq(|p, _t, _cov| {})
    ///     .out(|x, p, _t, _cov, y| { y[0] = x[0] / p[1]; })
    ///     .nstates(1)
    ///     .nouteqs(1)
    ///     .lag(|p, _t, _cov| lag! { 0 => p[2] })
    ///     .fa(|p, _t, _cov| fa! { 0 => 0.8 })
    ///     .init(|p, _t, _cov, x| { x[0] = p[3]; })
    ///     .build();
    /// ```
    pub fn builder() -> AnalyticalBuilder<Missing, Missing, Missing, Missing, Missing> {
        AnalyticalBuilder::new()
    }
}

// =============================================================================
// Type-State Builder Pattern
// =============================================================================

// Note: Missing and Provided marker types are defined in the parent module
// and imported via `use crate::{..., Missing, Provided, ...}`

/// Builder for constructing [`Analytical`] equations with compile-time validation.
///
/// This builder uses the type-state pattern to ensure all required fields
/// are set before `build()` can be called. Optional fields (`lag`, `fa`, `init`)
/// have sensible defaults.
///
/// # Required Fields (enforced at compile time)
/// - `eq`: The analytical equation function
/// - `seq_eq`: The secondary equation function
/// - `out`: Output equation function
/// - `nstates`: Number of state variables
/// - `nouteqs`: Number of output equations
///
/// # Optional Fields (with defaults)
/// - `lag`: Lag time function (defaults to no lag)
/// - `fa`: Bioavailability function (defaults to 100% bioavailability)
/// - `init`: Initial state function (defaults to zero initial state)
///
/// # Example
/// ```ignore
/// use pharmsol::prelude::*;
///
/// // Minimal example - only required fields
/// let analytical = Analytical::builder()
///     .eq(one_compartment)
///     .seq_eq(|p, _t, _cov| {})
///     .out(|x, p, _t, _cov, y| { y[0] = x[0] / p[1]; })
///     .nstates(1)
///     .nouteqs(1)
///     .build();
/// ```
pub struct AnalyticalBuilder<EqState, SeqEqState, OutState, NStatesState, NOuteqsState> {
    eq: Option<AnalyticalEq>,
    seq_eq: Option<SecEq>,
    lag: Option<Lag>,
    fa: Option<Fa>,
    init: Option<Init>,
    out: Option<Out>,
    nstates: Option<usize>,
    nouteqs: Option<usize>,
    _phantom: PhantomData<(EqState, SeqEqState, OutState, NStatesState, NOuteqsState)>,
}

impl AnalyticalBuilder<Missing, Missing, Missing, Missing, Missing> {
    /// Creates a new AnalyticalBuilder with all required fields unset.
    pub fn new() -> Self {
        Self {
            eq: None,
            seq_eq: None,
            lag: None,
            fa: None,
            init: None,
            out: None,
            nstates: None,
            nouteqs: None,
            _phantom: PhantomData,
        }
    }
}

impl Default for AnalyticalBuilder<Missing, Missing, Missing, Missing, Missing> {
    fn default() -> Self {
        Self::new()
    }
}

impl<EqState, SeqEqState, OutState, NStatesState, NOuteqsState>
    AnalyticalBuilder<EqState, SeqEqState, OutState, NStatesState, NOuteqsState>
{
    /// Sets the lag time function (optional).
    ///
    /// If not set, defaults to no lag for any compartment.
    pub fn lag(mut self, lag: Lag) -> Self {
        self.lag = Some(lag);
        self
    }

    /// Sets the bioavailability function (optional).
    ///
    /// If not set, defaults to 100% bioavailability for all compartments.
    pub fn fa(mut self, fa: Fa) -> Self {
        self.fa = Some(fa);
        self
    }

    /// Sets the initial state function (optional).
    ///
    /// If not set, defaults to zero initial state for all compartments.
    pub fn init(mut self, init: Init) -> Self {
        self.init = Some(init);
        self
    }
}

impl<SeqEqState, OutState, NStatesState, NOuteqsState>
    AnalyticalBuilder<Missing, SeqEqState, OutState, NStatesState, NOuteqsState>
{
    /// Sets the analytical equation function (required).
    pub fn eq(
        self,
        eq: AnalyticalEq,
    ) -> AnalyticalBuilder<Provided, SeqEqState, OutState, NStatesState, NOuteqsState> {
        AnalyticalBuilder {
            eq: Some(eq),
            seq_eq: self.seq_eq,
            lag: self.lag,
            fa: self.fa,
            init: self.init,
            out: self.out,
            nstates: self.nstates,
            nouteqs: self.nouteqs,
            _phantom: PhantomData,
        }
    }
}

impl<EqState, OutState, NStatesState, NOuteqsState>
    AnalyticalBuilder<EqState, Missing, OutState, NStatesState, NOuteqsState>
{
    /// Sets the secondary equation function (required).
    ///
    /// This function is used to update parameters at each time step.
    pub fn seq_eq(
        self,
        seq_eq: SecEq,
    ) -> AnalyticalBuilder<EqState, Provided, OutState, NStatesState, NOuteqsState> {
        AnalyticalBuilder {
            eq: self.eq,
            seq_eq: Some(seq_eq),
            lag: self.lag,
            fa: self.fa,
            init: self.init,
            out: self.out,
            nstates: self.nstates,
            nouteqs: self.nouteqs,
            _phantom: PhantomData,
        }
    }
}

impl<EqState, SeqEqState, NStatesState, NOuteqsState>
    AnalyticalBuilder<EqState, SeqEqState, Missing, NStatesState, NOuteqsState>
{
    /// Sets the output equation function (required).
    pub fn out(
        self,
        out: Out,
    ) -> AnalyticalBuilder<EqState, SeqEqState, Provided, NStatesState, NOuteqsState> {
        AnalyticalBuilder {
            eq: self.eq,
            seq_eq: self.seq_eq,
            lag: self.lag,
            fa: self.fa,
            init: self.init,
            out: Some(out),
            nstates: self.nstates,
            nouteqs: self.nouteqs,
            _phantom: PhantomData,
        }
    }
}

impl<EqState, SeqEqState, OutState, NOuteqsState>
    AnalyticalBuilder<EqState, SeqEqState, OutState, Missing, NOuteqsState>
{
    /// Sets the number of state variables (compartments) (required).
    pub fn nstates(
        self,
        nstates: usize,
    ) -> AnalyticalBuilder<EqState, SeqEqState, OutState, Provided, NOuteqsState> {
        AnalyticalBuilder {
            eq: self.eq,
            seq_eq: self.seq_eq,
            lag: self.lag,
            fa: self.fa,
            init: self.init,
            out: self.out,
            nstates: Some(nstates),
            nouteqs: self.nouteqs,
            _phantom: PhantomData,
        }
    }
}

impl<EqState, SeqEqState, OutState, NStatesState>
    AnalyticalBuilder<EqState, SeqEqState, OutState, NStatesState, Missing>
{
    /// Sets the number of output equations (required).
    pub fn nouteqs(
        self,
        nouteqs: usize,
    ) -> AnalyticalBuilder<EqState, SeqEqState, OutState, NStatesState, Provided> {
        AnalyticalBuilder {
            eq: self.eq,
            seq_eq: self.seq_eq,
            lag: self.lag,
            fa: self.fa,
            init: self.init,
            out: self.out,
            nstates: self.nstates,
            nouteqs: Some(nouteqs),
            _phantom: PhantomData,
        }
    }
}

impl<EqState, SeqEqState, OutState>
    AnalyticalBuilder<EqState, SeqEqState, OutState, Missing, Missing>
{
    /// Sets both nstates and nouteqs from a [`Neqs`] struct or tuple (required).
    pub fn neqs(
        self,
        neqs: impl Into<Neqs>,
    ) -> AnalyticalBuilder<EqState, SeqEqState, OutState, Provided, Provided> {
        let neqs = neqs.into();
        AnalyticalBuilder {
            eq: self.eq,
            seq_eq: self.seq_eq,
            lag: self.lag,
            fa: self.fa,
            init: self.init,
            out: self.out,
            nstates: Some(neqs.nstates),
            nouteqs: Some(neqs.nouteqs),
            _phantom: PhantomData,
        }
    }
}

/// Default lag function: no lag for any compartment
fn default_lag(_p: &V, _t: f64, _cov: &Covariates) -> HashMap<usize, f64> {
    HashMap::new()
}

/// Default fa function: 100% bioavailability for all compartments
fn default_fa(_p: &V, _t: f64, _cov: &Covariates) -> HashMap<usize, f64> {
    HashMap::new()
}

/// Default init function: zero initial state
fn default_init(_p: &V, _t: f64, _cov: &Covariates, _x: &mut V) {
    // State is already zero-initialized
}

impl AnalyticalBuilder<Provided, Provided, Provided, Provided, Provided> {
    /// Builds the [`Analytical`] equation.
    ///
    /// This method is only available when all required fields have been set:
    /// - `eq`
    /// - `seq_eq`
    /// - `out`
    /// - `nstates`
    /// - `nouteqs`
    ///
    /// Optional fields use defaults if not set:
    /// - `lag`: No lag (empty HashMap)
    /// - `fa`: 100% bioavailability (empty HashMap)
    /// - `init`: Zero initial state
    pub fn build(self) -> Analytical {
        Analytical {
            eq: self.eq.unwrap(),
            seq_eq: self.seq_eq.unwrap(),
            lag: self.lag.unwrap_or(default_lag),
            fa: self.fa.unwrap_or(default_fa),
            init: self.init.unwrap_or(default_init),
            out: self.out.unwrap(),
            neqs: Neqs::new(self.nstates.unwrap(), self.nouteqs.unwrap()),
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

    // #[inline(always)]
    // fn get_lag(&self, spp: &[f64]) -> Option<HashMap<usize, f64>> {
    //     Some((self.lag)(&V::from_vec(spp.to_owned())))
    // }

    // #[inline(always)]
    // fn get_fa(&self, spp: &[f64]) -> Option<HashMap<usize, f64>> {
    //     Some((self.fa)(&V::from_vec(spp.to_owned())))
    // }

    #[inline(always)]
    fn lag(&self) -> &Lag {
        &self.lag
    }

    #[inline(always)]
    fn fa(&self) -> &Fa {
        &self.fa
    }

    #[inline(always)]
    fn get_nstates(&self) -> usize {
        self.neqs.nstates
    }

    #[inline(always)]
    fn get_nouteqs(&self) -> usize {
        self.neqs.nouteqs
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

        // 1) Build and sort event times
        let mut ts = Vec::new();
        ts.push(ti);
        ts.push(tf);
        for inf in infusions {
            let t0 = inf.time();
            let t1 = t0 + inf.duration();
            if t0 > ti && t0 < tf {
                ts.push(t0)
            }
            if t1 > ti && t1 < tf {
                ts.push(t1)
            }
        }
        ts.sort_by(|a, b| a.partial_cmp(b).unwrap());
        ts.dedup_by(|a, b| (*a - *b).abs() < 1e-12);

        // 2) March over each sub-interval
        let mut current_t = ts[0];
        let mut sp = V::from_vec(support_point.to_owned(), NalgebraContext);
        let mut rateiv = V::zeros(self.get_nstates(), NalgebraContext);

        for &next_t in &ts[1..] {
            // prepare support and infusion rate for [current_t .. next_t]
            rateiv.fill(0.0);
            for inf in infusions {
                let s = inf.time();
                let e = s + inf.duration();
                if current_t >= s && next_t <= e {
                    rateiv[inf.input()] += inf.amount() / inf.duration();
                }
            }

            // advance the support-point to next_t
            (self.seq_eq)(&mut sp, next_t, covariates);

            // advance state by dt
            let dt = next_t - current_t;
            *x = (self.eq)(x, &sp, dt, rateiv.clone(), covariates);

            current_t = next_t;
        }

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
        let mut y = V::zeros(self.get_nouteqs(), NalgebraContext);
        let out = &self.out;
        (out)(
            x,
            &V::from_vec(support_point.clone(), NalgebraContext),
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
        let mut x = V::zeros(self.get_nstates(), NalgebraContext);
        if occasion_index == 0 {
            (init)(
                &V::from_vec(spp.to_vec(), NalgebraContext),
                0.0,
                covariates,
                &mut x,
            );
        }
        x
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SubjectBuilderExt;
    use std::collections::HashMap;

    #[test]
    fn secondary_equations_accumulate_within_single_solve() {
        let eq = |x: &V, p: &V, dt: f64, _rateiv: V, _cov: &Covariates| {
            let mut next = x.clone();
            next[0] += p[0] * dt;
            next
        };
        let seq_eq = |params: &mut V, _t: f64, _cov: &Covariates| {
            params[0] += 1.0;
        };
        let lag = |_p: &V, _t: f64, _cov: &Covariates| HashMap::new();
        let fa = |_p: &V, _t: f64, _cov: &Covariates| HashMap::new();
        let init = |_p: &V, _t: f64, _cov: &Covariates, x: &mut V| {
            x.fill(0.0);
        };
        let out = |x: &V, _p: &V, _t: f64, _cov: &Covariates, y: &mut V| {
            y[0] = x[0];
        };

        let analytical = Analytical::new(eq, seq_eq, lag, fa, init, out, (1, 1));
        let subject = Subject::builder("seq")
            .bolus(0.0, 0.0, 0)
            .infusion(0.25, 1.0, 0, 0.25)
            .observation(1.0, 0.0, 0)
            .build();

        let predictions = analytical
            .estimate_predictions(&subject, &vec![1.0])
            .unwrap();

        let value = predictions.predictions()[0].prediction();
        assert!((value - 2.5).abs() < 1e-12);
    }

    #[test]
    fn infusion_inputs_match_state_dimension() {
        let eq = |x: &V, _p: &V, dt: f64, rateiv: V, _cov: &Covariates| {
            let mut next = x.clone();
            next[0] += rateiv[3] * dt;
            next
        };
        let seq_eq = |_params: &mut V, _t: f64, _cov: &Covariates| {};
        let lag = |_p: &V, _t: f64, _cov: &Covariates| HashMap::new();
        let fa = |_p: &V, _t: f64, _cov: &Covariates| HashMap::new();
        let init = |_p: &V, _t: f64, _cov: &Covariates, x: &mut V| {
            x.fill(0.0);
        };
        let out = |x: &V, _p: &V, _t: f64, _cov: &Covariates, y: &mut V| {
            y[0] = x[0];
        };

        let analytical = Analytical::new(eq, seq_eq, lag, fa, init, out, (4, 1));
        let subject = Subject::builder("inf")
            .infusion(0.0, 4.0, 3, 1.0)
            .observation(1.0, 0.0, 0)
            .build();

        let predictions = analytical
            .estimate_predictions(&subject, &vec![0.0])
            .unwrap();

        assert_eq!(predictions.predictions()[0].prediction(), 4.0);
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

    fn estimate_log_likelihood(
        &self,
        subject: &Subject,
        support_point: &Vec<f64>,
        error_models: &ErrorModels,
        cache: bool,
    ) -> Result<f64, PharmsolError> {
        let ypred = if cache {
            _subject_predictions(self, subject, support_point)
        } else {
            _subject_predictions_no_cache(self, subject, support_point)
        }?;
        ypred.log_likelihood(error_models)
    }

    fn kind() -> crate::EqnKind {
        crate::EqnKind::Analytical
    }
}

/// Hash support points to a u64 for cache key generation.
/// Uses DefaultHasher for good distribution and collision resistance.
#[inline(always)]
fn spphash(spp: &[f64]) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::hash::DefaultHasher::new();
    for &value in spp {
        // Normalize -0.0 to 0.0 for consistent hashing
        let bits = if value == 0.0 { 0u64 } else { value.to_bits() };
        bits.hash(&mut hasher);
    }
    hasher.finish()
}

#[inline(always)]
#[cached(
    ty = "UnboundCache<(u64, u64), SubjectPredictions>",
    create = "{ UnboundCache::with_capacity(100_000) }",
    convert = r#"{ (subject.hash(), spphash(support_point)) }"#,
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
