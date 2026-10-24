//! Analytical models written as Rust closures.

use pharmsol_dsl::ModelKind;

use super::{Analytical, AnalyticalModel};
use crate::simulator::equation::model::{check_dimensions, AttachMetadataError, Model};
use crate::simulator::equation::{ModelMetadata, ValidatedModelMetadata};
use crate::simulator::{AnalyticalEq, Fa, Init, Lag, Neqs, Out, SecEq, V};
use crate::{Covariates, PharmsolError};

/// An analytical model made of handwritten functions.
///
/// Build one through [`Analytical::new`] or the
/// [`analytical!`](crate::analytical!) macro.
#[derive(Clone, Debug)]
pub struct NativeAnalytical {
    eq: AnalyticalEq,
    seq_eq: SecEq,
    lag: Lag,
    fa: Fa,
    init: Init,
    out: Out,
    neqs: Neqs,
    metadata: Option<ValidatedModelMetadata>,
}

impl Model for NativeAnalytical {
    type Workspace = ();

    fn workspace(&self) {}

    fn nstates(&self) -> usize {
        self.neqs.nstates
    }

    fn ninputs(&self) -> usize {
        self.neqs.ndrugs
    }

    fn noutputs(&self) -> usize {
        self.neqs.nout
    }

    fn metadata(&self) -> Option<&ValidatedModelMetadata> {
        self.metadata.as_ref()
    }

    fn lag(
        &self,
        _ws: &mut (),
        input: usize,
        time: f64,
        p: &V,
        cov: &Covariates,
    ) -> Result<f64, PharmsolError> {
        Ok((self.lag)(p, time, cov).get(&input).copied().unwrap_or(0.0))
    }

    fn bioavailability(
        &self,
        _ws: &mut (),
        input: usize,
        time: f64,
        p: &V,
        cov: &Covariates,
    ) -> Result<f64, PharmsolError> {
        Ok((self.fa)(p, time, cov).get(&input).copied().unwrap_or(1.0))
    }

    fn init(
        &self,
        _ws: &mut (),
        time: f64,
        p: &V,
        cov: &Covariates,
        x: &mut V,
    ) -> Result<(), PharmsolError> {
        (self.init)(p, time, cov, x);
        Ok(())
    }

    /// Boluses land in the state with the same index as the input.
    fn bolus(
        &self,
        _ws: &mut (),
        _time: f64,
        x: &mut V,
        _p: &V,
        _cov: &Covariates,
        input: usize,
        amount: f64,
    ) -> Result<(), PharmsolError> {
        x[input] += amount;
        Ok(())
    }

    fn outputs(
        &self,
        _ws: &mut (),
        time: f64,
        x: &V,
        p: &V,
        cov: &Covariates,
        _rates: &V,
        y: &mut V,
    ) -> Result<(), PharmsolError> {
        (self.out)(x, p, time, cov, y);
        Ok(())
    }
}

impl AnalyticalModel for NativeAnalytical {
    fn step(
        &self,
        _ws: &mut (),
        time: f64,
        dt: f64,
        x: &mut V,
        p: &mut V,
        cov: &Covariates,
        rates: &V,
    ) -> Result<(), PharmsolError> {
        (self.seq_eq)(p, time, cov);
        *x = (self.eq)(x, p, dt, rates, cov);
        Ok(())
    }
}

impl Analytical<NativeAnalytical> {
    /// Build an analytical model from handwritten functions.
    ///
    /// Set the dimensions with [`with_nstates`](Self::with_nstates),
    /// [`with_ndrugs`](Self::with_ndrugs) and [`with_nout`](Self::with_nout).
    pub fn new(eq: AnalyticalEq, seq_eq: SecEq, lag: Lag, fa: Fa, init: Init, out: Out) -> Self {
        Analytical::from_model(NativeAnalytical {
            eq,
            seq_eq,
            lag,
            fa,
            init,
            out,
            neqs: Neqs::default(),
            metadata: None,
        })
    }

    /// Set the number of state variables. Clears attached metadata.
    pub fn with_nstates(mut self, nstates: usize) -> Self {
        self.model_mut().neqs.nstates = nstates;
        self.model_mut().metadata = None;
        self
    }

    /// Set the number of drug inputs (size of `rateiv`). Clears attached
    /// metadata.
    pub fn with_ndrugs(mut self, ndrugs: usize) -> Self {
        self.model_mut().neqs.ndrugs = ndrugs;
        self.model_mut().metadata = None;
        self
    }

    /// Set the number of output equations. Clears attached metadata.
    pub fn with_nout(mut self, nout: usize) -> Self {
        self.model_mut().neqs.nout = nout;
        self.model_mut().metadata = None;
        self
    }

    /// Attach metadata, validated against this model's dimensions.
    pub fn with_metadata(mut self, metadata: ModelMetadata) -> Result<Self, AttachMetadataError> {
        let metadata = metadata.validate_for(ModelKind::Analytical)?;
        let neqs = &self.model.neqs;
        check_dimensions(&metadata, neqs.nstates, neqs.ndrugs, neqs.nout)?;
        self.model_mut().metadata = Some(metadata);
        Ok(self)
    }

    /// The attached metadata, if any.
    pub fn metadata(&self) -> Option<&ValidatedModelMetadata> {
        self.model.metadata.as_ref()
    }
}
