//! ODE models written as Rust closures.

use diffsol::{NalgebraContext, Vector};
use pharmsol_dsl::ModelKind;

use super::{OdeModel, ODE};
use crate::simulator::equation::model::{check_dimensions, AttachMetadataError, Model};
use crate::simulator::equation::{ModelMetadata, ValidatedModelMetadata};
use crate::simulator::{DiffEq, Fa, Init, Lag, Neqs, Out, V};
use crate::{Covariates, PharmsolError};

/// An ODE model made of handwritten functions.
///
/// Build one through [`ODE::new`] or the [`ode!`](crate::ode!) macro.
#[derive(Clone, Debug)]
pub struct NativeOde {
    diffeq: DiffEq,
    lag: Lag,
    fa: Fa,
    init: Init,
    out: Out,
    neqs: Neqs,
    metadata: Option<ValidatedModelMetadata>,
}

/// Scratch vectors for one [`NativeOde`] occasion.
#[derive(Debug)]
pub struct NativeOdeWorkspace {
    zero_inputs: V,
    bolus: V,
    dx_with_bolus: V,
    dx_without_bolus: V,
}

impl Model for NativeOde {
    type Workspace = NativeOdeWorkspace;

    fn workspace(&self) -> NativeOdeWorkspace {
        let ctx = NalgebraContext::new();
        NativeOdeWorkspace {
            zero_inputs: V::zeros(self.neqs.ndrugs, ctx),
            bolus: V::zeros(self.neqs.ndrugs, ctx),
            dx_with_bolus: V::zeros(self.neqs.nstates, ctx),
            dx_without_bolus: V::zeros(self.neqs.nstates, ctx),
        }
    }

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
        _ws: &mut NativeOdeWorkspace,
        input: usize,
        time: f64,
        p: &V,
        cov: &Covariates,
    ) -> Result<f64, PharmsolError> {
        Ok((self.lag)(p, time, cov).get(&input).copied().unwrap_or(0.0))
    }

    fn bioavailability(
        &self,
        _ws: &mut NativeOdeWorkspace,
        input: usize,
        time: f64,
        p: &V,
        cov: &Covariates,
    ) -> Result<f64, PharmsolError> {
        Ok((self.fa)(p, time, cov).get(&input).copied().unwrap_or(1.0))
    }

    fn init(
        &self,
        _ws: &mut NativeOdeWorkspace,
        time: f64,
        p: &V,
        cov: &Covariates,
        x: &mut V,
    ) -> Result<(), PharmsolError> {
        (self.init)(p, time, cov, x);
        Ok(())
    }

    /// A handwritten `diffeq` reads boluses from its `bolus` argument and may
    /// scale them (`bolus[oral] * f`). The state jump is the change in the
    /// derivative that the bolus causes.
    fn bolus(
        &self,
        ws: &mut NativeOdeWorkspace,
        time: f64,
        x: &mut V,
        p: &V,
        cov: &Covariates,
        input: usize,
        amount: f64,
    ) -> Result<(), PharmsolError> {
        ws.bolus.fill(0.0);
        ws.bolus[input] = amount;
        ws.dx_with_bolus.fill(0.0);
        ws.dx_without_bolus.fill(0.0);

        (self.diffeq)(
            x,
            p,
            time,
            &mut ws.dx_without_bolus,
            &ws.zero_inputs,
            &ws.zero_inputs,
            cov,
        );
        (self.diffeq)(
            x,
            p,
            time,
            &mut ws.dx_with_bolus,
            &ws.bolus,
            &ws.zero_inputs,
            cov,
        );

        ws.dx_with_bolus.axpy(-1.0, &ws.dx_without_bolus, 1.0);
        x.axpy(1.0, &ws.dx_with_bolus, 1.0);
        Ok(())
    }

    fn outputs(
        &self,
        _ws: &mut NativeOdeWorkspace,
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

impl OdeModel for NativeOde {
    fn rhs(
        &self,
        ws: &mut NativeOdeWorkspace,
        time: f64,
        x: &V,
        p: &V,
        cov: &Covariates,
        rates: &V,
        dx: &mut V,
    ) -> Result<(), PharmsolError> {
        (self.diffeq)(x, p, time, dx, &ws.zero_inputs, rates, cov);
        Ok(())
    }
}

impl ODE<NativeOde> {
    /// Build an ODE from handwritten functions.
    ///
    /// Set the dimensions with [`with_nstates`](Self::with_nstates),
    /// [`with_ndrugs`](Self::with_ndrugs) and [`with_nout`](Self::with_nout).
    pub fn new(diffeq: DiffEq, lag: Lag, fa: Fa, init: Init, out: Out) -> Self {
        ODE::from_model(NativeOde {
            diffeq,
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
        self.model.neqs.nstates = nstates;
        self.model.metadata = None;
        self
    }

    /// Set the number of drug inputs (size of `bolus` and `rateiv`). Clears
    /// attached metadata.
    pub fn with_ndrugs(mut self, ndrugs: usize) -> Self {
        self.model.neqs.ndrugs = ndrugs;
        self.model.metadata = None;
        self
    }

    /// Set the number of output equations. Clears attached metadata.
    pub fn with_nout(mut self, nout: usize) -> Self {
        self.model.neqs.nout = nout;
        self.model.metadata = None;
        self
    }

    /// Attach metadata, validated against this model's dimensions.
    pub fn with_metadata(mut self, metadata: ModelMetadata) -> Result<Self, AttachMetadataError> {
        let metadata = metadata.validate_for(ModelKind::Ode)?;
        let neqs = &self.model.neqs;
        check_dimensions(&metadata, neqs.nstates, neqs.ndrugs, neqs.nout)?;
        self.model.metadata = Some(metadata);
        Ok(self)
    }

    /// The attached metadata, if any.
    pub fn metadata(&self) -> Option<&ValidatedModelMetadata> {
        self.model.metadata.as_ref()
    }
}
