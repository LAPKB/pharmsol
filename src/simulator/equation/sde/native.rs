//! SDE models written as Rust closures.

use pharmsol_dsl::ModelKind;

use super::{SdeModel, SDE};
use crate::simulator::equation::model::{check_dimensions, AttachMetadataError, Model};
use crate::simulator::equation::{ModelMetadata, ValidatedModelMetadata};
use crate::simulator::{Diffusion, Drift, Fa, Init, Lag, Neqs, Out, V};
use crate::{Covariates, PharmsolError};

/// An SDE model made of handwritten functions.
///
/// Build one through [`SDE::new`] or the [`sde!`](crate::sde!) macro.
#[derive(Clone, Debug)]
pub struct NativeSde {
    drift: Drift,
    diffusion: Diffusion,
    lag: Lag,
    fa: Fa,
    init: Init,
    out: Out,
    neqs: Neqs,
    metadata: Option<ValidatedModelMetadata>,
    /// Destination state of each bolus input, when the macro injects it.
    bolus_destinations: Vec<Option<usize>>,
}

impl NativeSde {
    fn clear_metadata(&mut self) {
        self.metadata = None;
        self.bolus_destinations = vec![None; self.neqs.ndrugs];
    }
}

impl Model for NativeSde {
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

    /// A bolus lands in its injected destination, or else in the state with
    /// the same index as the input.
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
        let destination = self
            .bolus_destinations
            .get(input)
            .copied()
            .flatten()
            .unwrap_or(input);
        x[destination] += amount;
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

impl SdeModel for NativeSde {
    fn drift(
        &self,
        _ws: &mut (),
        time: f64,
        x: &V,
        p: &V,
        cov: &Covariates,
        rates: &V,
        dx: &mut V,
    ) -> Result<(), PharmsolError> {
        (self.drift)(x, p, time, dx, rates, cov);
        Ok(())
    }

    fn diffusion(
        &self,
        _ws: &mut (),
        _time: f64,
        _x: &V,
        p: &V,
        _cov: &Covariates,
        _rates: &V,
        sigma: &mut V,
    ) -> Result<(), PharmsolError> {
        (self.diffusion)(p, sigma);
        Ok(())
    }
}

impl SDE<NativeSde> {
    /// Build an SDE from handwritten functions, simulated with `nparticles`
    /// particles.
    ///
    /// Set the dimensions with [`with_nstates`](Self::with_nstates),
    /// [`with_ndrugs`](Self::with_ndrugs) and [`with_nout`](Self::with_nout).
    pub fn new(
        drift: Drift,
        diffusion: Diffusion,
        lag: Lag,
        fa: Fa,
        init: Init,
        out: Out,
        nparticles: usize,
    ) -> Self {
        SDE::from_model(
            NativeSde {
                drift,
                diffusion,
                lag,
                fa,
                init,
                out,
                neqs: Neqs::default(),
                metadata: None,
                bolus_destinations: Vec::new(),
            },
            nparticles,
        )
    }

    /// Set the number of state variables. Clears attached metadata.
    pub fn with_nstates(mut self, nstates: usize) -> Self {
        self.model_mut().neqs.nstates = nstates;
        self.model_mut().clear_metadata();
        self
    }

    /// Set the number of drug inputs (size of `rateiv`). Clears attached
    /// metadata.
    pub fn with_ndrugs(mut self, ndrugs: usize) -> Self {
        self.model_mut().neqs.ndrugs = ndrugs;
        self.model_mut().clear_metadata();
        self
    }

    /// Set the number of output equations. Clears attached metadata.
    pub fn with_nout(mut self, nout: usize) -> Self {
        self.model_mut().neqs.nout = nout;
        self.model_mut().clear_metadata();
        self
    }

    /// Attach metadata, validated against this model's dimensions and
    /// particle count.
    pub fn with_metadata(mut self, metadata: ModelMetadata) -> Result<Self, AttachMetadataError> {
        let metadata = metadata.validate_for_with_particles(ModelKind::Sde, self.nparticles)?;
        let neqs = &self.model.neqs;
        check_dimensions(&metadata, neqs.nstates, neqs.ndrugs, neqs.nout)?;
        self.model_mut().metadata = Some(metadata);
        Ok(self)
    }

    /// Route each bolus input straight into a destination state (used by the
    /// `sde!` macro).
    #[doc(hidden)]
    pub fn with_injected_bolus_inputs(mut self, destinations: &[Option<usize>]) -> Self {
        let ndrugs = self.model.neqs.ndrugs;
        let mut mapped = vec![None; ndrugs];
        for (input, destination) in destinations.iter().copied().take(ndrugs).enumerate() {
            mapped[input] = destination;
        }
        self.model_mut().bolus_destinations = mapped;
        self
    }

    /// The attached metadata, if any.
    pub fn metadata(&self) -> Option<&ValidatedModelMetadata> {
        self.model.metadata.as_ref()
    }
}
