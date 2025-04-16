use crate::{Covariates, Infusion};
use diffsol::{
    ConstantOp, LinearOp, NonLinearOp, NonLinearOpJacobian, OdeEquations, OdeEquationsRef, Op,
};
use nalgebra::DVector;
type T = f64;
type V = nalgebra::DVector<f64>;
type M = nalgebra::DMatrix<f64>;

pub struct PmRhs<'a, F>
where
    F: Fn(&V, &V, T, &mut V, V, &Covariates),
{
    nstates: usize,
    nparams: usize,
    infusions: Vec<&'a Infusion>,
    covariates: &'a Covariates,
    p: &'a Vec<f64>,
    func: &'a F,
}

impl<'a, F> Op for PmRhs<'a, F>
where
    F: Fn(&V, &V, T, &mut V, V, &Covariates),
{
    type T = T;
    type V = V;
    type M = M;
    fn nstates(&self) -> usize {
        self.nstates
    }
    fn nout(&self) -> usize {
        self.nstates
    }
    fn nparams(&self) -> usize {
        self.nparams
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
    fn nstates(&self) -> usize {
        self.nstates
    }
    fn nout(&self) -> usize {
        self.nout
    }
    fn nparams(&self) -> usize {
        self.nparams
    }
}

pub struct PmInit {
    nstates: usize,
    nout: usize,
    nparams: usize,
    init: V,
}

impl Op for PmInit {
    type T = T;
    type V = V;
    type M = M;
    fn nstates(&self) -> usize {
        self.nstates
    }
    fn nout(&self) -> usize {
        self.nout
    }
    fn nparams(&self) -> usize {
        self.nparams
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
    fn nstates(&self) -> usize {
        self.nstates
    }
    fn nout(&self) -> usize {
        self.nout
    }
    fn nparams(&self) -> usize {
        self.nparams
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
    fn nstates(&self) -> usize {
        self.nstates
    }
    fn nout(&self) -> usize {
        self.nout
    }
    fn nparams(&self) -> usize {
        self.nparams
    }
}

impl<'a, F> NonLinearOp for PmRhs<'a, F>
where
    F: Fn(&V, &V, T, &mut V, V, &Covariates),
{
    fn call_inplace(&self, x: &Self::V, t: Self::T, y: &mut Self::V) {
        let mut rateiv = Self::V::zeros(self.nstates);
        //TODO: This should be pre-calculated
        for infusion in &self.infusions {
            if t >= Self::T::from(infusion.time())
                && t <= Self::T::from(infusion.duration() + infusion.time())
            {
                rateiv[infusion.input()] += Self::T::from(infusion.amount() / infusion.duration());
            }
        }
        // self.statistics.borrow_mut().increment_call();
        let p = DVector::from_vec(self.p.clone());
        (self.func)(x, &p, t, y, rateiv, &self.covariates)
    }
}

impl<'a, F> NonLinearOpJacobian for PmRhs<'a, F>
where
    F: Fn(&V, &V, T, &mut V, V, &Covariates),
{
    fn jac_mul_inplace(&self, _x: &Self::V, t: Self::T, v: &Self::V, y: &mut Self::V) {
        let rateiv = Self::V::zeros(self.nstates);
        let p = DVector::from_vec(self.p.clone());
        (self.func)(v, &p, t, y, rateiv, &self.covariates);
    }
}

impl LinearOp for PmMass {
    fn gemv_inplace(&self, _x: &Self::V, _t: Self::T, _beta: Self::T, _y: &mut Self::V) {}
}

impl ConstantOp for PmInit {
    fn call_inplace(&self, _t: Self::T, y: &mut Self::V) {
        y.copy_from(&self.init);
    }
}

impl NonLinearOp for PmRoot {
    fn call_inplace(&self, _x: &Self::V, _t: Self::T, _y: &mut Self::V) {}
}

impl NonLinearOp for PmOut {
    fn call_inplace(&self, _x: &Self::V, _t: Self::T, _y: &mut Self::V) {}
}

// Completely revised PMProblem to fix lifetime issues
pub struct PMProblem<F>
where
    F: Fn(&V, &V, T, &mut V, V, &Covariates) + 'static,
{
    func: F,
    nstates: usize,
    nparams: usize,
    init: V,
    p: Vec<f64>,
    // Store owned copies to avoid lifetime issues
    covariates: Covariates,
    infusions: Vec<Infusion>,
}

impl<F> PMProblem<F>
where
    F: Fn(&V, &V, T, &mut V, V, &Covariates) + 'static,
{
    pub fn new(
        func: F,
        nstates: usize,
        p: Vec<f64>,
        covariates: &Covariates,
        infusions: Vec<&Infusion>,
        init: V,
    ) -> Self {
        let nparams = p.len();
        // Clone data to avoid lifetime issues
        let covariates = covariates.clone();
        let infusions = infusions.iter().map(|&i| i.clone()).collect();

        Self {
            func,
            nstates,
            nparams,
            init,
            p,
            covariates,
            infusions,
        }
    }
}

impl<F> Op for PMProblem<F>
where
    F: Fn(&V, &V, T, &mut V, V, &Covariates) + 'static,
{
    type T = T;
    type V = V;
    type M = M;
    fn nstates(&self) -> usize {
        self.nstates
    }
    fn nout(&self) -> usize {
        self.nstates
    }
    fn nparams(&self) -> usize {
        self.nparams
    }
}

// Implement OdeEquationsRef for PMProblem for any lifetime 'b
impl<'b, F> OdeEquationsRef<'b> for PMProblem<F>
where
    F: Fn(&V, &V, T, &mut V, V, &Covariates) + 'static,
{
    type Rhs = PmRhs<'b, F>;
    type Mass = PmMass;
    type Init = PmInit;
    type Root = PmRoot;
    type Out = PmOut;
}

// Implement OdeEquations with correct lifetime handling
impl<F> OdeEquations for PMProblem<F>
where
    F: Fn(&V, &V, T, &mut V, V, &Covariates) + 'static,
{
    fn rhs(&self) -> PmRhs<'_, F> {
        // Need to create references to the owned objects for this function call
        let infusion_refs: Vec<&Infusion> = self.infusions.iter().collect();

        PmRhs {
            nstates: self.nstates,
            nparams: self.nparams,
            infusions: infusion_refs,
            covariates: &self.covariates,
            p: &self.p,
            func: &self.func,
        }
    }

    fn mass(&self) -> Option<PmMass> {
        None
    }

    fn init(&self) -> PmInit {
        PmInit {
            nstates: self.nstates(),
            nout: self.nout(),
            nparams: self.nparams(),
            init: self.init.clone(),
        }
    }

    fn get_params(&self, p: &mut V) {
        p.copy_from(&DVector::from_vec(self.p.clone()));
    }

    fn root(&self) -> Option<PmRoot> {
        None
    }

    fn out(&self) -> Option<PmOut> {
        None
    }

    fn set_params(&mut self, p: &V) {
        self.p = p.iter().cloned().collect();
    }
}
