use crate::{Covariates, Infusion};
use diffsol::{
    ConstantOp, LinearOp, NonLinearOp, NonLinearOpJacobian, OdeEquations, OdeEquationsRef, Op,
};
use nalgebra::DVector;
use std::cell::RefCell;
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
    rateiv_buffer: &'a RefCell<V>,
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

pub struct PmInit<'a> {
    nstates: usize,
    nout: usize,
    nparams: usize,
    init: &'a V,
}

impl<'a> Op for PmInit<'a> {
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

impl<'a> ConstantOp for PmInit<'a> {
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
        {
            let mut rateiv = self.rateiv_buffer.borrow_mut();
            rateiv.fill(0.0);

            for infusion in &self.infusions {
                if t >= Self::T::from(infusion.time())
                    && t <= Self::T::from(infusion.duration() + infusion.time())
                {
                    rateiv[infusion.input()] +=
                        Self::T::from(infusion.amount() / infusion.duration());
                }
            }
        }

        let mut p_dvector = DVector::zeros(self.p.len());
        unsafe {
            std::ptr::copy_nonoverlapping(self.p.as_ptr(), p_dvector.as_mut_ptr(), self.p.len());
        }

        (self.func)(
            x,
            &p_dvector,
            t,
            y,
            self.rateiv_buffer.borrow().clone(),
            self.covariates,
        )
    }
}

impl<'a, F> NonLinearOpJacobian for PmRhs<'a, F>
where
    F: Fn(&V, &V, T, &mut V, V, &Covariates),
{
    fn jac_mul_inplace(&self, _x: &Self::V, t: Self::T, v: &Self::V, y: &mut Self::V) {
        let rateiv = V::zeros(self.nstates);

        let mut p_dvector = DVector::zeros(self.p.len());
        unsafe {
            std::ptr::copy_nonoverlapping(self.p.as_ptr(), p_dvector.as_mut_ptr(), self.p.len());
        }

        (self.func)(v, &p_dvector, t, y, rateiv, self.covariates);
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
pub struct PMProblem<F>
where
    F: Fn(&V, &V, T, &mut V, V, &Covariates) + 'static,
{
    func: F,
    nstates: usize,
    nparams: usize,
    init: V,
    p: Vec<f64>,
    covariates: Covariates,
    infusions: Vec<Infusion>,
    rateiv_buffer: RefCell<V>,
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
        let covariates = covariates.clone();
        let infusions: Vec<Infusion> = infusions.iter().map(|&i| i.clone()).collect();
        let rateiv_buffer = RefCell::new(V::zeros(nstates));

        Self {
            func,
            nstates,
            nparams,
            init,
            p,
            covariates,
            infusions,
            rateiv_buffer,
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
    type Init = PmInit<'b>;
    type Root = PmRoot;
    type Out = PmOut;
}

// Implement OdeEquations with correct lifetime handling
impl<F> OdeEquations for PMProblem<F>
where
    F: Fn(&V, &V, T, &mut V, V, &Covariates) + 'static,
{
    fn rhs(&self) -> PmRhs<'_, F> {
        let infusion_refs: Vec<&Infusion> = {
            let mut refs = Vec::with_capacity(self.infusions.len());
            for infusion in &self.infusions {
                refs.push(infusion);
            }
            refs
        };

        PmRhs {
            nstates: self.nstates,
            nparams: self.nparams,
            infusions: infusion_refs,
            covariates: &self.covariates,
            p: &self.p,
            func: &self.func,
            rateiv_buffer: &self.rateiv_buffer,
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
            p.copy_from(&DVector::from_vec(self.p.clone()));
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
            }
        } else {
            self.p = p.iter().cloned().collect();
        }
    }
}
