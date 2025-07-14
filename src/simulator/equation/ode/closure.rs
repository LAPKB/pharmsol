use crate::{Covariates, Infusion};
use diffsol::{
    ConstantOp, LinearOp, NonLinearOp, NonLinearOpJacobian, OdeEquations, OdeEquationsRef, Op,
    Vector,
};
use diffsol::{NalgebraContext, NalgebraMat, NalgebraVec};

use std::cell::RefCell;
pub type T = f64;
pub type V = NalgebraVec<T>;
pub type M = NalgebraMat<T>;
pub type C = NalgebraContext;

pub struct PmRhs<'a, F>
where
    F: Fn(&V, &V, T, &mut V, V, &Covariates),
{
    nstates: usize,
    nparams: usize,
    infusions: &'a [&'a Infusion], // Change from Vec to slice reference
    covariates: &'a Covariates,
    p: &'a Vec<f64>,
    func: &'a F,
    rateiv_buffer: &'a RefCell<V>,
}

impl<F> Op for PmRhs<'_, F>
where
    F: Fn(&V, &V, T, &mut V, V, &Covariates),
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
    F: Fn(&V, &V, T, &mut V, V, &Covariates),
{
    fn call_inplace(&self, x: &Self::V, t: Self::T, y: &mut Self::V) {
        // Compute rate IV at the current time
        let mut rateiv_ref = self.rateiv_buffer.borrow_mut();
        rateiv_ref.fill(0.0);

        for infusion in self.infusions {
            if t >= infusion.time() && t <= infusion.duration() + infusion.time() {
                rateiv_ref[infusion.input()] += infusion.amount() / infusion.duration();
            }
        }

        // We need to drop the mutable borrow before calling the function
        // to avoid potential conflicts with future borrows in the function
        let rateiv = rateiv_ref.clone();
        drop(rateiv_ref);

        let mut p = NalgebraVec::zeros(self.p.len(), NalgebraContext);
        for i in 0..self.p.len() {
            p[i] = self.p[i];
        }

        (self.func)(x, &p, t, y, rateiv, self.covariates)
    }
}

impl<F> NonLinearOpJacobian for PmRhs<'_, F>
where
    F: Fn(&V, &V, T, &mut V, V, &Covariates),
{
    fn jac_mul_inplace(&self, _x: &Self::V, t: Self::T, v: &Self::V, y: &mut Self::V) {
        let rateiv = V::zeros(self.nstates, NalgebraContext);

        let mut p = NalgebraVec::zeros(self.p.len(), NalgebraContext);
        for i in 0..self.p.len() {
            p[i] = self.p[i];
        }

        (self.func)(v, &p, t, y, rateiv, self.covariates);
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
    F: Fn(&V, &V, T, &mut V, V, &Covariates) + 'a,
{
    func: F,
    nstates: usize,
    nparams: usize,
    init: V,
    p: Vec<f64>,
    covariates: &'a Covariates,
    infusions: Vec<&'a Infusion>,
    rateiv_buffer: RefCell<V>,
}

impl<'a, F> PMProblem<'a, F>
where
    F: Fn(&V, &V, T, &mut V, V, &Covariates) + 'a,
{
    pub fn new(
        func: F,
        nstates: usize,
        p: Vec<f64>,
        covariates: &'a Covariates,
        infusions: Vec<&'a Infusion>,
        init: V,
    ) -> Self {
        let nparams = p.len();
        let rateiv_buffer = RefCell::new(V::zeros(nstates, NalgebraContext));

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

impl<'a, F> Op for PMProblem<'a, F>
where
    F: Fn(&V, &V, T, &mut V, V, &Covariates) + 'a,
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
    F: Fn(&V, &V, T, &mut V, V, &Covariates) + 'a,
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
    F: Fn(&V, &V, T, &mut V, V, &Covariates) + 'a,
{
    fn rhs(&self) -> PmRhs<'_, F> {
        PmRhs {
            nstates: self.nstates,
            nparams: self.nparams,
            infusions: &self.infusions, // Use reference instead of clone
            covariates: self.covariates,
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
            let p_vec = V::zeros(self.p.len(), NalgebraContext);
            p.copy_from(&p_vec);
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
            let mut p_new = vec![0.0; self.p.len()];
            for i in 0..self.p.len() {
                p_new[i] = p[i];
            }
            self.p = p_new;
        }
    }
}
