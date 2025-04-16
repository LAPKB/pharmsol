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
    infusions: &'a [&'a Infusion], // Change from Vec to slice reference
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
        // Compute rate IV at the current time
        let mut rateiv_ref = self.rateiv_buffer.borrow_mut();
        rateiv_ref.fill(0.0);

        for infusion in self.infusions {
            if t >= Self::T::from(infusion.time())
                && t <= Self::T::from(infusion.duration() + infusion.time())
            {
                rateiv_ref[infusion.input()] +=
                    Self::T::from(infusion.amount() / infusion.duration());
            }
        }

        // We need to drop the mutable borrow before calling the function
        // to avoid potential conflicts with future borrows in the function
        let rateiv = rateiv_ref.clone();
        drop(rateiv_ref);

        // Avoid creating a new DVector when possible
        let p_len = self.p.len();
        let mut p_dvector: DVector<f64>;
        let p_ref: &DVector<f64>;

        // Use stack allocation for small parameter vectors
        if p_len <= 16 {
            let mut stack_p = [0.0; 16];
            stack_p[..p_len].copy_from_slice(&self.p);
            p_dvector = DVector::from_row_slice(&stack_p[..p_len]);
            p_ref = &p_dvector;
        } else {
            // For larger vectors, use the more efficient approach with unsafe
            p_dvector = DVector::zeros(p_len);
            unsafe {
                std::ptr::copy_nonoverlapping(self.p.as_ptr(), p_dvector.as_mut_ptr(), p_len);
            }
            p_ref = &p_dvector;
        }

        (self.func)(x, p_ref, t, y, rateiv, self.covariates)
    }
}

impl<'a, F> NonLinearOpJacobian for PmRhs<'a, F>
where
    F: Fn(&V, &V, T, &mut V, V, &Covariates),
{
    fn jac_mul_inplace(&self, _x: &Self::V, t: Self::T, v: &Self::V, y: &mut Self::V) {
        let rateiv = V::zeros(self.nstates);

        // Avoid creating a new DVector when possible
        let p_len = self.p.len();
        let mut p_dvector: DVector<f64>;

        // Use stack allocation for small parameter vectors
        if p_len <= 16 {
            let mut stack_p = [0.0; 16];
            stack_p[..p_len].copy_from_slice(&self.p);
            p_dvector = DVector::from_row_slice(&stack_p[..p_len]);
        } else {
            // For larger vectors, use the more efficient approach with unsafe
            p_dvector = DVector::zeros(p_len);
            unsafe {
                std::ptr::copy_nonoverlapping(self.p.as_ptr(), p_dvector.as_mut_ptr(), p_len);
            }
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

impl<'a, F> Op for PMProblem<'a, F>
where
    F: Fn(&V, &V, T, &mut V, V, &Covariates) + 'a,
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
