use diffsol::matrix::sparsity::MatrixSparsity;
use diffsol::{
    jacobian::JacobianColoring,
    matrix::Matrix,
    op::{NonLinearOp, Op, OpStatistics},
    vector::Vector,
};
use std::{cell::RefCell, rc::Rc};

use crate::data::{Covariates, Infusion};
pub(crate) struct ODENetClosure<M>
where
    M: Matrix,
{
    linear: Vec<M>,
    nstates: usize,
    nout: usize,
    nparams: usize,
    p: Rc<M::V>,
    coloring: Option<JacobianColoring<M>>,
    sparsity: Option<M::Sparsity>,
    statistics: RefCell<OpStatistics>,
    covariates: Covariates,
    infusions: Vec<Infusion>,
}

impl<M> ODENetClosure<M>
where
    M: Matrix,
{
    pub(crate) fn new(
        linear: Vec<M>,
        nstates: usize,
        nout: usize,
        p: Rc<M::V>,
        covariates: Covariates,
        infusions: Vec<Infusion>,
    ) -> Self {
        let nparams = p.len();
        Self {
            linear,
            nstates,
            nout,
            nparams,
            p,
            statistics: RefCell::new(OpStatistics::default()),
            coloring: None,
            sparsity: None,
            covariates,
            infusions,
        }
    }
}

impl<M> Op for ODENetClosure<M>
where
    M: Matrix,
{
    type V = M::V;
    type T = M::T;
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
    fn sparsity(&self) -> Option<<Self::M as Matrix>::SparsityRef<'_>> {
        self.sparsity.as_ref().map(|s| s.as_ref())
    }
    fn statistics(&self) -> OpStatistics {
        self.statistics.borrow().clone()
    }
}

impl<M> NonLinearOp for ODENetClosure<M>
where
    M: Matrix,
{
    fn call_inplace(&self, x: &M::V, t: M::T, y: &mut M::V) {
        // let mut rateiv = Self::V::zeros(self.nstates);
        //TODO: This should be pre-calculated
        for infusion in &self.infusions {
            if t >= Self::T::from(infusion.time())
                && t <= Self::T::from(infusion.duration() + infusion.time())
            {
                y[infusion.input()] += Self::T::from(infusion.amount() / infusion.duration());
            }
        }
        self.statistics.borrow_mut().increment_call();
        //TODO: len(p) == len(self.linear) == self.nparams
        // Perform a matrix-vector multiplication `y = alpha * self * x + beta * y`.
        for i in 0..self.nparams {
            self.linear[i].gemv(self.p[i], x, Self::T::from(1.0), y);
        }
        // y += rateiv;
        // (self.func)(x, self.p.as_ref(), t, y, rateiv, &self.covariates)
    }
    fn jac_mul_inplace(&self, _x: &M::V, _t: M::T, v: &M::V, y: &mut M::V) {
        self.statistics.borrow_mut().increment_jac_mul();
        for i in 0..self.nparams {
            self.linear[i].gemv(self.p[i], v, M::T::from(1.0), y);
        }
        // y =
        // (self.func)(v, self.p.as_ref(), t, y, rateiv, &self.covariates);
        // (self.jacobian_action)(x, self.p.as_ref(), t, v, y)
    }
    fn jacobian_inplace(&self, x: &Self::V, t: Self::T, y: &mut Self::M) {
        self.statistics.borrow_mut().increment_matrix();
        if let Some(coloring) = self.coloring.as_ref() {
            coloring.jacobian_inplace(self, x, t, y);
        } else {
            self._default_jacobian_inplace(x, t, y);
        }
    }
}
