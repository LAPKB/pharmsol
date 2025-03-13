use diffsol::matrix::sparsity::MatrixSparsity;
use diffsol::op::nonlinear_op::NonLinearOp;
use diffsol::NonLinearOpJacobian;
use diffsol::{
    matrix::Matrix,
    op::{Op, OpStatistics},
    vector::Vector,
};

use std::{cell::RefCell, rc::Rc};

use crate::data::{Covariates, Infusion};

/// Closure wrapper for pharmacometric model ODE functions.
///
/// This structure adapts a pharmacometric model's differential equation function
/// to the interface expected by the diffsol ODE solver library. It handles the calculation
/// of infusion rates and provides access to covariates at each step of the integration.
///
/// # Type Parameters
///
/// * `M`: Matrix type used for linear algebra operations
/// * `F`: Type of the right-hand side function for the ODEs
pub(crate) struct PMClosure<M, F>
where
    M: Matrix,
    F: Fn(&M::V, &M::V, M::T, &mut M::V, M::V, &Covariates),
{
    func: F,
    nstates: usize,
    nout: usize,
    nparams: usize,
    p: Rc<M::V>,
    sparsity: Option<M::Sparsity>,
    statistics: RefCell<OpStatistics>,
    covariates: Covariates,
    infusions: Vec<Infusion>,
}

impl<M, F> PMClosure<M, F>
where
    M: Matrix,
    F: Fn(&M::V, &M::V, M::T, &mut M::V, M::V, &Covariates),
{
    /// Creates a new pharmacometric model closure.
    ///
    /// # Arguments
    ///
    /// * `func`: The differential equation function
    /// * `nstates`: Number of state variables in the system
    /// * `nout`: Number of output variables
    /// * `p`: Parameter vector for the model
    /// * `covariates`: Covariates that may influence the system
    /// * `infusions`: Vector of infusion events to be applied during simulation
    ///
    /// # Returns
    ///
    /// A configured closure ready to be used with diffsol
    pub(crate) fn new(
        func: F,
        nstates: usize,
        nout: usize,
        p: Rc<M::V>,
        covariates: Covariates,
        infusions: Vec<Infusion>,
    ) -> Self {
        let nparams = p.len();
        Self {
            func,
            nstates,
            nout,
            nparams,
            p,
            statistics: RefCell::new(OpStatistics::default()),
            sparsity: None,
            covariates,
            infusions,
        }
    }
}

/// Implementation of Op trait for PMClosure.
///
/// This provides information about the system dimensions and sparsity.
impl<M, F> Op for PMClosure<M, F>
where
    M: Matrix,
    F: Fn(&M::V, &M::V, M::T, &mut M::V, M::V, &Covariates),
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

/// Implementation of NonLinearOp for PMClosure.
///
/// This handles the evaluation of the right-hand side of the ODE system,
/// calculating infusion rates and applying the model function.
impl<M, F> NonLinearOp for PMClosure<M, F>
where
    M: Matrix,
    F: Fn(&M::V, &M::V, M::T, &mut M::V, M::V, &Covariates),
{
    fn call_inplace(&self, x: &M::V, t: M::T, y: &mut M::V) {
        let mut rateiv = Self::V::zeros(self.nstates);
        //TODO: This should be pre-calculated
        for infusion in &self.infusions {
            if t >= Self::T::from(infusion.time())
                && t <= Self::T::from(infusion.duration() + infusion.time())
            {
                rateiv[infusion.input()] += Self::T::from(infusion.amount() / infusion.duration());
            }
        }
        self.statistics.borrow_mut().increment_call();
        (self.func)(x, self.p.as_ref(), t, y, rateiv, &self.covariates)
    }
}

/// Implementation of NonLinearOpJacobian for PMClosure.
///
/// This handles the calculation of the Jacobian matrix action for implicit ODE solvers.
impl<M, F> NonLinearOpJacobian for PMClosure<M, F>
where
    M: Matrix,
    F: Fn(&M::V, &M::V, M::T, &mut M::V, M::V, &Covariates),
{
    fn jac_mul_inplace(&self, _x: &M::V, t: M::T, v: &M::V, y: &mut M::V) {
        let rateiv = Self::V::zeros(self.nstates);
        self.statistics.borrow_mut().increment_jac_mul();
        (self.func)(v, self.p.as_ref(), t, y, rateiv, &self.covariates);
        // (self.jacobian_action)(x, self.p.as_ref(), t, v, y)
    }
}
