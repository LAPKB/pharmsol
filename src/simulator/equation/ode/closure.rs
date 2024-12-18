use crate::{Covariates, Infusion, SupportPoint};
use diffsol::{
    ConstantOp, LinearOp, NonLinearOp, NonLinearOpJacobian, OdeEquations, OdeEquationsRef, Op,
};
type T = f64;
type V = nalgebra::DVector<f64>;
type M = nalgebra::DMatrix<f64>;
pub struct PmRhs<'a, F>
where
    F: Fn(&V, &SupportPoint, T, &mut V, V, &Covariates),
{
    nstates: &'a usize,
    nout: &'a usize,
    nparams: &'a usize,
    infusions: &'a Vec<Infusion>,
    covariates: &'a Covariates,
    p: &'a SupportPoint,
    func: &'a F,
}

impl<'a, F> Op for PmRhs<'a, F>
where
    F: Fn(&V, &SupportPoint, T, &mut V, V, &Covariates),
{
    type T = T;
    type V = V;
    type M = M;
    fn nstates(&self) -> usize {
        *self.nstates
    }
    fn nout(&self) -> usize {
        *self.nout
    }
    fn nparams(&self) -> usize {
        *self.nparams
    }
}

impl<'a, F> NonLinearOp for PmRhs<'a, F>
where
    F: Fn(&V, &SupportPoint, T, &mut V, V, &Covariates),
{
    fn call_inplace(&self, x: &Self::V, t: Self::T, y: &mut Self::V) {
        let mut rateiv = Self::V::zeros(*self.nstates);
        //TODO: This should be pre-calculated
        for infusion in self.infusions {
            if t >= Self::T::from(infusion.time())
                && t <= Self::T::from(infusion.duration() + infusion.time())
            {
                rateiv[infusion.input()] += Self::T::from(infusion.amount() / infusion.duration());
            }
        }
        // self.statistics.borrow_mut().increment_call();
        (self.func)(x, &self.p, t, y, rateiv, &self.covariates)
    }
}

impl<'a, F> NonLinearOpJacobian for PmRhs<'a, F>
where
    F: Fn(&V, &SupportPoint, T, &mut V, V, &Covariates),
{
    fn jac_mul_inplace(&self, _x: &Self::V, _t: Self::T, _v: &Self::V, _y: &mut Self::V) {}
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

impl LinearOp for PmMass {
    fn gemv_inplace(&self, _x: &Self::V, _t: Self::T, _beta: Self::T, _y: &mut Self::V) {}
}
pub struct PmInit {
    nstates: usize,
    nout: usize,
    nparams: usize,
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

impl ConstantOp for PmInit {
    fn call_inplace(&self, _t: Self::T, _y: &mut Self::V) {}
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

impl NonLinearOp for PmRoot {
    fn call_inplace(&self, _x: &Self::V, _t: Self::T, _y: &mut Self::V) {}
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

impl NonLinearOp for PmOut {
    fn call_inplace(&self, _x: &Self::V, _t: Self::T, _y: &mut Self::V) {}
}

pub struct PMProblem<F>
where
    F: Fn(&V, &SupportPoint, T, &mut V, V, &Covariates),
{
    // rhs: PmRhs<fn(&V, &SupportPoint, T, &mut V, V, &Covariates)>,
    func: F,
    nstates: usize,
    nout: usize,
    nparams: usize,
    p: SupportPoint,
    covariates: Covariates,
    infusions: Vec<Infusion>,
}

impl<F> PMProblem<F>
where
    F: Fn(&V, &SupportPoint, T, &mut V, V, &Covariates),
{
    pub fn new(
        func: F,
        nstates: usize,
        nout: usize,
        p: SupportPoint,
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
            covariates,
            infusions,
        }
    }
}

impl<F> Op for PMProblem<F>
where
    F: Fn(&V, &SupportPoint, T, &mut V, V, &Covariates),
{
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

impl<'a, F> OdeEquationsRef<'a> for PMProblem<F>
where
    F: Fn(&V, &SupportPoint, T, &mut V, V, &Covariates),
{
    type Rhs = PmRhs<'a, F>;
    type Mass = PmMass;
    type Init = PmInit;
    type Root = PmRoot;
    type Out = PmOut;
}

impl<F> OdeEquations for PMProblem<F>
where
    F: Fn(&V, &SupportPoint, T, &mut V, V, &Covariates),
{
    fn rhs(&self) -> <PMProblem<F> as OdeEquationsRef<'_>>::Rhs {
        PmRhs {
            nstates: &self.nstates,
            nout: &self.nout,
            nparams: &self.nparams,
            infusions: &self.infusions,
            covariates: &self.covariates,
            p: &self.p,
            func: &self.func,
        }
    }
    fn mass(&self) -> Option<<PMProblem<F> as OdeEquationsRef<'_>>::Mass> {
        None
    }
    fn init(&self) -> PmInit {
        PmInit {
            nstates: self.nstates(),
            nout: self.nout(),
            nparams: self.nparams(),
        }
    }
    fn root(&self) -> Option<<PMProblem<F> as OdeEquationsRef<'_>>::Root> {
        None
    }
    fn out(&self) -> Option<<PMProblem<F> as OdeEquationsRef<'_>>::Out> {
        None
    }
    fn set_params(&mut self, p: &Self::V) {
        let names = self.p.parameters();
        for (i, name) in names.iter().enumerate() {
            self.p.set(name, p[i]);
        }
    }
}
