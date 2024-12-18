use crate::{Covariates, Infusion, SupportPoint};
use diffsol::{
    ConstantOp, LinearOp, NonLinearOp, NonLinearOpJacobian, OdeEquations, OdeEquationsRef, Op,
};
type T = f64;
type V = nalgebra::DVector<f64>;
type M = nalgebra::DMatrix<f64>;
pub struct PmRhs<F>
where
    F: Fn(&V, &SupportPoint, T, &mut V, V, &Covariates),
{
    func: F,
    nstates: usize,
    nout: usize,
    nparams: usize,
    p: SupportPoint, //Rc<SupportPoint>,
    // sparsity: Option<M::Sparsity>,
    // statistics: RefCell<OpStatistics>,
    covariates: Covariates,
    infusions: Vec<Infusion>,
}
impl<F> PmRhs<F>
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
impl<F> Op for PmRhs<F>
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

impl<F> NonLinearOp for PmRhs<F>
where
    F: Fn(&V, &SupportPoint, T, &mut V, V, &Covariates),
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
        (self.func)(x, &self.p, t, y, rateiv, &self.covariates)
    }
}

impl<F> NonLinearOpJacobian for PmRhs<F>
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
    fn gemv_inplace(&self, x: &Self::V, t: Self::T, beta: Self::T, y: &mut Self::V) {}
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
    fn call_inplace(&self, t: Self::T, y: &mut Self::V) {}
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
    fn call_inplace(&self, x: &Self::V, t: Self::T, y: &mut Self::V) {}
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
    fn call_inplace(&self, x: &Self::V, t: Self::T, y: &mut Self::V) {}
}

pub struct PMProblem {
    rhs: PmRhs<fn(&V, &SupportPoint, T, &mut V, V, &Covariates)>,
    mass: Option<PmMass>,
    init: Option<PmInit>,
    root: Option<PmRoot>,
    out: Option<PmOut>,
}

impl PMProblem {
    pub fn new(
        rhs: PmRhs<fn(&V, &SupportPoint, T, &mut V, V, &Covariates)>,
        mass: Option<PmMass>,
        init: Option<PmInit>,
        root: Option<PmRoot>,
        out: Option<PmOut>,
    ) -> Self {
        Self {
            rhs,
            mass,
            init,
            root,
            out,
        }
    }
}

impl Op for PMProblem {
    type T = T;
    type V = V;
    type M = M;
    fn nstates(&self) -> usize {
        self.rhs.nstates
    }
    fn nout(&self) -> usize {
        self.rhs.nout
    }
    fn nparams(&self) -> usize {
        self.rhs.nparams
    }
}

impl OdeEquationsRef<'_> for PMProblem {
    type Rhs = PmRhs<fn(&V, &SupportPoint, T, &mut V, V, &Covariates)>;
    type Mass = PmMass;
    type Init = PmInit;
    type Root = PmRoot;
    type Out = PmOut;
}

impl OdeEquations for PMProblem {
    fn rhs(&self) -> <PMProblem as OdeEquationsRef<'_>>::Rhs {
        PmRhs {
            func: self.rhs.func,
            nstates: self.rhs.nstates,
            nout: self.rhs.nout,
            nparams: self.rhs.nparams,
            p: self.rhs.p.clone(),
            covariates: self.rhs.covariates.clone(),
            infusions: self.rhs.infusions.clone(),
        }
    }
    fn mass(&self) -> Option<<PMProblem as OdeEquationsRef<'_>>::Mass> {
        None
    }
    fn init(&self) -> PmInit {
        PmInit {
            nstates: self.nstates(),
            nout: self.nout(),
            nparams: self.nparams(),
        }
    }
    fn root(&self) -> Option<<PMProblem as OdeEquationsRef<'_>>::Root> {
        None
    }
    fn out(&self) -> Option<<PMProblem as OdeEquationsRef<'_>>::Out> {
        None
    }
    fn set_params(&mut self, p: &Self::V) {
        let names = self.rhs.p.parameters();
        for (i, name) in names.iter().enumerate() {
            self.rhs.p.set(name, p[i]);
        }
    }
}
