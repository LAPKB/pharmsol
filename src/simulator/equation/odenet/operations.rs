use std::collections::HashMap;
pub use Op::*;
pub use Operator::*;

use nalgebra::DVector;

use crate::Covariates;

#[derive(Clone, Debug)]
pub enum Operator {
    P(usize),
    Cov(String),
    X(usize),
    O(Box<Op>),
}

impl Operator {
    pub fn get(
        &self,
        p: &DVector<f64>,
        x: Option<&DVector<f64>>,
        cov: Option<&HashMap<String, f64>>,
    ) -> f64 {
        match self {
            Operator::P(i) => p[*i],
            Operator::Cov(s) => {
                if let Some(cov) = cov {
                    if let Some(v) = cov.get(s) {
                        *v
                    } else {
                        panic!("Covariate {} not found in covariates {:?}", s, cov);
                    }
                } else {
                    panic!("Covariates cannot be accessed in this context");
                }
            }
            Operator::X(i) => {
                if let Some(x) = x {
                    x[*i]
                } else {
                    panic!("State cannot be accessed in this context");
                }
            }
            Operator::O(op) => op.apply(p, x, cov),
        }
    }
}

#[derive(Clone, Debug)]
pub enum Op {
    Equal(Operator),
    Sum(Operator, Operator),
    Sub(Operator, Operator),
    Mul(Operator, Operator),
    Div(Operator, Operator),
}

impl Op {
    pub fn apply(
        &self,
        p: &DVector<f64>,
        x: Option<&DVector<f64>>,
        cov: Option<&HashMap<String, f64>>,
    ) -> f64 {
        match self {
            Op::Equal(o) => o.get(p, x, cov),
            Op::Sum(a, b) => a.get(p, x, cov) + b.get(p, x, cov),
            Op::Sub(a, b) => a.get(p, x, cov) - b.get(p, x, cov),
            Op::Mul(a, b) => a.get(p, x, cov) * b.get(p, x, cov),
            Op::Div(n, d) => {
                if d.get(p, x, cov) == 0.0 {
                    panic!("Division by zero when applying operation Div");
                } else {
                    n.get(p, x, cov) / d.get(p, x, cov)
                }
            }
        }
    }
}

//Init
#[derive(Clone, Debug)]
pub struct Init {
    state_index: usize,
    operation: Op,
}

impl Init {
    pub fn new(state_index: usize, operation: Op) -> Self {
        Self {
            state_index,
            operation,
        }
    }
    pub fn apply(&self, state: &mut DVector<f64>, p: &DVector<f64>, covs: &Covariates) {
        //This is a hardcoded assumption, that the initial time is 0.0
        let covs = covs.to_hashmap(0.0);
        state[self.state_index] = self.operation.apply(p, None, Some(&covs));
    }
}

//Lag & Fa

#[derive(Clone, Debug)]
pub struct Lag {
    state_index: usize,
    operation: Op,
}

pub type Fa = Lag;

impl Lag {
    pub fn new(state_index: usize, operation: Op) -> Self {
        Self {
            state_index,
            operation,
        }
    }
    pub fn apply(&self, lag: &mut HashMap<usize, f64>, p: &DVector<f64>) {
        lag.insert(self.state_index, self.operation.apply(p, None, None));
    }
}

// Output Equations
#[derive(Clone, Debug)]
pub struct OutEq {
    output_index: usize,
    operation: Op,
}

impl OutEq {
    pub fn new(output_index: usize, operation: Op) -> Self {
        Self {
            output_index,
            operation,
        }
    }
    pub fn apply(
        &self,
        y: &mut DVector<f64>,
        p: &DVector<f64>,
        x: &DVector<f64>,
        cov: &HashMap<String, f64>,
    ) {
        y[self.output_index] = self.operation.apply(p, Some(x), Some(cov));
    }
}
