use nalgebra::DVector;

use crate::{Equation, Subject, ODE};

//states

pub struct Model<'a, Eqn: Equation> {
    equation: &'a Eqn,
    data: &'a Subject,
    state: Eqn::S,
}

impl<'a> Model<'a, ODE> {
    pub fn new(equation: &'a ODE, data: &'a Subject) -> Self {
        Self {
            equation,
            data,
            state: DVector::default(),
        }
    }
}
