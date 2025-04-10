use nalgebra::DVector;

use crate::{Equation, Subject, ODE};

// Define Model as a trait
pub trait Model<'a> {
    type Eq: Equation;

    fn new(equation: &'a Self::Eq, data: &'a Subject) -> Self
    where
        Self: Sized;

    fn equation(&self) -> &Self::Eq;
    fn data(&self) -> &Subject;
    fn state(&self) -> &<Self::Eq as Equation>::S;
}

// Implementation for ODE
pub struct ODEModel<'a> {
    equation: &'a ODE,
    data: &'a Subject,
    state: <ODE as Equation>::S,
}

impl<'a> Model<'a> for ODEModel<'a> {
    type Eq = ODE;

    fn new(equation: &'a ODE, data: &'a Subject) -> Self {
        Self {
            equation,
            data,
            state: DVector::default(),
        }
    }

    fn equation(&self) -> &ODE {
        self.equation
    }

    fn data(&self) -> &Subject {
        self.data
    }

    fn state(&self) -> &<ODE as Equation>::S {
        &self.state
    }
}
