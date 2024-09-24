use nalgebra::DVector;

#[derive(Clone, Debug)]
pub struct OutEq {
    state_index: usize,
    output_index: usize,
    operation: Op,
}

#[derive(Clone, Debug)]
pub enum Op {
    None,
    Div(usize),
}

impl OutEq {
    pub fn new(output_index: usize, state_index: usize, operation: Op) -> Self {
        Self {
            state_index,
            output_index,
            operation,
        }
    }
    pub fn apply(&self, y: &mut DVector<f64>, p: &DVector<f64>, x: &DVector<f64>) {
        match self.operation {
            Op::None => {
                y[self.output_index] = x[self.state_index];
            }
            Op::Div(divisor) => {
                y[self.output_index] = x[self.state_index] / p[divisor];
            }
        }
    }
}
