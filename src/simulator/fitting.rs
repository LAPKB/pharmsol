use argmin::{
    core::{CostFunction, Error, Executor},
    solver::neldermead::NelderMead,
};
use ndarray::Array1;

use crate::data::{error_model::ErrorModel, Subject};

use super::Equation;

struct SppOptimizer<'a> {
    equation: &'a Equation,
    subject: &'a Subject,
    error_model: &'a ErrorModel<'a>,
}

impl<'a> CostFunction for SppOptimizer<'a> {
    type Param = Array1<f64>;
    type Output = f64;

    fn cost(&self, point: &Self::Param) -> Result<Self::Output, Error> {
        Ok(-self
            .equation
            .simulate_subject(self.subject, point.to_vec().as_ref())
            .likelihood(self.error_model))
    }
}

impl<'a> SppOptimizer<'a> {
    fn new(equation: &'a Equation, subject: &'a Subject, error_model: &'a ErrorModel<'a>) -> Self {
        Self {
            equation,
            subject,
            error_model,
        }
    }

    fn optimize(self, point: Array1<f64>) -> Array1<f64> {
        let simplex = create_initial_simplex(&point);
        let solver = NelderMead::new(simplex)
            .with_sd_tolerance(1e-2)
            .expect("Error setting up the solver");
        let res = Executor::new(self, solver)
            .configure(|state| state.max_iters(10))
            .run()
            .unwrap();
        res.state.best_param.unwrap()
    }
}

fn create_initial_simplex(initial_point: &Array1<f64>) -> Vec<Array1<f64>> {
    let num_dimensions = initial_point.len();
    let perturbation_percentage = 0.008;

    // Initialize a Vec to store the vertices of the simplex
    let mut vertices = Vec::new();

    // Add the initial point to the vertices
    vertices.push(initial_point.to_owned());

    // Calculate perturbation values for each component
    for i in 0..num_dimensions {
        let perturbation = if initial_point[i] == 0.0 {
            0.00025 // Special case for components equal to 0
        } else {
            perturbation_percentage * initial_point[i]
        };

        let mut perturbed_point = initial_point.to_owned();
        perturbed_point[i] += perturbation;
        vertices.push(perturbed_point);
    }

    vertices
}

trait OptimalSupportPoint {
    fn optimal_support_point(
        &self,
        equation: &Equation,
        error_model: &ErrorModel,
        point: Array1<f64>,
    ) -> Array1<f64>;
}

impl OptimalSupportPoint for Subject {
    fn optimal_support_point(
        &self,
        equation: &Equation,
        error_model: &ErrorModel,
        point: Array1<f64>,
    ) -> Array1<f64> {
        SppOptimizer::new(equation, self, error_model).optimize(point)
    }
}
