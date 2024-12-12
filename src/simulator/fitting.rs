use argmin::{
    core::{CostFunction, Error, Executor},
    solver::neldermead::NelderMead,
};
use ndarray::{Array1, Array2, Axis};

use crate::{
    data::{Data, Subject},
    Equation, Predictions,
};

use super::SupportPoint;

struct SppOptimizer<'a, E: Equation> {
    equation: &'a E,
    subject: &'a Subject,
    parameters: Vec<String>,
}

impl<'a, E: Equation> CostFunction for SppOptimizer<'a, E> {
    type Param = Array1<f64>;
    type Output = f64;

    fn cost(&self, point: &Self::Param) -> Result<Self::Output, Error> {
        let support_point = SupportPoint::from_vec(point.to_vec(), self.parameters.clone());
        let (concentrations, _) =
            self.equation
                .simulate_subject(self.subject, &support_point, None);
        Ok(concentrations.squared_error()) //TODO: Change this to use the D function
    }
}

impl<'a, E: Equation> SppOptimizer<'a, E> {
    fn new(equation: &'a E, subject: &'a Subject, parameters: Vec<String>) -> Self {
        Self {
            equation,
            subject,
            parameters,
        }
    }

    fn optimize(self, point: &SupportPoint) -> Array1<f64> {
        let point = Array1::from_vec(point.to_vec());
        let simplex = create_initial_simplex(&point);
        let solver = NelderMead::new(simplex)
            .with_sd_tolerance(1e-2)
            .expect("Error setting up the solver");
        let res = Executor::new(self, solver)
            // .configure(|state| state.max_iters(10))
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

pub trait OptimalSupportPoint<E: Equation> {
    fn optimal_support_point(&self, equation: &E, point: &SupportPoint) -> Array1<f64>;
}

impl<E: Equation> OptimalSupportPoint<E> for Subject {
    fn optimal_support_point(&self, equation: &E, point: &SupportPoint) -> Array1<f64> {
        SppOptimizer::new(equation, self, point.parameters()).optimize(point)
    }
}

pub trait EstimateTheta<E: Equation> {
    fn estimate_theta(&self, equation: &E, point: &SupportPoint) -> Array2<f64>;
}

impl<E: Equation> EstimateTheta<E> for Data {
    fn estimate_theta(&self, equation: &E, point: &SupportPoint) -> Array2<f64> {
        let n_sub = self.len();
        let mut theta = Array2::zeros((n_sub, point.len()));

        theta
            .axis_iter_mut(Axis(0))
            .zip(self.get_subjects().iter())
            .for_each(|(mut row, subject)| {
                row.assign(&subject.optimal_support_point(equation, point));
            });

        theta
    }
}
