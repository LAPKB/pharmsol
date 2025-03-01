use argmin::{
    core::{CostFunction, Error, Executor},
    solver::neldermead::NelderMead,
};
use ndarray::{Array1, Array2, Axis};

use crate::{
    data::{Data, Subject},
    Equation, Predictions,
};

struct SppOptimizer<'a, E: Equation> {
    equation: &'a E,
    subject: &'a Subject,
}

impl<'a, E: Equation> CostFunction for SppOptimizer<'a, E> {
    type Param = Vec<f64>;
    type Output = f64;

    fn cost(&self, point: &Self::Param) -> Result<Self::Output, Error> {
        let (concentrations, _) =
            self.equation
                .simulate_subject(self.subject, point.as_ref(), None);
        Ok(concentrations.squared_error()) //TODO: Change this to use the D function
    }
}

impl<'a, E: Equation> SppOptimizer<'a, E> {
    fn new(equation: &'a E, subject: &'a Subject) -> Self {
        Self { equation, subject }
    }

    fn optimize(self, point: &Array1<f64>) -> Array1<f64> {
        let simplex = create_initial_simplex(point)
            .into_iter()
            .map(|x| x.to_vec())
            .collect();
        let solver = NelderMead::new(simplex)
            .with_sd_tolerance(1e-2)
            .expect("Error setting up the solver");
        let res = Executor::new(self, solver)
            // .configure(|state| state.max_iters(10))
            .run()
            .unwrap();
        Array1::from_vec(res.state.best_param.unwrap())
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
    fn optimal_support_point(&self, equation: &E, point: &Array1<f64>) -> Array1<f64>;
}

impl<E: Equation> OptimalSupportPoint<E> for Subject {
    fn optimal_support_point(&self, equation: &E, point: &Array1<f64>) -> Array1<f64> {
        SppOptimizer::new(equation, self).optimize(point)
    }
}

pub trait EstimateTheta<E: Equation> {
    fn estimate_theta(&self, equation: &E, point: &Array1<f64>) -> Array2<f64>;
}

impl<E: Equation> EstimateTheta<E> for Data {
    fn estimate_theta(&self, equation: &E, point: &Array1<f64>) -> Array2<f64> {
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
