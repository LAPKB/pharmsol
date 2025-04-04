use argmin::{
    core::{CostFunction, Error, Executor},
    solver::neldermead::NelderMead,
};
use ndarray::{Array1, Array2, Axis};

use crate::{
    data::{Data, Subject},
    Equation, Predictions,
};

/// Optimizer that finds the best subject-specific parameters
///
/// This struct wraps the equation model and subject data to create
/// a cost function that can be optimized to find the best parameter
/// fit for a specific individual.
struct SppOptimizer<'a, E: Equation> {
    equation: &'a E,
    subject: &'a Subject,
}

/// Implementation of the CostFunction trait for parameter optimization
///
/// This allows the SppOptimizer to be used with optimization algorithms
/// from the argmin library to find optimal subject-specific parameters.
impl<E: Equation> CostFunction for SppOptimizer<'_, E> {
    type Param = Vec<f64>;
    type Output = f64;

    /// Calculates the cost (error) between model predictions and observations
    ///
    /// # Arguments
    ///
    /// * `point` - Parameter vector to evaluate
    ///
    /// # Returns
    ///
    /// The squared error between model predictions and observations, or an error if calculation fails
    fn cost(&self, point: &Self::Param) -> Result<Self::Output, Error> {
        let (concentrations, _) =
            self.equation
                .simulate_subject(self.subject, point.as_ref(), None);
        Ok(concentrations.squared_error()) //TODO: Change this to use the D function
    }
}

impl<'a, E: Equation> SppOptimizer<'a, E> {
    /// Creates a new SppOptimizer for individual parameter optimization
    ///
    /// # Arguments
    ///
    /// * `equation` - Reference to the equation model to use for simulation
    /// * `subject` - Reference to the subject data to optimize against
    ///
    /// # Returns
    ///
    /// A new SppOptimizer instance
    fn new(equation: &'a E, subject: &'a Subject) -> Self {
        Self { equation, subject }
    }

    /// Performs optimization to find the best parameter values
    ///
    /// Uses the Nelder-Mead algorithm to find parameter values that
    /// minimize the error between model predictions and observations.
    ///
    /// # Arguments
    ///
    /// * `point` - Initial parameter values to start optimization from
    ///
    /// # Returns
    ///
    /// Optimized parameter values as an ndarray Array1
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

/// Trait for finding the optimal support point for a subject
pub trait OptimalSupportPoint<E: Equation> {
    /// Finds the optimal support point for a subject
    ///
    /// # Arguments
    ///
    /// * `equation` - Reference to the equation model to use for simulation
    /// * `point` - Initial parameter values to start optimization from
    ///
    /// # Returns
    ///
    /// Optimized parameter values as an ndarray Array1
    fn optimal_support_point(&self, equation: &E, point: &Array1<f64>) -> Array1<f64>;
}

impl<E: Equation> OptimalSupportPoint<E> for Subject {
    fn optimal_support_point(&self, equation: &E, point: &Array1<f64>) -> Array1<f64> {
        SppOptimizer::new(equation, self).optimize(point)
    }
}

/// Trait for estimating the theta parameter for a dataset
pub trait EstimateTheta<E: Equation> {
    /// Estimates the theta parameter for a dataset
    ///
    /// # Arguments
    ///
    /// * `equation` - Reference to the equation model to use for simulation
    /// * `point` - Initial parameter values to start optimization from
    ///
    /// # Returns
    ///
    /// Estimated theta parameter values as an ndarray Array2
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
