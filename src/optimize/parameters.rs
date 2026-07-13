//! Nelder‑Mead parameter refinement for pharmacometric models.
//!
//! This module provides a [`ParameterOptimizer`] that refines a single parameter
//! Given an [`Equation`], observed [`Data`], and [`AssayErrorModels`] via
//! Nelder‑Mead optimization in log‑space. The optimizer finds the parameter vector
//! that minimizes the negative log-likelihood of the model predictions against the data,
//! as measured by the provided error models.

use argmin::{
    core::{CostFunction, Error, Executor},
    solver::neldermead::NelderMead,
};

use ndarray::{Array1, Axis};

use crate::{prelude::simulator::log_likelihood_matrix, AssayErrorModels, Data, Equation};

/// Optimizer that refines a single parameter vector against observed data.
pub struct ParameterOptimizer<'a, E: Equation> {
    equation: &'a E,
    data: &'a Data,
    sig: &'a AssayErrorModels,
    pyl: &'a Array1<f64>,
}

impl<E: Equation> CostFunction for ParameterOptimizer<'_, E> {
    type Param = Vec<f64>;
    type Output = f64;

    fn cost(&self, parameters: &Self::Param) -> Result<Self::Output, Error> {
        let theta = Array1::from(parameters.clone()).insert_axis(Axis(0));

        let log_psi = log_likelihood_matrix(self.equation, self.data, &theta, self.sig, false)?;
        let psi = log_psi.mapv(f64::exp);

        if psi.ncols() > 1 {
            tracing::error!("Psi in ParameterOptimizer has more than one column");
        }
        if psi.nrows() != self.pyl.len() {
            tracing::error!(
                "Psi in ParameterOptimizer has {} rows, but pyl has {}",
                psi.nrows(),
                self.pyl.len()
            );
        }
        let nsub = psi.nrows() as f64;
        let mut sum = -nsub;
        for (p_i, pyl_i) in psi.iter().zip(self.pyl.iter()) {
            sum += p_i / pyl_i;
        }
        Ok(-sum)
    }
}

impl<'a, E: Equation> ParameterOptimizer<'a, E> {
    /// Create a new optimizer.
    ///
    /// * `equation` — the model to evaluate.
    /// * `data` — observed subject data.
    /// * `sig` — assay error models per output.
    /// * `pyl` — reference (target) likelihood vector.
    pub fn new(
        equation: &'a E,
        data: &'a Data,
        sig: &'a AssayErrorModels,
        pyl: &'a Array1<f64>,
    ) -> Self {
        Self {
            equation,
            data,
            sig,
            pyl,
        }
    }

    /// Optimize the parameters to minimize the negative log-likelihood against the data.

    pub fn optimize_point(self, parameters: Array1<f64>) -> Result<Array1<f64>, Error> {
        let simplex = create_initial_simplex(&parameters.to_vec());
        let solver: NelderMead<Vec<f64>, f64> = NelderMead::new(simplex).with_sd_tolerance(1e-2)?;
        let res = Executor::new(self, solver)
            .configure(|state| state.max_iters(5))
            // .add_observer(SlogLogger::term(), ObserverMode::Always)
            .run()?;
        Ok(Array1::from(res.state.best_param.unwrap()))
    }
}

fn create_initial_simplex(initial_point: &[f64]) -> Vec<Vec<f64>> {
    let num_dimensions = initial_point.len();
    let perturbation_percentage = 0.008;

    let mut vertices = Vec::new();
    vertices.push(initial_point.to_vec());

    for i in 0..num_dimensions {
        let perturbation = if initial_point[i] == 0.0 {
            0.00025
        } else {
            perturbation_percentage * initial_point[i]
        };

        let mut perturbed_point = initial_point.to_owned();
        perturbed_point[i] += perturbation;
        vertices.push(perturbed_point);
    }

    vertices
}
