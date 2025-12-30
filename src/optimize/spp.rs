use argmin::{
    core::{CostFunction, Error, Executor},
    solver::neldermead::NelderMead,
};

use ndarray::{Array1, Axis};

use crate::{prelude::simulator::psi, Data, Equation, ErrorModels};

pub struct SppOptimizer<'a, E: Equation> {
    equation: &'a E,
    data: &'a Data,
    sig: &'a ErrorModels,
    log_pyl: &'a Array1<f64>,
}

impl<E: Equation> CostFunction for SppOptimizer<'_, E> {
    type Param = Vec<f64>;
    type Output = f64;
    fn cost(&self, spp: &Self::Param) -> Result<Self::Output, Error> {
        let theta = Array1::from(spp.clone()).insert_axis(Axis(0));

        let log_psi = psi(self.equation, self.data, &theta, self.sig, false, false)?;

        if log_psi.ncols() > 1 {
            tracing::error!("log_psi in SppOptimizer has more than one column");
        }
        if log_psi.nrows() != self.log_pyl.len() {
            tracing::error!(
                "log_psi in SppOptimizer has {} rows, but spp has {}",
                log_psi.nrows(),
                self.log_pyl.len()
            );
        }
        let nsub = log_psi.nrows() as f64;
        let mut sum = -nsub;
        // Convert log-likelihoods back to likelihood ratio: exp(log_psi - log_pyl)
        for (log_p_i, log_pyl_i) in log_psi.iter().zip(self.log_pyl.iter()) {
            sum += (log_p_i - log_pyl_i).exp();
        }
        Ok(-sum)
    }
}

impl<'a, E: Equation> SppOptimizer<'a, E> {
    pub fn new(
        equation: &'a E,
        data: &'a Data,
        sig: &'a ErrorModels,
        log_pyl: &'a Array1<f64>,
    ) -> Self {
        Self {
            equation,
            data,
            sig,
            log_pyl,
        }
    }
    pub fn optimize_point(self, spp: Array1<f64>) -> Result<Array1<f64>, Error> {
        let simplex = create_initial_simplex(&spp.to_vec());
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

    // Initialize a Vec to store the vertices of the simplex
    let mut vertices = Vec::new();

    // Add the initial point to the vertices
    vertices.push(initial_point.to_vec());

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
