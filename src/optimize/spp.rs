use argmin::{
    core::{CostFunction, Error, Executor},
    solver::neldermead::NelderMead,
};

use faer::Mat;

use crate::{prelude::simulator::psi, Data, Equation, ErrorModels};

pub struct SppOptimizer<'a, E: Equation> {
    equation: &'a E,
    data: &'a Data,
    sig: &'a ErrorModels,
    pyl: &'a Vec<f64>,
}

impl<E: Equation> CostFunction for SppOptimizer<'_, E> {
    type Param = Vec<f64>;
    type Output = f64;
    fn cost(&self, spp: &Self::Param) -> Result<Self::Output, Error> {
        // Create a 1xN matrix (1 support point with N parameters)
        let theta = Mat::from_fn(1, spp.len(), |_, j| spp[j]);

        let psi = psi(self.equation, self.data, &theta, self.sig, false, false)?;

        if psi.ncols() > 1 {
            tracing::error!("Psi in SppOptimizer has more than one column");
        }
        if psi.nrows() != self.pyl.len() {
            tracing::error!(
                "Psi in SppOptimizer has {} rows, but spp has {}",
                psi.nrows(),
                self.pyl.len()
            );
        }
        let nsub = psi.nrows() as f64;
        let mut sum = -nsub;
        // Iterate through each row of psi (which has only 1 column)
        for (i, pyl_i) in self.pyl.iter().enumerate() {
            let p_i = psi[(i, 0)];
            sum += p_i / pyl_i;
        }
        Ok(-sum)
    }
}

impl<'a, E: Equation> SppOptimizer<'a, E> {
    pub fn new(equation: &'a E, data: &'a Data, sig: &'a ErrorModels, pyl: &'a Vec<f64>) -> Self {
        Self {
            equation,
            data,
            sig,
            pyl,
        }
    }
    pub fn optimize_point(self, spp: Vec<f64>) -> Result<Vec<f64>, Error> {
        let simplex = create_initial_simplex(&spp);
        let solver: NelderMead<Vec<f64>, f64> = NelderMead::new(simplex).with_sd_tolerance(1e-2)?;
        let res = Executor::new(self, solver)
            .configure(|state| state.max_iters(5))
            // .add_observer(SlogLogger::term(), ObserverMode::Always)
            .run()?;
        Ok(res.state.best_param.unwrap())
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
