#[cfg(feature = "native")]
use argmin::{
    core::{CostFunction, Error, Executor},
    solver::neldermead::NelderMead,
};

#[cfg(feature = "native")]
use ndarray::{Array1, Axis};

#[cfg(feature = "native")]
use crate::{prelude::simulator::psi, Data, Equation, ErrorModels};

#[cfg(feature = "native")]
pub struct SppOptimizer<'a, E: Equation> {
    equation: &'a E,
    data: &'a Data,
    sig: &'a ErrorModels,
    pyl: &'a Array1<f64>,
}

#[cfg(feature = "native")]
impl<E: Equation> CostFunction for SppOptimizer<'_, E> {
    type Param = Vec<f64>;
    type Output = f64;
    fn cost(&self, spp: &Self::Param) -> Result<Self::Output, Error> {
        let theta = Array1::from(spp.clone()).insert_axis(Axis(0));

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
        for (p_i, pyl_i) in psi.iter().zip(self.pyl.iter()) {
            sum += p_i / pyl_i;
        }
        Ok(-sum)
    }
}

#[cfg(feature = "native")]
impl<'a, E: Equation> SppOptimizer<'a, E> {
    pub fn new(
        equation: &'a E,
        data: &'a Data,
        sig: &'a ErrorModels,
        pyl: &'a Array1<f64>,
    ) -> Self {
        Self {
            equation,
            data,
            sig,
            pyl,
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

#[cfg(feature = "native")]
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

// WASM / non-native stubs
#[cfg(not(feature = "native"))]
use ndarray::Array1;

#[cfg(not(feature = "native"))]
use crate::{Data, Equation, ErrorModels};

#[cfg(not(feature = "native"))]
pub struct SppOptimizer<'a, E> {
    _phantom: std::marker::PhantomData<&'a E>,
}

#[cfg(not(feature = "native"))]
impl<'a, E: Equation> SppOptimizer<'a, E> {
    pub fn new(
        _equation: &'a E,
        _data: &'a Data,
        _sig: &'a ErrorModels,
        _pyl: &'a Array1<f64>,
    ) -> Self {
        SppOptimizer {
            _phantom: std::marker::PhantomData,
        }
    }

    pub fn optimize_point(
        self,
        _spp: Array1<f64>,
    ) -> Result<Array1<f64>, Box<dyn std::error::Error>> {
        Err(Box::new(std::io::Error::new(
            std::io::ErrorKind::Other,
            "SppOptimizer is disabled when feature `native` is not enabled",
        )))
    }
}
