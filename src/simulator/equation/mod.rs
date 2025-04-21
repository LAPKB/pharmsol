pub mod analytical;
pub mod meta;
pub mod ode;
pub mod sde;
pub use analytical::*;
pub use meta::*;
pub use ode::*;
pub use sde::*;

use crate::Subject;

use super::{likelihood::Prediction, model::Model};

/// Trait defining the associated types for equations.
// pub trait EquationTypes {
//     /// The state vector type
//     type S: State;
//     /// The predictions container type
//     type P: Predictions;
// }

/// Trait for model equations that can be simulated.
///
/// This trait defines the interface for different types of model equations
/// (ODE, SDE, analytical) that can be simulated to generate predictions
/// and estimate parameters.
///
pub trait Predictions: Default {
    /// Create a new prediction container with specified capacity.
    ///
    /// # Parameters
    /// - `nparticles`: Number of particles (for SDE)
    ///
    /// # Returns
    /// A new Outputs container
    fn empty(nparticles: usize) -> Self;

    /// Calculate the sum of squared errors for all Outputs.
    ///
    /// # Returns
    /// The sum of squared errors
    fn squared_error(&self) -> f64;

    /// Get all Outputs as a vector.
    ///
    /// # Returns
    /// Vector of prediction objects
    fn get_predictions(&self) -> Vec<Prediction>;
}

pub trait State {
    fn add_bolus(&mut self, input: usize, amount: f64);
}
#[allow(private_bounds)]
pub trait Equation<'a>: 'static + Clone + Sync {
    /// The state vector type
    type S: State;
    /// The Outputs container type
    type P: Predictions;

    type Mod: Model<'a, Eq = Self>;

    fn get_nstates(&self) -> usize;
    fn get_nouteqs(&self) -> usize;

    #[allow(dead_code)]
    fn is_sde(&self) -> bool {
        false
    }

    fn initialize_model(&'a self, subject: &'a Subject, spp: Vec<f64>) -> Self::Mod;

    fn nparticles(&self) -> usize {
        1
    }
}
