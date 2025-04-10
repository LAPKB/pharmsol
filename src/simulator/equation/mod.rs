pub mod analytical;
pub mod meta;
pub mod ode;
pub mod sde;
pub use analytical::*;
pub use meta::*;
pub use ode::*;
pub use sde::*;

use crate::Subject;

use super::model::Model;

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
#[allow(private_bounds)]
pub trait Equation: 'static + Clone + Sync {
    // fn get_init(&self) -> &Init;
    // fn get_out(&self) -> &Out;

    fn get_nstates(&self) -> usize;
    fn get_nouteqs(&self) -> usize;

    #[allow(dead_code)]
    fn is_sde(&self) -> bool {
        false
    }

    fn initialize_model(&self, subject: &Subject, spp: &[f64]) {
        Model::new(self, subject, spp)
    }
}
