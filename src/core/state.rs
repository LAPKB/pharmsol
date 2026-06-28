/// Trait for state vectors that can receive bolus doses.
///
/// Implemented by the state types used by each backend:
/// - [`V`](nalgebra::DVector) for ODE and Analytical
/// - `Vec<DVector<f64>>` for SDE (one per particle)
pub trait State {
    /// Add a bolus dose to the state at the specified resolved input index.
    ///
    /// # Parameters
    /// - `input`: The resolved dense input index used by the execution layer
    /// - `amount`: The bolus amount
    fn add_bolus(&mut self, input: usize, amount: f64);
}
