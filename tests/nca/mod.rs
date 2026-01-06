// NCA Test Module
// Comprehensive test suite for Non-Compartmental Analysis algorithms

pub mod test_auc;
pub mod test_params;
pub mod test_quality;
pub mod test_terminal;
pub mod validation;

// Re-export common test utilities
pub use validation::{compare_results, load_validation_dataset, ValidationDataset};
