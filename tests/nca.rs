//! NCA Integration Tests
//!
//! Tests for the public NCA API using Subject::builder().nca()

// Include test modules from nca/ directory
#[path = "nca/test_auc.rs"]
mod test_auc;

#[path = "nca/test_params.rs"]
mod test_params;

#[path = "nca/test_quality.rs"]
mod test_quality;

#[path = "nca/test_terminal.rs"]
mod test_terminal;
