//! The `exa` module provides functionality for building and loading pharmacological models.
//!
//! This module is split into two submodules:
//! - `build`: Contains functions for compiling models from source text.
//! - `load`: Contains functions for loading compiled models.

pub mod build;
pub mod load;

#[repr(C)]
#[derive(Clone, Debug)]
pub enum EqnKind {
    ODE = 0,
    Analytical = 1,
    SDE = 2,
}

impl EqnKind {
    pub fn to_str(&self) -> String {
        match self {
            Self::ODE => "EqnKind::ODE".to_string(),
            Self::Analytical => "EqnKind::Analytical".to_string(),
            Self::SDE => "EqnKind::SDE".to_string(),
        }
    }
}
