//! Main JSON Model struct

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::json::errors::JsonModelError;
use crate::json::types::*;

/// Supported schema versions
pub const SUPPORTED_SCHEMA_VERSIONS: &[&str] = &["1.0"];

/// A pharmacometric model defined in JSON
///
/// This is the main struct that represents a parsed JSON model file.
/// It supports all three equation types (analytical, ODE, SDE) and
/// includes optional fields for covariates, error models, and UI metadata.
///
/// # Example
///
/// ```ignore
/// use pharmsol::json::JsonModel;
///
/// let json = r#"{
///     "schema": "1.0",
///     "id": "pk_1cmt_oral",
///     "type": "analytical",
///     "analytical": "one_compartment_with_absorption",
///     "parameters": ["ka", "ke", "V"],
///     "output": "x[1] / V"
/// }"#;
///
/// let model = JsonModel::from_str(json)?;
/// assert_eq!(model.id, "pk_1cmt_oral");
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct JsonModel {
    // ─────────────────────────────────────────────────────────────────────────
    // Layer 1: Identity (always required)
    // ─────────────────────────────────────────────────────────────────────────
    /// Schema version (e.g., "1.0")
    pub schema: String,

    /// Unique model identifier (snake_case)
    pub id: String,

    /// Model equation type
    #[serde(rename = "type")]
    pub model_type: ModelType,

    /// Library model ID to inherit from
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extends: Option<String>,

    /// Model version (semver)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,

    /// Alternative names (e.g., NONMEM ADVAN codes)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub aliases: Option<Vec<String>>,

    // ─────────────────────────────────────────────────────────────────────────
    // Layer 2: Structural Model
    // ─────────────────────────────────────────────────────────────────────────
    /// Parameter names in fetch order
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parameters: Option<Vec<String>>,

    /// Compartment names (indexed in declaration order)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub compartments: Option<Vec<String>>,

    /// State variable names (for SDE)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub states: Option<Vec<String>>,

    // ─────────────────────────────────────────────────────────────────────────
    // Equation Fields (type-dependent)
    // ─────────────────────────────────────────────────────────────────────────
    /// Built-in analytical solution function (for analytical type)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub analytical: Option<AnalyticalFunction>,

    /// Differential equations (for ODE type)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub diffeq: Option<DiffEqSpec>,

    /// SDE drift term (deterministic part)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub drift: Option<DiffEqSpec>,

    /// SDE diffusion coefficients
    #[serde(skip_serializing_if = "Option::is_none")]
    pub diffusion: Option<HashMap<String, ExpressionOrNumber>>,

    /// Secondary equations (for analytical)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub secondary: Option<String>,

    // ─────────────────────────────────────────────────────────────────────────
    // Output
    // ─────────────────────────────────────────────────────────────────────────
    /// Single output equation
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output: Option<String>,

    /// Multiple output definitions
    #[serde(skip_serializing_if = "Option::is_none")]
    pub outputs: Option<Vec<OutputDefinition>>,

    // ─────────────────────────────────────────────────────────────────────────
    // Optional Features
    // ─────────────────────────────────────────────────────────────────────────
    /// Initial conditions
    #[serde(skip_serializing_if = "Option::is_none")]
    pub init: Option<InitSpec>,

    /// Lag times per input compartment
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lag: Option<HashMap<String, ExpressionOrNumber>>,

    /// Bioavailability per input compartment
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fa: Option<HashMap<String, ExpressionOrNumber>>,

    /// [num_states, num_outputs]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub neqs: Option<(usize, usize)>,

    /// Number of particles for SDE simulation
    #[serde(skip_serializing_if = "Option::is_none")]
    pub particles: Option<usize>,

    // ─────────────────────────────────────────────────────────────────────────
    // Layer 3: Model Extensions
    // ─────────────────────────────────────────────────────────────────────────
    /// Derived parameters (computed from primary parameters)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub derived: Option<Vec<DerivedParameter>>,

    /// Enabled optional features
    #[serde(skip_serializing_if = "Option::is_none")]
    pub features: Option<Vec<Feature>>,

    /// Covariate definitions
    #[serde(skip_serializing_if = "Option::is_none")]
    pub covariates: Option<Vec<CovariateDefinition>>,

    /// Covariate effect specifications
    #[serde(rename = "covariateEffects", skip_serializing_if = "Option::is_none")]
    pub covariate_effects: Option<Vec<CovariateEffect>>,

    // ─────────────────────────────────────────────────────────────────────────
    // Layer 4: UI Metadata (ignored by compiler)
    // ─────────────────────────────────────────────────────────────────────────
    /// UI display information
    #[serde(skip_serializing_if = "Option::is_none")]
    pub display: Option<DisplayInfo>,

    /// Visual diagram layout
    #[serde(skip_serializing_if = "Option::is_none")]
    pub layout: Option<HashMap<String, Position>>,

    /// Rich documentation
    #[serde(skip_serializing_if = "Option::is_none")]
    pub documentation: Option<Documentation>,
}

impl JsonModel {
    /// Parse a JSON string into a JsonModel
    pub fn from_str(json: &str) -> Result<Self, JsonModelError> {
        let model: Self = serde_json::from_str(json)?;
        model.check_schema_version()?;
        Ok(model)
    }

    /// Parse from a JSON Value
    pub fn from_value(value: serde_json::Value) -> Result<Self, JsonModelError> {
        let model: Self = serde_json::from_value(value)?;
        model.check_schema_version()?;
        Ok(model)
    }

    /// Serialize to a JSON string
    pub fn to_json(&self) -> Result<String, JsonModelError> {
        Ok(serde_json::to_string_pretty(self)?)
    }

    /// Check if the schema version is supported
    fn check_schema_version(&self) -> Result<(), JsonModelError> {
        if !SUPPORTED_SCHEMA_VERSIONS.contains(&self.schema.as_str()) {
            return Err(JsonModelError::UnsupportedSchema {
                version: self.schema.clone(),
                supported: SUPPORTED_SCHEMA_VERSIONS.join(", "),
            });
        }
        Ok(())
    }

    /// Get the number of states (inferred or explicit)
    pub fn num_states(&self) -> usize {
        if let Some((nstates, _)) = self.neqs {
            return nstates;
        }

        match self.model_type {
            ModelType::Analytical => {
                if let Some(func) = &self.analytical {
                    func.num_states()
                } else {
                    1
                }
            }
            ModelType::Ode => {
                if let Some(compartments) = &self.compartments {
                    compartments.len()
                } else if let Some(DiffEqSpec::Object(map)) = &self.diffeq {
                    map.len()
                } else {
                    // Try to count from dx[n] in the string
                    1
                }
            }
            ModelType::Sde => {
                if let Some(states) = &self.states {
                    states.len()
                } else if let Some(DiffEqSpec::Object(map)) = &self.drift {
                    map.len()
                } else {
                    1
                }
            }
        }
    }

    /// Get the number of outputs (inferred or explicit)
    pub fn num_outputs(&self) -> usize {
        if let Some((_, nout)) = self.neqs {
            return nout;
        }

        if let Some(outputs) = &self.outputs {
            outputs.len()
        } else if self.output.is_some() {
            1
        } else {
            1
        }
    }

    /// Get the neqs tuple
    pub fn get_neqs(&self) -> (usize, usize) {
        self.neqs.unwrap_or((self.num_states(), self.num_outputs()))
    }

    /// Get compartment-to-index mapping
    pub fn compartment_map(&self) -> HashMap<String, usize> {
        let mut map = HashMap::new();
        if let Some(compartments) = &self.compartments {
            for (i, name) in compartments.iter().enumerate() {
                map.insert(name.clone(), i);
            }
        }
        map
    }

    /// Get state-to-index mapping (for SDE)
    pub fn state_map(&self) -> HashMap<String, usize> {
        let mut map = HashMap::new();
        if let Some(states) = &self.states {
            for (i, name) in states.iter().enumerate() {
                map.insert(name.clone(), i);
            }
        }
        map
    }

    /// Check if the model uses covariates
    pub fn has_covariates(&self) -> bool {
        self.covariates.is_some() && !self.covariates.as_ref().unwrap().is_empty()
    }

    /// Check if the model uses lag times
    pub fn has_lag(&self) -> bool {
        self.lag.is_some() && !self.lag.as_ref().unwrap().is_empty()
    }

    /// Check if the model uses bioavailability
    pub fn has_fa(&self) -> bool {
        self.fa.is_some() && !self.fa.as_ref().unwrap().is_empty()
    }

    /// Check if the model has initial conditions
    pub fn has_init(&self) -> bool {
        self.init.is_some()
    }

    /// Get the parameters as a vector (guaranteed non-empty after validation)
    pub fn get_parameters(&self) -> Vec<String> {
        self.parameters.clone().unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_minimal_analytical() {
        let json = r#"{
            "schema": "1.0",
            "id": "pk_1cmt_iv",
            "type": "analytical",
            "analytical": "one_compartment",
            "parameters": ["ke", "V"],
            "output": "x[0] / V"
        }"#;

        let model = JsonModel::from_str(json).unwrap();
        assert_eq!(model.id, "pk_1cmt_iv");
        assert_eq!(model.model_type, ModelType::Analytical);
        assert_eq!(model.analytical, Some(AnalyticalFunction::OneCompartment));
        assert_eq!(model.num_states(), 1);
        assert_eq!(model.num_outputs(), 1);
    }

    #[test]
    fn test_parse_minimal_ode() {
        let json = r#"{
            "schema": "1.0",
            "id": "pk_2cmt_ode",
            "type": "ode",
            "compartments": ["depot", "central", "peripheral"],
            "parameters": ["ka", "ke", "k12", "k21", "V"],
            "diffeq": {
                "depot": "-ka * x[0]",
                "central": "ka * x[0] - ke * x[1] - k12 * x[1] + k21 * x[2] + rateiv[1]",
                "peripheral": "k12 * x[1] - k21 * x[2]"
            },
            "output": "x[1] / V",
            "neqs": [3, 1]
        }"#;

        let model = JsonModel::from_str(json).unwrap();
        assert_eq!(model.id, "pk_2cmt_ode");
        assert_eq!(model.model_type, ModelType::Ode);
        assert_eq!(model.num_states(), 3);
        assert_eq!(model.compartment_map().get("central"), Some(&1));
    }

    #[test]
    fn test_parse_sde() {
        let json = r#"{
            "schema": "1.0",
            "id": "pk_1cmt_sde",
            "type": "sde",
            "parameters": ["ke0", "sigma_ke", "V"],
            "states": ["amount", "ke"],
            "drift": {
                "amount": "-ke * x[0]",
                "ke": "-0.5 * (ke - ke0)"
            },
            "diffusion": {
                "ke": "sigma_ke"
            },
            "init": {
                "ke": "ke0"
            },
            "output": "x[0] / V",
            "neqs": [2, 1],
            "particles": 1000
        }"#;

        let model = JsonModel::from_str(json).unwrap();
        assert_eq!(model.model_type, ModelType::Sde);
        assert_eq!(model.particles, Some(1000));
        assert_eq!(model.state_map().get("ke"), Some(&1));
    }

    #[test]
    fn test_unsupported_schema() {
        let json = r#"{
            "schema": "999.0",
            "id": "test",
            "type": "ode",
            "parameters": ["ke"],
            "diffeq": "dx[0] = -ke * x[0];",
            "output": "x[0]"
        }"#;

        let result = JsonModel::from_str(json);
        assert!(matches!(
            result,
            Err(JsonModelError::UnsupportedSchema { .. })
        ));
    }

    #[test]
    fn test_unknown_field_rejected() {
        let json = r#"{
            "schema": "1.0",
            "id": "test",
            "type": "ode",
            "parameters": ["ke"],
            "diffeq": "dx[0] = -ke * x[0];",
            "output": "x[0]",
            "unknown_field": "should fail"
        }"#;

        let result = JsonModel::from_str(json);
        assert!(result.is_err());
    }
}
