//! Main JSON Model struct

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::json::errors::JsonModelError;
use crate::json::types::*;

/// Supported schema versions
pub const SUPPORTED_SCHEMA_VERSIONS: &[&str] = &["2.0"];

/// A pharmacometric model defined in JSON.
///
/// This is the main struct that represents a parsed JSON model file.
/// It supports all three equation types (analytical, ODE, SDE) and
/// includes optional fields for covariates, secondary calculations, and
/// authoring metadata.
///
/// # Example
///
/// ```ignore
/// use pharmsol::json::JsonModel;
///
/// let json = r#"{
///     "schema": "2.0",
///     "id": "pk/1cmt-oral",
///     "type": "analytical",
///     "compartments": ["depot", "central"],
///     "analytical": "one_compartment_with_absorption",
///     "parameters": ["ka", "ke", "V"],
///     "outputs": [
///         { "id": "cp", "equation": "central / V" }
///     ]
/// }"#;
///
/// let model = JsonModel::from_str(json)?;
/// assert_eq!(model.id, "pk/1cmt-oral");
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct JsonModel {
    // Layer 1: Identity
    /// Schema version (`2.0`)
    pub schema: String,

    /// Unique model identifier
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

    // Layer 2: Structural Model
    /// Parameter names in fetch order
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parameters: Option<Vec<String>>,

    /// Compartment names (indexed in declaration order)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub compartments: Option<Vec<String>>,

    /// State variable names (for SDE)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub states: Option<Vec<String>>,

    // Equation Fields
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

    /// Secondary equations (ordered name→expression pairs)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub secondary: Option<Vec<NamedEquation>>,

    // Output
    /// Output definitions
    #[serde(skip_serializing_if = "Option::is_none")]
    pub outputs: Option<Vec<OutputDefinition>>,

    // Optional Features
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

    // Layer 3: Extensions and Metadata
    /// Enabled optional features
    #[serde(skip_serializing_if = "Option::is_none")]
    pub features: Option<Vec<Feature>>,

    /// Covariates used in this model.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub covariates: Option<Vec<CovariateDefinition>>,

    /// Metadata container for authoring documents
    #[serde(skip_serializing_if = "Option::is_none")]
    pub editor: Option<EditorInfo>,
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
            ModelType::Analytical => self.analytical.map_or(1, |func| func.num_states()),
            ModelType::Ode => self
                .compartments
                .as_ref()
                .map(|compartments| compartments.len())
                .or_else(|| {
                    self.diffeq.as_ref().map(|spec| match spec {
                        DiffEqSpec::Object(map) => map.len(),
                    })
                })
                .unwrap_or(1),
            ModelType::Sde => self
                .states
                .as_ref()
                .map(|states| states.len())
                .or_else(|| {
                    self.drift.as_ref().map(|spec| match spec {
                        DiffEqSpec::Object(map) => map.len(),
                    })
                })
                .unwrap_or(1),
        }
    }

    /// Get the number of outputs (inferred or explicit)
    pub fn num_outputs(&self) -> usize {
        if let Some((_, nout)) = self.neqs {
            return nout;
        }

        self.outputs.as_ref().map_or(1, |outputs| outputs.len())
    }

    /// Get the neqs tuple
    pub fn get_neqs(&self) -> (usize, usize) {
        self.neqs.unwrap_or((self.num_states(), self.num_outputs()))
    }

    /// Get compartment-to-index mapping
    pub fn compartment_map(&self) -> HashMap<String, usize> {
        let mut map = HashMap::new();
        if let Some(compartments) = &self.compartments {
            for (index, name) in compartments.iter().enumerate() {
                map.insert(name.clone(), index);
            }
        }
        map
    }

    /// Get state-to-index mapping (for SDE)
    pub fn state_map(&self) -> HashMap<String, usize> {
        let mut map = HashMap::new();
        if let Some(states) = &self.states {
            for (index, name) in states.iter().enumerate() {
                map.insert(name.clone(), index);
            }
        }
        map
    }

    /// Check if the model uses covariates
    pub fn has_covariates(&self) -> bool {
        self.covariates
            .as_ref()
            .is_some_and(|covariates| !covariates.is_empty())
    }

    /// Check if the model has secondary equations
    pub fn has_secondary(&self) -> bool {
        self.secondary
            .as_ref()
            .is_some_and(|secondary| !secondary.is_empty())
    }

    /// Get covariate symbols in declaration order.
    pub fn covariate_names(&self) -> Vec<String> {
        self.covariates
            .as_ref()
            .map(|covariates| covariates.iter().map(|cov| cov.id.clone()).collect())
            .unwrap_or_default()
    }

    /// Get display information from editor metadata.
    pub fn display_info(&self) -> Option<&DisplayInfo> {
        self.editor
            .as_ref()
            .and_then(|editor| editor.display.as_ref())
    }

    /// Get layout information from editor metadata.
    pub fn layout_info(&self) -> Option<&HashMap<String, Position>> {
        self.editor
            .as_ref()
            .and_then(|editor| editor.layout.as_ref())
    }

    /// Get documentation from editor metadata.
    pub fn documentation_info(&self) -> Option<&Documentation> {
        self.editor
            .as_ref()
            .and_then(|editor| editor.documentation.as_ref())
    }

    /// Check if the model uses lag times
    pub fn has_lag(&self) -> bool {
        self.lag.as_ref().is_some_and(|lag| !lag.is_empty())
    }

    /// Check if the model uses bioavailability
    pub fn has_fa(&self) -> bool {
        self.fa.as_ref().is_some_and(|fa| !fa.is_empty())
    }

    /// Check if the model has initial conditions
    pub fn has_init(&self) -> bool {
        self.init.is_some()
    }

    /// Get the parameters as a vector (guaranteed non-empty after validation)
    pub fn get_parameters(&self) -> Vec<String> {
        self.parameters.clone().unwrap_or_default()
    }

    /// Get outputs in canonical executable form.
    pub fn normalized_outputs(&self) -> Result<Vec<OutputDefinition>, JsonModelError> {
        self.outputs.clone().ok_or(JsonModelError::MissingOutput)
    }

    /// Get executable calculations in evaluation order.
    pub fn executable_calculations(&self) -> Vec<NamedEquation> {
        self.secondary.clone().unwrap_or_default()
    }

    /// Build the normalized executable representation for compile-time consumers.
    pub fn to_executable_model(&self) -> Result<ExecutableModel, JsonModelError> {
        Ok(ExecutableModel {
            id: self.id.clone(),
            model_type: self.model_type,
            parameters: self.parameters.clone().unwrap_or_default(),
            compartments: self.compartments.clone().unwrap_or_default(),
            states: self.states.clone().unwrap_or_default(),
            analytical: self.analytical,
            diffeq: self.diffeq.clone(),
            drift: self.drift.clone(),
            diffusion: self.diffusion.clone(),
            calculations: self.executable_calculations(),
            outputs: self.normalized_outputs()?,
            init: self.init.clone(),
            lag: self.lag.clone(),
            fa: self.fa.clone(),
            neqs: self.neqs,
            particles: self.particles,
            covariates: self.covariates.clone().unwrap_or_default(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_minimal_analytical() {
        let json = r#"{
            "schema": "2.0",
            "id": "pk/1cmt-iv",
            "type": "analytical",
            "analytical": "one_compartment",
            "parameters": ["ke", "V"],
            "compartments": ["central"],
            "outputs": [
                { "id": "cp", "equation": "central / V" }
            ]
        }"#;

        let model = JsonModel::from_str(json).unwrap();
        assert_eq!(model.id, "pk/1cmt-iv");
        assert_eq!(model.model_type, ModelType::Analytical);
        assert_eq!(model.analytical, Some(AnalyticalFunction::OneCompartment));
        assert_eq!(model.num_states(), 1);
        assert_eq!(model.num_outputs(), 1);
    }

    #[test]
    fn test_parse_minimal_ode() {
        let json = r#"{
            "schema": "2.0",
            "id": "pk/2cmt-ode",
            "type": "ode",
            "compartments": ["depot", "central", "peripheral"],
            "parameters": ["ka", "ke", "k12", "k21", "V"],
            "diffeq": {
                "depot": "-ka * depot",
                "central": "ka * depot - ke * central - k12 * central + k21 * peripheral + rateiv[1]",
                "peripheral": "k12 * central - k21 * peripheral"
            },
            "outputs": [
                { "id": "cp", "equation": "central / V" }
            ],
            "neqs": [3, 1]
        }"#;

        let model = JsonModel::from_str(json).unwrap();
        assert_eq!(model.id, "pk/2cmt-ode");
        assert_eq!(model.model_type, ModelType::Ode);
        assert_eq!(model.num_states(), 3);
        assert_eq!(model.compartment_map().get("central"), Some(&1));
    }

    #[test]
    fn test_parse_sde() {
        let json = r#"{
            "schema": "2.0",
            "id": "pk/1cmt-sde",
            "type": "sde",
            "parameters": ["ke0", "sigma_ke", "V"],
            "states": ["amount", "ke"],
            "drift": {
                "amount": "-ke * amount",
                "ke": "-0.5 * (ke - ke0)"
            },
            "diffusion": {
                "ke": "sigma_ke"
            },
            "init": {
                "ke": "ke0"
            },
            "outputs": [
                { "id": "cp", "equation": "amount / V" }
            ],
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
            "compartments": ["central"],
            "diffeq": { "central": "-ke * central" },
            "outputs": [{ "id": "cp", "equation": "central" }]
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
            "schema": "2.0",
            "id": "test",
            "type": "ode",
            "parameters": ["ke"],
            "compartments": ["central"],
            "diffeq": { "central": "-ke * central" },
            "outputs": [{ "id": "cp", "equation": "central" }],
            "unknown_field": "should fail"
        }"#;

        let result = JsonModel::from_str(json);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_with_covariates_and_editor() {
        let json = r#"{
            "schema": "2.0",
            "id": "pk/1cmt-oral",
            "type": "analytical",
            "analytical": "one_compartment_with_absorption",
            "parameters": ["ka", "ke", "V"],
            "compartments": ["depot", "central"],
            "outputs": [
                { "id": "cp", "equation": "central / V" }
            ],
            "secondary": [
                { "id": "ke_scaled", "equation": "ke * 1.0" }
            ],
            "covariates": [
                { "id": "wt", "column": "WT", "reference": 70.0 }
            ],
            "editor": {
                "display": { "name": "One Compartment Oral" }
            }
        }"#;

        let model = JsonModel::from_str(json).unwrap();
        assert_eq!(model.schema, "2.0");
        assert_eq!(model.id, "pk/1cmt-oral");
        assert_eq!(model.outputs.as_ref().unwrap()[0].id, "cp");
        assert_eq!(model.covariate_names(), vec!["wt".to_string()]);
        assert_eq!(model.executable_calculations()[0].id, "ke_scaled");
        assert_eq!(
            model
                .display_info()
                .and_then(|display| display.name.as_deref()),
            Some("One Compartment Oral")
        );
    }

    #[test]
    fn test_to_executable_model_preserves_outputs_and_calculations() {
        let json = r#"{
            "schema": "2.0",
            "id": "pk/1cmt-exec",
            "type": "ode",
            "parameters": ["CL", "V"],
            "compartments": ["central"],
            "diffeq": { "central": "-ke * central" },
            "outputs": [
                { "id": "cp", "equation": "central / V" }
            ],
            "secondary": [
                { "id": "ke", "equation": "CL / V" },
                { "id": "half_life", "equation": "0.693 / ke" }
            ],
            "covariates": [
                { "id": "wt", "column": "WT" }
            ]
        }"#;

        let model = JsonModel::from_str(json).unwrap();
        let executable = model.to_executable_model().unwrap();

        assert_eq!(executable.outputs[0].id, "cp");
        assert_eq!(
            executable
                .calculations
                .iter()
                .map(|entry| entry.id.as_str())
                .collect::<Vec<_>>(),
            vec!["ke", "half_life"]
        );
        assert_eq!(executable.covariates[0].column_name(), "WT");
    }
}
