//! Validation for JSON models

use std::collections::HashSet;

use crate::json::errors::JsonModelError;
use crate::json::model::JsonModel;
use crate::json::types::*;

/// A validated JSON model
///
/// This wrapper type guarantees that the contained model has passed
/// all validation checks and is ready for code generation.
#[derive(Debug, Clone)]
pub struct ValidatedModel(JsonModel);

impl ValidatedModel {
    /// Get the inner JsonModel
    pub fn inner(&self) -> &JsonModel {
        &self.0
    }

    /// Consume the wrapper and return the inner JsonModel
    pub fn into_inner(self) -> JsonModel {
        self.0
    }
}

/// Validator for JSON models
pub struct Validator {
    /// Whether to treat warnings as errors
    strict: bool,
}

impl Default for Validator {
    fn default() -> Self {
        Self::new()
    }
}

impl Validator {
    /// Create a new validator
    pub fn new() -> Self {
        Self { strict: false }
    }

    /// Create a strict validator that treats warnings as errors
    pub fn strict() -> Self {
        Self { strict: true }
    }

    /// Validate a JSON model
    pub fn validate(&self, model: &JsonModel) -> Result<ValidatedModel, JsonModelError> {
        // 1. Validate type-specific requirements
        self.validate_type_requirements(model)?;

        // 2. Validate parameters
        self.validate_parameters(model)?;

        // 3. Validate output
        self.validate_output(model)?;

        // 4. Validate compartments/states
        self.validate_compartments(model)?;

        // 5. Validate covariates
        self.validate_covariates(model)?;

        // 6. Validate covariate effects
        self.validate_covariate_effects(model)?;

        // 7. Validate analytical function parameters
        if let Some(func) = &model.analytical {
            self.validate_analytical_params(model, func)?;
        }

        Ok(ValidatedModel(model.clone()))
    }

    /// Validate type-specific field requirements
    fn validate_type_requirements(&self, model: &JsonModel) -> Result<(), JsonModelError> {
        match model.model_type {
            ModelType::Analytical => {
                // Must have analytical function
                if model.analytical.is_none() {
                    return Err(JsonModelError::missing_field("analytical", "analytical"));
                }
                // Must not have ODE/SDE fields
                if model.diffeq.is_some() {
                    return Err(JsonModelError::invalid_field("diffeq", "analytical"));
                }
                if model.drift.is_some() {
                    return Err(JsonModelError::invalid_field("drift", "analytical"));
                }
                if model.diffusion.is_some() {
                    return Err(JsonModelError::invalid_field("diffusion", "analytical"));
                }
            }
            ModelType::Ode => {
                // Must have diffeq
                if model.diffeq.is_none() {
                    return Err(JsonModelError::missing_field("diffeq", "ode"));
                }
                // Must not have analytical/SDE fields
                if model.analytical.is_some() {
                    return Err(JsonModelError::invalid_field("analytical", "ode"));
                }
                if model.drift.is_some() {
                    return Err(JsonModelError::invalid_field("drift", "ode"));
                }
                if model.diffusion.is_some() {
                    return Err(JsonModelError::invalid_field("diffusion", "ode"));
                }
            }
            ModelType::Sde => {
                // Must have drift and diffusion
                if model.drift.is_none() {
                    return Err(JsonModelError::missing_field("drift", "sde"));
                }
                if model.diffusion.is_none() {
                    return Err(JsonModelError::missing_field("diffusion", "sde"));
                }
                // Must not have analytical/ODE fields
                if model.analytical.is_some() {
                    return Err(JsonModelError::invalid_field("analytical", "sde"));
                }
                if model.diffeq.is_some() {
                    return Err(JsonModelError::invalid_field("diffeq", "sde"));
                }
            }
        }
        Ok(())
    }

    /// Validate parameters
    fn validate_parameters(&self, model: &JsonModel) -> Result<(), JsonModelError> {
        // Parameters required unless using extends
        if model.extends.is_none() && model.parameters.is_none() {
            return Err(JsonModelError::MissingParameters);
        }

        if let Some(params) = &model.parameters {
            // Check for duplicates
            let mut seen = HashSet::new();
            for param in params {
                if !seen.insert(param.clone()) {
                    return Err(JsonModelError::DuplicateParameter {
                        name: param.clone(),
                    });
                }
            }

            // Check for empty parameters
            if params.is_empty() && model.extends.is_none() {
                return Err(JsonModelError::MissingParameters);
            }
        }

        Ok(())
    }

    /// Validate output
    fn validate_output(&self, model: &JsonModel) -> Result<(), JsonModelError> {
        // Output required unless using extends
        if model.extends.is_none() && model.output.is_none() && model.outputs.is_none() {
            return Err(JsonModelError::MissingOutput);
        }

        // Check for empty output
        if let Some(output) = &model.output {
            if output.trim().is_empty() {
                return Err(JsonModelError::EmptyExpression {
                    context: "output".to_string(),
                });
            }
        }

        // Check outputs array
        if let Some(outputs) = &model.outputs {
            for (i, out) in outputs.iter().enumerate() {
                if out.equation.trim().is_empty() {
                    return Err(JsonModelError::EmptyExpression {
                        context: format!("outputs[{}]", i),
                    });
                }
            }
        }

        Ok(())
    }

    /// Validate compartments
    fn validate_compartments(&self, model: &JsonModel) -> Result<(), JsonModelError> {
        if let Some(compartments) = &model.compartments {
            let mut seen = HashSet::new();
            for cmt in compartments {
                if !seen.insert(cmt.clone()) {
                    return Err(JsonModelError::DuplicateCompartment { name: cmt.clone() });
                }
            }
        }

        if let Some(states) = &model.states {
            let mut seen = HashSet::new();
            for state in states {
                if !seen.insert(state.clone()) {
                    return Err(JsonModelError::DuplicateCompartment {
                        name: state.clone(),
                    });
                }
            }
        }

        Ok(())
    }

    /// Validate covariate definitions
    fn validate_covariates(&self, model: &JsonModel) -> Result<(), JsonModelError> {
        if let Some(covariates) = &model.covariates {
            let mut seen = HashSet::new();
            for cov in covariates {
                if !seen.insert(cov.id.clone()) {
                    return Err(JsonModelError::UndefinedCovariate {
                        name: format!("duplicate covariate: {}", cov.id),
                    });
                }
            }
        }
        Ok(())
    }

    /// Validate covariate effects
    fn validate_covariate_effects(&self, model: &JsonModel) -> Result<(), JsonModelError> {
        if let Some(effects) = &model.covariate_effects {
            let params: HashSet<_> = model
                .parameters
                .as_ref()
                .map(|p| p.iter().cloned().collect())
                .unwrap_or_default();

            let covariates: HashSet<_> = model
                .covariates
                .as_ref()
                .map(|c| c.iter().map(|cov| cov.id.clone()).collect())
                .unwrap_or_default();

            for effect in effects {
                // Check that target parameter exists
                if !params.is_empty() && !params.contains(&effect.on) {
                    return Err(JsonModelError::InvalidCovariateEffectTarget {
                        parameter: effect.on.clone(),
                    });
                }

                // Check type-specific requirements
                match effect.effect_type {
                    CovariateEffectType::Allometric => {
                        if effect.covariate.is_none() {
                            return Err(JsonModelError::MissingCovariateEffectField {
                                effect_type: "allometric".to_string(),
                                field: "covariate".to_string(),
                            });
                        }
                        if effect.exponent.is_none() {
                            return Err(JsonModelError::MissingCovariateEffectField {
                                effect_type: "allometric".to_string(),
                                field: "exponent".to_string(),
                            });
                        }
                    }
                    CovariateEffectType::Linear | CovariateEffectType::Exponential => {
                        if effect.covariate.is_none() {
                            return Err(JsonModelError::MissingCovariateEffectField {
                                effect_type: format!("{:?}", effect.effect_type).to_lowercase(),
                                field: "covariate".to_string(),
                            });
                        }
                        if effect.slope.is_none() {
                            return Err(JsonModelError::MissingCovariateEffectField {
                                effect_type: format!("{:?}", effect.effect_type).to_lowercase(),
                                field: "slope".to_string(),
                            });
                        }
                    }
                    CovariateEffectType::Custom => {
                        if effect.expression.is_none() {
                            return Err(JsonModelError::MissingCovariateEffectField {
                                effect_type: "custom".to_string(),
                                field: "expression".to_string(),
                            });
                        }
                    }
                    CovariateEffectType::Categorical => {
                        if effect.covariate.is_none() {
                            return Err(JsonModelError::MissingCovariateEffectField {
                                effect_type: "categorical".to_string(),
                                field: "covariate".to_string(),
                            });
                        }
                        if effect.levels.is_none() {
                            return Err(JsonModelError::MissingCovariateEffectField {
                                effect_type: "categorical".to_string(),
                                field: "levels".to_string(),
                            });
                        }
                    }
                    CovariateEffectType::Proportional => {
                        if effect.covariate.is_none() {
                            return Err(JsonModelError::MissingCovariateEffectField {
                                effect_type: "proportional".to_string(),
                                field: "covariate".to_string(),
                            });
                        }
                    }
                }

                // Check that referenced covariate exists
                if let Some(cov_name) = &effect.covariate {
                    if !covariates.is_empty() && !covariates.contains(cov_name) {
                        return Err(JsonModelError::UndefinedCovariate {
                            name: cov_name.clone(),
                        });
                    }
                }
            }
        }
        Ok(())
    }

    /// Validate analytical function parameters
    fn validate_analytical_params(
        &self,
        model: &JsonModel,
        func: &AnalyticalFunction,
    ) -> Result<(), JsonModelError> {
        let expected = func.expected_parameters();
        let actual = model.get_parameters();

        // Check if expected parameters are present at the start (in order)
        // Extra parameters (like V, tlag) are allowed after
        if self.strict && actual.len() >= expected.len() {
            let actual_prefix: Vec<_> = actual.iter().take(expected.len()).cloned().collect();
            let expected_vec: Vec<_> = expected.iter().map(|s| s.to_string()).collect();

            if actual_prefix != expected_vec {
                return Err(JsonModelError::ParameterOrderWarning {
                    function: func.rust_name().to_string(),
                    expected: expected_vec,
                    actual: actual_prefix,
                });
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_missing_analytical() {
        let json = r#"{
            "schema": "1.0",
            "id": "test",
            "type": "analytical",
            "parameters": ["ke"],
            "output": "x[0]"
        }"#;

        let model = JsonModel::from_str(json).unwrap();
        let result = Validator::new().validate(&model);
        assert!(matches!(
            result,
            Err(JsonModelError::MissingField { field, .. }) if field == "analytical"
        ));
    }

    #[test]
    fn test_validate_missing_diffeq() {
        let json = r#"{
            "schema": "1.0",
            "id": "test",
            "type": "ode",
            "parameters": ["ke"],
            "output": "x[0]"
        }"#;

        let model = JsonModel::from_str(json).unwrap();
        let result = Validator::new().validate(&model);
        assert!(matches!(
            result,
            Err(JsonModelError::MissingField { field, .. }) if field == "diffeq"
        ));
    }

    #[test]
    fn test_validate_invalid_field_for_type() {
        let json = r#"{
            "schema": "1.0",
            "id": "test",
            "type": "analytical",
            "analytical": "one_compartment",
            "diffeq": "dx[0] = -ke * x[0];",
            "parameters": ["ke"],
            "output": "x[0]"
        }"#;

        let model = JsonModel::from_str(json).unwrap();
        let result = Validator::new().validate(&model);
        assert!(matches!(
            result,
            Err(JsonModelError::InvalidFieldForType { field, .. }) if field == "diffeq"
        ));
    }

    #[test]
    fn test_validate_duplicate_parameter() {
        let json = r#"{
            "schema": "1.0",
            "id": "test",
            "type": "ode",
            "parameters": ["ke", "V", "ke"],
            "diffeq": "dx[0] = -ke * x[0];",
            "output": "x[0]"
        }"#;

        let model = JsonModel::from_str(json).unwrap();
        let result = Validator::new().validate(&model);
        assert!(matches!(
            result,
            Err(JsonModelError::DuplicateParameter { name }) if name == "ke"
        ));
    }

    #[test]
    fn test_validate_valid_model() {
        let json = r#"{
            "schema": "1.0",
            "id": "pk_1cmt_oral",
            "type": "analytical",
            "analytical": "one_compartment_with_absorption",
            "parameters": ["ka", "ke", "V"],
            "output": "x[1] / V"
        }"#;

        let model = JsonModel::from_str(json).unwrap();
        let result = Validator::new().validate(&model);
        assert!(result.is_ok());
    }
}
