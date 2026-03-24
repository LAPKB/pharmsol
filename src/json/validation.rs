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

        // 6. Validate expressions parse correctly
        self.validate_expressions(model)?;

        // 7. Validate equation keys match declared compartments/states
        self.validate_equation_keys(model)?;

        // 8. Validate expression identifiers reference declared names
        self.validate_expression_identifiers(model)?;

        // 9. Validate analytical function parameters
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

    /// Validate covariate definitions (just a list of names now)
    fn validate_covariates(&self, model: &JsonModel) -> Result<(), JsonModelError> {
        if let Some(covariates) = &model.covariates {
            let mut seen = HashSet::new();
            for cov in covariates {
                if !seen.insert(cov.clone()) {
                    return Err(JsonModelError::DuplicateCovariate { name: cov.clone() });
                }
            }
        }
        Ok(())
    }

    /// Validate that all expressions in the model parse correctly
    fn validate_expressions(&self, model: &JsonModel) -> Result<(), JsonModelError> {
        use crate::json::expression;

        // Validate output expression
        if let Some(output) = &model.output {
            expression::parse(output).map_err(|e| JsonModelError::ExpressionParseError {
                context: "output".to_string(),
                message: e.to_string(),
            })?;
        }

        // Validate multiple outputs
        if let Some(outputs) = &model.outputs {
            for (i, out) in outputs.iter().enumerate() {
                expression::parse(&out.equation).map_err(|e| {
                    JsonModelError::ExpressionParseError {
                        context: format!("outputs[{}]", i),
                        message: e.to_string(),
                    }
                })?;
            }
        }

        // Validate diffeq expressions
        if let Some(DiffEqSpec::Object(map)) = &model.diffeq {
            for (name, expr) in map {
                expression::parse(expr).map_err(|e| JsonModelError::ExpressionParseError {
                    context: format!("diffeq.{}", name),
                    message: e.to_string(),
                })?;
            }
        }

        // Validate drift expressions (SDE)
        if let Some(DiffEqSpec::Object(map)) = &model.drift {
            for (name, expr) in map {
                expression::parse(expr).map_err(|e| JsonModelError::ExpressionParseError {
                    context: format!("drift.{}", name),
                    message: e.to_string(),
                })?;
            }
        }

        // Validate secondary equations
        if let Some(secondary) = &model.secondary {
            for (name, expr) in secondary {
                expression::parse(expr).map_err(|e| JsonModelError::ExpressionParseError {
                    context: format!("secondary.{}", name),
                    message: e.to_string(),
                })?;
            }
        }

        // Validate derived parameter expressions
        if let Some(derived) = &model.derived {
            for d in derived {
                expression::parse(&d.expression).map_err(|e| {
                    JsonModelError::ExpressionParseError {
                        context: format!("derived.{}", d.symbol),
                        message: e.to_string(),
                    }
                })?;
            }
        }

        Ok(())
    }

    /// Validate that diffeq/drift keys match declared compartments/states
    fn validate_equation_keys(&self, model: &JsonModel) -> Result<(), JsonModelError> {
        // Validate diffeq keys against compartments
        if let Some(DiffEqSpec::Object(map)) = &model.diffeq {
            if let Some(compartments) = &model.compartments {
                for name in map.keys() {
                    // Allow numeric indices as well as named compartments
                    if name.parse::<usize>().is_err() && !compartments.contains(name) {
                        return Err(JsonModelError::UndefinedCompartment { name: name.clone() });
                    }
                }
            }
        }

        // Validate drift keys against states
        if let Some(DiffEqSpec::Object(map)) = &model.drift {
            if let Some(states) = &model.states {
                for name in map.keys() {
                    if name.parse::<usize>().is_err() && !states.contains(name) {
                        // Also check compartments as fallback
                        let in_compartments = model
                            .compartments
                            .as_ref()
                            .is_some_and(|c| c.contains(name));
                        if !in_compartments {
                            return Err(JsonModelError::UndefinedCompartment {
                                name: name.clone(),
                            });
                        }
                    }
                }
            }
        }

        // Validate lag keys against compartments
        if let Some(lag) = &model.lag {
            if let Some(compartments) = &model.compartments {
                for name in lag.keys() {
                    if name.parse::<usize>().is_err() && !compartments.contains(name) {
                        return Err(JsonModelError::UndefinedCompartment { name: name.clone() });
                    }
                }
            }
        }

        // Validate fa keys against compartments
        if let Some(fa) = &model.fa {
            if let Some(compartments) = &model.compartments {
                for name in fa.keys() {
                    if name.parse::<usize>().is_err() && !compartments.contains(name) {
                        return Err(JsonModelError::UndefinedCompartment { name: name.clone() });
                    }
                }
            }
        }

        Ok(())
    }

    /// Validate that expression identifiers reference declared names
    fn validate_expression_identifiers(&self, model: &JsonModel) -> Result<(), JsonModelError> {
        use crate::json::expression;
        use std::collections::HashSet;

        // Build the set of known identifiers
        let mut known: HashSet<String> = HashSet::new();

        // Parameters
        if let Some(params) = &model.parameters {
            known.extend(params.iter().cloned());
        }

        // Compartments
        if let Some(compartments) = &model.compartments {
            known.extend(compartments.iter().cloned());
        }

        // States
        if let Some(states) = &model.states {
            known.extend(states.iter().cloned());
        }

        // Covariates
        if let Some(covariates) = &model.covariates {
            known.extend(covariates.iter().cloned());
        }

        // Built-in array names that can be indexed
        known.extend(
            ["x", "dx", "rateiv", "y", "t", "d", "p"]
                .iter()
                .map(|s| s.to_string()),
        );

        // Helper to validate a single expression's identifiers
        let validate_expr = |expr_str: &str,
                             context: &str,
                             extra: &HashSet<String>|
         -> Result<(), JsonModelError> {
            let ast = match expression::parse(expr_str) {
                Ok(ast) => ast,
                Err(_) => return Ok(()), // Syntax errors caught by validate_expressions
            };
            let ids = expression::collect_identifiers(&ast);
            let mut local_known = known.clone();
            local_known.extend(extra.iter().cloned());
            for id in &ids {
                if !local_known.contains(id) && !expression::is_known_function(id) {
                    return Err(JsonModelError::UndefinedParameter {
                        name: id.clone(),
                        context: context.to_string(),
                    });
                }
            }
            Ok(())
        };

        // Secondary equations — each can reference earlier ones
        let mut secondary_names: HashSet<String> = HashSet::new();
        if let Some(secondary) = &model.secondary {
            for (name, expr) in secondary {
                validate_expr(expr, &format!("secondary.{}", name), &secondary_names)?;
                secondary_names.insert(name.clone());
            }
        }

        // For remaining expressions, all secondary names are available
        let all_secondary = &secondary_names;

        // Derived parameters
        if let Some(derived) = &model.derived {
            let mut derived_names: HashSet<String> = HashSet::new();
            for d in derived {
                let mut extra = all_secondary.clone();
                extra.extend(derived_names.iter().cloned());
                validate_expr(&d.expression, &format!("derived.{}", d.symbol), &extra)?;
                derived_names.insert(d.symbol.clone());
            }
        }

        // Output
        if let Some(output) = &model.output {
            validate_expr(output, "output", all_secondary)?;
        }
        if let Some(outputs) = &model.outputs {
            for (i, out) in outputs.iter().enumerate() {
                validate_expr(&out.equation, &format!("outputs[{}]", i), all_secondary)?;
            }
        }

        // Diffeq
        if let Some(DiffEqSpec::Object(map)) = &model.diffeq {
            for (name, expr) in map {
                validate_expr(expr, &format!("diffeq.{}", name), all_secondary)?;
            }
        }

        // Drift
        if let Some(DiffEqSpec::Object(map)) = &model.drift {
            for (name, expr) in map {
                validate_expr(expr, &format!("drift.{}", name), all_secondary)?;
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

    #[test]
    fn test_validate_duplicate_covariate() {
        let json = r#"{
            "schema": "1.0",
            "id": "test",
            "type": "analytical",
            "analytical": "one_compartment",
            "parameters": ["ke", "V"],
            "covariates": ["wt", "wt"],
            "output": "x[0] / V"
        }"#;

        let model = JsonModel::from_str(json).unwrap();
        let result = Validator::new().validate(&model);
        assert!(
            matches!(result, Err(JsonModelError::DuplicateCovariate { name }) if name == "wt"),
            "Should reject duplicate covariates with DuplicateCovariate variant"
        );
    }

    #[test]
    fn test_validate_undefined_identifier_in_output() {
        let json = r#"{
            "schema": "1.0",
            "id": "test",
            "type": "analytical",
            "analytical": "one_compartment",
            "parameters": ["ke", "V"],
            "output": "x[0] / Vv"
        }"#;

        let model = JsonModel::from_str(json).unwrap();
        let result = Validator::new().validate(&model);
        assert!(
            matches!(result, Err(JsonModelError::UndefinedParameter { ref name, .. }) if name == "Vv"),
            "Should catch typo 'Vv' when 'V' is declared: {:?}",
            result
        );
    }

    #[test]
    fn test_validate_undefined_identifier_in_diffeq() {
        let json = r#"{
            "schema": "1.0",
            "id": "test",
            "type": "ode",
            "compartments": ["central"],
            "parameters": ["ke", "V"],
            "diffeq": { "central": "-ke * central + rateiv[0] + TYPO" },
            "output": "central / V"
        }"#;

        let model = JsonModel::from_str(json).unwrap();
        let result = Validator::new().validate(&model);
        assert!(
            matches!(result, Err(JsonModelError::UndefinedParameter { name, .. }) if name == "TYPO"),
            "Should catch undefined identifier 'TYPO' in diffeq"
        );
    }

    #[test]
    fn test_validate_secondary_can_reference_earlier() {
        let json = r#"{
            "schema": "1.0",
            "id": "test",
            "type": "analytical",
            "analytical": "one_compartment",
            "parameters": ["CL", "V"],
            "secondary": {
                "ke": "CL / V"
            },
            "output": "x[0] / V"
        }"#;

        let model = JsonModel::from_str(json).unwrap();
        let result = Validator::new().validate(&model);
        assert!(
            result.is_ok(),
            "Secondary can reference declared parameters"
        );
    }

    #[test]
    fn test_validate_secondary_order_matters() {
        let json = r#"{
            "schema": "1.0",
            "id": "test",
            "type": "analytical",
            "analytical": "one_compartment",
            "parameters": ["CLs", "Vs"],
            "covariates": ["wt"],
            "secondary": {
                "CL": "CLs * (wt / 70)^0.75",
                "V": "Vs * (wt / 70)",
                "ke": "CL / V"
            },
            "output": "x[0] / V"
        }"#;

        let model = JsonModel::from_str(json).unwrap();
        let result = Validator::new().validate(&model);
        assert!(
            result.is_ok(),
            "Secondary 'ke' can reference earlier 'CL' and 'V'"
        );
    }

    #[test]
    fn test_validate_diffeq_key_not_in_compartments() {
        let json = r#"{
            "schema": "1.0",
            "id": "test",
            "type": "ode",
            "compartments": ["central"],
            "parameters": ["ke", "V"],
            "diffeq": { "nonexistent": "-ke * x[0]" },
            "output": "x[0] / V"
        }"#;

        let model = JsonModel::from_str(json).unwrap();
        let result = Validator::new().validate(&model);
        assert!(
            matches!(result, Err(JsonModelError::UndefinedCompartment { name }) if name == "nonexistent"),
            "Should reject diffeq key not in compartments"
        );
    }

    #[test]
    fn test_validate_lag_key_not_in_compartments() {
        let json = r#"{
            "schema": "1.0",
            "id": "test",
            "type": "ode",
            "compartments": ["depot", "central"],
            "parameters": ["ka", "ke", "V", "tlag"],
            "diffeq": {
                "depot": "-ka * depot",
                "central": "ka * depot - ke * central"
            },
            "lag": { "nonexistent": "tlag" },
            "output": "central / V"
        }"#;

        let model = JsonModel::from_str(json).unwrap();
        let result = Validator::new().validate(&model);
        assert!(
            matches!(result, Err(JsonModelError::UndefinedCompartment { name }) if name == "nonexistent"),
            "Should reject lag key not in compartments"
        );
    }

    #[test]
    fn test_validate_allows_numeric_diffeq_keys() {
        let json = r#"{
            "schema": "1.0",
            "id": "test",
            "type": "ode",
            "compartments": ["central"],
            "parameters": ["ke", "V"],
            "diffeq": { "0": "-ke * x[0]" },
            "output": "x[0] / V"
        }"#;

        let model = JsonModel::from_str(json).unwrap();
        let result = Validator::new().validate(&model);
        assert!(result.is_ok(), "Numeric diffeq keys should be allowed");
    }

    #[test]
    fn test_validate_known_function_not_flagged() {
        let json = r#"{
            "schema": "1.0",
            "id": "test",
            "type": "analytical",
            "analytical": "one_compartment",
            "parameters": ["ke", "V"],
            "output": "exp(-ke * t) / V"
        }"#;

        let model = JsonModel::from_str(json).unwrap();
        let result = Validator::new().validate(&model);
        assert!(
            result.is_ok(),
            "Built-in functions like exp should not be flagged"
        );
    }
}
