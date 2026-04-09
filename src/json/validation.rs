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

    /// Get the normalized executable model.
    pub fn executable(&self) -> Result<ExecutableModel, JsonModelError> {
        self.0.to_executable_model()
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
        self.validate_type_requirements(model)?;
        self.validate_parameters(model)?;
        self.validate_output(model)?;
        self.validate_compartments(model)?;
        self.validate_covariates(model)?;
        self.validate_extra_schema_conventions(model)?;
        self.validate_expressions(model)?;
        self.validate_equation_keys(model)?;
        self.validate_expression_identifiers(model)?;

        if let Some(func) = &model.analytical {
            self.validate_analytical_params(model, func)?;
        }

        Ok(ValidatedModel(model.clone()))
    }

    /// Validate type-specific field requirements
    fn validate_type_requirements(&self, model: &JsonModel) -> Result<(), JsonModelError> {
        match model.model_type {
            ModelType::Analytical => {
                if model.analytical.is_none() {
                    return Err(JsonModelError::missing_field("analytical", "analytical"));
                }
                if model.compartments.is_none() {
                    return Err(JsonModelError::missing_field("compartments", "analytical"));
                }
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
                if model.diffeq.is_none() {
                    return Err(JsonModelError::missing_field("diffeq", "ode"));
                }
                if model.compartments.is_none() {
                    return Err(JsonModelError::missing_field("compartments", "ode"));
                }
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
                if model.drift.is_none() {
                    return Err(JsonModelError::missing_field("drift", "sde"));
                }
                if model.diffusion.is_none() {
                    return Err(JsonModelError::missing_field("diffusion", "sde"));
                }
                if model.states.is_none() {
                    return Err(JsonModelError::missing_field("states", "sde"));
                }
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
        if model.extends.is_none() && model.parameters.is_none() {
            return Err(JsonModelError::MissingParameters);
        }

        if let Some(params) = &model.parameters {
            let mut seen = HashSet::new();
            for param in params {
                if !seen.insert(param.clone()) {
                    return Err(JsonModelError::DuplicateParameter {
                        name: param.clone(),
                    });
                }
            }

            if params.is_empty() && model.extends.is_none() {
                return Err(JsonModelError::MissingParameters);
            }
        }

        Ok(())
    }

    /// Validate outputs
    fn validate_output(&self, model: &JsonModel) -> Result<(), JsonModelError> {
        if model.extends.is_none() && model.outputs.is_none() {
            return Err(JsonModelError::MissingOutput);
        }

        if model.extends.is_some() && model.outputs.is_none() {
            return Ok(());
        }

        let outputs = model.normalized_outputs()?;
        let mut seen_ids = HashSet::new();

        for (index, output) in outputs.iter().enumerate() {
            if output.equation.trim().is_empty() {
                return Err(JsonModelError::EmptyExpression {
                    context: format!("outputs[{}]", index),
                });
            }

            if !seen_ids.insert(output.id.clone()) {
                return Err(JsonModelError::DuplicateOutput {
                    id: output.id.clone(),
                });
            }
        }

        Ok(())
    }

    /// Validate compartments and states
    fn validate_compartments(&self, model: &JsonModel) -> Result<(), JsonModelError> {
        if let Some(compartments) = &model.compartments {
            let mut seen = HashSet::new();
            for name in compartments {
                if !seen.insert(name.clone()) {
                    return Err(JsonModelError::DuplicateCompartment { name: name.clone() });
                }
            }
        }

        if let Some(states) = &model.states {
            let mut seen = HashSet::new();
            for name in states {
                if !seen.insert(name.clone()) {
                    return Err(JsonModelError::DuplicateCompartment { name: name.clone() });
                }
            }
        }

        Ok(())
    }

    /// Validate covariates
    fn validate_covariates(&self, model: &JsonModel) -> Result<(), JsonModelError> {
        if let Some(covariates) = &model.covariates {
            let mut seen = HashSet::new();
            for covariate in covariates {
                if !seen.insert(covariate.id.clone()) {
                    return Err(JsonModelError::DuplicateCovariate {
                        name: covariate.id.clone(),
                    });
                }
            }
        }

        Ok(())
    }

    /// Reject duplicate secondary ids and numeric init/lag/fa keys.
    fn validate_extra_schema_conventions(&self, model: &JsonModel) -> Result<(), JsonModelError> {
        if let Some(secondary) = &model.secondary {
            let mut seen = HashSet::new();
            for entry in secondary {
                if !seen.insert(entry.id.clone()) {
                    return Err(JsonModelError::schema_rule(
                        "secondary",
                        &model.schema,
                        format!("duplicate secondary identifier '{}'", entry.id),
                    ));
                }
            }
        }

        if let Some(init) = &model.init {
            let InitSpec::Object(map) = init;
            for key in map.keys() {
                if key.parse::<usize>().is_ok() {
                    return Err(JsonModelError::schema_rule(
                        "init",
                        &model.schema,
                        format!(
                            "numeric key '{}' is not canonical; use compartment or state ids",
                            key
                        ),
                    ));
                }
            }
        }

        if let Some(lag) = &model.lag {
            for key in lag.keys() {
                if key.parse::<usize>().is_ok() {
                    return Err(JsonModelError::schema_rule(
                        "lag",
                        &model.schema,
                        format!(
                            "numeric key '{}' is not canonical; use compartment ids",
                            key
                        ),
                    ));
                }
            }
        }

        if let Some(fa) = &model.fa {
            for key in fa.keys() {
                if key.parse::<usize>().is_ok() {
                    return Err(JsonModelError::schema_rule(
                        "fa",
                        &model.schema,
                        format!(
                            "numeric key '{}' is not canonical; use compartment ids",
                            key
                        ),
                    ));
                }
            }
        }

        Ok(())
    }

    /// Validate that all expressions in the model parse correctly
    fn validate_expressions(&self, model: &JsonModel) -> Result<(), JsonModelError> {
        use crate::json::expression;

        if let Some(outputs) = &model.outputs {
            for (index, output) in outputs.iter().enumerate() {
                expression::parse(&output.equation).map_err(|error| {
                    JsonModelError::ExpressionParseError {
                        context: format!("outputs[{}]", index),
                        message: error.to_string(),
                    }
                })?;
            }
        }

        if let Some(diffeq) = &model.diffeq {
            let DiffEqSpec::Object(map) = diffeq;
            for (name, expr) in map {
                expression::parse(expr).map_err(|error| JsonModelError::ExpressionParseError {
                    context: format!("diffeq.{}", name),
                    message: error.to_string(),
                })?;
            }
        }

        if let Some(drift) = &model.drift {
            let DiffEqSpec::Object(map) = drift;
            for (name, expr) in map {
                expression::parse(expr).map_err(|error| JsonModelError::ExpressionParseError {
                    context: format!("drift.{}", name),
                    message: error.to_string(),
                })?;
            }
        }

        if let Some(diffusion) = &model.diffusion {
            for (name, expr) in diffusion {
                if let ExpressionOrNumber::Expression(expr) = expr {
                    expression::parse(expr).map_err(|error| {
                        JsonModelError::ExpressionParseError {
                            context: format!("diffusion.{}", name),
                            message: error.to_string(),
                        }
                    })?;
                }
            }
        }

        if let Some(secondary) = &model.secondary {
            for entry in secondary {
                expression::parse(&entry.equation).map_err(|error| {
                    JsonModelError::ExpressionParseError {
                        context: format!("secondary.{}", entry.id),
                        message: error.to_string(),
                    }
                })?;
            }
        }

        if let Some(init) = &model.init {
            let InitSpec::Object(map) = init;
            for (name, expr) in map {
                if let ExpressionOrNumber::Expression(expr) = expr {
                    expression::parse(expr).map_err(|error| {
                        JsonModelError::ExpressionParseError {
                            context: format!("init.{}", name),
                            message: error.to_string(),
                        }
                    })?;
                }
            }
        }

        if let Some(lag) = &model.lag {
            for (name, expr) in lag {
                if let ExpressionOrNumber::Expression(expr) = expr {
                    expression::parse(expr).map_err(|error| {
                        JsonModelError::ExpressionParseError {
                            context: format!("lag.{}", name),
                            message: error.to_string(),
                        }
                    })?;
                }
            }
        }

        if let Some(fa) = &model.fa {
            for (name, expr) in fa {
                if let ExpressionOrNumber::Expression(expr) = expr {
                    expression::parse(expr).map_err(|error| {
                        JsonModelError::ExpressionParseError {
                            context: format!("fa.{}", name),
                            message: error.to_string(),
                        }
                    })?;
                }
            }
        }

        Ok(())
    }

    /// Validate equation keys against declared names
    fn validate_equation_keys(&self, model: &JsonModel) -> Result<(), JsonModelError> {
        if let (Some(diffeq), Some(compartments)) = (&model.diffeq, &model.compartments) {
            let DiffEqSpec::Object(map) = diffeq;
            for name in map.keys() {
                if !compartments.contains(name) {
                    return Err(JsonModelError::UndefinedCompartment { name: name.clone() });
                }
            }
        }

        if let (Some(drift), Some(states)) = (&model.drift, &model.states) {
            let DiffEqSpec::Object(map) = drift;
            for name in map.keys() {
                if !states.contains(name) {
                    return Err(JsonModelError::UndefinedCompartment { name: name.clone() });
                }
            }
        }

        if let (Some(diffusion), Some(states)) = (&model.diffusion, &model.states) {
            for name in diffusion.keys() {
                if !states.contains(name) {
                    return Err(JsonModelError::UndefinedCompartment { name: name.clone() });
                }
            }
        }

        if let (Some(lag), Some(compartments)) = (&model.lag, &model.compartments) {
            for name in lag.keys() {
                if !compartments.contains(name) {
                    return Err(JsonModelError::UndefinedCompartment { name: name.clone() });
                }
            }
        }

        if let (Some(fa), Some(compartments)) = (&model.fa, &model.compartments) {
            for name in fa.keys() {
                if !compartments.contains(name) {
                    return Err(JsonModelError::UndefinedCompartment { name: name.clone() });
                }
            }
        }

        if let Some(init) = &model.init {
            let InitSpec::Object(map) = init;
            let mut valid_keys = HashSet::new();
            if let Some(compartments) = &model.compartments {
                valid_keys.extend(compartments.iter().cloned());
            }
            if let Some(states) = &model.states {
                valid_keys.extend(states.iter().cloned());
            }

            for name in map.keys() {
                if !valid_keys.contains(name) {
                    return Err(JsonModelError::UndefinedCompartment { name: name.clone() });
                }
            }
        }

        Ok(())
    }

    /// Validate that expression identifiers reference declared names
    fn validate_expression_identifiers(&self, model: &JsonModel) -> Result<(), JsonModelError> {
        use crate::json::expression;

        let mut known: HashSet<String> = HashSet::new();

        if let Some(params) = &model.parameters {
            known.extend(params.iter().cloned());
        }
        if let Some(compartments) = &model.compartments {
            known.extend(compartments.iter().cloned());
        }
        if let Some(states) = &model.states {
            known.extend(states.iter().cloned());
        }
        if let Some(covariates) = &model.covariates {
            known.extend(covariates.iter().map(|cov| cov.id.clone()));
        }

        known.extend(
            ["x", "dx", "b", "rateiv", "y", "t", "d", "p"]
                .iter()
                .map(|name| name.to_string()),
        );

        let validate_expr = |expr_str: &str,
                             context: &str,
                             extra: &HashSet<String>|
         -> Result<(), JsonModelError> {
            let ast = match expression::parse(expr_str) {
                Ok(ast) => ast,
                Err(_) => return Ok(()),
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

        let mut calculation_names: HashSet<String> = HashSet::new();
        for entry in model.executable_calculations() {
            validate_expr(
                &entry.equation,
                &format!("calculation.{}", entry.id),
                &calculation_names,
            )?;
            calculation_names.insert(entry.id);
        }

        if let Some(outputs) = &model.outputs {
            for (index, output) in outputs.iter().enumerate() {
                validate_expr(
                    &output.equation,
                    &format!("outputs[{}]", index),
                    &calculation_names,
                )?;
            }
        }

        if let Some(diffeq) = &model.diffeq {
            let DiffEqSpec::Object(map) = diffeq;
            for (name, expr) in map {
                validate_expr(expr, &format!("diffeq.{}", name), &calculation_names)?;
            }
        }

        if let Some(drift) = &model.drift {
            let DiffEqSpec::Object(map) = drift;
            for (name, expr) in map {
                validate_expr(expr, &format!("drift.{}", name), &calculation_names)?;
            }
        }

        if let Some(diffusion) = &model.diffusion {
            for (name, expr) in diffusion {
                if let ExpressionOrNumber::Expression(expr) = expr {
                    validate_expr(expr, &format!("diffusion.{}", name), &calculation_names)?;
                }
            }
        }

        if let Some(init) = &model.init {
            let InitSpec::Object(map) = init;
            for (name, expr) in map {
                if let ExpressionOrNumber::Expression(expr) = expr {
                    validate_expr(expr, &format!("init.{}", name), &calculation_names)?;
                }
            }
        }

        if let Some(lag) = &model.lag {
            for (name, expr) in lag {
                if let ExpressionOrNumber::Expression(expr) = expr {
                    validate_expr(expr, &format!("lag.{}", name), &calculation_names)?;
                }
            }
        }

        if let Some(fa) = &model.fa {
            for (name, expr) in fa {
                if let ExpressionOrNumber::Expression(expr) = expr {
                    validate_expr(expr, &format!("fa.{}", name), &calculation_names)?;
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

        if self.strict && actual.len() >= expected.len() {
            let actual_prefix: Vec<_> = actual.iter().take(expected.len()).cloned().collect();
            let expected_vec: Vec<_> = expected.iter().map(|name| name.to_string()).collect();

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
            "schema": "2.0",
            "id": "test",
            "type": "analytical",
            "parameters": ["ke", "V"],
            "compartments": ["central"],
            "outputs": [{ "id": "cp", "equation": "central / V" }]
        }"#;

        let model = JsonModel::from_str(json).unwrap();
        let result = Validator::new().validate(&model);
        assert!(matches!(
            result,
            Err(JsonModelError::MissingField { field, .. }) if field == "analytical"
        ));
    }

    #[test]
    fn test_validate_missing_outputs() {
        let json = r#"{
            "schema": "2.0",
            "id": "test",
            "type": "ode",
            "parameters": ["ke", "V"],
            "compartments": ["central"],
            "diffeq": { "central": "-ke * central" }
        }"#;

        let model = JsonModel::from_str(json).unwrap();
        let result = Validator::new().validate(&model);
        assert!(matches!(result, Err(JsonModelError::MissingOutput)));
    }

    #[test]
    fn test_validate_duplicate_outputs() {
        let json = r#"{
            "schema": "2.0",
            "id": "test",
            "type": "ode",
            "parameters": ["ke", "V"],
            "compartments": ["central"],
            "diffeq": { "central": "-ke * central" },
            "outputs": [
                { "id": "cp", "equation": "central / V" },
                { "id": "cp", "equation": "central / V" }
            ]
        }"#;

        let model = JsonModel::from_str(json).unwrap();
        let result = Validator::new().validate(&model);
        assert!(matches!(
            result,
            Err(JsonModelError::DuplicateOutput { .. })
        ));
    }

    #[test]
    fn test_validate_rejects_numeric_lag_keys() {
        let json = r#"{
            "schema": "2.0",
            "id": "test",
            "type": "analytical",
            "analytical": "one_compartment_with_absorption",
            "parameters": ["ka", "ke", "V", "tlag"],
            "compartments": ["depot", "central"],
            "lag": { "0": "tlag" },
            "outputs": [{ "id": "cp", "equation": "central / V" }]
        }"#;

        let model = JsonModel::from_str(json).unwrap();
        let result = Validator::new().validate(&model);
        assert!(matches!(
            result,
            Err(JsonModelError::SchemaRuleViolation { .. })
        ));
    }

    #[test]
    fn test_validate_accepts_minimal_v2_model() {
        let json = r#"{
            "schema": "2.0",
            "id": "pk/1cmt-iv-ode",
            "type": "ode",
            "parameters": ["ke", "V"],
            "compartments": ["central"],
            "diffeq": { "central": "-ke * central + rateiv[0]" },
            "outputs": [{ "id": "cp", "equation": "central / V" }]
        }"#;

        let model = JsonModel::from_str(json).unwrap();
        Validator::new().validate(&model).unwrap();
    }

    #[test]
    fn test_validate_accepts_bolus_symbol() {
        let json = r#"{
            "schema": "2.0",
            "id": "pk/1cmt-bolus",
            "type": "ode",
            "parameters": ["ke", "V"],
            "compartments": ["central"],
            "diffeq": { "central": "-ke * central + b[0]" },
            "outputs": [{ "id": "cp", "equation": "central / V" }]
        }"#;

        let model = JsonModel::from_str(json).unwrap();
        Validator::new().validate(&model).unwrap();
    }

    #[test]
    fn test_validate_secondary_can_feed_outputs_and_diffeq() {
        let json = r#"{
            "schema": "2.0",
            "id": "pk/1cmt-derived",
            "type": "ode",
            "parameters": ["CL", "V"],
            "compartments": ["central"],
            "secondary": [
                { "id": "ke", "equation": "CL / V" }
            ],
            "diffeq": { "central": "-ke * central" },
            "outputs": [{ "id": "cp", "equation": "central / V" }]
        }"#;

        let model = JsonModel::from_str(json).unwrap();
        Validator::new().validate(&model).unwrap();
    }

    #[test]
    fn test_validate_rejects_unknown_compartment_reference() {
        let json = r#"{
            "schema": "2.0",
            "id": "test",
            "type": "ode",
            "parameters": ["ke", "V"],
            "compartments": ["central"],
            "diffeq": { "peripheral": "-ke * peripheral" },
            "outputs": [{ "id": "cp", "equation": "central / V" }]
        }"#;

        let model = JsonModel::from_str(json).unwrap();
        let result = Validator::new().validate(&model);
        assert!(matches!(
            result,
            Err(JsonModelError::UndefinedCompartment { .. })
        ));
    }

    #[test]
    fn test_validate_rejects_invalid_expression() {
        let json = r#"{
            "schema": "2.0",
            "id": "test",
            "type": "ode",
            "parameters": ["ke", "V"],
            "compartments": ["central"],
            "diffeq": { "central": "ke * central +" },
            "outputs": [{ "id": "cp", "equation": "central / V" }]
        }"#;

        let model = JsonModel::from_str(json).unwrap();
        let result = Validator::new().validate(&model);
        assert!(matches!(
            result,
            Err(JsonModelError::ExpressionParseError { .. })
        ));
    }
}
