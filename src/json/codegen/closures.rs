//! Closure generation for model equations
//!
//! This module generates the closure functions that are passed to
//! equation constructors (Analytical, ODE, SDE).
//!
//! All math expressions in the JSON model are written in a language-agnostic
//! format and transpiled to Rust via the expression parser.

use std::collections::{HashMap, HashSet};

use crate::json::errors::JsonModelError;
use crate::json::expression;
use crate::json::model::JsonModel;
use crate::json::types::*;

/// Generator for closure functions
pub struct ClosureGenerator<'a> {
    model: &'a JsonModel,
    compartment_map: HashMap<String, usize>,
    state_map: HashMap<String, usize>,
}

fn parse_bound_output_index(id: &str) -> Option<usize> {
    let raw_index = id.strip_prefix("out_")?;
    let parsed_index = raw_index.parse::<usize>().ok()?;
    parsed_index.checked_sub(1)
}

impl<'a> ClosureGenerator<'a> {
    /// Create a new closure generator
    pub fn new(model: &'a JsonModel) -> Self {
        Self {
            model,
            compartment_map: model.compartment_map(),
            state_map: model.state_map(),
        }
    }

    /// Get the combined name→index map (compartments + states)
    fn name_map(&self) -> HashMap<String, usize> {
        let mut map = self.compartment_map.clone();
        map.extend(self.state_map.iter().map(|(k, v)| (k.clone(), *v)));
        map
    }

    /// Resolve a declared compartment/state name to its index.
    /// Returns an error for unknown names.
    fn resolve_index(&self, name: &str, context: &str) -> Result<usize, JsonModelError> {
        if let Some(&idx) = self.compartment_map.get(name) {
            return Ok(idx);
        }
        if let Some(&idx) = self.state_map.get(name) {
            return Ok(idx);
        }
        Err(JsonModelError::CodeGenError(format!(
            "Unknown compartment or state '{}' in {}",
            name, context
        )))
    }

    /// Transpile a math expression to Rust code, resolving named compartment indices
    fn transpile(&self, expr: &str) -> Result<String, JsonModelError> {
        let name_map = self.name_map();
        expression::to_rust_with_names(expr, &name_map).map_err(|e| {
            JsonModelError::ExpressionParseError {
                context: "codegen".to_string(),
                message: e.to_string(),
            }
        })
    }

    /// Generate the fetch_params! macro call
    fn fetch_params(&self) -> String {
        let params = self.model.get_parameters();
        if params.is_empty() {
            return String::new();
        }
        format!("fetch_params!(p, {});", params.join(", "))
    }

    /// Generate compartment bindings (e.g., let central = x[0];)
    fn generate_compartment_bindings(&self) -> String {
        if self.compartment_map.is_empty() {
            return String::new();
        }

        let mut bindings: Vec<_> = self
            .compartment_map
            .iter()
            .map(|(name, &idx)| format!("let {} = x[{}];", name, idx))
            .collect();
        bindings.sort(); // Consistent ordering
        bindings.join("\n        ")
    }

    /// Generate state bindings for SDE (e.g., let state0 = x[0];)
    fn generate_state_bindings(&self) -> String {
        if self.state_map.is_empty() {
            return String::new();
        }

        let mut bindings: Vec<_> = self
            .state_map
            .iter()
            .map(|(name, &idx)| format!("let {} = x[{}];", name, idx))
            .collect();
        bindings.sort(); // Consistent ordering
        bindings.join("\n        ")
    }

    /// Generate fetch_cov! calls for covariates declared in the model
    fn fetch_covariates(&self) -> String {
        let Some(covariates) = &self.model.covariates else {
            return String::new();
        };

        if covariates.is_empty() {
            return String::new();
        }

        let fetch_lines: Vec<_> = covariates
            .iter()
            .map(|covariate| {
                format!(
                    "let {} = cov.get_covariate(\"{}\", t).unwrap();",
                    covariate.symbol(),
                    covariate.column_name()
                )
            })
            .collect();

        fetch_lines.join("\n        ")
    }

    /// Generate normalized executable calculations (ordered let statements).
    fn generate_calculations(&self) -> Result<String, JsonModelError> {
        let calculations = self.model.executable_calculations();

        if calculations.is_empty() {
            return Ok(String::new());
        }

        let mut lines = Vec::new();
        for entry in calculations {
            let rust_expr = self.transpile(&entry.equation)?;
            lines.push(format!("let {} = {};", entry.id, rust_expr));
        }
        Ok(lines.join("\n        "))
    }

    /// Generate the common preamble (params + bindings + covariates + calculations).
    fn generate_preamble(
        &self,
        include_compartments: bool,
        include_states: bool,
    ) -> Result<String, JsonModelError> {
        let mut parts = Vec::new();

        let fetch_params = self.fetch_params();
        if !fetch_params.is_empty() {
            parts.push(fetch_params);
        }

        if include_compartments {
            let bindings = self.generate_compartment_bindings();
            if !bindings.is_empty() {
                parts.push(bindings);
            }
        }

        if include_states {
            let bindings = self.generate_state_bindings();
            if !bindings.is_empty() {
                parts.push(bindings);
            }
        }

        if self.model.has_covariates() {
            let fetch_cov = self.fetch_covariates();
            if !fetch_cov.is_empty() {
                parts.push(fetch_cov);
            }
        }

        let calculations = self.generate_calculations()?;
        if !calculations.is_empty() {
            parts.push(calculations);
        }

        Ok(parts.join("\n        "))
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Closure Generators
    // ═══════════════════════════════════════════════════════════════════════════

    /// Generate the output closure
    /// Signature: fn(&V, &V, T, &Covariates, &mut V)
    pub fn generate_output(&self) -> Result<String, JsonModelError> {
        let preamble = self.generate_preamble(true, false)?;
        let outputs = self.model.normalized_outputs()?;
        let mut seen_indices = HashSet::new();
        let mut indexed_outputs = Vec::with_capacity(outputs.len());

        for (fallback_index, output) in outputs.iter().enumerate() {
            let output_index = parse_bound_output_index(&output.id).unwrap_or(fallback_index);
            if !seen_indices.insert(output_index) {
                return Err(JsonModelError::CodeGenError(format!(
                    "Duplicate output index {} resolved from output '{}'",
                    output_index, output.id
                )));
            }

            let rust_expr = self.transpile(&output.equation)?;
            indexed_outputs.push((
                output_index,
                format!("y[{}] = {};", output_index, rust_expr),
            ));
        }

        indexed_outputs.sort_by_key(|(index, _)| *index);
        let body = indexed_outputs
            .into_iter()
            .map(|(_, line)| line)
            .collect::<Vec<_>>()
            .join("\n        ");

        // Use _t or t depending on whether covariates need time
        let t_param = if self.model.has_covariates() {
            "t"
        } else {
            "_t"
        };
        let cov_param = if self.model.has_covariates() {
            "cov"
        } else {
            "_cov"
        };

        Ok(format!(
            r#"|x, p, {t_param}, {cov_param}, y| {{
        {preamble}
        {body}
    }}"#
        ))
    }

    /// Generate the differential equation closure
    /// Signature: fn(&V, &V, T, &mut V, &V, &V, &Covariates)
    pub fn generate_diffeq(&self) -> Result<String, JsonModelError> {
        let diffeq = self
            .model
            .diffeq
            .as_ref()
            .ok_or_else(|| JsonModelError::missing_field("diffeq", "ode"))?;

        let preamble = self.generate_preamble(true, false)?;

        let DiffEqSpec::Object(map) = diffeq;
        let mut indexed_lines = Vec::new();
        for (name, expr) in map {
            let idx = self.resolve_index(name, "diffeq")?;
            let rust_expr = self.transpile(expr)?;
            indexed_lines.push((idx, format!("dx[{}] = {};", idx, rust_expr)));
        }
        indexed_lines.sort_by_key(|(idx, _)| *idx);
        let body = indexed_lines
            .into_iter()
            .map(|(_, line)| line)
            .collect::<Vec<_>>()
            .join("\n        ");

        let t_param = if self.model.has_covariates() {
            "t"
        } else {
            "_t"
        };
        let cov_param = if self.model.has_covariates() {
            "cov"
        } else {
            "_cov"
        };

        Ok(format!(
            r#"|x, p, {t_param}, dx, b, rateiv, {cov_param}| {{
        {preamble}
        {body}
    }}"#
        ))
    }

    /// Generate the drift closure for SDE
    /// Signature: fn(&V, &V, T, &mut V, V, &Covariates)
    pub fn generate_drift(&self) -> Result<String, JsonModelError> {
        let drift = self
            .model
            .drift
            .as_ref()
            .ok_or_else(|| JsonModelError::missing_field("drift", "sde"))?;

        let preamble = self.generate_preamble(false, true)?;

        let DiffEqSpec::Object(map) = drift;
        let mut indexed_lines = Vec::new();
        for (name, expr) in map {
            let idx = self.resolve_index(name, "drift")?;
            let rust_expr = self.transpile(expr)?;
            indexed_lines.push((idx, format!("dx[{}] = {};", idx, rust_expr)));
        }
        indexed_lines.sort_by_key(|(idx, _)| *idx);
        let body = indexed_lines
            .into_iter()
            .map(|(_, line)| line)
            .collect::<Vec<_>>()
            .join("\n        ");

        let t_param = if self.model.has_covariates() {
            "t"
        } else {
            "_t"
        };
        let cov_param = if self.model.has_covariates() {
            "cov"
        } else {
            "_cov"
        };

        Ok(format!(
            r#"|x, p, {t_param}, dx, rateiv, {cov_param}| {{
        {preamble}
        {body}
    }}"#
        ))
    }

    /// Generate the diffusion closure for SDE
    /// Signature: fn(&V, &mut V)
    pub fn generate_diffusion(&self) -> Result<String, JsonModelError> {
        let diffusion = self
            .model
            .diffusion
            .as_ref()
            .ok_or_else(|| JsonModelError::missing_field("diffusion", "sde"))?;

        let fetch_params = self.fetch_params();
        let states = self.generate_state_bindings();

        let mut indexed_lines = Vec::new();
        for (name, expr) in diffusion {
            let idx = self.resolve_index(name, "diffusion")?;
            let rust_expr = match expr {
                ExpressionOrNumber::Number(n) => format!("{:.6}", n),
                ExpressionOrNumber::Expression(s) => self.transpile(s)?,
            };
            indexed_lines.push((idx, format!("d[{}] = {};", idx, rust_expr)));
        }
        indexed_lines.sort_by_key(|(idx, _)| *idx);
        let body = indexed_lines
            .into_iter()
            .map(|(_, line)| line)
            .collect::<Vec<_>>()
            .join("\n        ");

        Ok(format!(
            r#"|x, p, d| {{
        {fetch_params}
        {states}
        {body}
    }}"#
        ))
    }

    /// Generate the lag closure
    /// Signature: fn(&V, T, &Covariates) -> HashMap<usize, T>
    pub fn generate_lag(&self) -> Result<String, JsonModelError> {
        let Some(lag) = &self.model.lag else {
            return Ok("|_p, _t, _cov| lag! {}".to_string());
        };

        if lag.is_empty() {
            return Ok("|_p, _t, _cov| lag! {}".to_string());
        }

        let fetch_params = self.fetch_params();

        let mut entries = Vec::new();
        for (name, expr) in lag {
            let idx = self.resolve_index(name, "lag")?;
            let rust_expr = match expr {
                ExpressionOrNumber::Number(n) => format!("{:.6}", n),
                ExpressionOrNumber::Expression(s) => self.transpile(s)?,
            };
            entries.push(format!("{} => {}", idx, rust_expr));
        }

        Ok(format!(
            r#"|p, _t, _cov| {{
        {fetch_params}
        lag! {{ {} }}
    }}"#,
            entries.join(", ")
        ))
    }

    /// Generate the fa (bioavailability) closure
    /// Signature: fn(&V, T, &Covariates) -> HashMap<usize, T>
    pub fn generate_fa(&self) -> Result<String, JsonModelError> {
        let Some(fa) = &self.model.fa else {
            return Ok("|_p, _t, _cov| fa! {}".to_string());
        };

        if fa.is_empty() {
            return Ok("|_p, _t, _cov| fa! {}".to_string());
        }

        let fetch_params = self.fetch_params();

        let mut entries = Vec::new();
        for (name, expr) in fa {
            let idx = self.resolve_index(name, "fa")?;
            let rust_expr = match expr {
                ExpressionOrNumber::Number(n) => format!("{:.6}", n),
                ExpressionOrNumber::Expression(s) => self.transpile(s)?,
            };
            entries.push(format!("{} => {}", idx, rust_expr));
        }

        Ok(format!(
            r#"|p, _t, _cov| {{
        {fetch_params}
        fa! {{ {} }}
    }}"#,
            entries.join(", ")
        ))
    }

    /// Generate the init closure
    /// Signature: fn(&V, T, &Covariates, &mut V)
    pub fn generate_init(&self) -> Result<String, JsonModelError> {
        let Some(init) = &self.model.init else {
            return Ok("|_p, _t, _cov, _x| {}".to_string());
        };

        let InitSpec::Object(map) = init;
        let mut lines = Vec::new();
        for (name, expr) in map {
            let idx = self.resolve_index(name, "init")?;
            let rust_expr = match expr {
                ExpressionOrNumber::Number(n) => format!("{:.6}", n),
                ExpressionOrNumber::Expression(s) => self.transpile(s)?,
            };
            lines.push(format!("x[{}] = {};", idx, rust_expr));
        }
        let body = lines.join("\n        ");

        let fetch_params = self.fetch_params();

        Ok(format!(
            r#"|p, _t, _cov, x| {{
        {fetch_params}
        {body}
    }}"#
        ))
    }

    /// Generate the secondary equation closure (for analytical)
    /// Signature: fn(&mut V, T, &Covariates)
    pub fn generate_secondary_closure(&self) -> Result<String, JsonModelError> {
        if self.model.executable_calculations().is_empty() && !self.model.has_covariates() {
            return Ok("|_p, _t, _cov| {}".to_string());
        }

        let preamble = self.generate_preamble(false, false)?;

        let t_param = if self.model.has_covariates() {
            "t"
        } else {
            "_t"
        };
        let cov_param = if self.model.has_covariates() {
            "cov"
        } else {
            "_cov"
        };

        Ok(format!(
            r#"|p, {t_param}, {cov_param}| {{
        {preamble}
    }}"#
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_output() {
        let json = r#"{
            "schema": "2.0",
            "id": "pk/1cmt-iv",
            "type": "analytical",
            "analytical": "one_compartment",
            "parameters": ["ke", "V"],
            "compartments": ["central"],
            "outputs": [{ "id": "cp", "equation": "central / V" }]
        }"#;

        let model = JsonModel::from_str(json).unwrap();
        let cg = ClosureGenerator::new(&model);
        let output = cg.generate_output().unwrap();

        assert!(output.contains("fetch_params!(p, ke, V)"));
        assert!(output.contains("let central = x[0];"));
        assert!(output.contains("y[0] = central / V"));
    }

    #[test]
    fn test_generate_output_with_secondary() {
        let json = r#"{
            "schema": "2.0",
            "id": "pk/1cmt-sec",
            "type": "analytical",
            "analytical": "one_compartment",
            "parameters": ["CL", "V"],
            "compartments": ["central"],
            "secondary": [{ "id": "ke", "equation": "CL / V" }],
            "outputs": [{ "id": "cp", "equation": "central / V" }]
        }"#;

        let model = JsonModel::from_str(json).unwrap();
        let cg = ClosureGenerator::new(&model);
        let output = cg.generate_output().unwrap();

        assert!(output.contains("let ke = CL / V;"));
        assert!(output.contains("let central = x[0];"));
        assert!(output.contains("y[0] = central / V"));
    }

    #[test]
    fn test_generate_lag() {
        let json = r#"{
            "schema": "2.0",
            "id": "pk/1cmt-lag",
            "type": "analytical",
            "analytical": "one_compartment_with_absorption",
            "parameters": ["ka", "ke", "V", "tlag"],
            "compartments": ["depot", "central"],
            "lag": { "depot": "tlag" },
            "outputs": [{ "id": "cp", "equation": "central / V" }]
        }"#;

        let model = JsonModel::from_str(json).unwrap();
        let cg = ClosureGenerator::new(&model);
        let lag = cg.generate_lag().unwrap();

        assert!(lag.contains("lag!"));
        assert!(lag.contains("0 => tlag"));
    }

    #[test]
    fn test_generate_diffeq_object() {
        let json = r#"{
            "schema": "2.0",
            "id": "pk/2cmt-ode",
            "type": "ode",
            "compartments": ["depot", "central"],
            "parameters": ["ka", "ke", "V"],
            "diffeq": {
                "depot": "-ka * depot",
                "central": "ka * depot - ke * central + rateiv[1]"
            },
            "outputs": [{ "id": "cp", "equation": "central / V" }]
        }"#;

        let model = JsonModel::from_str(json).unwrap();
        let cg = ClosureGenerator::new(&model);
        let diffeq = cg.generate_diffeq().unwrap();

        assert!(diffeq.contains("dx[0]"));
        assert!(diffeq.contains("dx[1]"));
        // expression parser transpiles -ka to -(ka)
        assert!(diffeq.contains("ka"));
        assert!(diffeq.contains("depot"));
    }

    #[test]
    fn test_generate_with_covariates() {
        let json = r#"{
            "schema": "2.0",
            "id": "pk/1cmt-cov",
            "type": "analytical",
            "analytical": "one_compartment",
            "parameters": ["ke", "V"],
            "compartments": ["central"],
            "covariates": [{ "id": "wt" }],
            "secondary": [{ "id": "V", "equation": "V * (wt / 70)^0.75" }],
            "outputs": [{ "id": "cp", "equation": "central / V" }]
        }"#;

        let model = JsonModel::from_str(json).unwrap();
        let cg = ClosureGenerator::new(&model);
        let output = cg.generate_output().unwrap();

        // Should fetch covariates
        assert!(output.contains("cov.get_covariate(\"wt\", t)"));
        // Should have secondary equation with allometric scaling
        assert!(output.contains("let V = V * (wt / 70.0).powf(0.75);"));
    }

    #[test]
    fn test_generate_empty_lag_fa() {
        let json = r#"{
            "schema": "2.0",
            "id": "pk/1cmt-empty",
            "type": "analytical",
            "analytical": "one_compartment",
            "parameters": ["ke", "V"],
            "compartments": ["central"],
            "outputs": [{ "id": "cp", "equation": "central / V" }]
        }"#;

        let model = JsonModel::from_str(json).unwrap();
        let cg = ClosureGenerator::new(&model);

        let lag = cg.generate_lag().unwrap();
        let fa = cg.generate_fa().unwrap();

        assert!(lag.contains("lag! {}"));
        assert!(fa.contains("fa! {}"));
    }

    #[test]
    fn test_lag_propagates_transpile_error() {
        let json = r#"{
            "schema": "2.0",
            "id": "pk/2cmt-bad-lag",
            "type": "ode",
            "compartments": ["depot", "central"],
            "parameters": ["ka", "ke", "V"],
            "diffeq": {
                "depot": "-ka * depot",
                "central": "ka * depot - ke * central"
            },
            "lag": { "depot": "++ invalid" },
            "outputs": [{ "id": "cp", "equation": "central / V" }]
        }"#;

        let model = JsonModel::from_str(json).unwrap();
        let cg = ClosureGenerator::new(&model);
        let result = cg.generate_lag();

        assert!(result.is_err(), "Should propagate transpile error for lag");
    }

    #[test]
    fn test_fa_propagates_transpile_error() {
        let json = r#"{
            "schema": "2.0",
            "id": "pk/2cmt-bad-fa",
            "type": "ode",
            "compartments": ["depot", "central"],
            "parameters": ["ka", "ke", "V"],
            "diffeq": {
                "depot": "-ka * depot",
                "central": "ka * depot - ke * central"
            },
            "fa": { "depot": "++ invalid" },
            "outputs": [{ "id": "cp", "equation": "central / V" }]
        }"#;

        let model = JsonModel::from_str(json).unwrap();
        let cg = ClosureGenerator::new(&model);
        let result = cg.generate_fa();

        assert!(result.is_err(), "Should propagate transpile error for fa");
    }

    #[test]
    fn test_unknown_compartment_in_diffeq_errors() {
        let json = r#"{
            "schema": "2.0",
            "id": "pk/unknown-diffeq",
            "type": "ode",
            "compartments": ["central"],
            "parameters": ["ke", "V"],
            "diffeq": {
                "nonexistent": "-ke * x[0]"
            },
            "outputs": [{ "id": "cp", "equation": "central / V" }]
        }"#;

        let model = JsonModel::from_str(json).unwrap();
        let cg = ClosureGenerator::new(&model);
        let result = cg.generate_diffeq();

        assert!(result.is_err(), "Should error on unknown compartment name");
    }

    #[test]
    fn test_unknown_compartment_in_lag_errors() {
        let json = r#"{
            "schema": "2.0",
            "id": "pk/unknown-lag",
            "type": "ode",
            "compartments": ["central"],
            "parameters": ["ke", "V", "tlag"],
            "diffeq": { "central": "-ke * central" },
            "lag": { "nonexistent": "tlag" },
            "outputs": [{ "id": "cp", "equation": "central / V" }]
        }"#;

        let model = JsonModel::from_str(json).unwrap();
        let cg = ClosureGenerator::new(&model);
        let result = cg.generate_lag();

        assert!(
            result.is_err(),
            "Should error on unknown compartment in lag"
        );
    }

    #[test]
    fn test_diffeq_output_is_sorted_by_index() {
        let json = r#"{
            "schema": "2.0",
            "id": "pk/sorted-diffeq",
            "type": "ode",
            "compartments": ["depot", "central"],
            "parameters": ["ka", "ke", "V"],
            "diffeq": {
                "central": "ka * depot - ke * central",
                "depot": "-ka * depot"
            },
            "outputs": [{ "id": "cp", "equation": "central / V" }]
        }"#;

        let model = JsonModel::from_str(json).unwrap();
        let cg = ClosureGenerator::new(&model);
        let diffeq = cg.generate_diffeq().unwrap();

        // dx[0] (depot) should appear before dx[1] (central)
        let pos0 = diffeq.find("dx[0]").expect("Should contain dx[0]");
        let pos1 = diffeq.find("dx[1]").expect("Should contain dx[1]");
        assert!(pos0 < pos1, "dx[0] should come before dx[1]");
    }
}
