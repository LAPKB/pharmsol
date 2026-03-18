//! Closure generation for model equations
//!
//! This module generates the closure functions that are passed to
//! equation constructors (Analytical, ODE, SDE).
//!
//! All math expressions in the JSON model are written in a language-agnostic
//! format and transpiled to Rust via the expression parser.

use std::collections::HashMap;

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
            .map(|name| {
                format!(
                    "let {} = cov.get_covariate(\"{}\", t).unwrap_or(0.0);",
                    name, name
                )
            })
            .collect();

        fetch_lines.join("\n        ")
    }

    /// Generate secondary equation bindings (ordered let statements)
    fn generate_secondary_equations(&self) -> Result<String, JsonModelError> {
        let Some(secondary) = &self.model.secondary else {
            return Ok(String::new());
        };

        if secondary.is_empty() {
            return Ok(String::new());
        }

        let mut lines = Vec::new();
        for (name, expr) in secondary {
            let rust_expr = self.transpile(expr)?;
            lines.push(format!("let {} = {};", name, rust_expr));
        }
        Ok(lines.join("\n        "))
    }

    /// Generate derived parameters code
    fn generate_derived_params(&self) -> Result<String, JsonModelError> {
        if let Some(derived) = &self.model.derived {
            let mut lines = Vec::new();
            for d in derived {
                let rust_expr = self.transpile(&d.expression)?;
                lines.push(format!("let {} = {};", d.symbol, rust_expr));
            }
            return Ok(lines.join("\n        "));
        }
        Ok(String::new())
    }

    /// Generate the common preamble (params + compartment bindings + covariates + derived + secondary)
    fn generate_preamble(&self, include_compartments: bool, include_states: bool) -> Result<String, JsonModelError> {
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

        let derived = self.generate_derived_params()?;
        if !derived.is_empty() {
            parts.push(derived);
        }

        let secondary = self.generate_secondary_equations()?;
        if !secondary.is_empty() {
            parts.push(secondary);
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

        let body = if let Some(output) = &self.model.output {
            let rust_expr = self.transpile(output)?;
            format!("y[0] = {};", rust_expr)
        } else if let Some(outputs) = &self.model.outputs {
            outputs
                .iter()
                .enumerate()
                .map(|(i, o)| {
                    let rust_expr = self.transpile(&o.equation)?;
                    Ok(format!("y[{}] = {};", i, rust_expr))
                })
                .collect::<Result<Vec<_>, JsonModelError>>()?
                .join("\n        ")
        } else {
            return Err(JsonModelError::MissingOutput);
        };

        // Use _t or t depending on whether covariates need time
        let t_param = if self.model.has_covariates() { "t" } else { "_t" };
        let cov_param = if self.model.has_covariates() { "cov" } else { "_cov" };

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

        let body = match diffeq {
            DiffEqSpec::String(s) => {
                // Raw string format — pass through as-is (for backwards compatibility)
                s.clone()
            }
            DiffEqSpec::Object(map) => {
                let mut lines = Vec::new();
                for (name, expr) in map {
                    let idx = self.compartment_map.get(name).copied().unwrap_or_else(|| {
                        name.parse::<usize>().unwrap_or(0)
                    });
                    let rust_expr = self.transpile(expr)?;
                    lines.push(format!("dx[{}] = {};", idx, rust_expr));
                }
                lines.join("\n        ")
            }
        };

        let t_param = if self.model.has_covariates() { "t" } else { "_t" };
        let cov_param = if self.model.has_covariates() { "cov" } else { "_cov" };

        Ok(format!(
            r#"|x, p, {t_param}, dx, _b, rateiv, {cov_param}| {{
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

        let body = match drift {
            DiffEqSpec::String(s) => s.clone(),
            DiffEqSpec::Object(map) => {
                let mut lines = Vec::new();
                for (name, expr) in map {
                    let idx = self.state_map.get(name).copied().unwrap_or_else(|| {
                        self.compartment_map
                            .get(name)
                            .copied()
                            .unwrap_or_else(|| name.parse::<usize>().unwrap_or(0))
                    });
                    let rust_expr = self.transpile(expr)?;
                    lines.push(format!("dx[{}] = {};", idx, rust_expr));
                }
                lines.join("\n        ")
            }
        };

        let t_param = if self.model.has_covariates() { "t" } else { "_t" };
        let cov_param = if self.model.has_covariates() { "cov" } else { "_cov" };

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

        let mut lines = Vec::new();
        for (name, expr) in diffusion {
            let idx = self.state_map.get(name).copied().unwrap_or_else(|| {
                self.compartment_map
                    .get(name)
                    .copied()
                    .unwrap_or_else(|| name.parse::<usize>().unwrap_or(0))
            });
            let rust_expr = match expr {
                ExpressionOrNumber::Number(n) => format!("{:.6}", n),
                ExpressionOrNumber::Expression(s) => self.transpile(s)?,
            };
            lines.push(format!("d[{}] = {};", idx, rust_expr));
        }
        let body = lines.join("\n        ");

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

        let entries: Vec<_> = lag
            .iter()
            .map(|(name, expr)| {
                let idx = self
                    .compartment_map
                    .get(name)
                    .copied()
                    .unwrap_or_else(|| name.parse::<usize>().unwrap_or(0));
                let rust_expr = match expr {
                    ExpressionOrNumber::Number(n) => format!("{:.6}", n),
                    ExpressionOrNumber::Expression(s) => {
                        self.transpile(s).unwrap_or_else(|_| s.clone())
                    }
                };
                format!("{} => {}", idx, rust_expr)
            })
            .collect();

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

        let entries: Vec<_> = fa
            .iter()
            .map(|(name, expr)| {
                let idx = self
                    .compartment_map
                    .get(name)
                    .copied()
                    .unwrap_or_else(|| name.parse::<usize>().unwrap_or(0));
                let rust_expr = match expr {
                    ExpressionOrNumber::Number(n) => format!("{:.6}", n),
                    ExpressionOrNumber::Expression(s) => {
                        self.transpile(s).unwrap_or_else(|_| s.clone())
                    }
                };
                format!("{} => {}", idx, rust_expr)
            })
            .collect();

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

        let body = match init {
            InitSpec::String(s) => s.clone(),
            InitSpec::Object(map) => {
                let mut lines = Vec::new();
                for (name, expr) in map {
                    let idx = self.state_map.get(name).copied().unwrap_or_else(|| {
                        self.compartment_map
                            .get(name)
                            .copied()
                            .unwrap_or_else(|| name.parse::<usize>().unwrap_or(0))
                    });
                    let rust_expr = match expr {
                        ExpressionOrNumber::Number(n) => format!("{:.6}", n),
                        ExpressionOrNumber::Expression(s) => self.transpile(s)?,
                    };
                    lines.push(format!("x[{}] = {};", idx, rust_expr));
                }
                lines.join("\n        ")
            }
        };

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
        if !self.model.has_secondary() && !self.model.has_covariates() {
            return Ok("|_p, _t, _cov| {}".to_string());
        }

        let preamble = self.generate_preamble(false, false)?;

        let t_param = if self.model.has_covariates() { "t" } else { "_t" };
        let cov_param = if self.model.has_covariates() { "cov" } else { "_cov" };

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
            "schema": "1.0",
            "id": "test",
            "type": "analytical",
            "analytical": "one_compartment",
            "parameters": ["ke", "V"],
            "output": "x[0] / V"
        }"#;

        let model = JsonModel::from_str(json).unwrap();
        let gen = ClosureGenerator::new(&model);
        let output = gen.generate_output().unwrap();

        assert!(output.contains("fetch_params!(p, ke, V)"));
        assert!(output.contains("y[0] = x[0] / V"));
    }

    #[test]
    fn test_generate_output_with_secondary() {
        let json = r#"{
            "schema": "1.0",
            "id": "test",
            "type": "analytical",
            "analytical": "one_compartment",
            "parameters": ["CL", "V"],
            "secondary": { "ke": "CL / V" },
            "output": "x[0] / V"
        }"#;

        let model = JsonModel::from_str(json).unwrap();
        let gen = ClosureGenerator::new(&model);
        let output = gen.generate_output().unwrap();

        assert!(output.contains("let ke = CL / V;"));
        assert!(output.contains("y[0] = x[0] / V"));
    }

    #[test]
    fn test_generate_lag() {
        let json = r#"{
            "schema": "1.0",
            "id": "test",
            "type": "analytical",
            "analytical": "one_compartment_with_absorption",
            "parameters": ["ka", "ke", "V", "tlag"],
            "lag": { "0": "tlag" },
            "output": "x[1] / V"
        }"#;

        let model = JsonModel::from_str(json).unwrap();
        let gen = ClosureGenerator::new(&model);
        let lag = gen.generate_lag().unwrap();

        assert!(lag.contains("lag!"));
        assert!(lag.contains("0 => tlag"));
    }

    #[test]
    fn test_generate_diffeq_object() {
        let json = r#"{
            "schema": "1.0",
            "id": "test",
            "type": "ode",
            "compartments": ["depot", "central"],
            "parameters": ["ka", "ke", "V"],
            "diffeq": {
                "depot": "-ka * depot",
                "central": "ka * depot - ke * central + rateiv[1]"
            },
            "output": "central / V"
        }"#;

        let model = JsonModel::from_str(json).unwrap();
        let gen = ClosureGenerator::new(&model);
        let diffeq = gen.generate_diffeq().unwrap();

        assert!(diffeq.contains("dx[0]"));
        assert!(diffeq.contains("dx[1]"));
        // expression parser transpiles -ka to -(ka)
        assert!(diffeq.contains("ka"));
        assert!(diffeq.contains("depot"));
    }

    #[test]
    fn test_generate_with_covariates() {
        let json = r#"{
            "schema": "1.0",
            "id": "test",
            "type": "analytical",
            "analytical": "one_compartment",
            "parameters": ["ke", "V"],
            "covariates": ["wt"],
            "secondary": { "V": "V * (wt / 70)^0.75" },
            "output": "x[0] / V"
        }"#;

        let model = JsonModel::from_str(json).unwrap();
        let gen = ClosureGenerator::new(&model);
        let output = gen.generate_output().unwrap();

        // Should fetch covariates
        assert!(output.contains("cov.get_covariate(\"wt\", t)"));
        // Should have secondary equation with allometric scaling
        assert!(output.contains("let V = V * (wt / 70.0).powf(0.75);"));
    }

    #[test]
    fn test_generate_empty_lag_fa() {
        let json = r#"{
            "schema": "1.0",
            "id": "test",
            "type": "analytical",
            "analytical": "one_compartment",
            "parameters": ["ke", "V"],
            "output": "x[0] / V"
        }"#;

        let model = JsonModel::from_str(json).unwrap();
        let gen = ClosureGenerator::new(&model);

        let lag = gen.generate_lag().unwrap();
        let fa = gen.generate_fa().unwrap();

        assert!(lag.contains("lag! {}"));
        assert!(fa.contains("fa! {}"));
    }
}
