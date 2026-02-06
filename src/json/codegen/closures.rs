//! Closure generation for model equations
//!
//! This module generates the closure functions that are passed to
//! equation constructors (Analytical, ODE, SDE).

use std::collections::HashMap;

use crate::json::errors::JsonModelError;
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

    /// Generate fetch_cov! macro call for covariates used in covariate effects
    fn fetch_covariates(&self) -> String {
        // Collect all covariate names used in effects
        let Some(effects) = &self.model.covariate_effects else {
            return String::new();
        };

        let cov_names: Vec<_> = effects
            .iter()
            .filter_map(|e| e.covariate.as_ref())
            .map(|c| c.as_str())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();

        if cov_names.is_empty() {
            return String::new();
        }

        // Generate code to fetch each covariate
        let fetch_lines: Vec<_> = cov_names
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

    /// Generate covariate effect code to inject before equations
    fn generate_covariate_effects(&self) -> String {
        let Some(effects) = &self.model.covariate_effects else {
            return String::new();
        };

        if effects.is_empty() {
            return String::new();
        }

        // First, fetch all covariates used
        let fetch_cov = self.fetch_covariates();

        let mut lines = Vec::new();

        for effect in effects {
            let param = &effect.on;
            let code = match effect.effect_type {
                CovariateEffectType::Allometric => {
                    let cov = effect.covariate.as_ref().unwrap();
                    let exp = effect.exponent.unwrap_or(0.75);
                    let reference = effect.reference.unwrap_or(70.0);
                    format!(
                        "let {param} = {param} * ({cov} / {:.1}).powf({:.4});",
                        reference, exp
                    )
                }
                CovariateEffectType::Linear => {
                    let cov = effect.covariate.as_ref().unwrap();
                    let slope = effect.slope.unwrap_or(0.0);
                    let reference = effect.reference.unwrap_or(0.0);
                    format!(
                        "let {param} = {param} * (1.0 + {:.6} * ({cov} - {:.6}));",
                        slope, reference
                    )
                }
                CovariateEffectType::Exponential => {
                    let cov = effect.covariate.as_ref().unwrap();
                    let slope = effect.slope.unwrap_or(0.0);
                    let reference = effect.reference.unwrap_or(0.0);
                    format!(
                        "let {param} = {param} * ({:.6} * ({cov} - {:.6})).exp();",
                        slope, reference
                    )
                }
                CovariateEffectType::Proportional => {
                    let cov = effect.covariate.as_ref().unwrap();
                    let slope = effect.slope.unwrap_or(0.0);
                    format!("let {param} = {param} * (1.0 + {:.6} * {cov});", slope)
                }
                CovariateEffectType::Custom => {
                    let expr = effect.expression.as_ref().unwrap();
                    format!("let {param} = {expr};")
                }
                CovariateEffectType::Categorical => {
                    // Categorical effects require match statement
                    let cov = effect.covariate.as_ref().unwrap();
                    if let Some(levels) = &effect.levels {
                        let arms: Vec<_> = levels
                            .iter()
                            .map(|(k, v)| format!("\"{}\" => {:.6}", k, v))
                            .collect();
                        format!(
                            "let {param} = {param} * match {cov} {{ {}, _ => 1.0 }};",
                            arms.join(", ")
                        )
                    } else {
                        String::new()
                    }
                }
            };
            if !code.is_empty() {
                lines.push(code);
            }
        }

        // Prepend fetch code
        if !fetch_cov.is_empty() {
            return format!("{}\n        {}", fetch_cov, lines.join("\n        "));
        }

        lines.join("\n        ")
    }

    /// Generate derived parameters code
    fn generate_derived_params(&self) -> String {
        // Use model-level derived parameters
        if let Some(derived) = &self.model.derived {
            let lines: Vec<_> = derived
                .iter()
                .map(|d| format!("let {} = {};", d.symbol, d.expression))
                .collect();
            return lines.join("\n        ");
        }
        String::new()
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Closure Generators
    // ═══════════════════════════════════════════════════════════════════════════

    /// Generate the output closure
    /// Signature: fn(&V, &V, T, &Covariates, &mut V)
    pub fn generate_output(&self) -> Result<String, JsonModelError> {
        let output_expr = if let Some(output) = &self.model.output {
            output.clone()
        } else if let Some(outputs) = &self.model.outputs {
            // Multiple outputs
            outputs
                .iter()
                .enumerate()
                .map(|(i, o)| format!("y[{}] = {};", i, o.equation))
                .collect::<Vec<_>>()
                .join("\n        ")
        } else {
            return Err(JsonModelError::MissingOutput);
        };

        let fetch_params = self.fetch_params();
        let derived = self.generate_derived_params();
        let cov_effects = self.generate_covariate_effects();

        // Determine if we have a single expression or multiple statements
        let body = if output_expr.contains("y[") {
            // Already has y[] assignments
            output_expr
        } else {
            // Single expression, wrap it
            format!("y[0] = {};", output_expr)
        };

        let compartments = self.generate_compartment_bindings();

        Ok(format!(
            r#"|x, p, _t, _cov, y| {{
        {fetch_params}
        {compartments}
        {derived}
        {cov_effects}
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

        let body = match diffeq {
            DiffEqSpec::String(s) => s.clone(),
            DiffEqSpec::Object(map) => {
                // Convert named compartments to dx[n] format
                let mut lines = Vec::new();
                for (name, expr) in map {
                    let idx = self.compartment_map.get(name).copied().unwrap_or_else(|| {
                        // Try parsing as number
                        name.parse::<usize>().unwrap_or(0)
                    });
                    lines.push(format!("dx[{}] = {};", idx, expr));
                }
                lines.join("\n        ")
            }
        };

        let fetch_params = self.fetch_params();
        let compartments = self.generate_compartment_bindings();
        let derived = self.generate_derived_params();
        let cov_effects = self.generate_covariate_effects();

        Ok(format!(
            r#"|x, p, _t, dx, _b, rateiv, _cov| {{
        {fetch_params}
        {compartments}
        {derived}
        {cov_effects}
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
                    lines.push(format!("dx[{}] = {};", idx, expr));
                }
                lines.join("\n        ")
            }
        };

        let fetch_params = self.fetch_params();
        let states = self.generate_state_bindings();
        let derived = self.generate_derived_params();
        let cov_effects = self.generate_covariate_effects();

        Ok(format!(
            r#"|x, p, _t, dx, rateiv, _cov| {{
        {fetch_params}
        {states}
        {derived}
        {cov_effects}
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
            lines.push(format!("d[{}] = {};", idx, expr.to_rust_expr()));
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
                // Convert compartment name to index
                let idx = self
                    .compartment_map
                    .get(name)
                    .copied()
                    .unwrap_or_else(|| name.parse::<usize>().unwrap_or(0));
                format!("{} => {}", idx, expr.to_rust_expr())
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
                // Convert compartment name to index
                let idx = self
                    .compartment_map
                    .get(name)
                    .copied()
                    .unwrap_or_else(|| name.parse::<usize>().unwrap_or(0));
                format!("{} => {}", idx, expr.to_rust_expr())
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
                    lines.push(format!("x[{}] = {};", idx, expr.to_rust_expr()));
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
    pub fn generate_secondary(&self) -> Result<String, JsonModelError> {
        let Some(secondary) = &self.model.secondary else {
            return Ok("|_p, _t, _cov| {}".to_string());
        };

        let fetch_params = self.fetch_params();
        let cov_effects = self.generate_covariate_effects();

        Ok(format!(
            r#"|p, _t, _cov| {{
        {fetch_params}
        {cov_effects}
        {secondary}
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
                "depot": "-ka * x[0]",
                "central": "ka * x[0] - ke * x[1] + rateiv[1]"
            },
            "output": "x[1] / V"
        }"#;

        let model = JsonModel::from_str(json).unwrap();
        let gen = ClosureGenerator::new(&model);
        let diffeq = gen.generate_diffeq().unwrap();

        assert!(diffeq.contains("dx[0] = -ka * x[0]"));
        assert!(diffeq.contains("dx[1] = ka * x[0] - ke * x[1] + rateiv[1]"));
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
