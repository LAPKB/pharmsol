//! Code generation from JSON models to Rust code
//!
//! This module transforms validated JSON models into Rust code strings
//! that can be compiled by the `exa` module.

mod analytical;
mod closures;
mod ode;
mod sde;

use crate::json::errors::JsonModelError;
use crate::json::model::JsonModel;
use crate::json::types::*;
use crate::simulator::equation::EqnKind;

pub use closures::ClosureGenerator;

/// Generated Rust code ready for compilation
#[derive(Debug, Clone)]
pub struct GeneratedCode {
    /// The complete equation constructor code
    pub equation_code: String,

    /// Parameter names in fetch order
    pub parameters: Vec<String>,

    /// The equation kind (ODE, Analytical, SDE)
    pub kind: EqnKind,
}

/// Code generator for JSON models
pub struct CodeGenerator<'a> {
    model: &'a JsonModel,
    closure_gen: ClosureGenerator<'a>,
}

impl<'a> CodeGenerator<'a> {
    /// Create a new code generator for a model
    pub fn new(model: &'a JsonModel) -> Self {
        Self {
            model,
            closure_gen: ClosureGenerator::new(model),
        }
    }

    /// Generate the complete Rust code
    pub fn generate(&self) -> Result<GeneratedCode, JsonModelError> {
        let (equation_code, kind) = match self.model.model_type {
            ModelType::Analytical => {
                let code = self.generate_analytical()?;
                (code, EqnKind::Analytical)
            }
            ModelType::Ode => {
                let code = self.generate_ode()?;
                (code, EqnKind::ODE)
            }
            ModelType::Sde => {
                let code = self.generate_sde()?;
                (code, EqnKind::SDE)
            }
        };

        Ok(GeneratedCode {
            equation_code,
            parameters: self.model.get_parameters(),
            kind,
        })
    }

    /// Generate analytical model code
    fn generate_analytical(&self) -> Result<String, JsonModelError> {
        let func = self
            .model
            .analytical
            .as_ref()
            .ok_or_else(|| JsonModelError::missing_field("analytical", "analytical"))?;

        let seq_eq = self.closure_gen.generate_secondary()?;
        let lag = self.closure_gen.generate_lag()?;
        let fa = self.closure_gen.generate_fa()?;
        let init = self.closure_gen.generate_init()?;
        let out = self.closure_gen.generate_output()?;
        let neqs = self.model.get_neqs();

        Ok(format!(
            r#"equation::Analytical::new(
    {func_name},
    {seq_eq},
    {lag},
    {fa},
    {init},
    {out},
    ({nstates}, {nouts}),
)"#,
            func_name = func.rust_name(),
            seq_eq = seq_eq,
            lag = lag,
            fa = fa,
            init = init,
            out = out,
            nstates = neqs.0,
            nouts = neqs.1,
        ))
    }

    /// Generate ODE model code
    fn generate_ode(&self) -> Result<String, JsonModelError> {
        let diffeq = self.closure_gen.generate_diffeq()?;
        let lag = self.closure_gen.generate_lag()?;
        let fa = self.closure_gen.generate_fa()?;
        let init = self.closure_gen.generate_init()?;
        let out = self.closure_gen.generate_output()?;
        let neqs = self.model.get_neqs();

        Ok(format!(
            r#"equation::ODE::new(
    {diffeq},
    {lag},
    {fa},
    {init},
    {out},
    ({nstates}, {nouts}),
)"#,
            diffeq = diffeq,
            lag = lag,
            fa = fa,
            init = init,
            out = out,
            nstates = neqs.0,
            nouts = neqs.1,
        ))
    }

    /// Generate SDE model code
    fn generate_sde(&self) -> Result<String, JsonModelError> {
        let drift = self.closure_gen.generate_drift()?;
        let diffusion = self.closure_gen.generate_diffusion()?;
        let lag = self.closure_gen.generate_lag()?;
        let fa = self.closure_gen.generate_fa()?;
        let init = self.closure_gen.generate_init()?;
        let out = self.closure_gen.generate_output()?;
        let neqs = self.model.get_neqs();
        let particles = self.model.particles.unwrap_or(1000);

        Ok(format!(
            r#"equation::SDE::new(
    {drift},
    {diffusion},
    {lag},
    {fa},
    {init},
    {out},
    ({nstates}, {nouts}),
    {particles},
)"#,
            drift = drift,
            diffusion = diffusion,
            lag = lag,
            fa = fa,
            init = init,
            out = out,
            nstates = neqs.0,
            nouts = neqs.1,
            particles = particles,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_analytical() {
        let json = r#"{
            "schema": "1.0",
            "id": "pk_1cmt_oral",
            "type": "analytical",
            "analytical": "one_compartment_with_absorption",
            "parameters": ["ka", "ke", "V"],
            "output": "x[1] / V"
        }"#;

        let model = JsonModel::from_str(json).unwrap();
        let generator = CodeGenerator::new(&model);
        let result = generator.generate().unwrap();

        assert!(result
            .equation_code
            .contains("one_compartment_with_absorption"));
        assert!(result.equation_code.contains("equation::Analytical::new"));
        assert_eq!(result.parameters, vec!["ka", "ke", "V"]);
    }

    #[test]
    fn test_generate_ode() {
        let json = r#"{
            "schema": "1.0",
            "id": "pk_1cmt_ode",
            "type": "ode",
            "parameters": ["ke", "V"],
            "diffeq": "dx[0] = -ke * x[0] + rateiv[0];",
            "output": "x[0] / V",
            "neqs": [1, 1]
        }"#;

        let model = JsonModel::from_str(json).unwrap();
        let generator = CodeGenerator::new(&model);
        let result = generator.generate().unwrap();

        assert!(result.equation_code.contains("equation::ODE::new"));
        assert!(result.equation_code.contains("-ke * x[0]"));
    }

    #[test]
    fn test_generate_with_lag() {
        let json = r#"{
            "schema": "1.0",
            "id": "pk_1cmt_oral_lag",
            "type": "analytical",
            "analytical": "one_compartment_with_absorption",
            "parameters": ["ka", "ke", "V", "tlag"],
            "lag": { "0": "tlag" },
            "output": "x[1] / V",
            "neqs": [2, 1]
        }"#;

        let model = JsonModel::from_str(json).unwrap();
        let generator = CodeGenerator::new(&model);
        let result = generator.generate().unwrap();

        assert!(result.equation_code.contains("lag!"));
        assert!(result.equation_code.contains("0 => tlag"));
    }
}
