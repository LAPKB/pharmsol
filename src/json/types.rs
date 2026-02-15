//! Core type definitions for JSON models

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ═══════════════════════════════════════════════════════════════════════════════
// Model Type
// ═══════════════════════════════════════════════════════════════════════════════

/// The type of equation system used by the model
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ModelType {
    /// Analytical (closed-form) solution
    Analytical,
    /// Ordinary differential equations
    Ode,
    /// Stochastic differential equations
    Sde,
}

impl std::fmt::Display for ModelType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Analytical => write!(f, "analytical"),
            Self::Ode => write!(f, "ode"),
            Self::Sde => write!(f, "sde"),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Analytical Functions
// ═══════════════════════════════════════════════════════════════════════════════

/// Built-in analytical solution functions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AnalyticalFunction {
    /// One compartment IV (ke)
    OneCompartment,
    /// One compartment with first-order absorption (ka, ke)
    OneCompartmentWithAbsorption,
    /// Two compartments IV (ke, kcp, kpc)
    TwoCompartments,
    /// Two compartments with absorption (ke, ka, kcp, kpc)
    TwoCompartmentsWithAbsorption,
    /// Three compartments IV (k10, k12, k13, k21, k31)
    ThreeCompartments,
    /// Three compartments with absorption (ka, k10, k12, k13, k21, k31)
    ThreeCompartmentsWithAbsorption,
}

impl AnalyticalFunction {
    /// Get the Rust function name for code generation
    pub fn rust_name(&self) -> &'static str {
        match self {
            Self::OneCompartment => "one_compartment",
            Self::OneCompartmentWithAbsorption => "one_compartment_with_absorption",
            Self::TwoCompartments => "two_compartments",
            Self::TwoCompartmentsWithAbsorption => "two_compartments_with_absorption",
            Self::ThreeCompartments => "three_compartments",
            Self::ThreeCompartmentsWithAbsorption => "three_compartments_with_absorption",
        }
    }

    /// Get the expected parameter names for this function (in order)
    pub fn expected_parameters(&self) -> Vec<&'static str> {
        match self {
            Self::OneCompartment => vec!["ke"],
            Self::OneCompartmentWithAbsorption => vec!["ka", "ke"],
            Self::TwoCompartments => vec!["ke", "kcp", "kpc"],
            Self::TwoCompartmentsWithAbsorption => vec!["ke", "ka", "kcp", "kpc"],
            Self::ThreeCompartments => vec!["k10", "k12", "k13", "k21", "k31"],
            Self::ThreeCompartmentsWithAbsorption => {
                vec!["ka", "k10", "k12", "k13", "k21", "k31"]
            }
        }
    }

    /// Get the number of states for this function
    pub fn num_states(&self) -> usize {
        match self {
            Self::OneCompartment => 1,
            Self::OneCompartmentWithAbsorption => 2,
            Self::TwoCompartments => 2,
            Self::TwoCompartmentsWithAbsorption => 3,
            Self::ThreeCompartments => 3,
            Self::ThreeCompartmentsWithAbsorption => 4,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Expression Types
// ═══════════════════════════════════════════════════════════════════════════════

/// A Rust expression string
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Expression(pub String);

impl Expression {
    /// Create a new expression
    pub fn new(s: impl Into<String>) -> Self {
        Self(s.into())
    }

    /// Get the expression string
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Check if the expression is empty
    pub fn is_empty(&self) -> bool {
        self.0.trim().is_empty()
    }
}

impl From<String> for Expression {
    fn from(s: String) -> Self {
        Self(s)
    }
}

impl From<&str> for Expression {
    fn from(s: &str) -> Self {
        Self(s.to_string())
    }
}

impl AsRef<str> for Expression {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

/// Either an expression or a numeric value
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ExpressionOrNumber {
    /// A numeric constant
    Number(f64),
    /// A Rust expression
    Expression(String),
}

impl ExpressionOrNumber {
    /// Convert to a Rust expression string
    pub fn to_rust_expr(&self) -> String {
        match self {
            Self::Number(n) => format!("{:.6}", n),
            Self::Expression(s) => s.clone(),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Differential Equation Specification
// ═══════════════════════════════════════════════════════════════════════════════

/// Differential equation specification (string or object format)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum DiffEqSpec {
    /// Single string with all equations
    String(String),
    /// Map of compartment name to equation
    Object(HashMap<String, String>),
}

impl DiffEqSpec {
    /// Check if empty
    pub fn is_empty(&self) -> bool {
        match self {
            Self::String(s) => s.trim().is_empty(),
            Self::Object(m) => m.is_empty(),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Initial Conditions
// ═══════════════════════════════════════════════════════════════════════════════

/// Initial condition specification
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum InitSpec {
    /// Single string with all init code
    String(String),
    /// Map of compartment/state name to initial value
    Object(HashMap<String, ExpressionOrNumber>),
}

// ═══════════════════════════════════════════════════════════════════════════════
// Output Definition
// ═══════════════════════════════════════════════════════════════════════════════

/// Definition of a model output
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OutputDefinition {
    /// Output identifier
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,

    /// Output equation expression
    pub equation: String,

    /// Human-readable name
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,

    /// Output units
    #[serde(skip_serializing_if = "Option::is_none")]
    pub units: Option<String>,
}

// ═══════════════════════════════════════════════════════════════════════════════
// Derived Parameters
// ═══════════════════════════════════════════════════════════════════════════════

/// Derived parameter definition
///
/// Derived parameters are computed from primary parameters using expressions.
/// For example, ke = CL / V computes elimination rate constant from
/// clearance and volume.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DerivedParameter {
    /// Symbol for the derived parameter
    pub symbol: String,

    /// Expression to compute the derived parameter
    pub expression: String,
}

// ═══════════════════════════════════════════════════════════════════════════════
// Covariates
// ═══════════════════════════════════════════════════════════════════════════════

/// Covariate type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum CovariateType {
    /// Continuous covariate
    #[default]
    Continuous,
    /// Categorical covariate
    Categorical,
}

/// Interpolation method for time-varying covariates
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum InterpolationMethod {
    /// Linear interpolation
    #[default]
    Linear,
    /// Constant (use value at time point)
    Constant,
    /// Last observation carried forward
    Locf,
}

/// Covariate definition
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CovariateDefinition {
    /// Covariate identifier
    pub id: String,

    /// Human-readable name
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,

    /// Covariate type
    #[serde(rename = "type", default)]
    pub cov_type: CovariateType,

    /// Units for continuous covariates
    #[serde(skip_serializing_if = "Option::is_none")]
    pub units: Option<String>,

    /// Reference value for centering
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reference: Option<f64>,

    /// Interpolation method
    #[serde(default)]
    pub interpolation: InterpolationMethod,

    /// Possible values for categorical covariates
    #[serde(skip_serializing_if = "Option::is_none")]
    pub levels: Option<Vec<String>>,
}

/// Covariate effect type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CovariateEffectType {
    /// Allometric scaling: P * (cov/ref)^exp
    Allometric,
    /// Linear effect: P * (1 + slope * (cov - ref))
    Linear,
    /// Exponential effect: P * exp(slope * (cov - ref))
    Exponential,
    /// Proportional effect: P * (1 + slope * cov)
    Proportional,
    /// Categorical effect: P * theta_level
    Categorical,
    /// Custom expression
    Custom,
}

/// Covariate effect specification
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CovariateEffect {
    /// Parameter affected by this covariate
    pub on: String,

    /// Covariate ID
    #[serde(skip_serializing_if = "Option::is_none")]
    pub covariate: Option<String>,

    /// Effect type
    #[serde(rename = "type")]
    pub effect_type: CovariateEffectType,

    /// Exponent for allometric scaling
    #[serde(skip_serializing_if = "Option::is_none")]
    pub exponent: Option<f64>,

    /// Slope for linear/exponential effects
    #[serde(skip_serializing_if = "Option::is_none")]
    pub slope: Option<f64>,

    /// Reference value for centering
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reference: Option<f64>,

    /// Custom expression
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expression: Option<String>,

    /// Multipliers for categorical levels
    #[serde(skip_serializing_if = "Option::is_none")]
    pub levels: Option<HashMap<String, f64>>,
}

// ═══════════════════════════════════════════════════════════════════════════════
// Error Model Type (hint only, values provided by PMcore Settings)
// ═══════════════════════════════════════════════════════════════════════════════

/// Error model type (for documentation/hints only)
///
/// Note: The actual error model parameters (σ values) should be configured
/// in PMcore's Settings struct, not in the JSON model. This enum is kept
/// for documentation purposes and to indicate the intended error structure.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ErrorModelType {
    /// Additive error: σ = a
    Additive,
    /// Proportional error: σ = b × f
    Proportional,
    /// Combined error: σ = √(a² + b²×f²)
    Combined,
    /// Polynomial error: σ = c₀ + c₁f + c₂f² + c₃f³
    Polynomial,
}

// ═══════════════════════════════════════════════════════════════════════════════
// UI Metadata (ignored by compiler)
// ═══════════════════════════════════════════════════════════════════════════════

/// Model complexity level
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Complexity {
    Basic,
    Intermediate,
    Advanced,
}

/// Model category
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Category {
    Pk,
    Pd,
    Pkpd,
    Disease,
    Other,
}

/// Position for layout
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Position {
    pub x: f64,
    pub y: f64,
}

/// Display information for UI
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct DisplayInfo {
    /// Human-readable model name
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,

    /// Abbreviated name
    #[serde(skip_serializing_if = "Option::is_none")]
    pub short_name: Option<String>,

    /// Model category
    #[serde(skip_serializing_if = "Option::is_none")]
    pub category: Option<Category>,

    /// Model subcategory
    #[serde(skip_serializing_if = "Option::is_none")]
    pub subcategory: Option<String>,

    /// Complexity level
    #[serde(skip_serializing_if = "Option::is_none")]
    pub complexity: Option<Complexity>,

    /// Icon identifier
    #[serde(skip_serializing_if = "Option::is_none")]
    pub icon: Option<String>,

    /// Searchable tags
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tags: Option<Vec<String>>,
}

/// Literature reference
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Reference {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub authors: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub journal: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub year: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub doi: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pmid: Option<String>,
}

/// LaTeX equations for display
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct EquationDocs {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub differential: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub solution: Option<String>,
}

/// Rich documentation
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct Documentation {
    /// One-line summary
    #[serde(skip_serializing_if = "Option::is_none")]
    pub summary: Option<String>,

    /// Detailed description
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,

    /// LaTeX equations
    #[serde(skip_serializing_if = "Option::is_none")]
    pub equations: Option<EquationDocs>,

    /// Model assumptions
    #[serde(skip_serializing_if = "Option::is_none")]
    pub assumptions: Option<Vec<String>>,

    /// When to use this model
    #[serde(skip_serializing_if = "Option::is_none")]
    pub when_to_use: Option<Vec<String>>,

    /// When NOT to use this model
    #[serde(skip_serializing_if = "Option::is_none")]
    pub when_not_to_use: Option<Vec<String>>,

    /// Literature references
    #[serde(skip_serializing_if = "Option::is_none")]
    pub references: Option<Vec<Reference>>,
}

/// Optional features that can be enabled
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Feature {
    LagTime,
    Bioavailability,
    InitialConditions,
}
