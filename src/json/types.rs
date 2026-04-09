//! Core type definitions for JSON models

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

fn default_required() -> bool {
    true
}

fn is_default_required(value: &bool) -> bool {
    *value
}

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

/// Differential equation specification keyed by compartment or state id.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum DiffEqSpec {
    /// Map of compartment name to equation
    Object(HashMap<String, String>),
}

impl DiffEqSpec {
    /// Check if empty
    pub fn is_empty(&self) -> bool {
        match self {
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
    pub id: String,

    /// Output equation expression
    pub equation: String,

    /// Human-readable name
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,

    /// Output units
    #[serde(skip_serializing_if = "Option::is_none")]
    pub units: Option<String>,
}

/// Ordered named equation used for canonical secondary expressions.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NamedEquation {
    /// Symbol introduced by this equation.
    pub id: String,

    /// Expression assigned to the symbol.
    pub equation: String,
}

/// Covariate definition used by model expressions and UI consumers.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CovariateDefinition {
    /// Symbol used inside expressions.
    pub id: String,

    /// Dataset column backing the symbol. Defaults to `id` when omitted.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub column: Option<String>,

    /// Whether the covariate must be present in incoming data.
    #[serde(
        default = "default_required",
        skip_serializing_if = "is_default_required"
    )]
    pub required: bool,

    /// Optional reference value used by UI consumers or scaling conventions.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reference: Option<f64>,
}

impl CovariateDefinition {
    pub fn symbol(&self) -> &str {
        &self.id
    }

    pub fn column_name(&self) -> &str {
        self.column.as_deref().unwrap_or(&self.id)
    }
}

/// Normalized executable representation consumed by validation and code generation.
///
/// This provides the compile-time shape after validation and normalization.
/// Compiler-ignored editor metadata is excluded.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ExecutableModel {
    /// Unique model identifier.
    pub id: String,

    /// Model equation type.
    #[serde(rename = "type")]
    pub model_type: ModelType,

    /// Parameter names in fetch order.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub parameters: Vec<String>,

    /// Compartment names in declaration order.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub compartments: Vec<String>,

    /// State names in declaration order.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub states: Vec<String>,

    /// Built-in analytical function, when applicable.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub analytical: Option<AnalyticalFunction>,

    /// Differential equations for ODE models.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub diffeq: Option<DiffEqSpec>,

    /// Drift equations for SDE models.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub drift: Option<DiffEqSpec>,

    /// Diffusion definitions for SDE models.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub diffusion: Option<HashMap<String, ExpressionOrNumber>>,

    /// Ordered executable calculations evaluated before outputs and equations.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub calculations: Vec<NamedEquation>,

    /// Canonicalized outputs with stable identifiers.
    pub outputs: Vec<OutputDefinition>,

    /// Initial conditions.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub init: Option<InitSpec>,

    /// Lag times keyed by compartment/state id.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lag: Option<HashMap<String, ExpressionOrNumber>>,

    /// Bioavailability keyed by compartment/state id.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fa: Option<HashMap<String, ExpressionOrNumber>>,

    /// Optional explicit equation dimensions.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub neqs: Option<(usize, usize)>,

    /// Number of particles for SDE simulation.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub particles: Option<usize>,

    /// Covariates used by executable expressions.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub covariates: Vec<CovariateDefinition>,
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
    #[serde(
        alias = "shortName",
        rename = "shortName",
        skip_serializing_if = "Option::is_none"
    )]
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
    #[serde(
        alias = "whenToUse",
        rename = "whenToUse",
        skip_serializing_if = "Option::is_none"
    )]
    pub when_to_use: Option<Vec<String>>,

    /// When NOT to use this model
    #[serde(
        alias = "whenNotToUse",
        rename = "whenNotToUse",
        skip_serializing_if = "Option::is_none"
    )]
    pub when_not_to_use: Option<Vec<String>>,

    /// Literature references
    #[serde(skip_serializing_if = "Option::is_none")]
    pub references: Option<Vec<Reference>>,
}

/// Compiler-ignored editor metadata used by richer consumers like papir.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct EditorInfo {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub display: Option<DisplayInfo>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub layout: Option<HashMap<String, Position>>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub documentation: Option<Documentation>,
}

/// Optional features that can be enabled
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Feature {
    LagTime,
    Bioavailability,
    InitialConditions,
}
