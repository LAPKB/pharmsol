//! JSON Model Definition and Code Generation
//!
//! This module provides functionality for defining pharmacometric models using JSON
//! and generating Rust code that can be compiled by the `exa` module.
//!
//! # Overview
//!
//! The JSON model system provides a declarative way to define pharmacometric models
//! without writing Rust code directly. Models are defined in JSON following a
//! structured schema, then validated and compiled to native code.
//!
//! The system supports three equation types:
//! - **Analytical**: Built-in closed-form solutions (fastest execution)
//! - **ODE**: Custom ordinary differential equations
//! - **SDE**: Stochastic differential equations with particle filtering
//!
//! # Quick Start
//!
//! ```ignore
//! use pharmsol::json::{parse_json, validate_json, generate_code};
//!
//! // Define a model in JSON
//! let json = r#"{
//!     "schema": "1.0",
//!     "id": "pk_1cmt_oral",
//!     "type": "analytical",
//!     "analytical": "one_compartment_with_absorption",
//!     "parameters": ["ka", "ke", "V"],
//!     "output": "x[1] / V"
//! }"#;
//!
//! // Parse and validate
//! let validated = validate_json(json)?;
//!
//! // Generate Rust code
//! let code = generate_code(json)?;
//! println!("Generated: {}", code.equation_code);
//! ```
//!
//! # Using the Model Library
//!
//! The library provides pre-built standard PK models:
//!
//! ```ignore
//! use pharmsol::json::ModelLibrary;
//!
//! let library = ModelLibrary::builtin();
//!
//! // List available models
//! for id in library.list() {
//!     println!("Available: {}", id);
//! }
//!
//! // Get a specific model
//! let model = library.get("pk/1cmt-oral").unwrap();
//!
//! // Search by keyword
//! let oral_models = library.search("oral");
//!
//! // Filter by type
//! let ode_models = library.filter_by_type(ModelType::Ode);
//! ```
//!
//! # Model Inheritance
//!
//! Models can extend base models to add customizations:
//!
//! ```ignore
//! use pharmsol::json::{JsonModel, ModelLibrary};
//!
//! let mut library = ModelLibrary::builtin();
//!
//! // Define a model that extends a library model
//! let derived = JsonModel::from_str(r#"{
//!     "schema": "1.0",
//!     "id": "pk_1cmt_wt",
//!     "extends": "pk/1cmt-oral",
//!     "type": "analytical",
//!     "analytical": "one_compartment_with_absorption",
//!     "parameters": ["ka", "ke", "V"],
//!     "covariates": [{ "id": "WT", "reference": 70.0 }],
//!     "covariateEffects": [{
//!         "on": "V",
//!         "covariate": "WT",
//!         "type": "allometric",
//!         "exponent": 1.0,
//!         "reference": 70.0
//!     }]
//! }"#)?;
//!
//! // Resolve inherits base model's output expression
//! let resolved = library.resolve(&derived)?;
//! ```
//!
//! # JSON Schema
//!
//! ## Required Fields
//!
//! | Field | Description |
//! |-------|-------------|
//! | `schema` | Schema version (currently `"1.0"`) |
//! | `id` | Unique model identifier |
//! | `type` | Equation type: `"analytical"`, `"ode"`, or `"sde"` |
//!
//! ## Model Type Specific Fields
//!
//! ### Analytical Models
//! - `analytical`: One of the built-in functions (e.g., `"one_compartment_with_absorption"`)
//! - `parameters`: Parameter names in order expected by the analytical function
//! - `output`: Output equation expression
//!
//! ### ODE Models
//! - `compartments`: List of compartment names
//! - `diffeq`: Differential equations (object or string)
//! - `parameters`: Parameter names
//! - `output`: Output equation expression
//!
//! ### SDE Models
//! - `states`: List of state variable names
//! - `drift`: Drift equations (deterministic part)
//! - `diffusion`: Diffusion coefficients
//! - `particles`: Number of particles for simulation
//!
//! ## Optional Features
//!
//! - `lag`: Lag times per compartment
//! - `fa`: Bioavailability factors
//! - `init`: Initial conditions
//! - `covariates`: Covariate definitions
//! - `covariateEffects`: Covariate effect specifications
//! - `errorModel`: Residual error model
//!
//! # Available Analytical Functions
//!
//! | Function | Parameters | States |
//! |----------|------------|--------|
//! | `one_compartment` | ke | 1 |
//! | `one_compartment_with_absorption` | ka, ke | 2 |
//! | `two_compartments` | ke, kcp, kpc | 2 |
//! | `two_compartments_with_absorption` | ke, ka, kcp, kpc | 3 |
//! | `three_compartments` | k10, k12, k13, k21, k31 | 3 |
//! | `three_compartments_with_absorption` | ka, k10, k12, k13, k21, k31 | 4 |
//!
//! # Error Handling
//!
//! All functions return `Result<T, JsonModelError>` with descriptive errors:
//!
//! ```ignore
//! match validate_json(json) {
//!     Ok(model) => println!("Valid model: {}", model.inner().id),
//!     Err(JsonModelError::MissingField { field, model_type }) => {
//!         eprintln!("Missing {} for {} model", field, model_type);
//!     }
//!     Err(JsonModelError::UnsupportedSchema { version, .. }) => {
//!         eprintln!("Schema {} not supported", version);
//!     }
//!     Err(e) => eprintln!("Error: {}", e),
//! }
//! ```

mod codegen;
mod errors;
pub mod library;
mod model;
mod types;
mod validation;

pub use codegen::{CodeGenerator, GeneratedCode};
pub use errors::JsonModelError;
pub use library::ModelLibrary;
pub use model::JsonModel;
pub use types::*;
pub use validation::{ValidatedModel, Validator};

/// Parse a JSON string into a JsonModel
pub fn parse_json(json: &str) -> Result<JsonModel, JsonModelError> {
    JsonModel::from_str(json)
}

/// Parse and validate a JSON model
pub fn validate_json(json: &str) -> Result<ValidatedModel, JsonModelError> {
    let model = JsonModel::from_str(json)?;
    let validator = Validator::new();
    validator.validate(&model)
}

/// Parse, validate, and generate code from a JSON model
pub fn generate_code(json: &str) -> Result<GeneratedCode, JsonModelError> {
    let model = JsonModel::from_str(json)?;
    let validator = Validator::new();
    let validated = validator.validate(&model)?;
    let generator = CodeGenerator::new(validated.inner());
    generator.generate()
}

/// Compile a JSON model to a dynamic library
///
/// This is the high-level API that combines parsing, validation,
/// code generation, and compilation into a single call.
///
/// Requires the `exa` feature to be enabled.
#[cfg(feature = "exa")]
pub fn compile_json<E: crate::Equation>(
    json: &str,
    output_path: Option<std::path::PathBuf>,
    template_path: std::path::PathBuf,
    event_callback: impl Fn(String, String) + Send + Sync + 'static,
) -> Result<String, JsonModelError> {
    let generated = generate_code(json)?;

    crate::exa::build::compile::<E>(
        generated.equation_code,
        output_path,
        generated.parameters,
        template_path,
        event_callback,
    )
    .map_err(|e| JsonModelError::CompilationError(e.to_string()))
}
