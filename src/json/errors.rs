//! Error types for JSON model parsing and code generation

use thiserror::Error;

/// Errors that can occur when working with JSON models
#[derive(Debug, Error)]
pub enum JsonModelError {
    // ─────────────────────────────────────────────────────────────────────────
    // Parsing Errors
    // ─────────────────────────────────────────────────────────────────────────
    /// Failed to parse JSON
    #[error("Failed to parse JSON: {0}")]
    ParseError(#[from] serde_json::Error),

    /// Unsupported schema version
    #[error("Unsupported schema version '{version}'. Supported versions: {supported}")]
    UnsupportedSchema { version: String, supported: String },

    // ─────────────────────────────────────────────────────────────────────────
    // Structural Errors
    // ─────────────────────────────────────────────────────────────────────────
    /// Missing required field for model type
    #[error("Missing required field '{field}' for {model_type} models")]
    MissingField { field: String, model_type: String },

    /// Invalid field for model type
    #[error("Field '{field}' is not valid for {model_type} models")]
    InvalidFieldForType { field: String, model_type: String },

    /// Missing output definitions
    #[error("Model must have an 'outputs' field")]
    MissingOutput,

    /// Missing parameters
    #[error("Model must have 'parameters' field (unless using 'extends')")]
    MissingParameters,

    // ─────────────────────────────────────────────────────────────────────────
    // Semantic Errors
    // ─────────────────────────────────────────────────────────────────────────
    /// Undefined parameter used in expression
    #[error("Undefined parameter '{name}' used in {context}")]
    UndefinedParameter { name: String, context: String },

    /// Undefined compartment
    #[error("Undefined compartment '{name}'")]
    UndefinedCompartment { name: String },

    /// Undefined covariate
    #[error("Undefined covariate '{name}' referenced in covariate effect")]
    UndefinedCovariate { name: String },

    /// Parameter order mismatch for analytical function
    #[error(
        "Parameter order warning for '{function}': expected parameters in order {expected:?}, \
         but got {actual:?}. This may cause incorrect model behavior."
    )]
    ParameterOrderWarning {
        function: String,
        expected: Vec<String>,
        actual: Vec<String>,
    },

    /// Duplicate parameter name
    #[error("Duplicate parameter name: '{name}'")]
    DuplicateParameter { name: String },

    /// Duplicate compartment name
    #[error("Duplicate compartment name: '{name}'")]
    DuplicateCompartment { name: String },

    /// Duplicate covariate name
    #[error("Duplicate covariate name: '{name}'")]
    DuplicateCovariate { name: String },

    /// Duplicate output identifier
    #[error("Duplicate output identifier: '{id}'")]
    DuplicateOutput { id: String },

    /// Invalid neqs specification
    #[error("Invalid neqs: expected [num_states, num_outputs], got {0:?}")]
    InvalidNeqs(Vec<usize>),

    /// Schema-specific rule violation
    #[error("Field '{field}' violates schema {schema}: {message}")]
    SchemaRuleViolation {
        field: String,
        schema: String,
        message: String,
    },

    // ─────────────────────────────────────────────────────────────────────────
    // Expression Errors
    // ─────────────────────────────────────────────────────────────────────────
    /// Invalid expression syntax
    #[error("Invalid expression in {context}: {message}")]
    InvalidExpression { context: String, message: String },

    /// Empty expression
    #[error("Empty expression in {context}")]
    EmptyExpression { context: String },

    /// Expression parse error
    #[error("Failed to parse expression in {context}: {message}")]
    ExpressionParseError { context: String, message: String },

    // ─────────────────────────────────────────────────────────────────────────
    // Library Errors
    // ─────────────────────────────────────────────────────────────────────────
    /// Model not found in library
    #[error("Model '{0}' not found in library")]
    ModelNotFound(String),

    /// Circular inheritance detected
    #[error("Circular inheritance detected: {0}")]
    CircularInheritance(String),

    /// General library error (file I/O, etc.)
    #[error("Library error: {0}")]
    LibraryError(String),

    // ─────────────────────────────────────────────────────────────────────────
    // Code Generation Errors
    // ─────────────────────────────────────────────────────────────────────────
    /// Code generation failed
    #[error("Code generation failed: {0}")]
    CodeGenError(String),

    /// Compilation failed
    #[error("Compilation failed: {0}")]
    CompilationError(String),
}

impl JsonModelError {
    /// Create a missing field error
    pub fn missing_field(field: impl Into<String>, model_type: impl Into<String>) -> Self {
        Self::MissingField {
            field: field.into(),
            model_type: model_type.into(),
        }
    }

    /// Create an invalid field error
    pub fn invalid_field(field: impl Into<String>, model_type: impl Into<String>) -> Self {
        Self::InvalidFieldForType {
            field: field.into(),
            model_type: model_type.into(),
        }
    }

    /// Create an undefined parameter error
    pub fn undefined_param(name: impl Into<String>, context: impl Into<String>) -> Self {
        Self::UndefinedParameter {
            name: name.into(),
            context: context.into(),
        }
    }

    /// Create an invalid expression error
    pub fn invalid_expr(context: impl Into<String>, message: impl Into<String>) -> Self {
        Self::InvalidExpression {
            context: context.into(),
            message: message.into(),
        }
    }

    /// Create a schema rule violation error.
    pub fn schema_rule(
        field: impl Into<String>,
        schema: impl Into<String>,
        message: impl Into<String>,
    ) -> Self {
        Self::SchemaRuleViolation {
            field: field.into(),
            schema: schema.into(),
            message: message.into(),
        }
    }
}
