//! Model Library
//!
//! Provides a registry of built-in pharmacometric models that can be:
//! - Used directly via their ID
//! - Extended via the `extends` field for customization
//!
//! # Example
//!
//! ```rust,ignore
//! use pharmsol::json::library::ModelLibrary;
//!
//! let library = ModelLibrary::builtin();
//!
//! // List available models
//! for id in library.list() {
//!     println!("Available: {}", id);
//! }
//!
//! // Get a model
//! if let Some(model) = library.get("pk/1cmt-iv") {
//!     println!("Found model: {}", model.id);
//! }
//! ```

use crate::json::errors::JsonModelError;
use crate::json::model::JsonModel;
use crate::json::types::{
    CovariateDefinition, DisplayInfo, Documentation, EditorInfo, ExecutableModel, ModelType,
    NamedEquation,
};
use crate::json::Validator;
use std::collections::HashMap;
use std::path::Path;

/// A registry of JSON model definitions
#[derive(Debug, Clone)]
pub struct ModelLibrary {
    models: HashMap<String, JsonModel>,
}

// Embed built-in models at compile time
mod embedded {
    // PK Analytical Models
    pub const PK_1CMT_IV: &str = include_str!("models/pk_1cmt_iv.json");
    pub const PK_1CMT_ORAL: &str = include_str!("models/pk_1cmt_oral.json");
    pub const PK_2CMT_IV: &str = include_str!("models/pk_2cmt_iv.json");
    pub const PK_2CMT_ORAL: &str = include_str!("models/pk_2cmt_oral.json");
    pub const PK_3CMT_IV: &str = include_str!("models/pk_3cmt_iv.json");
    pub const PK_3CMT_ORAL: &str = include_str!("models/pk_3cmt_oral.json");

    // PK ODE Models
    pub const PK_1CMT_IV_ODE: &str = include_str!("models/pk_1cmt_iv_ode.json");
    pub const PK_1CMT_ORAL_ODE: &str = include_str!("models/pk_1cmt_oral_ode.json");
    pub const PK_2CMT_IV_ODE: &str = include_str!("models/pk_2cmt_iv_ode.json");
    pub const PK_2CMT_ORAL_ODE: &str = include_str!("models/pk_2cmt_oral_ode.json");
}

impl ModelLibrary {
    /// Create a new empty library
    pub fn new() -> Self {
        Self {
            models: HashMap::new(),
        }
    }

    /// Create a library with all built-in models
    pub fn builtin() -> Self {
        let mut library = Self::new();

        // Load embedded models
        let embedded_models = [
            embedded::PK_1CMT_IV,
            embedded::PK_1CMT_ORAL,
            embedded::PK_2CMT_IV,
            embedded::PK_2CMT_ORAL,
            embedded::PK_3CMT_IV,
            embedded::PK_3CMT_ORAL,
            embedded::PK_1CMT_IV_ODE,
            embedded::PK_1CMT_ORAL_ODE,
            embedded::PK_2CMT_IV_ODE,
            embedded::PK_2CMT_ORAL_ODE,
        ];

        for json in embedded_models {
            if let Ok(model) = JsonModel::from_str(json) {
                library.models.insert(model.id.clone(), model);
            }
        }

        library
    }

    /// Load models from a directory (recursively searches for .json files)
    pub fn from_dir(path: &Path) -> Result<Self, JsonModelError> {
        let mut library = Self::new();
        library.load_dir(path)?;
        Ok(library)
    }

    /// Load models from a directory into this library
    pub fn load_dir(&mut self, path: &Path) -> Result<(), JsonModelError> {
        if !path.exists() {
            return Err(JsonModelError::LibraryError(format!(
                "Directory not found: {}",
                path.display()
            )));
        }

        Self::load_dir_recursive(path, &mut self.models)?;
        Ok(())
    }

    fn load_dir_recursive(
        path: &Path,
        models: &mut HashMap<String, JsonModel>,
    ) -> Result<(), JsonModelError> {
        let entries = std::fs::read_dir(path).map_err(|e| {
            JsonModelError::LibraryError(format!("Failed to read directory: {}", e))
        })?;

        for entry in entries {
            let entry = entry.map_err(|e| {
                JsonModelError::LibraryError(format!("Failed to read entry: {}", e))
            })?;
            let file_path = entry.path();

            if file_path.is_dir() {
                Self::load_dir_recursive(&file_path, models)?;
            } else if file_path.extension().is_some_and(|ext| ext == "json") {
                let content = std::fs::read_to_string(&file_path).map_err(|e| {
                    JsonModelError::LibraryError(format!(
                        "Failed to read {}: {}",
                        file_path.display(),
                        e
                    ))
                })?;

                match JsonModel::from_str(&content) {
                    Ok(model) => {
                        models.insert(model.id.clone(), model);
                    }
                    Err(e) => {
                        // Log warning but continue loading other models
                        eprintln!("Warning: Failed to parse {}: {}", file_path.display(), e);
                    }
                }
            }
        }

        Ok(())
    }

    /// Get a model by ID
    pub fn get(&self, id: &str) -> Option<&JsonModel> {
        self.models.get(id)
    }

    /// Check if a model exists
    pub fn contains(&self, id: &str) -> bool {
        self.models.contains_key(id)
    }

    /// Add a model to the library
    pub fn add(&mut self, model: JsonModel) {
        self.models.insert(model.id.clone(), model);
    }

    /// Remove a model from the library
    pub fn remove(&mut self, id: &str) -> Option<JsonModel> {
        self.models.remove(id)
    }

    /// List all model IDs
    pub fn list(&self) -> Vec<&str> {
        let mut ids: Vec<&str> = self.models.keys().map(|s| s.as_str()).collect();
        ids.sort();
        ids
    }

    /// Get the number of models
    pub fn len(&self) -> usize {
        self.models.len()
    }

    /// Check if the library is empty
    pub fn is_empty(&self) -> bool {
        self.models.is_empty()
    }

    /// Search models by partial ID or name match
    pub fn search(&self, query: &str) -> Vec<&JsonModel> {
        let query_lower = query.to_lowercase();
        self.models
            .values()
            .filter(|model| {
                // Match by ID
                if model.id.to_lowercase().contains(&query_lower) {
                    return true;
                }
                // Match by name in display info
                if let Some(display) = model.display_info() {
                    if let Some(ref name) = display.name {
                        if name.to_lowercase().contains(&query_lower) {
                            return true;
                        }
                    }
                }
                false
            })
            .collect()
    }

    /// Filter models by type
    pub fn filter_by_type(&self, model_type: ModelType) -> Vec<&JsonModel> {
        self.models
            .values()
            .filter(|m| m.model_type == model_type)
            .collect()
    }

    /// Filter models by tag (from display info)
    pub fn filter_by_tag(&self, tag: &str) -> Vec<&JsonModel> {
        let tag_lower = tag.to_lowercase();
        self.models
            .values()
            .filter(|model| {
                if let Some(display) = model.display_info() {
                    if let Some(ref tags) = display.tags {
                        return tags.iter().any(|t| t.to_lowercase() == tag_lower);
                    }
                }
                false
            })
            .collect()
    }

    /// Resolve a model's inheritance chain, returning a fully resolved model
    ///
    /// This processes the `extends` field to merge base model properties
    /// with the derived model's overrides.
    pub fn resolve(&self, model: &JsonModel) -> Result<JsonModel, JsonModelError> {
        self.resolve_with_chain(model, &mut Vec::new())
    }

    /// Resolve a model's inheritance chain and normalize it into executable form.
    pub fn resolve_executable(&self, model: &JsonModel) -> Result<ExecutableModel, JsonModelError> {
        let resolved = self.resolve(model)?;
        let validated = Validator::new().validate(&resolved)?;
        validated.executable()
    }

    fn resolve_with_chain(
        &self,
        model: &JsonModel,
        chain: &mut Vec<String>,
    ) -> Result<JsonModel, JsonModelError> {
        // Check for circular inheritance
        if chain.contains(&model.id) {
            return Err(JsonModelError::CircularInheritance(format!(
                "{} -> {}",
                chain.join(" -> "),
                model.id
            )));
        }

        // If no base, return model as-is
        let Some(ref base_id) = model.extends else {
            return Ok(model.clone());
        };

        // Track inheritance chain
        chain.push(model.id.clone());

        // Get base model
        let base = self
            .get(base_id)
            .ok_or_else(|| JsonModelError::ModelNotFound(base_id.clone()))?;

        // Recursively resolve base
        let resolved_base = self.resolve_with_chain(base, chain)?;

        // Merge: derived model overrides base
        Ok(merge_models(&resolved_base, model))
    }
}

impl Default for ModelLibrary {
    fn default() -> Self {
        Self::new()
    }
}

/// Merge two models, with derived overriding base
fn merge_models(base: &JsonModel, derived: &JsonModel) -> JsonModel {
    JsonModel {
        // ─────────────────────────────────────────────────────────────────────
        // Layer 1: Identity (derived always owns these)
        // ─────────────────────────────────────────────────────────────────────
        schema: derived.schema.clone(),
        id: derived.id.clone(),
        model_type: derived.model_type,
        extends: None, // Clear extends after resolution
        version: derived.version.clone().or_else(|| base.version.clone()),
        aliases: merge_option_vec_dedup(&base.aliases, &derived.aliases),

        // ─────────────────────────────────────────────────────────────────────
        // Layer 2: Structural Model
        // ─────────────────────────────────────────────────────────────────────
        parameters: derived
            .parameters
            .clone()
            .or_else(|| base.parameters.clone()),
        compartments: derived
            .compartments
            .clone()
            .or_else(|| base.compartments.clone()),
        states: derived.states.clone().or_else(|| base.states.clone()),

        // ─────────────────────────────────────────────────────────────────────
        // Equation Fields
        // ─────────────────────────────────────────────────────────────────────
        analytical: derived.analytical.or(base.analytical),
        diffeq: derived.diffeq.clone().or_else(|| base.diffeq.clone()),
        drift: derived.drift.clone().or_else(|| base.drift.clone()),
        diffusion: derived.diffusion.clone().or_else(|| base.diffusion.clone()),
        secondary: merge_secondary(&base.secondary, &derived.secondary),

        // ─────────────────────────────────────────────────────────────────────
        // Output
        // ─────────────────────────────────────────────────────────────────────
        outputs: derived.outputs.clone().or_else(|| base.outputs.clone()),

        // ─────────────────────────────────────────────────────────────────────
        // Optional Features
        // ─────────────────────────────────────────────────────────────────────
        init: derived.init.clone().or_else(|| base.init.clone()),
        lag: derived.lag.clone().or_else(|| base.lag.clone()),
        fa: derived.fa.clone().or_else(|| base.fa.clone()),
        neqs: derived.neqs.or(base.neqs),
        particles: derived.particles.or(base.particles),

        // ─────────────────────────────────────────────────────────────────────
        // Layer 3: Model Extensions
        // ─────────────────────────────────────────────────────────────────────
        features: merge_option_vec(&base.features, &derived.features),
        covariates: merge_covariates(&base.covariates, &derived.covariates),

        // ─────────────────────────────────────────────────────────────────────
        // Layer 4: UI Metadata
        // ─────────────────────────────────────────────────────────────────────
        editor: merge_editor(&base.editor, &derived.editor),
    }
}

/// Merge optional vectors (append derived items)
fn merge_option_vec<T: Clone>(base: &Option<Vec<T>>, derived: &Option<Vec<T>>) -> Option<Vec<T>> {
    match (base, derived) {
        (None, None) => None,
        (Some(b), None) => Some(b.clone()),
        (None, Some(d)) => Some(d.clone()),
        (Some(b), Some(d)) => {
            let mut merged = b.clone();
            merged.extend(d.iter().cloned());
            Some(merged)
        }
    }
}

/// Merge optional string vectors, deduplicating entries
fn merge_option_vec_dedup(
    base: &Option<Vec<String>>,
    derived: &Option<Vec<String>>,
) -> Option<Vec<String>> {
    match (base, derived) {
        (None, None) => None,
        (Some(b), None) => Some(b.clone()),
        (None, Some(d)) => Some(d.clone()),
        (Some(b), Some(d)) => {
            let mut merged = b.clone();
            for item in d {
                if !merged.contains(item) {
                    merged.push(item.clone());
                }
            }
            Some(merged)
        }
    }
}

/// Merge covariates by symbol, preserving order and allowing derived overrides.
fn merge_covariates(
    base: &Option<Vec<CovariateDefinition>>,
    derived: &Option<Vec<CovariateDefinition>>,
) -> Option<Vec<CovariateDefinition>> {
    match (base, derived) {
        (None, None) => None,
        (Some(b), None) => Some(b.clone()),
        (None, Some(d)) => Some(d.clone()),
        (Some(b), Some(d)) => {
            let mut merged = b.clone();
            for item in d {
                if let Some(existing) = merged.iter_mut().find(|existing| existing.id == item.id) {
                    *existing = item.clone();
                } else {
                    merged.push(item.clone());
                }
            }
            Some(merged)
        }
    }
}

/// Merge secondary equations: derived keys override base keys, preserving order
fn merge_secondary(
    base: &Option<Vec<NamedEquation>>,
    derived: &Option<Vec<NamedEquation>>,
) -> Option<Vec<NamedEquation>> {
    match (base, derived) {
        (None, None) => None,
        (Some(b), None) => Some(b.clone()),
        (None, Some(d)) => Some(d.clone()),
        (Some(b), Some(d)) => {
            // Start with base, then override or append from derived
            let mut merged = b.clone();
            for entry in d {
                if let Some(existing) = merged.iter_mut().find(|existing| existing.id == entry.id) {
                    existing.equation = entry.equation.clone();
                } else {
                    merged.push(entry.clone());
                }
            }
            Some(merged)
        }
    }
}

/// Merge optional HashMaps (derived overrides base keys)
fn merge_option_hashmap<K: Clone + std::cmp::Eq + std::hash::Hash, V: Clone>(
    base: &Option<HashMap<K, V>>,
    derived: &Option<HashMap<K, V>>,
) -> Option<HashMap<K, V>> {
    match (base, derived) {
        (None, None) => None,
        (Some(b), None) => Some(b.clone()),
        (None, Some(d)) => Some(d.clone()),
        (Some(b), Some(d)) => {
            let mut merged = b.clone();
            merged.extend(d.iter().map(|(k, v)| (k.clone(), v.clone())));
            Some(merged)
        }
    }
}

/// Merge display info (derived overrides base)
fn merge_display(base: &Option<DisplayInfo>, derived: &Option<DisplayInfo>) -> Option<DisplayInfo> {
    match (base, derived) {
        (None, None) => None,
        (Some(b), None) => Some(b.clone()),
        (None, Some(d)) => Some(d.clone()),
        (Some(b), Some(d)) => Some(DisplayInfo {
            name: d.name.clone().or_else(|| b.name.clone()),
            short_name: d.short_name.clone().or_else(|| b.short_name.clone()),
            category: d.category.or(b.category),
            subcategory: d.subcategory.clone().or_else(|| b.subcategory.clone()),
            complexity: d.complexity.or(b.complexity),
            icon: d.icon.clone().or_else(|| b.icon.clone()),
            tags: merge_option_vec(&b.tags, &d.tags),
        }),
    }
}

/// Merge documentation (derived overrides base)
fn merge_documentation(
    base: &Option<Documentation>,
    derived: &Option<Documentation>,
) -> Option<Documentation> {
    match (base, derived) {
        (None, None) => None,
        (Some(b), None) => Some(b.clone()),
        (None, Some(d)) => Some(d.clone()),
        (Some(b), Some(d)) => Some(Documentation {
            summary: d.summary.clone().or_else(|| b.summary.clone()),
            description: d.description.clone().or_else(|| b.description.clone()),
            equations: d.equations.clone().or_else(|| b.equations.clone()),
            assumptions: merge_option_vec(&b.assumptions, &d.assumptions),
            when_to_use: merge_option_vec(&b.when_to_use, &d.when_to_use),
            when_not_to_use: merge_option_vec(&b.when_not_to_use, &d.when_not_to_use),
            references: merge_option_vec(&b.references, &d.references),
        }),
    }
}

fn merge_editor(base: &Option<EditorInfo>, derived: &Option<EditorInfo>) -> Option<EditorInfo> {
    match (base, derived) {
        (None, None) => None,
        (Some(b), None) => Some(b.clone()),
        (None, Some(d)) => Some(d.clone()),
        (Some(b), Some(d)) => Some(EditorInfo {
            display: merge_display(&b.display, &d.display),
            layout: merge_option_hashmap(&b.layout, &d.layout),
            documentation: merge_documentation(&b.documentation, &d.documentation),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builtin_library() {
        let library = ModelLibrary::builtin();
        assert!(!library.is_empty());
        assert!(library.contains("pk/1cmt-iv"));
        assert!(library.contains("pk/1cmt-iv-ode"));
    }

    #[test]
    fn test_search() {
        let library = ModelLibrary::builtin();

        // Search by ID
        let results = library.search("1cmt");
        assert!(!results.is_empty());
    }

    #[test]
    fn test_resolve_simple() {
        let mut library = ModelLibrary::new();

        let base = JsonModel::from_str(
            r#"{
            "schema": "2.0",
            "id": "base-model",
            "type": "analytical",
            "analytical": "one_compartment",
            "parameters": ["ke", "V"],
            "compartments": ["central"],
            "outputs": [
                { "id": "cp", "equation": "central / V" }
            ],
            "editor": {
                "display": { "name": "Base" }
            }
        }"#,
        )
        .unwrap();
        library.add(base);

        let derived = JsonModel::from_str(
            r#"{
            "schema": "2.0",
            "id": "derived-model",
            "extends": "base-model",
            "type": "analytical",
            "analytical": "one_compartment",
            "parameters": ["ke", "V", "extra"],
            "compartments": ["central"]
        }"#,
        )
        .unwrap();

        let resolved = library.resolve(&derived).unwrap();
        assert_eq!(resolved.parameters.as_ref().unwrap().len(), 3);
        assert_eq!(resolved.outputs.as_ref().unwrap()[0].id, "cp");
        assert_eq!(
            resolved
                .display_info()
                .and_then(|display| display.name.as_deref()),
            Some("Base")
        );
    }

    #[test]
    fn test_resolve_executable_uses_inherited_outputs() {
        let mut library = ModelLibrary::new();

        let base = JsonModel::from_str(
            r#"{
            "schema": "2.0",
            "id": "base-model",
            "type": "ode",
            "parameters": ["ke", "V"],
            "compartments": ["central"],
            "diffeq": { "central": "-ke * central" },
            "outputs": [
                { "id": "cp", "equation": "central / V" }
            ]
        }"#,
        )
        .unwrap();
        library.add(base);

        let derived = JsonModel::from_str(
            r#"{
            "schema": "2.0",
            "id": "derived-model",
            "extends": "base-model",
            "type": "ode",
            "parameters": ["ke", "V", "scale"],
            "compartments": ["central"],
            "secondary": [
                { "id": "cp_scaled", "equation": "central / V * scale" }
            ]
        }"#,
        )
        .unwrap();

        let executable = library.resolve_executable(&derived).unwrap();
        assert_eq!(executable.outputs[0].id, "cp");
        assert_eq!(
            executable
                .calculations
                .iter()
                .map(|entry| entry.id.as_str())
                .collect::<Vec<_>>(),
            vec!["cp_scaled"]
        );
    }

    #[test]
    fn test_circular_inheritance() {
        let mut library = ModelLibrary::new();

        let model_a = JsonModel::from_str(
            r#"{
            "schema": "2.0",
            "id": "model-a",
            "extends": "model-b",
            "type": "analytical",
            "analytical": "one_compartment",
            "parameters": ["ke", "V"],
            "compartments": ["central"],
            "outputs": [{ "id": "cp", "equation": "central / V" }]
        }"#,
        )
        .unwrap();

        let model_b = JsonModel::from_str(
            r#"{
            "schema": "2.0",
            "id": "model-b",
            "extends": "model-a",
            "type": "analytical",
            "analytical": "one_compartment",
            "parameters": ["ke", "V"],
            "compartments": ["central"],
            "outputs": [{ "id": "cp", "equation": "central / V" }]
        }"#,
        )
        .unwrap();

        library.add(model_a.clone());
        library.add(model_b);

        // Should detect circular inheritance
        let result = library.resolve(&model_a);
        assert!(matches!(
            result,
            Err(JsonModelError::CircularInheritance(_))
        ));
    }

    #[test]
    fn test_merge_secondary_overrides_by_key() {
        let mut library = ModelLibrary::new();

        let base = JsonModel::from_str(
            r#"{
            "schema": "2.0",
            "id": "base",
            "type": "analytical",
            "analytical": "one_compartment",
            "parameters": ["CL", "V"],
            "compartments": ["central"],
            "secondary": [{ "id": "ke", "equation": "CL / V" }],
            "outputs": [{ "id": "cp", "equation": "central / V" }]
        }"#,
        )
        .unwrap();
        library.add(base);

        let derived = JsonModel::from_str(
            r#"{
            "schema": "2.0",
            "id": "derived",
            "extends": "base",
            "type": "analytical",
            "analytical": "one_compartment",
            "parameters": ["CL", "V"],
            "compartments": ["central"],
            "secondary": [{ "id": "ke", "equation": "CL / V * 0.9" }]
        }"#,
        )
        .unwrap();

        let resolved = library.resolve(&derived).unwrap();
        let secondary = resolved.secondary.as_ref().unwrap();

        // Should have exactly one "ke", not two
        assert_eq!(
            secondary.len(),
            1,
            "Derived 'ke' should override base 'ke', not duplicate"
        );
        assert_eq!(secondary[0].id, "ke");
        assert_eq!(secondary[0].equation, "CL / V * 0.9");
    }

    #[test]
    fn test_merge_secondary_appends_new_keys() {
        let mut library = ModelLibrary::new();

        let base = JsonModel::from_str(
            r#"{
            "schema": "2.0",
            "id": "base",
            "type": "analytical",
            "analytical": "one_compartment",
            "parameters": ["CL", "V"],
            "compartments": ["central"],
            "secondary": [{ "id": "ke", "equation": "CL / V" }],
            "outputs": [{ "id": "cp", "equation": "central / V" }]
        }"#,
        )
        .unwrap();
        library.add(base);

        let derived = JsonModel::from_str(
            r#"{
            "schema": "2.0",
            "id": "derived",
            "extends": "base",
            "type": "analytical",
            "analytical": "one_compartment",
            "parameters": ["CL", "V"],
            "compartments": ["central"],
            "secondary": [{ "id": "halflife", "equation": "0.693 / ke" }]
        }"#,
        )
        .unwrap();

        let resolved = library.resolve(&derived).unwrap();
        let secondary = resolved.secondary.as_ref().unwrap();

        // Should have both base "ke" and derived "halflife"
        assert_eq!(secondary.len(), 2);
        assert_eq!(secondary[0].id, "ke");
        assert_eq!(secondary[1].id, "halflife");
    }

    #[test]
    fn test_merge_covariates_deduplicates() {
        let mut library = ModelLibrary::new();

        let base = JsonModel::from_str(
            r#"{
            "schema": "2.0",
            "id": "base",
            "type": "analytical",
            "analytical": "one_compartment",
            "parameters": ["ke", "V"],
            "compartments": ["central"],
            "covariates": [{ "id": "wt" }, { "id": "age" }],
            "outputs": [{ "id": "cp", "equation": "central / V" }]
        }"#,
        )
        .unwrap();
        library.add(base);

        let derived = JsonModel::from_str(
            r#"{
            "schema": "2.0",
            "id": "derived",
            "extends": "base",
            "type": "analytical",
            "analytical": "one_compartment",
            "parameters": ["ke", "V"],
            "compartments": ["central"],
            "covariates": [{ "id": "wt", "column": "WT" }, { "id": "sex" }]
        }"#,
        )
        .unwrap();

        let resolved = library.resolve(&derived).unwrap();
        let covariates = resolved.covariates.as_ref().unwrap();

        // "wt" should appear only once
        assert_eq!(
            covariates.len(),
            3,
            "Should have wt, age, sex (wt deduplicated)"
        );
        assert!(covariates.iter().any(|cov| cov.id == "wt"));
        assert!(covariates.iter().any(|cov| cov.id == "age"));
        assert!(covariates.iter().any(|cov| cov.id == "sex"));
    }
}
