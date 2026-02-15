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
use crate::json::types::{DisplayInfo, Documentation, ModelType};
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
                if let Some(ref display) = model.display {
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
                if let Some(ref display) = model.display {
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
        aliases: merge_option_vec(&base.aliases, &derived.aliases),

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
        secondary: derived.secondary.clone().or_else(|| base.secondary.clone()),

        // ─────────────────────────────────────────────────────────────────────
        // Output
        // ─────────────────────────────────────────────────────────────────────
        output: derived.output.clone().or_else(|| base.output.clone()),
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
        derived: merge_option_vec(&base.derived, &derived.derived),
        features: merge_option_vec(&base.features, &derived.features),
        covariates: merge_option_vec(&base.covariates, &derived.covariates),
        covariate_effects: merge_option_vec(&base.covariate_effects, &derived.covariate_effects),

        // ─────────────────────────────────────────────────────────────────────
        // Layer 4: UI Metadata
        // ─────────────────────────────────────────────────────────────────────
        display: merge_display(&base.display, &derived.display),
        layout: merge_option_hashmap(&base.layout, &derived.layout),
        documentation: merge_documentation(&base.documentation, &derived.documentation),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builtin_library() {
        let library = ModelLibrary::builtin();
        assert!(!library.is_empty());

        // Should have analytical models
        let analytical = library.filter_by_type(ModelType::Analytical);
        assert!(!analytical.is_empty());
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

        // Add a base model
        let base = JsonModel::from_str(
            r#"{
            "schema": "1.0",
            "id": "base-model",
            "type": "analytical",
            "analytical": "one_compartment",
            "parameters": ["ke", "V"],
            "output": "x[0] / V"
        }"#,
        )
        .unwrap();
        library.add(base);

        // Add a derived model
        let derived = JsonModel::from_str(
            r#"{
            "schema": "1.0",
            "id": "derived-model",
            "extends": "base-model",
            "type": "analytical",
            "analytical": "one_compartment",
            "parameters": ["ke", "V", "extra"]
        }"#,
        )
        .unwrap();

        // Resolve should merge
        let resolved = library.resolve(&derived).unwrap();
        assert_eq!(resolved.parameters.as_ref().unwrap().len(), 3);
        assert!(resolved.output.is_some()); // Inherited from base
    }

    #[test]
    fn test_circular_inheritance() {
        let mut library = ModelLibrary::new();

        let model_a = JsonModel::from_str(
            r#"{
            "schema": "1.0",
            "id": "model-a",
            "extends": "model-b",
            "type": "analytical",
            "analytical": "one_compartment",
            "parameters": ["ke", "V"]
        }"#,
        )
        .unwrap();

        let model_b = JsonModel::from_str(
            r#"{
            "schema": "1.0",
            "id": "model-b",
            "extends": "model-a",
            "type": "analytical",
            "analytical": "one_compartment",
            "parameters": ["ke", "V"]
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
}
