//! File-based parsers and parser-facing row utilities.
//!
//! Use this module when your source data starts as files or parser-shaped rows.
//! It provides format-specific loaders such as [`read_pmetrics`] and keeps the
//! existing Pmetrics row API available to callers.
//!
//! Choose the entrypoint by source shape:
//! - Use [`DataRow`] or [`build_data`] when you already mapped external data into
//!   Pmetrics row fields yourself.
//! - Use [`read_pmetrics`] when the source file already follows the Pmetrics CSV
//!   convention.

pub mod pmetrics;

pub use pmetrics::*;
