//! File-based parsers and parser-facing row utilities.
//!
//! Use this module when your source data starts as files or parser-shaped rows.
//! It re-exports the row ingestion API from [`crate::data::row`] and provides
//! format-specific loaders such as [`read_pmetrics`].
//!
//! Choose the entrypoint by source shape:
//! - Use [`DataRow`] or [`build_data`] when you already mapped external data into
//!   canonical row fields yourself.
//! - Use [`read_pmetrics`] when the source file already follows the Pmetrics CSV
//!   convention.

pub mod pmetrics;

pub use crate::data::row::{build_data, DataError, DataRow, DataRowBuilder};
pub use pmetrics::*;
