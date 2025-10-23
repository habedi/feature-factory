//! ## Foundation Module
//!
//! This module contains the foundational components of the Feature Factory library.
//! All other high-level modules should depend only on this foundation module to maintain
//! proper architectural separation.
//!
//! ### Submodules
//!
//! - [`errors`]: Error types and result aliases used throughout the library.
//! - [`traits`]: Core trait definitions for transformers.
//! - [`types`]: Common type definitions and utilities.

pub mod errors;
pub mod traits;
pub mod types;
