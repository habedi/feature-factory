//! ## Transformer Implementations
//!
//! This module includes submodules for different categories of transformers.
//!
//! ### Available Submodules
//!
//! - [`categorical`]: Transformers for encoding categorical variables.
//! - [`datetime`]: Transformers for extracting features from datetime columns.
//! - [`discretization`]: Transformers for converting continuous variables into discrete bins.
//! - [`feature_creation`]: Transformers for generating new features from existing data.
//! - [`feature_selection`]: Transformers for selecting relevant features based on statistical or heuristic methods.
//! - [`imputation`]: Transformers for handling missing values.
//! - [`numerical`]: Transformers for applying numerical transformations such as logarithmic or power transformations.
//! - [`outliers`]: Transformers for identifying and handling outliers.

pub mod categorical;
pub mod datetime;
pub mod discretization;
pub mod feature_creation;
pub mod feature_selection;
pub mod imputation;
pub mod numerical;
pub mod outliers;
