pub mod core;
mod logging;
pub mod transformers;

// Re-export core components for a cleaner API
pub use core::errors::{FeatureFactoryError, FeatureFactoryResult};
pub use core::pipeline::Pipeline;
pub use core::traits::Transformer;
