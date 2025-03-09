//! ## Feature Factory Pipeline
//!
//! This module provides core abstractions for building, fitting, and transforming data using
//! composable pipelines of transformers in the Feature Factory library.
//!
//! ### Overview
//!
//! - The [`Transformer`] trait defines a common interface for implementing data transformation steps,
//!   supporting both stateful (requiring fitting) and stateless transformations.
//! - The [`Pipeline`] struct enables chaining multiple transformers into a cohesive data transformation pipeline,
//!   supporting both fitting and transforming operations.
//! - Macros [`crate::impl_transformer`] and [`crate::make_pipeline`] simplify the creation and implementation
//!   of transformers and pipelines.

use crate::exceptions::{FeatureFactoryError, FeatureFactoryResult};
use async_trait::async_trait;
use datafusion::prelude::*;
use std::time::Instant;

/// Trait for components used in the data transformation pipeline.
///
/// Every transformer must provide a `fit` method (which may collect data to compute parameters)
/// and a `transform` method (which updates the DataFrame’s logical plan without triggering execution).
#[async_trait]
pub trait Transformer {
    /// Fit the transformer given a DataFrame.
    ///
    /// # Arguments
    ///
    /// * `df` - The input DataFrame.
    ///
    /// # Returns
    ///
    /// * `FeatureFactoryResult<()>` - Returns Ok if successful, or an error otherwise.
    async fn fit(&mut self, df: &DataFrame) -> FeatureFactoryResult<()>;

    /// Transform the input DataFrame, returning a new DataFrame with the transformation applied.
    ///
    /// # Arguments
    ///
    /// * `df` - The input DataFrame.
    ///
    /// # Returns
    ///
    /// * `FeatureFactoryResult<DataFrame>` - The transformed DataFrame or an error if transformation fails.
    fn transform(&self, df: DataFrame) -> FeatureFactoryResult<DataFrame>;

    /// Returns true if the transformer is stateful (i.e. requires a call to fit before transform can be called).
    fn is_stateful(&self) -> bool;
}

/// Macro to implement the [`Transformer`] trait for Feature Factory transformers.
///
/// The type must already have inherent methods:
/// - `async fn fit(&mut self, &DataFrame) -> FeatureFactoryResult<()>`
/// - `fn transform(&self, DataFrame) -> FeatureFactoryResult<DataFrame>`
/// - **`fn inherent_is_stateful(&self) -> bool`**
///
/// # Example
///
/// ```rust,no_run
/// use feature_factory::exceptions::FeatureFactoryResult;
/// use datafusion::prelude::DataFrame;
/// // Import the macro.
/// use feature_factory::impl_transformer;
///
/// // Suppose you have a transformer type `MyTransformer` defined elsewhere:
/// pub struct MyTransformer { /* ... */ }
///
/// impl MyTransformer {
///     pub async fn fit(&mut self, df: &DataFrame) -> FeatureFactoryResult<()> {
///         // Implementation here...
///         Ok(())
///     }
///
///     pub fn transform(&self, df: DataFrame) -> FeatureFactoryResult<DataFrame> {
///         // Implementation here...
///         Ok(df)
///     }
///
///     // Note the different name for the inherent method.
///     pub fn inherent_is_stateful(&self) -> bool {
///         true // or false
///     }
/// }
///
/// // Then simply invoke the macro to implement the Transformer trait:
/// impl_transformer!(MyTransformer);
/// ```
#[macro_export]
macro_rules! impl_transformer {
    ($ty:ty) => {
        #[async_trait::async_trait]
        impl $crate::pipeline::Transformer for $ty {
            async fn fit(
                &mut self,
                df: &datafusion::prelude::DataFrame,
            ) -> $crate::exceptions::FeatureFactoryResult<()> {
                <$ty>::fit(self, df).await
            }
            fn transform(
                &self,
                df: datafusion::prelude::DataFrame,
            ) -> $crate::exceptions::FeatureFactoryResult<datafusion::prelude::DataFrame> {
                <$ty>::transform(self, df)
            }
            fn is_stateful(&self) -> bool {
                <$ty>::inherent_is_stateful(self)
            }
        }
    };
}

/// A pipeline that chains a sequence of transformers.
///
/// Each transformer’s output (a new logical plan) is passed as input to the next transformer.
/// This design allows lazy chaining of transformations until a terminal action (like `collect`) is called.
pub struct Pipeline {
    steps: Vec<(String, Box<dyn Transformer + Send + Sync>)>,
    verbose: bool,
}

impl Pipeline {
    /// Creates a new pipeline.
    ///
    /// # Arguments
    ///
    /// * `steps` - A vector of (name, transformer) pairs (each transformer is already boxed).
    /// * `verbose` - If true, prints timing information.
    pub fn new(steps: Vec<(String, Box<dyn Transformer + Send + Sync>)>, verbose: bool) -> Self {
        Self { steps, verbose }
    }

    /// Fits each transformer (sequentially) and updates the logical plan.
    pub async fn fit(&mut self, df: &DataFrame) -> FeatureFactoryResult<DataFrame> {
        if self.steps.is_empty() {
            return Err(FeatureFactoryError::InvalidParameter(
                "Pipeline must have at least one transformer.".to_string(),
            ));
        }
        let mut current_df = df.clone();
        for (name, step) in self.steps.iter_mut() {
            if self.verbose {
                println!("Fitting step: {}", name);
            }
            let start = Instant::now();
            step.fit(&current_df).await.map_err(|e| {
                FeatureFactoryError::InvalidParameter(format!(
                    "Error fitting transformer '{}': {:?}",
                    name, e
                ))
            })?;
            current_df = step.transform(current_df).map_err(|e| {
                FeatureFactoryError::InvalidParameter(format!(
                    "Error transforming in '{}': {:?}",
                    name, e
                ))
            })?;
            if self.verbose {
                println!("Step '{}' completed in {:?}", name, start.elapsed());
            }
        }
        Ok(current_df)
    }

    /// Applies the `transform` method of each transformer (without fitting).
    pub fn transform(&self, df: DataFrame) -> FeatureFactoryResult<DataFrame> {
        if self.steps.is_empty() {
            return Err(FeatureFactoryError::InvalidParameter(
                "Pipeline must have at least one transformer.".to_string(),
            ));
        }
        let mut current_df = df;
        for (name, step) in self.steps.iter() {
            if self.verbose {
                println!("Applying transformer: {}", name);
            }
            current_df = step.transform(current_df).map_err(|e| {
                FeatureFactoryError::InvalidParameter(format!(
                    "Error in transformer '{}': {:?}",
                    name, e
                ))
            })?;
        }
        Ok(current_df)
    }

    /// Convenience method to call `fit` and then return the final transformed DataFrame.
    pub async fn fit_transform(&mut self, df: &DataFrame) -> FeatureFactoryResult<DataFrame> {
        self.fit(df).await
    }
}

/// Macro to simplify pipeline creation by automatically boxing transformers.
///
/// # Example
///
/// ```rust,no_run
/// use feature_factory::make_pipeline;
/// use feature_factory::transformers::imputation::DropMissingData;
///
/// // Create a pipeline with a single step.
/// let pipeline = make_pipeline!(false,
///     ("step1", DropMissingData::new()),
/// );
/// ```
#[macro_export]
macro_rules! make_pipeline {
    ($verbose:expr, $(($name:expr, $transformer:expr)),+ $(,)?) => {
        {
            let steps: Vec<(String, Box<dyn $crate::pipeline::Transformer + Send + Sync>)> = vec![
                $(
                    ($name.to_string(), Box::new($transformer)),
                )+
            ];
            $crate::pipeline::Pipeline::new(steps, $verbose)
        }
    };
}
