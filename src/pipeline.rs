//! ## Feature Factory Pipeline
//!
//! This module provides core abstractions for building, fitting, and transforming data using
//! composable pipelines of transformers in the Feature Factory library.
//!
//! ### Overview
//!
//! - The [`crate::foundation::traits::Transformer`] trait defines a common interface for implementing data transformation steps,
//!   supporting both stateful (requiring fitting) and stateless transformations.
//! - The [`Pipeline`] struct enables chaining multiple transformers into a cohesive data transformation pipeline,
//!   supporting both fitting and transforming operations.
//! - Macros [`crate::impl_transformer`] and [`crate::make_pipeline`] simplify the creation and implementation
//!   of transformers and pipelines.

use crate::foundation::errors::{FeatureFactoryError, FeatureFactoryResult};
use crate::foundation::traits::Transformer;
use datafusion::prelude::*;
use std::time::Instant;

/// Macro to implement the [`crate::foundation::traits::Transformer`] trait for Feature Factory transformers.
///
/// The type must already have inherent methods:
/// - `async fn fit(&mut self, &DataFrame) -> FeatureFactoryResult<()>`
/// - `fn transform(&self, DataFrame) -> FeatureFactoryResult<DataFrame>`
/// - **`fn inherent_is_stateful(&self) -> bool`**
///
/// # Example
///
/// ```rust,no_run
/// use feature_factory::FeatureFactoryResult;
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
        impl $crate::foundation::traits::Transformer for $ty {
            async fn fit(
                &mut self,
                df: &datafusion::prelude::DataFrame,
            ) -> $crate::foundation::errors::FeatureFactoryResult<()> {
                <$ty>::fit(self, df).await
            }
            fn transform(
                &self,
                df: datafusion::prelude::DataFrame,
            ) -> $crate::foundation::errors::FeatureFactoryResult<datafusion::prelude::DataFrame>
            {
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
/// Each transformer's output (a new logical plan) is passed as input to the next transformer.
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

    /// Returns the number of steps in the pipeline.
    pub fn len(&self) -> usize {
        self.steps.len()
    }

    /// Returns true if the pipeline has no steps.
    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
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
            let steps: Vec<(String, Box<dyn $crate::foundation::traits::Transformer + Send + Sync>)> = vec![
                $(
                    ($name.to_string(), Box::new($transformer)),
                )+
            ];
            $crate::pipeline::Pipeline::new(steps, $verbose)
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::foundation::errors::FeatureFactoryError;

    struct DummyTransformer {
        fitted: bool,
    }

    #[async_trait::async_trait]
    impl Transformer for DummyTransformer {
        async fn fit(&mut self, _df: &DataFrame) -> FeatureFactoryResult<()> {
            self.fitted = true;
            Ok(())
        }

        fn transform(&self, df: DataFrame) -> FeatureFactoryResult<DataFrame> {
            if !self.fitted {
                return Err(FeatureFactoryError::FitNotCalled);
            }
            Ok(df)
        }

        fn is_stateful(&self) -> bool {
            true
        }
    }

    #[tokio::test]
    async fn test_pipeline_empty() {
        let ctx = SessionContext::new();
        let df = ctx.sql("SELECT 1 as a").await.unwrap();

        let mut pipeline = Pipeline::new(vec![], false);
        assert!(pipeline.is_empty());
        assert_eq!(pipeline.len(), 0);
        assert!(pipeline.fit(&df).await.is_err());
    }

    #[tokio::test]
    async fn test_pipeline_single_step() {
        let ctx = SessionContext::new();
        let df = ctx.sql("SELECT 1 as a").await.unwrap();

        let transformer = DummyTransformer { fitted: false };
        let mut pipeline = Pipeline::new(vec![("step1".to_string(), Box::new(transformer))], false);

        assert!(!pipeline.is_empty());
        assert_eq!(pipeline.len(), 1);

        let result = pipeline.fit(&df).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_pipeline_len_and_is_empty() {
        let pipeline = Pipeline::new(vec![], false);
        assert!(pipeline.is_empty());
        assert_eq!(pipeline.len(), 0);
    }
}
