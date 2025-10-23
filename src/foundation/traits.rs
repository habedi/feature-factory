//! ## Foundation Traits
//!
//! This module defines the core traits used throughout the Feature Factory library.

use crate::foundation::errors::FeatureFactoryResult;
use async_trait::async_trait;
use datafusion::prelude::DataFrame;

/// Trait for components used in the data transformation pipeline.
///
/// Every transformer must provide a `fit` method (which may collect data to compute parameters)
/// and a `transform` method (which updates the DataFrame's logical plan without triggering execution).
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::foundation::errors::FeatureFactoryError;

    struct DummyTransformer {
        fitted: bool,
    }

    #[async_trait]
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
    async fn test_transformer_trait() {
        use datafusion::prelude::*;
        let ctx = SessionContext::new();
        let df = ctx.sql("SELECT 1 as a").await.unwrap();

        let mut transformer = DummyTransformer { fitted: false };
        assert!(transformer.is_stateful());

        // Should fail before fit
        assert!(transformer.transform(df.clone()).is_err());

        // Fit the transformer
        transformer.fit(&df).await.unwrap();

        // Should succeed after fit
        assert!(transformer.transform(df).is_ok());
    }
}
