use std::sync::Arc;

use arrow::array::{Array, ArrayRef, Float64Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use datafusion::datasource::memory::MemTable;
use datafusion::prelude::*;
use tokio;

use feature_factory::exceptions::FeatureFactoryResult;
use feature_factory::transformers::variable_discretization::{
    ArbitraryDiscretizer, EqualFrequencyDiscretizer, EqualWidthDiscretizer,
    GeometricWidthDiscretizer,
};

/// Helper function to create a DataFrame with a single column "value" of type Float64.
async fn create_df(values: &[f64]) -> DataFrame {
    let schema = Arc::new(Schema::new(vec![Field::new(
        "value",
        DataType::Float64,
        false,
    )]));
    let array: ArrayRef = Arc::new(Float64Array::from(values.to_vec()));
    let batch = RecordBatch::try_new(schema.clone(), vec![array]).unwrap();
    let mem_table = MemTable::try_new(schema, vec![vec![batch]]).unwrap();
    let ctx = SessionContext::new();
    ctx.register_table("t", Arc::new(mem_table)).unwrap();
    ctx.table("t").await.unwrap()
}

/// Helper function to create a DataFrame with a single column "other" of type Float64.
async fn create_df_other(values: &[f64]) -> DataFrame {
    let schema = Arc::new(Schema::new(vec![Field::new(
        "other",
        DataType::Float64,
        false,
    )]));
    let array: ArrayRef = Arc::new(Float64Array::from(values.to_vec()));
    let batch = RecordBatch::try_new(schema.clone(), vec![array]).unwrap();
    let mem_table = MemTable::try_new(schema, vec![vec![batch]]).unwrap();
    let ctx = SessionContext::new();
    ctx.register_table("t", Arc::new(mem_table)).unwrap();
    ctx.table("t").await.unwrap()
}

#[tokio::test]
async fn test_arbitrary_discretiser() -> FeatureFactoryResult<()> {
    let df = create_df(&[2.0, 6.0, 11.0]).await;
    let mut intervals = std::collections::HashMap::new();
    intervals.insert(
        "value".to_string(),
        vec![
            (0.0, 5.0, "low".to_string()),
            (5.0, 10.0, "medium".to_string()),
            (10.0, 15.0, "high".to_string()),
        ],
    );
    let discretiser = ArbitraryDiscretizer::new(vec!["value".to_string()], intervals);
    let transformed = discretiser.transform(df)?;
    let batches = transformed.collect().await?;
    let batch = batches.first().expect("Expected at least one batch");
    let schema = batch.schema();
    let value_col = batch
        .column(schema.index_of("value").unwrap())
        .as_any()
        .downcast_ref::<StringArray>()
        .expect("Expected StringArray for discretized column");
    let expected = vec!["low", "medium", "high"];
    for (i, exp) in expected.into_iter().enumerate() {
        assert_eq!(value_col.value(i), exp, "Row {}: expected {}", i, exp);
    }
    Ok(())
}

#[tokio::test]
async fn test_equal_frequency_discretiser() -> FeatureFactoryResult<()> {
    let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let df = create_df(&values).await;
    let mut discretiser = EqualFrequencyDiscretizer::new(vec!["value".to_string()], 3);
    discretiser.fit(&df).await?;
    let transformed = discretiser.transform(df)?;
    let batches = transformed.collect().await?;
    let batch = batches.first().expect("Expected at least one batch");
    let schema = batch.schema();
    let value_col = batch
        .column(schema.index_of("value").unwrap())
        .as_any()
        .downcast_ref::<StringArray>()
        .expect("Expected StringArray for equal-frequency discretised column");
    for i in 0..value_col.len() {
        assert!(!value_col.is_null(i), "Row {} is null", i);
    }
    Ok(())
}

#[tokio::test]
async fn test_equal_width_discretiser() -> FeatureFactoryResult<()> {
    let values = vec![0.0, 5.0, 10.0, 15.0, 20.0];
    let df = create_df(&values).await;
    let mut discretiser = EqualWidthDiscretizer::new(vec!["value".to_string()], 4);
    discretiser.fit(&df).await?;
    let transformed = discretiser.transform(df)?;
    let batches = transformed.collect().await?;
    let batch = batches.first().expect("Expected at least one batch");
    let schema = batch.schema();
    let value_col = batch
        .column(schema.index_of("value").unwrap())
        .as_any()
        .downcast_ref::<StringArray>()
        .expect("Expected StringArray for equal-width discretised column");
    for i in 0..value_col.len() {
        assert!(!value_col.is_null(i), "Row {} is null", i);
    }
    Ok(())
}

#[tokio::test]
async fn test_geometric_width_discretiser() -> FeatureFactoryResult<()> {
    let values = vec![1.0, 2.0, 4.0, 8.0, 16.0];
    let df = create_df(&values).await;
    let mut discretiser = GeometricWidthDiscretizer::new(vec!["value".to_string()], 3);
    discretiser.fit(&df).await?;
    let transformed = discretiser.transform(df)?;
    let batches = transformed.collect().await?;
    let batch = batches.first().expect("Expected at least one batch");
    let schema = batch.schema();
    let value_col = batch
        .column(schema.index_of("value").unwrap())
        .as_any()
        .downcast_ref::<StringArray>()
        .expect("Expected StringArray for geometric-width discretized column");
    for i in 0..value_col.len() {
        assert!(!value_col.is_null(i), "Row {} is null", i);
    }
    Ok(())
}

/// Additional tests for error conditions

#[tokio::test]
async fn test_arbitrary_discretiser_invalid_intervals() -> FeatureFactoryResult<()> {
    let df = create_df(&[1.0, 2.0, 3.0]).await;
    // Define invalid intervals: first interval has lower == upper.
    let mut intervals = std::collections::HashMap::new();
    intervals.insert(
        "value".to_string(),
        vec![
            (5.0, 5.0, "equal".to_string()),
            (5.0, 10.0, "medium".to_string()),
        ],
    );
    let mut discretiser = ArbitraryDiscretizer::new(vec!["value".to_string()], intervals);
    let result = discretiser.fit(&df).await;
    assert!(
        result.is_err(),
        "Expected error due to invalid intervals (lower >= upper)"
    );
    Ok(())
}

#[tokio::test]
async fn test_equal_frequency_discretiser_invalid_bins() -> FeatureFactoryResult<()> {
    let df = create_df(&[1.0, 2.0, 3.0]).await;
    let mut discretiser = EqualFrequencyDiscretizer::new(vec!["value".to_string()], 0);
    let result = discretiser.fit(&df).await;
    assert!(result.is_err(), "Expected error for bins < 1");
    Ok(())
}

#[tokio::test]
async fn test_equal_width_discretiser_constant_column() -> FeatureFactoryResult<()> {
    // Create a DataFrame where all values are constant.
    let df = create_df(&[5.0, 5.0, 5.0]).await;
    let mut discretiser = EqualWidthDiscretizer::new(vec!["value".to_string()], 3);
    let result = discretiser.fit(&df).await;
    assert!(
        result.is_err(),
        "Expected error for constant column in equal-width discretisation"
    );
    Ok(())
}

#[tokio::test]
async fn test_geometric_width_discretiser_non_positive() -> FeatureFactoryResult<()> {
    // Create a DataFrame with a non-positive value.
    let df = create_df(&[0.0, 1.0, 2.0]).await;
    let mut discretiser = GeometricWidthDiscretizer::new(vec!["value".to_string()], 3);
    let result = discretiser.fit(&df).await;
    assert!(
        result.is_err(),
        "Expected error due to non-positive value in geometric discretisation"
    );
    Ok(())
}

#[tokio::test]
async fn test_discretiser_missing_column() -> FeatureFactoryResult<()> {
    // Create a DataFrame with column "other" instead of "value".
    let df = create_df_other(&[1.0, 2.0, 3.0]).await;
    let mut discretiser = EqualWidthDiscretizer::new(vec!["value".to_string()], 3);
    let result = discretiser.fit(&df).await;
    assert!(
        result.is_err(),
        "Expected error because column 'value' is missing"
    );
    Ok(())
}
