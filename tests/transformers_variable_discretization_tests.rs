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

/// Helper: create a DataFrame with a single column "value" of type Float64.
/// The values are provided as a slice.
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

#[tokio::test]
async fn test_arbitrary_discretiser() -> FeatureFactoryResult<()> {
    // Create a DataFrame with 3 values.
    let df = create_df(&[2.0, 6.0, 11.0]).await;
    // User-defined intervals for column "value".
    // Intervals: [0,5) -> "low", [5,10) -> "medium", [10,15) -> "high".
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
    // The "value" column should now be a string column with labels.
    let value_col = batch
        .column(schema.index_of("value").unwrap())
        .as_any()
        .downcast_ref::<StringArray>()
        .expect("Expected StringArray for discretized column");
    // Our original values: 2.0 -> "low", 6.0 -> "medium", 11.0 -> "high".
    let expected = vec!["low", "medium", "high"];
    for (i, exp) in expected.into_iter().enumerate() {
        assert_eq!(value_col.value(i), exp, "Row {}: expected {}", i, exp);
    }
    Ok(())
}

#[tokio::test]
async fn test_equal_frequency_discretiser() -> FeatureFactoryResult<()> {
    // Create a DataFrame with 10 values.
    let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let df = create_df(&values).await;
    let mut discretiser = EqualFrequencyDiscretizer::new(vec!["value".to_string()], 3);
    discretiser.fit(&df).await?;
    let transformed = discretiser.transform(df)?;
    let batches = transformed.collect().await?;
    let batch = batches.first().expect("Expected at least one batch");
    let schema = batch.schema();
    // After discretization, "value" should be a StringArray with bin labels.
    let value_col = batch
        .column(schema.index_of("value").unwrap())
        .as_any()
        .downcast_ref::<StringArray>()
        .expect("Expected StringArray for equal-frequency discretised column");
    // Check that each row has a non-null label (we don't assert exact boundaries as they depend on quantile computation).
    for i in 0..value_col.len() {
        assert!(!value_col.is_null(i), "Row {} is null", i);
    }
    Ok(())
}

#[tokio::test]
async fn test_equal_width_discretiser() -> FeatureFactoryResult<()> {
    // Create a DataFrame with values ranging from 0 to 20.
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
    // Check that labels are not null.
    for i in 0..value_col.len() {
        assert!(!value_col.is_null(i), "Row {} is null", i);
    }
    Ok(())
}

#[tokio::test]
async fn test_geometric_width_discretiser() -> FeatureFactoryResult<()> {
    // Create a DataFrame with positive values.
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
    // Check that each row has a non-null label.
    for i in 0..value_col.len() {
        assert!(!value_col.is_null(i), "Row {} is null", i);
    }
    Ok(())
}
