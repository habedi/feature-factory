use arrow::array::{ArrayRef, Float64Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use datafusion::datasource::memory::MemTable;
use datafusion::prelude::*;
use std::sync::Arc;

use feature_factory::exceptions::FeatureFactoryResult;
use feature_factory::transformers::outliers::{ArbitraryOutlierCapper, OutlierTrimmer, Winsorizer};

/// Helper function to create a DataFrame with a single Float64 column "value" using the provided values.
async fn create_df(values: Vec<f64>) -> DataFrame {
    let schema = Arc::new(Schema::new(vec![Field::new(
        "value",
        DataType::Float64,
        false,
    )]));
    let array: ArrayRef = Arc::new(Float64Array::from(values));
    let batch = RecordBatch::try_new(schema.clone(), vec![array]).unwrap();
    let mem_table = MemTable::try_new(schema, vec![vec![batch]]).unwrap();
    let ctx = SessionContext::new();
    ctx.register_table("t", Arc::new(mem_table)).unwrap();
    ctx.table("t").await.unwrap()
}

#[tokio::test]
async fn test_arbitrary_outlier_capper() -> FeatureFactoryResult<()> {
    // Input values with outliers.
    let df = create_df(vec![1.0, 5.0, 10.0, 20.0]).await;
    // Define user caps: values below 2.0 should be capped at 2.0, above 15.0 capped at 15.0.
    let mut lower_caps = std::collections::HashMap::new();
    lower_caps.insert("value".to_string(), 2.0);
    let mut upper_caps = std::collections::HashMap::new();
    upper_caps.insert("value".to_string(), 15.0);
    let capper = ArbitraryOutlierCapper::new(vec!["value".to_string()], lower_caps, upper_caps);
    let transformed = capper.transform(df)?;
    let batches = transformed.collect().await?;
    let batch = batches.first().expect("Expected at least one batch");
    let schema = batch.schema();
    let col_idx = schema.index_of("value").unwrap();
    let array = batch
        .column(col_idx)
        .as_any()
        .downcast_ref::<Float64Array>()
        .expect("Expected Float64Array");
    // Expected: 1.0 -> 2.0, 5.0 unchanged, 10.0 unchanged, 20.0 -> 15.0.
    let expected = [2.0, 5.0, 10.0, 15.0];
    for i in 0..array.len() {
        assert!(
            (array.value(i) - expected[i]).abs() < 1e-6,
            "Row {}: expected {}, got {}",
            i,
            expected[i],
            array.value(i)
        );
    }
    Ok(())
}

#[tokio::test]
async fn test_winsorizer() -> FeatureFactoryResult<()> {
    // Create a DataFrame with values 1.0 .. 10.0
    let df = create_df((1..=10).map(|v| v as f64).collect()).await;
    // Use 0.2 and 0.8 percentiles.
    let mut winsorizer = Winsorizer::new(vec!["value".to_string()], 0.2, 0.8);
    winsorizer.fit(&df).await?;
    // Get computed thresholds for checking.
    let (lower, upper) = winsorizer
        .thresholds
        .get("value")
        .cloned()
        .expect("Thresholds not computed");
    let transformed = winsorizer.transform(df)?;
    let batches = transformed.collect().await?;
    let batch = batches.first().expect("Expected at least one batch");
    let schema = batch.schema();
    let array = batch
        .column(schema.index_of("value").unwrap())
        .as_any()
        .downcast_ref::<Float64Array>()
        .expect("Expected Float64Array");
    for i in 0..array.len() {
        let val = array.value(i);
        assert!(
            val >= lower - 1e-6,
            "Row {}: value {} is less than lower threshold {}",
            i,
            val,
            lower
        );
        assert!(
            val <= upper + 1e-6,
            "Row {}: value {} is greater than upper threshold {}",
            i,
            val,
            upper
        );
    }
    Ok(())
}

#[tokio::test]
async fn test_outlier_trimmer() -> FeatureFactoryResult<()> {
    // Create a DataFrame with values 1.0 .. 10.0
    let df = create_df((1..=10).map(|v| v as f64).collect()).await;
    // Set percentile thresholds to trim values outside the 0.3 and 0.7 percentiles.
    let mut trimmer = OutlierTrimmer::new(vec!["value".to_string()], 0.3, 0.7);
    trimmer.fit(&df).await?;
    let transformed = trimmer.transform(df)?;
    let batches = transformed.collect().await?;
    let batch = batches.first().expect("Expected at least one batch");
    let schema = batch.schema();
    let array = batch
        .column(schema.index_of("value").unwrap())
        .as_any()
        .downcast_ref::<Float64Array>()
        .expect("Expected Float64Array");
    // All remaining values should be between the computed thresholds.
    let (lower, upper) = trimmer
        .thresholds
        .get("value")
        .cloned()
        .expect("Thresholds not computed");
    for i in 0..array.len() {
        let val = array.value(i);
        assert!(
            val >= lower - 1e-6,
            "Row {}: value {} is less than lower threshold {}",
            i,
            val,
            lower
        );
        assert!(
            val <= upper + 1e-6,
            "Row {}: value {} is greater than upper threshold {}",
            i,
            val,
            upper
        );
    }
    // Check that some rows were trimmed (i.e. the number of rows is less than 10).
    assert!(array.len() < 10, "Expected some rows to be trimmed");
    Ok(())
}

//
// Additional tests for invalid percentile inputs
//

#[tokio::test]
async fn test_winsorizer_invalid_lower_percentile() -> FeatureFactoryResult<()> {
    let df = create_df((1..=10).map(|v| v as f64).collect()).await;
    // Create a Winsorizer with an invalid lower percentile (< 0).
    let mut winsorizer = Winsorizer::new(vec!["value".to_string()], -0.1, 0.8);
    let result = winsorizer.fit(&df).await;
    assert!(result.is_err(), "Expected error for lower_percentile < 0");
    if let Err(e) = result {
        let err_str = format!("{}", e);
        assert!(
            err_str.contains("lower_percentile"),
            "Expected error message about lower_percentile, got: {}",
            err_str
        );
    }
    Ok(())
}

#[tokio::test]
async fn test_winsorizer_invalid_upper_percentile() -> FeatureFactoryResult<()> {
    let df = create_df((1..=10).map(|v| v as f64).collect()).await;
    // Create a Winsorizer with an invalid upper percentile (> 1).
    let mut winsorizer = Winsorizer::new(vec!["value".to_string()], 0.2, 1.2);
    let result = winsorizer.fit(&df).await;
    assert!(result.is_err(), "Expected error for upper_percentile > 1");
    if let Err(e) = result {
        let err_str = format!("{}", e);
        assert!(
            err_str.contains("upper_percentile"),
            "Expected error message about upper_percentile, got: {}",
            err_str
        );
    }
    Ok(())
}

#[tokio::test]
async fn test_winsorizer_invalid_percentile_order() -> FeatureFactoryResult<()> {
    let df = create_df((1..=10).map(|v| v as f64).collect()).await;
    // Create a Winsorizer with lower_percentile >= upper_percentile.
    let mut winsorizer = Winsorizer::new(vec!["value".to_string()], 0.8, 0.2);
    let result = winsorizer.fit(&df).await;
    assert!(
        result.is_err(),
        "Expected error for lower_percentile >= upper_percentile"
    );
    if let Err(e) = result {
        let err_str = format!("{}", e);
        assert!(
            err_str.contains("lower_percentile"),
            "Expected error message about percentile order, got: {}",
            err_str
        );
    }
    Ok(())
}

#[tokio::test]
async fn test_outlier_trimmer_invalid_upper_percentile() -> FeatureFactoryResult<()> {
    let df = create_df((1..=10).map(|v| v as f64).collect()).await;
    // Create an OutlierTrimmer with an invalid upper percentile (> 1).
    let mut trimmer = OutlierTrimmer::new(vec!["value".to_string()], 0.3, 1.5);
    let result = trimmer.fit(&df).await;
    assert!(result.is_err(), "Expected error for upper_percentile > 1");
    if let Err(e) = result {
        let err_str = format!("{}", e);
        assert!(
            err_str.contains("upper_percentile"),
            "Expected error message about upper_percentile, got: {}",
            err_str
        );
    }
    Ok(())
}

#[tokio::test]
async fn test_outlier_trimmer_invalid_percentile_order() -> FeatureFactoryResult<()> {
    let df = create_df((1..=10).map(|v| v as f64).collect()).await;
    // Create an OutlierTrimmer with lower_percentile >= upper_percentile.
    let mut trimmer = OutlierTrimmer::new(vec!["value".to_string()], 0.7, 0.3);
    let result = trimmer.fit(&df).await;
    assert!(
        result.is_err(),
        "Expected error for lower_percentile >= upper_percentile"
    );
    if let Err(e) = result {
        let err_str = format!("{}", e);
        assert!(
            err_str.contains("lower_percentile"),
            "Expected error message about percentile order, got: {}",
            err_str
        );
    }
    Ok(())
}
