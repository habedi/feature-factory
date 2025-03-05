use std::sync::Arc;

use arrow::array::{Array, ArrayRef, Float64Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use datafusion::datasource::memory::MemTable;
use datafusion::prelude::*;

// Import your imputation modules and custom error/result types.
// Adjust the module path according to your project structure.
use feature_factory::exceptions::{FeatureFactoryError, FeatureFactoryResult};
use feature_factory::transformers::imputation::{
    AddMissingIndicator, ArbitraryNumberImputer, CategoricalImputer, DropMissingData,
    EndTailImputer, ImputeStrategy, MeanMedianImputer,
};

/// Creates an in-memory DataFrame with two columns:
///   - "a": Float64 with some missing values.
///   - "b": Utf8 with some missing values.
async fn create_dataframe() -> DataFrame {
    let schema = Arc::new(Schema::new(vec![
        Field::new("a", DataType::Float64, true),
        Field::new("b", DataType::Utf8, true),
    ]));

    let a_array: ArrayRef = Arc::new(Float64Array::from(vec![
        Some(1.0),
        Some(2.0),
        None,
        Some(4.0),
    ]));
    let b_array: ArrayRef = Arc::new(StringArray::from(vec![
        Some("x"),
        None,
        Some("x"),
        Some("y"),
    ]));

    let batch = RecordBatch::try_new(schema.clone(), vec![a_array, b_array]).unwrap();

    let mem_table = MemTable::try_new(schema, vec![vec![batch]]).unwrap();
    let ctx = SessionContext::new();
    ctx.register_table("t", Arc::new(mem_table)).unwrap();
    ctx.table("t").await.unwrap()
}

#[tokio::test]
async fn test_mean_imputation() -> FeatureFactoryResult<()> {
    // Create DataFrame.
    let df = create_dataframe().await;

    // Use DFMeanMedianImputer with Mean strategy on column "a".
    let mut imputer = MeanMedianImputer::new(vec!["a".to_string()], ImputeStrategy::Mean);
    imputer.fit(&df).await?;

    // Transform DataFrame and collect the results.
    let transformed = imputer.transform(df)?;
    let batches = transformed
        .collect()
        .await
        .map_err(FeatureFactoryError::from)?;
    // We'll work with the first record batch.
    let batch = batches.first().expect("Expected at least one batch");

    // Get column "a" after imputation.
    let a_array = batch
        .column(batch.schema().index_of("a").unwrap())
        .as_any()
        .downcast_ref::<Float64Array>()
        .expect("Expected Float64Array");

    // The original column "a" had values [1.0, 2.0, null, 4.0].
    // The computed mean should be (1.0 + 2.0 + 4.0) / 3 = 7/3 ≈ 2.3333333.
    // The imputed value should replace the null.
    let expected = vec![Some(1.0), Some(2.0), Some(7.0 / 3.0), Some(4.0)];
    for (i, exp) in expected.iter().enumerate() {
        let value = if a_array.is_null(i) {
            None
        } else {
            Some(a_array.value(i))
        };
        assert!(
            (value.unwrap_or(0.0) - exp.unwrap_or(0.0)).abs() < 1e-6,
            "row {}: expected {:?}, got {:?}",
            i,
            exp,
            value
        );
    }
    Ok(())
}

#[tokio::test]
async fn test_arbitrary_number_imputation() -> FeatureFactoryResult<()> {
    let df = create_dataframe().await;

    // Use DFArbitraryNumberImputer on column "a" with a chosen imputation value.
    let imputer = ArbitraryNumberImputer::new(vec!["a".to_string()], 99.0);
    let transformed = imputer.transform(df)?;
    let batches = transformed.collect().await?;
    let batch = batches.first().expect("Expected at least one batch");

    let a_array = batch
        .column(batch.schema().index_of("a").unwrap())
        .as_any()
        .downcast_ref::<Float64Array>()
        .expect("Expected Float64Array");

    // Expect that nulls are replaced by 99.0 while non-null values remain unchanged.
    let expected = vec![Some(1.0), Some(2.0), Some(99.0), Some(4.0)];
    for (i, exp) in expected.iter().enumerate() {
        let value = if a_array.is_null(i) {
            None
        } else {
            Some(a_array.value(i))
        };
        assert!(
            (value.unwrap_or(0.0) - exp.unwrap_or(0.0)).abs() < 1e-6,
            "row {}: expected {:?}, got {:?}",
            i,
            exp,
            value
        );
    }
    Ok(())
}

#[tokio::test]
async fn test_end_tail_imputation() -> FeatureFactoryResult<()> {
    let df = create_dataframe().await;

    // Use DFEndTailImputer on column "a" with a given percentile.
    let mut imputer = EndTailImputer::new(vec!["a".to_string()], 0.5);
    imputer.fit(&df).await?;
    let transformed = imputer.transform(df)?;
    let batches = transformed.collect().await?;
    let batch = batches.first().expect("Expected at least one batch");

    let a_array = batch
        .column(batch.schema().index_of("a").unwrap())
        .as_any()
        .downcast_ref::<Float64Array>()
        .expect("Expected Float64Array");

    // With our data [1.0, 2.0, 4.0] (ignoring null), the median is 2.0.
    // The computed value from approx_percentile_cont should be close to 2.0.
    let expected = vec![Some(1.0), Some(2.0), Some(2.0), Some(4.0)];
    for (i, exp) in expected.iter().enumerate() {
        let value = if a_array.is_null(i) {
            None
        } else {
            Some(a_array.value(i))
        };
        // We allow for a small floating point error.
        assert!(
            (value.unwrap_or(0.0) - exp.unwrap_or(0.0)).abs() < 1e-6,
            "row {}: expected {:?}, got {:?}",
            i,
            exp,
            value
        );
    }
    Ok(())
}

#[tokio::test]
async fn test_categorical_imputation() -> FeatureFactoryResult<()> {
    let df = create_dataframe().await;

    // Use DFCategoricalImputer on column "b". Do not supply a default so that it computes the mode.
    let mut imputer = CategoricalImputer::new(vec!["b".to_string()], None);
    imputer.fit(&df).await?;
    let transformed = imputer.transform(df)?;
    let batches = transformed.collect().await?;
    let batch = batches.first().expect("Expected at least one batch");

    let b_array = batch
        .column(batch.schema().index_of("b").unwrap())
        .as_any()
        .downcast_ref::<StringArray>()
        .expect("Expected StringArray");

    // For column "b", our values are ["x", null, "x", "y"]. The mode should be "x".
    let expected = vec![Some("x"), Some("x"), Some("x"), Some("y")];
    for (i, exp) in expected.iter().enumerate() {
        let value = if b_array.is_null(i) {
            None
        } else {
            Some(b_array.value(i))
        };
        assert_eq!(
            value, *exp,
            "row {}: expected {:?}, got {:?}",
            i, exp, value
        );
    }
    Ok(())
}

#[tokio::test]
async fn test_add_missing_indicator() -> FeatureFactoryResult<()> {
    let df = create_dataframe().await;

    // Use DFAddMissingIndicator on column "a" with a custom suffix.
    let imputer = AddMissingIndicator::new(vec!["a".to_string()], Some("_is_null".to_string()));
    let transformed = imputer.transform(df)?;
    let batches = transformed.collect().await?;
    let batch = batches.first().expect("Expected at least one batch");

    // The transformed DataFrame should now have an additional column "a_is_null".
    let schema = batch.schema();
    assert!(
        schema.field_with_name("a_is_null").is_ok(),
        "Missing indicator column not found"
    );

    // Check that the indicator column is computed correctly.
    let indicator_array = batch
        .column(schema.index_of("a_is_null").unwrap())
        .as_any()
        .downcast_ref::<arrow::array::BooleanArray>()
        .expect("Expected BooleanArray");
    // Original "a" was [1.0,2.0,null,4.0] so indicator should be [false, false, true, false]
    let expected = vec![false, false, true, false];
    for i in 0..expected.len() {
        assert_eq!(
            indicator_array.value(i),
            expected[i],
            "row {}: expected {}, got {}",
            i,
            expected[i],
            indicator_array.value(i)
        );
    }
    Ok(())
}

#[tokio::test]
async fn test_drop_missing_data() -> FeatureFactoryResult<()> {
    let df = create_dataframe().await;

    // DFDropMissingData should remove rows with any null values.
    let imputer = DropMissingData::new();
    let filtered = imputer.transform(df)?;
    let batches = filtered.collect().await?;
    let batch = batches.first().expect("Expected at least one batch");

    // Our original DataFrame had 4 rows:
    // Row 1: (1.0, "x") -> no missing.
    // Row 2: (2.0, null) -> missing.
    // Row 3: (null, "x") -> missing.
    // Row 4: (4.0, "y") -> no missing.
    // So we expect only rows 1 and 4.
    assert_eq!(
        batch.num_rows(),
        2,
        "Expected 2 rows after dropping missing data"
    );

    // Check that the remaining rows have no nulls.
    for col in batch.columns() {
        for i in 0..col.len() {
            assert!(!col.is_null(i), "Found a null value in filtered data");
        }
    }
    Ok(())
}
