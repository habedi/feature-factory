use std::sync::Arc;

use arrow::array::{Array, ArrayRef, Float64Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use datafusion::datasource::memory::MemTable;
use datafusion::prelude::*;

use feature_factory::exceptions::FeatureFactoryResult;
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
    let df = create_dataframe().await;
    let mut imputer = MeanMedianImputer::new(vec!["a".to_string()], ImputeStrategy::Mean);
    imputer.fit(&df).await?;
    let transformed = imputer.transform(df)?;
    let batches = transformed.collect().await?;
    let batch = batches.first().expect("Expected at least one batch");

    let a_array = batch
        .column(batch.schema().index_of("a").unwrap())
        .as_any()
        .downcast_ref::<Float64Array>()
        .expect("Expected Float64Array");

    // The original column "a" had values [1.0, 2.0, null, 4.0].
    // The computed mean should be (1.0 + 2.0 + 4.0) / 3 ≈ 2.3333333.
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
    let imputer = ArbitraryNumberImputer::new(vec!["a".to_string()], 99.0);
    let transformed = imputer.transform(df)?;
    let batches = transformed.collect().await?;
    let batch = batches.first().expect("Expected at least one batch");

    let a_array = batch
        .column(batch.schema().index_of("a").unwrap())
        .as_any()
        .downcast_ref::<Float64Array>()
        .expect("Expected Float64Array");

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
    let expected = vec![Some(1.0), Some(2.0), Some(2.0), Some(4.0)];
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
async fn test_categorical_imputation() -> FeatureFactoryResult<()> {
    let df = create_dataframe().await;
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
    let imputer = AddMissingIndicator::new(vec!["a".to_string()], Some("_is_null".to_string()));
    let transformed = imputer.transform(df)?;
    let batches = transformed.collect().await?;
    let batch = batches.first().expect("Expected at least one batch");

    let schema = batch.schema();
    assert!(
        schema.field_with_name("a_is_null").is_ok(),
        "Missing indicator column not found"
    );

    let indicator_array = batch
        .column(schema.index_of("a_is_null").unwrap())
        .as_any()
        .downcast_ref::<arrow::array::BooleanArray>()
        .expect("Expected BooleanArray");

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
    let imputer = DropMissingData::new();
    let filtered = imputer.transform(df)?;
    let batches = filtered.collect().await?;
    let batch = batches.first().expect("Expected at least one batch");

    // Original rows:
    // Row 1: (1.0, "x") -> no missing.
    // Row 2: (2.0, null) -> missing.
    // Row 3: (null, "x") -> missing.
    // Row 4: (4.0, "y") -> no missing.
    // Expect only rows 1 and 4.
    assert_eq!(
        batch.num_rows(),
        2,
        "Expected 2 rows after dropping missing data"
    );
    for col in batch.columns() {
        for i in 0..col.len() {
            assert!(!col.is_null(i), "Found a null value in filtered data");
        }
    }
    Ok(())
}

//
// Additional tests for input validation
//

#[tokio::test]
async fn test_mean_imputer_missing_column() -> FeatureFactoryResult<()> {
    let df = create_dataframe().await;
    // Use a non-existent column "c"
    let mut imputer = MeanMedianImputer::new(vec!["c".to_string()], ImputeStrategy::Mean);
    let result = imputer.fit(&df).await;
    assert!(
        result.is_err(),
        "Expected error when fitting on missing column"
    );
    if let Err(e) = result {
        let err_str = format!("{}", e);
        assert!(
            err_str.contains("Column 'c'"),
            "Expected error message about missing column, got: {}",
            err_str
        );
    }
    Ok(())
}

#[tokio::test]
async fn test_arbitrary_number_imputer_missing_column() -> FeatureFactoryResult<()> {
    let df = create_dataframe().await;
    let imputer = ArbitraryNumberImputer::new(vec!["c".to_string()], 42.0);
    let result = imputer.transform(df);
    assert!(
        result.is_err(),
        "Expected error when transforming with a missing column"
    );
    if let Err(e) = result {
        let err_str = format!("{}", e);
        assert!(
            err_str.contains("Column 'c'"),
            "Expected error message about missing column, got: {}",
            err_str
        );
    }
    Ok(())
}

#[tokio::test]
async fn test_arbitrary_number_imputer_invalid_number() -> FeatureFactoryResult<()> {
    let df = create_dataframe().await;
    // Use an invalid number (NaN)
    let mut imputer = ArbitraryNumberImputer::new(vec!["a".to_string()], f64::NAN);
    let result = imputer.fit(&df).await;
    assert!(
        result.is_err(),
        "Expected error when using an invalid (NaN) number"
    );
    if let Err(e) = result {
        let err_str = format!("{}", e);
        assert!(
            err_str.contains("Fixed number"),
            "Expected error message about fixed number, got: {}",
            err_str
        );
    }
    Ok(())
}

#[tokio::test]
async fn test_end_tail_imputer_invalid_percentile() -> FeatureFactoryResult<()> {
    let df = create_dataframe().await;
    // Use an invalid percentile (< 0)
    let mut imputer = EndTailImputer::new(vec!["a".to_string()], -0.1);
    let result = imputer.fit(&df).await;
    assert!(
        result.is_err(),
        "Expected error when using a percentile less than 0"
    );
    if let Err(e) = result {
        let err_str = format!("{}", e);
        assert!(
            err_str.contains("Percentile"),
            "Expected error message about percentile, got: {}",
            err_str
        );
    }

    // Use an invalid percentile (> 1)
    let mut imputer2 = EndTailImputer::new(vec!["a".to_string()], 1.1);
    let result2 = imputer2.fit(&df).await;
    assert!(
        result2.is_err(),
        "Expected error when using a percentile greater than 1"
    );
    if let Err(e) = result2 {
        let err_str = format!("{}", e);
        assert!(
            err_str.contains("Percentile"),
            "Expected error message about percentile, got: {}",
            err_str
        );
    }
    Ok(())
}

#[tokio::test]
async fn test_categorical_imputer_missing_column() -> FeatureFactoryResult<()> {
    let df = create_dataframe().await;
    let mut imputer = CategoricalImputer::new(vec!["c".to_string()], None);
    let result = imputer.fit(&df).await;
    assert!(
        result.is_err(),
        "Expected error when fitting categorical imputer on missing column"
    );
    if let Err(e) = result {
        let err_str = format!("{}", e);
        assert!(
            err_str.contains("Column 'c'"),
            "Expected error message about missing column, got: {}",
            err_str
        );
    }
    Ok(())
}

#[tokio::test]
async fn test_add_missing_indicator_missing_column() -> FeatureFactoryResult<()> {
    let df = create_dataframe().await;
    let imputer = AddMissingIndicator::new(vec!["c".to_string()], None);
    let result = imputer.transform(df);
    assert!(
        result.is_err(),
        "Expected error when adding missing indicator for a missing column"
    );
    if let Err(e) = result {
        let err_str = format!("{}", e);
        assert!(
            err_str.contains("Column 'c'"),
            "Expected error message about missing column, got: {}",
            err_str
        );
    }
    Ok(())
}

/// Helper function to create a DataFrame with two columns:
/// "a" (Float64) and "b" (Utf8). Rows with missing values are included.
async fn create_missing_df() -> DataFrame {
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
async fn test_drop_missing_data_default_all_columns() -> FeatureFactoryResult<()> {
    let df = create_missing_df().await;
    // Use default DropMissingData (checks all columns)
    let transformer = DropMissingData::new();
    let filtered = transformer.transform(df)?;
    let batches = filtered.collect().await?;
    let batch = batches.first().expect("Expected at least one batch");

    // Our test DF has 4 rows:
    // Row 0: a=1.0, b="x"       -> complete.
    // Row 1: a=2.0, b=null      -> missing in b.
    // Row 2: a=null, b="x"      -> missing in a.
    // Row 3: a=4.0, b="y"       -> complete.
    // Expect only rows 0 and 3.
    assert_eq!(
        batch.num_rows(),
        2,
        "Expected 2 rows after dropping missing values"
    );
    for col in batch.columns() {
        for i in 0..col.len() {
            assert!(
                !col.is_null(i),
                "Found a null value in filtered data at row {}",
                i
            );
        }
    }
    Ok(())
}

#[tokio::test]
async fn test_drop_missing_data_specific_columns() -> FeatureFactoryResult<()> {
    let df = create_missing_df().await;
    // Drop rows only if column "a" is missing.
    let transformer = DropMissingData::with_columns(vec!["a".to_string()]);
    let filtered = transformer.transform(df)?;
    let batches = filtered.collect().await?;
    let batch = batches.first().expect("Expected at least one batch");

    // Original DF rows:
    // Row 0: a=1.0, b="x"       -> kept (a present)
    // Row 1: a=2.0, b=null      -> kept (a present, even though b is missing)
    // Row 2: a=null, b="x"      -> dropped (a missing)
    // Row 3: a=4.0, b="y"       -> kept
    assert_eq!(
        batch.num_rows(),
        3,
        "Expected 3 rows after dropping rows with missing 'a'"
    );
    // Now check that all remaining rows have non-null "a".
    let schema = batch.schema();
    let a_array = batch
        .column(schema.index_of("a").unwrap())
        .as_any()
        .downcast_ref::<Float64Array>()
        .expect("Expected Float64Array for column 'a'");
    for i in 0..a_array.len() {
        assert!(
            !a_array.is_null(i),
            "Row {}: column 'a' should not be null",
            i
        );
    }
    Ok(())
}

#[tokio::test]
async fn test_drop_missing_data_missing_column() -> FeatureFactoryResult<()> {
    let df = create_missing_df().await;
    // Try to drop rows for a non-existent column.
    let transformer = DropMissingData::with_columns(vec!["nonexistent".to_string()]);
    let result = transformer.transform(df);
    assert!(
        result.is_err(),
        "Expected an error because column 'nonexistent' does not exist"
    );
    Ok(())
}
