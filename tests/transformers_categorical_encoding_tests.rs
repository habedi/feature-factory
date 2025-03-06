use std::sync::Arc;

use arrow::array::{Array, ArrayRef, Float64Array, Int64Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use datafusion::datasource::memory::MemTable;
use datafusion::prelude::*;
use tokio;

use feature_factory::exceptions::FeatureFactoryResult;
use feature_factory::transformers::categorical_encoding::{
    CountFrequencyEncoder, MeanEncoder, OneHotEncoder, OrdinalEncoder, RareLabelEncoder, WoEEncoder,
};

/// Helper function to create a DataFrame for testing categorical encoders.
/// This DataFrame has a single categorical column "color" and (optionally) a target column "target".
async fn create_categorical_df(with_target: bool) -> DataFrame {
    // Create schema: "color" and optionally "target"
    let fields = if with_target {
        vec![
            Field::new("color", DataType::Utf8, true),
            Field::new("target", DataType::Float64, true),
        ]
    } else {
        vec![Field::new("color", DataType::Utf8, true)]
    };
    let schema = Arc::new(Schema::new(fields));

    // Data for "color": repeated values.
    let colors = vec![
        Some("red"),
        Some("blue"),
        Some("red"),
        Some("green"),
        Some("blue"),
        Some("red"),
    ];
    let color_array: ArrayRef = Arc::new(StringArray::from(colors));

    let batch = if with_target {
        // For MeanEncoder and WoEEncoder tests.
        // Let target be: for "red" assign 10, for "blue" assign 20, for "green" assign 30.
        let target_vals = vec![
            Some(10.0),
            Some(20.0),
            Some(10.0),
            Some(30.0),
            Some(20.0),
            Some(10.0),
        ];
        let target_array: ArrayRef = Arc::new(Float64Array::from(target_vals));
        RecordBatch::try_new(schema.clone(), vec![color_array, target_array]).unwrap()
    } else {
        RecordBatch::try_new(schema.clone(), vec![color_array]).unwrap()
    };

    let mem_table = MemTable::try_new(schema, vec![vec![batch]]).unwrap();
    let ctx = SessionContext::new();
    ctx.register_table("t", Arc::new(mem_table)).unwrap();
    ctx.table("t").await.unwrap()
}

#[tokio::test]
async fn test_one_hot_encoder() -> FeatureFactoryResult<()> {
    let df = create_categorical_df(false).await;
    let mut encoder = OneHotEncoder::new(vec!["color".to_string()]);
    encoder.fit(&df).await?;
    let transformed = encoder.transform(df)?;
    let batches = transformed.collect().await?;
    let batch = batches.first().expect("Expected at least one batch");

    // Original column "color" should exist, and new binary columns for each category.
    let schema = batch.schema();
    // Expected categories from test data: "red", "blue", "green"
    for cat in &["red", "blue", "green"] {
        let col_name = format!("color_{}", cat);
        assert!(
            schema.field_with_name(&col_name).is_ok(),
            "Missing one-hot column {}",
            col_name
        );
    }
    // Check values for one row (e.g. row 0 is "red")
    let red_col = batch
        .column(schema.index_of("color_red").unwrap())
        .as_any()
        .downcast_ref::<arrow::array::Int32Array>()
        .expect("Expected Int32Array for one-hot column");
    assert_eq!(red_col.value(0), 1, "Row 0 should be encoded as 1 for red");
    assert_eq!(red_col.value(1), 0, "Row 1 should be encoded as 0 for red");
    Ok(())
}

#[tokio::test]
async fn test_count_frequency_encoder() -> FeatureFactoryResult<()> {
    let df = create_categorical_df(false).await;
    let mut encoder = CountFrequencyEncoder::new(vec!["color".to_string()]);
    encoder.fit(&df).await?;
    let transformed = encoder.transform(df)?;
    let batches = transformed.collect().await?;
    let batch = batches.first().expect("Expected at least one batch");

    // Expected counts: red appears 3 times, blue 2 times, green 1 time.
    let schema = batch.schema();
    let color_array = batch
        .column(schema.index_of("color").unwrap())
        .as_any()
        .downcast_ref::<Int64Array>()
        .expect("Expected Int64Array after frequency encoding");
    let expected = vec![3, 2, 3, 1, 2, 3];
    for (i, exp) in expected.into_iter().enumerate() {
        assert_eq!(color_array.value(i), exp, "Row {}: expected {}", i, exp);
    }
    Ok(())
}

#[tokio::test]
async fn test_ordinal_encoder() -> FeatureFactoryResult<()> {
    let df = create_categorical_df(false).await;
    let mut encoder = OrdinalEncoder::new(vec!["color".to_string()]);
    encoder.fit(&df).await?;
    let transformed = encoder.transform(df)?;
    let batches = transformed.collect().await?;
    let batch = batches.first().expect("Expected at least one batch");

    // Distinct values sorted alphabetically: blue, green, red.
    // Therefore, blue -> 0, green -> 1, red -> 2.
    let schema = batch.schema();
    let color_array = batch
        .column(schema.index_of("color").unwrap())
        .as_any()
        .downcast_ref::<Int64Array>()
        .expect("Expected Int64Array for ordinal encoding");
    let expected = vec![2, 0, 2, 1, 0, 2];
    for (i, exp) in expected.into_iter().enumerate() {
        assert_eq!(color_array.value(i), exp, "Row {}: expected {}", i, exp);
    }
    Ok(())
}

#[tokio::test]
async fn test_mean_encoder() -> FeatureFactoryResult<()> {
    let df = create_categorical_df(true).await;
    let mut encoder = MeanEncoder::new(vec!["color".to_string()], "target".to_string());
    encoder.fit(&df).await?;
    let transformed = encoder.transform(df)?;
    let batches = transformed.collect().await?;
    let batch = batches.first().expect("Expected at least one batch");

    // For our test data:
    // color: red, blue, red, green, blue, red
    // target: 10, 20, 10, 30, 20, 10
    // Mean for red = (10+10+10)/3 = 10, blue = (20+20)/2 = 20, green = 30.
    let schema = batch.schema();
    let color_array = batch
        .column(schema.index_of("color").unwrap())
        .as_any()
        .downcast_ref::<Float64Array>()
        .expect("Expected Float64Array for mean encoding");
    let expected = vec![10.0, 20.0, 10.0, 30.0, 20.0, 10.0];
    for (i, exp) in expected.into_iter().enumerate() {
        let val = color_array.value(i);
        assert!(
            (val - exp).abs() < 1e-6,
            "Row {}: expected {}, got {}",
            i,
            exp,
            val
        );
    }
    Ok(())
}

#[tokio::test]
async fn test_woe_encoder() -> FeatureFactoryResult<()> {
    // Create a DataFrame with a binary target (Int64).
    let schema = Arc::new(Schema::new(vec![
        Field::new("color", DataType::Utf8, true),
        Field::new("target", DataType::Int64, true),
    ]));
    let colors = vec![
        Some("red"),
        Some("blue"),
        Some("red"),
        Some("green"),
        Some("blue"),
        Some("red"),
    ];
    let color_array: ArrayRef = Arc::new(StringArray::from(colors));
    let target_vals = vec![Some(1), Some(0), Some(1), Some(0), Some(0), Some(1)];
    let target_array: ArrayRef = Arc::new(Int64Array::from(target_vals));
    let batch = RecordBatch::try_new(schema.clone(), vec![color_array, target_array]).unwrap();
    let mem_table = MemTable::try_new(schema, vec![vec![batch]]).unwrap();
    let ctx = SessionContext::new();
    ctx.register_table("t", Arc::new(mem_table)).unwrap();
    let df = ctx.table("t").await.unwrap();

    let mut encoder = WoEEncoder::new(vec!["color".to_string()], "target".to_string());
    encoder.fit(&df).await?;
    let transformed = encoder.transform(df)?;
    let batches = transformed.collect().await?;
    let batch = batches.first().expect("Expected at least one batch");
    let schema = batch.schema();
    let color_array = batch
        .column(schema.index_of("color").unwrap())
        .as_any()
        .downcast_ref::<Float64Array>()
        .expect("Expected Float64Array for WoE encoding");
    for i in 0..color_array.len() {
        assert!(
            !color_array.is_null(i),
            "Row {}: WoE encoding should not be null",
            i
        );
    }
    Ok(())
}

#[tokio::test]
async fn test_rare_label_encoder() -> FeatureFactoryResult<()> {
    let df = create_categorical_df(false).await;
    // With test data, frequencies: red: 3/6, blue: 2/6, green: 1/6.
    // Using a threshold of 0.5 should mark blue and green as "rare", red remains unchanged.
    let mut encoder = RareLabelEncoder::new(vec!["color".to_string()], 0.5);
    encoder.fit(&df).await?;
    let transformed = encoder.transform(df)?;
    let batches = transformed.collect().await?;
    let batch = batches.first().expect("Expected at least one batch");
    let schema = batch.schema();
    let color_array = batch
        .column(schema.index_of("color").unwrap())
        .as_any()
        .downcast_ref::<StringArray>()
        .expect("Expected StringArray for rare label encoding");
    let expected = vec!["red", "rare", "red", "rare", "rare", "red"];
    for (i, exp) in expected.into_iter().enumerate() {
        let val = color_array.value(i);
        assert_eq!(val, exp, "Row {}: expected {}, got {}", i, exp, val);
    }
    Ok(())
}

/// Additional tests to check error conditions.

#[tokio::test]
async fn test_one_hot_encoder_missing_column() -> FeatureFactoryResult<()> {
    let df = create_categorical_df(false).await;
    let mut encoder = OneHotEncoder::new(vec!["nonexistent".to_string()]);
    let result = encoder.fit(&df).await;
    assert!(
        result.is_err(),
        "Expected error for missing column in OneHotEncoder"
    );
    Ok(())
}

#[tokio::test]
async fn test_count_frequency_encoder_missing_column() -> FeatureFactoryResult<()> {
    let df = create_categorical_df(false).await;
    let mut encoder = CountFrequencyEncoder::new(vec!["nonexistent".to_string()]);
    let result = encoder.fit(&df).await;
    assert!(
        result.is_err(),
        "Expected error for missing column in CountFrequencyEncoder"
    );
    Ok(())
}

#[tokio::test]
async fn test_ordinal_encoder_missing_column() -> FeatureFactoryResult<()> {
    let df = create_categorical_df(false).await;
    let mut encoder = OrdinalEncoder::new(vec!["nonexistent".to_string()]);
    let result = encoder.fit(&df).await;
    assert!(
        result.is_err(),
        "Expected error for missing column in OrdinalEncoder"
    );
    Ok(())
}

#[tokio::test]
async fn test_mean_encoder_invalid_target() -> FeatureFactoryResult<()> {
    // Create a DF with target as Utf8 (should be numeric).
    let schema = Arc::new(Schema::new(vec![
        Field::new("color", DataType::Utf8, true),
        Field::new("target", DataType::Utf8, true),
    ]));
    let colors = vec![Some("red"), Some("blue"), Some("red")];
    let color_array: ArrayRef = Arc::new(StringArray::from(colors));
    let targets = vec![Some("high"), Some("low"), Some("high")];
    let target_array: ArrayRef = Arc::new(StringArray::from(targets));
    let batch = RecordBatch::try_new(schema.clone(), vec![color_array, target_array]).unwrap();
    let mem_table = MemTable::try_new(schema, vec![vec![batch]]).unwrap();
    let ctx = SessionContext::new();
    ctx.register_table("t", Arc::new(mem_table)).unwrap();
    let df = ctx.table("t").await.unwrap();

    let mut encoder = MeanEncoder::new(vec!["color".to_string()], "target".to_string());
    let result = encoder.fit(&df).await;
    assert!(
        result.is_err(),
        "Expected error for invalid target type in MeanEncoder"
    );
    Ok(())
}

#[tokio::test]
async fn test_woe_encoder_invalid_target() -> FeatureFactoryResult<()> {
    // Create a DF with target as Utf8 (should be numeric).
    let schema = Arc::new(Schema::new(vec![
        Field::new("color", DataType::Utf8, true),
        Field::new("target", DataType::Utf8, true),
    ]));
    let colors = vec![Some("red"), Some("blue"), Some("red")];
    let color_array: ArrayRef = Arc::new(StringArray::from(colors));
    let targets = vec![Some("good"), Some("bad"), Some("good")];
    let target_array: ArrayRef = Arc::new(StringArray::from(targets));
    let batch = RecordBatch::try_new(schema.clone(), vec![color_array, target_array]).unwrap();
    let mem_table = MemTable::try_new(schema, vec![vec![batch]]).unwrap();
    let ctx = SessionContext::new();
    ctx.register_table("t", Arc::new(mem_table)).unwrap();
    let df = ctx.table("t").await.unwrap();

    let mut encoder = WoEEncoder::new(vec!["color".to_string()], "target".to_string());
    let result = encoder.fit(&df).await;
    assert!(
        result.is_err(),
        "Expected error for invalid target type in WoEEncoder"
    );
    Ok(())
}

#[tokio::test]
async fn test_rare_label_encoder_invalid_threshold() -> FeatureFactoryResult<()> {
    let df = create_categorical_df(false).await;
    let mut encoder = RareLabelEncoder::new(vec!["color".to_string()], -0.1);
    let result = encoder.fit(&df).await;
    assert!(result.is_err(), "Expected error for threshold less than 0");

    let mut encoder2 = RareLabelEncoder::new(vec!["color".to_string()], 1.5);
    let result2 = encoder2.fit(&df).await;
    assert!(
        result2.is_err(),
        "Expected error for threshold greater than 1"
    );
    Ok(())
}
