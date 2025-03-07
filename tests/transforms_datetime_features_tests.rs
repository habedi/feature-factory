use approx::assert_abs_diff_eq;
use arrow::array::{ArrayRef, Float64Array, Int32Array, TimestampNanosecondArray};
use arrow::datatypes::{DataType, Field, Schema, TimeUnit as ArrowTimeUnit};
use arrow::record_batch::RecordBatch;
use datafusion::datasource::MemTable;
use datafusion::prelude::*;
use feature_factory::transformers::datetime_features::{
    DatetimeFeatures, DatetimeSubtraction, TimeUnit,
};
use std::sync::Arc;

/// Helper function to extract an array's values as f64.
fn extract_as_f64(array: &ArrayRef) -> Vec<f64> {
    if let Some(arr) = array.as_any().downcast_ref::<Float64Array>() {
        (0..arr.len()).map(|i| arr.value(i)).collect()
    } else if let Some(arr) = array.as_any().downcast_ref::<Int32Array>() {
        (0..arr.len()).map(|i| arr.value(i) as f64).collect()
    } else {
        panic!("Array is not Float64Array or Int32Array");
    }
}

/// Create a DataFrame with one timestamp column "ts" for testing DatetimeFeatures.
async fn create_datetime_features_df() -> DataFrame {
    let schema = Arc::new(Schema::new(vec![Field::new(
        "ts",
        DataType::Timestamp(ArrowTimeUnit::Nanosecond, None),
        false,
    )]));

    // Three timestamps:
    // Row0: 2023-03-01T12:34:56Z
    // Row1: 2022-12-31T23:59:59Z
    // Row2: 2021-01-01T00:00:00Z
    let ts_values = vec![
        1677674096000000000,
        1672531199000000000,
        1609459200000000000,
    ];
    let ts_array = TimestampNanosecondArray::from(ts_values);
    let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(ts_array) as ArrayRef]).unwrap();
    let mem_table = MemTable::try_new(schema, vec![vec![batch]]).unwrap();

    let ctx = SessionContext::new();
    ctx.register_table("dt", Arc::new(mem_table)).unwrap();
    ctx.table("dt").await.unwrap()
}

/// Create a DataFrame with two timestamp columns "ts1" and "ts2" for testing DatetimeSubtraction.
async fn create_datetime_subtraction_df() -> DataFrame {
    let schema = Arc::new(Schema::new(vec![
        Field::new(
            "ts1",
            DataType::Timestamp(ArrowTimeUnit::Nanosecond, None),
            false,
        ),
        Field::new(
            "ts2",
            DataType::Timestamp(ArrowTimeUnit::Nanosecond, None),
            false,
        ),
    ]));

    // Two rows:
    // Row0: ts1 = 2023-03-01T12:34:56Z, ts2 = 2023-03-01T12:30:00Z
    // Row1: ts1 = 2023-03-01T00:00:00Z, ts2 = 2023-02-28T23:00:00Z
    let ts1_values = vec![1677674096000000000, 1677628800000000000];
    let ts2_values = vec![1677673800000000000, 1677625200000000000];
    let ts1_array = TimestampNanosecondArray::from(ts1_values);
    let ts2_array = TimestampNanosecondArray::from(ts2_values);
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(ts1_array) as ArrayRef,
            Arc::new(ts2_array) as ArrayRef,
        ],
    )
    .unwrap();
    let mem_table = MemTable::try_new(schema, vec![vec![batch]]).unwrap();

    let ctx = SessionContext::new();
    ctx.register_table("sub_dt", Arc::new(mem_table)).unwrap();
    ctx.table("sub_dt").await.unwrap()
}

/// ------------------ Normal Operation Tests ------------------

#[tokio::test]
async fn test_datetime_features_extraction() {
    let df = create_datetime_features_df().await;
    let mut transformer = DatetimeFeatures::new(vec!["ts".to_string()]);
    transformer.fit(&df).await.unwrap();
    let transformed_df = transformer.transform(df).unwrap();
    let batches = transformed_df.collect().await.unwrap();
    let batch = &batches[0];

    // Expected new columns (after original "ts"):
    // "ts_year", "ts_month", "ts_day", "ts_hour", "ts_minute", "ts_second", "ts_weekday"
    let ts_year = extract_as_f64(&batch.column(1));
    let ts_month = extract_as_f64(&batch.column(2));
    let ts_day = extract_as_f64(&batch.column(3));
    let ts_hour = extract_as_f64(&batch.column(4));
    let ts_minute = extract_as_f64(&batch.column(5));
    let ts_second = extract_as_f64(&batch.column(6));
    let ts_weekday = extract_as_f64(&batch.column(7));

    let expected_year = vec![2023.0, 2022.0, 2021.0];
    let expected_month = vec![3.0, 12.0, 1.0];
    let expected_day = vec![1.0, 31.0, 1.0];
    let expected_hour = vec![12.0, 23.0, 0.0];
    let expected_minute = vec![34.0, 59.0, 0.0];
    let expected_second = vec![56.0, 59.0, 0.0];
    let expected_weekday = vec![3.0, 6.0, 5.0];

    for i in 0..ts_year.len() {
        assert_abs_diff_eq!(ts_year[i], expected_year[i], epsilon = 1e-6);
        assert_abs_diff_eq!(ts_month[i], expected_month[i], epsilon = 1e-6);
        assert_abs_diff_eq!(ts_day[i], expected_day[i], epsilon = 1e-6);
        assert_abs_diff_eq!(ts_hour[i], expected_hour[i], epsilon = 1e-6);
        assert_abs_diff_eq!(ts_minute[i], expected_minute[i], epsilon = 1e-6);
        assert_abs_diff_eq!(ts_second[i], expected_second[i], epsilon = 1e-6);
        assert_abs_diff_eq!(ts_weekday[i], expected_weekday[i], epsilon = 1e-6);
    }
}

#[tokio::test]
async fn test_datetime_subtraction() {
    let df = create_datetime_subtraction_df().await;
    let mut transformer = DatetimeSubtraction::new(vec![(
        "diff_min".to_string(),
        "ts1".to_string(),
        "ts2".to_string(),
        TimeUnit::Minute,
    )]);
    transformer.fit(&df).await.unwrap();
    let transformed_df = transformer.transform(df).unwrap();
    let batches = transformed_df.collect().await.unwrap();
    let batch = &batches[0];

    // Expected new column "diff_min" is the difference in minutes.
    let diff_min = extract_as_f64(&batch.column(2));
    // Expected:
    // Row0: 2023-03-01T12:34:56Z - 2023-03-01T12:30:00Z = 296 sec → 296/60 ≈ 4.933333
    // Row1: 2023-03-01T00:00:00Z - 2023-02-28T23:00:00Z = 3600 sec → 3600/60 = 60.0
    let expected_diff = vec![296.0 / 60.0, 3600.0 / 60.0];

    for i in 0..diff_min.len() {
        assert_abs_diff_eq!(diff_min[i], expected_diff[i], epsilon = 1e-6);
    }
}

/// ------------------ Error and Edge Case Tests ------------------

#[tokio::test]
async fn test_datetime_features_missing_column() {
    let df = create_datetime_features_df().await;
    // Attempt to extract features from a non-existent column.
    let mut transformer = DatetimeFeatures::new(vec!["nonexistent".to_string()]);
    let result = transformer.fit(&df).await;
    assert!(
        result.is_err(),
        "Expected error for missing datetime column"
    );
}

#[tokio::test]
async fn test_datetime_features_invalid_type() {
    // Create a DataFrame with a column "ts" of type Float64 (not a datetime type).
    let schema = Arc::new(Schema::new(vec![Field::new(
        "ts",
        DataType::Float64,
        false,
    )]));
    let ts_array: ArrayRef = Arc::new(Float64Array::from(vec![1.0_f64, 2.0_f64, 3.0_f64]));
    let batch = RecordBatch::try_new(schema.clone(), vec![ts_array]).unwrap();
    let mem_table = MemTable::try_new(schema, vec![vec![batch]]).unwrap();
    let ctx = SessionContext::new();
    ctx.register_table("t", Arc::new(mem_table)).unwrap();
    let df = ctx.table("t").await.unwrap();

    let mut transformer = DatetimeFeatures::new(vec!["ts".to_string()]);
    let result = transformer.fit(&df).await;
    assert!(
        result.is_err(),
        "Expected error for non-datetime column in DatetimeFeatures"
    );
}

#[tokio::test]
async fn test_datetime_subtraction_missing_column() {
    let df = create_datetime_subtraction_df().await;
    // Attempt to subtract using a non-existent column.
    let mut transformer = DatetimeSubtraction::new(vec![(
        "diff".to_string(),
        "ts1".to_string(),
        "nonexistent".to_string(),
        TimeUnit::Hour,
    )]);
    let result = transformer.fit(&df).await;
    assert!(
        result.is_err(),
        "Expected error for missing column in DatetimeSubtraction"
    );
}

#[tokio::test]
async fn test_datetime_subtraction_invalid_type() {
    // Create a DataFrame where one column is not a datetime type.
    let schema = Arc::new(Schema::new(vec![
        Field::new("ts1", DataType::Float64, false),
        Field::new(
            "ts2",
            DataType::Timestamp(ArrowTimeUnit::Nanosecond, None),
            false,
        ),
    ]));
    let ts1: ArrayRef = Arc::new(Float64Array::from(vec![1.0_f64, 2.0_f64]));
    let ts2_values = vec![1677674096000000000, 1677674096000000000]; // valid timestamps
    let ts2: ArrayRef = Arc::new(TimestampNanosecondArray::from(ts2_values));
    let batch = RecordBatch::try_new(schema.clone(), vec![ts1, ts2]).unwrap();
    let mem_table = MemTable::try_new(schema, vec![vec![batch]]).unwrap();
    let ctx = SessionContext::new();
    ctx.register_table("t", Arc::new(mem_table)).unwrap();
    let df = ctx.table("t").await.unwrap();

    let mut transformer = DatetimeSubtraction::new(vec![(
        "diff".to_string(),
        "ts1".to_string(), // invalid type
        "ts2".to_string(),
        TimeUnit::Minute,
    )]);
    let result = transformer.fit(&df).await;
    assert!(
        result.is_err(),
        "Expected error for non-datetime column in DatetimeSubtraction"
    );
}
