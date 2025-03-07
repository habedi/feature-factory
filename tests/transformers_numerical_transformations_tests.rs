use approx::assert_abs_diff_eq;
use arrow::array::{ArrayRef, Float64Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use datafusion::datasource::MemTable;
use datafusion::prelude::*;
use feature_factory::transformers::numerical_transformations::{
    ArcsinTransformer, BoxCoxTransformer, LogCpTransformer, LogTransformer, PowerTransformer,
    ReciprocalTransformer, YeoJohnsonTransformer,
};
use std::sync::Arc;

/// Create a test DataFrame using a MemTable. The table "t" has three columns:
/// "a", "b", and "c" with simple floating‑point values.
async fn create_test_dataframe() -> DataFrame {
    let schema = Arc::new(Schema::new(vec![
        Field::new("a", DataType::Float64, false),
        Field::new("b", DataType::Float64, false),
        Field::new("c", DataType::Float64, false),
    ]));
    let a: ArrayRef = Arc::new(Float64Array::from(vec![1.0_f64, 2.0_f64, 10.0_f64]));
    let b: ArrayRef = Arc::new(Float64Array::from(vec![0.5_f64, 1.5_f64, 2.5_f64]));
    let c: ArrayRef = Arc::new(Float64Array::from(vec![2.0_f64, 3.0_f64, 4.0_f64]));
    let batch = RecordBatch::try_new(schema.clone(), vec![a, b, c]).unwrap();

    let mem_table = MemTable::try_new(schema, vec![vec![batch]]).unwrap();
    let ctx = SessionContext::new();
    ctx.register_table("t", Arc::new(mem_table)).unwrap();
    ctx.table("t").await.unwrap()
}

#[tokio::test]
async fn test_log_transformer() {
    let df = create_test_dataframe().await;
    let mut transformer = LogTransformer::new(vec!["a".to_string()]);
    transformer.fit(&df).await.unwrap();
    let transformed_df = transformer.transform(df).unwrap();
    let batches = transformed_df.collect().await.unwrap();
    let batch = &batches[0];
    let a_array = batch
        .column(0)
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap();

    let expected = vec![1.0_f64.ln(), 2.0_f64.ln(), 10.0_f64.ln()];
    for i in 0..a_array.len() {
        assert_abs_diff_eq!(a_array.value(i), expected[i], epsilon = 1e-6);
    }
}

#[tokio::test]
async fn test_log_cp_transformer() {
    let df = create_test_dataframe().await;
    let mut transformer = LogCpTransformer::new(vec!["b".to_string()], 1.0_f64);
    transformer.fit(&df).await.unwrap();
    let transformed_df = transformer.transform(df).unwrap();
    let batches = transformed_df.collect().await.unwrap();
    let batch = &batches[0];
    let b_array = batch
        .column(1)
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap();

    let expected = vec![
        (0.5_f64 + 1.0_f64).ln(),
        (1.5_f64 + 1.0_f64).ln(),
        (2.5_f64 + 1.0_f64).ln(),
    ];
    for i in 0..b_array.len() {
        assert_abs_diff_eq!(b_array.value(i), expected[i], epsilon = 1e-6);
    }
}

#[tokio::test]
async fn test_reciprocal_transformer() {
    let df = create_test_dataframe().await;
    let mut transformer = ReciprocalTransformer::new(vec!["c".to_string()]);
    transformer.fit(&df).await.unwrap();
    let transformed_df = transformer.transform(df).unwrap();
    let batches = transformed_df.collect().await.unwrap();
    let batch = &batches[0];
    let c_array = batch
        .column(2)
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap();

    let expected = vec![1.0_f64 / 2.0_f64, 1.0_f64 / 3.0_f64, 1.0_f64 / 4.0_f64];
    for i in 0..c_array.len() {
        assert_abs_diff_eq!(c_array.value(i), expected[i], epsilon = 1e-6);
    }
}

#[tokio::test]
async fn test_power_transformer() {
    let df = create_test_dataframe().await;
    let mut transformer = PowerTransformer::new(vec!["a".to_string()], 2.0_f64);
    transformer.fit(&df).await.unwrap();
    let transformed_df = transformer.transform(df).unwrap();
    let batches = transformed_df.collect().await.unwrap();
    let batch = &batches[0];
    let a_array = batch
        .column(0)
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap();

    let expected = vec![1.0_f64.powf(2.0), 2.0_f64.powf(2.0), 10.0_f64.powf(2.0)];
    for i in 0..a_array.len() {
        assert_abs_diff_eq!(a_array.value(i), expected[i], epsilon = 1e-6);
    }
}

#[tokio::test]
async fn test_boxcox_transformer_lambda_nonzero() {
    let df = create_test_dataframe().await;
    let mut transformer = BoxCoxTransformer::new(vec!["a".to_string()], 2.0_f64);
    transformer.fit(&df).await.unwrap();
    let transformed_df = transformer.transform(df).unwrap();
    let batches = transformed_df.collect().await.unwrap();
    let batch = &batches[0];
    let a_array = batch
        .column(0)
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap();

    let expected = vec![
        (1.0_f64.powf(2.0) - 1.0) / 2.0,
        (2.0_f64.powf(2.0) - 1.0) / 2.0,
        (10.0_f64.powf(2.0) - 1.0) / 2.0,
    ];
    for i in 0..a_array.len() {
        assert_abs_diff_eq!(a_array.value(i), expected[i], epsilon = 1e-6);
    }
}

#[tokio::test]
async fn test_boxcox_transformer_lambda_zero() {
    let df = create_test_dataframe().await;
    let mut transformer = BoxCoxTransformer::new(vec!["a".to_string()], 0.0_f64);
    transformer.fit(&df).await.unwrap();
    let transformed_df = transformer.transform(df).unwrap();
    let batches = transformed_df.collect().await.unwrap();
    let batch = &batches[0];
    let a_array = batch
        .column(0)
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap();

    let expected = vec![1.0_f64.ln(), 2.0_f64.ln(), 10.0_f64.ln()];
    for i in 0..a_array.len() {
        assert_abs_diff_eq!(a_array.value(i), expected[i], epsilon = 1e-6);
    }
}

#[tokio::test]
async fn test_yeo_johnson_transformer() {
    // Create a DataFrame with one column "d" having both negative and non-negative values.
    let schema = Arc::new(Schema::new(vec![Field::new("d", DataType::Float64, false)]));
    let d: ArrayRef = Arc::new(Float64Array::from(vec![
        -1.0_f64, 0.0_f64, 1.0_f64, 2.0_f64,
    ]));
    let batch = RecordBatch::try_new(schema.clone(), vec![d]).unwrap();
    let mem_table = MemTable::try_new(schema, vec![vec![batch]]).unwrap();
    let ctx = SessionContext::new();
    ctx.register_table("u", Arc::new(mem_table)).unwrap();
    let df = ctx.table("u").await.unwrap();

    // For lambda=1, the Yeo–Johnson transformation should be near the identity.
    let mut transformer = YeoJohnsonTransformer::new(vec!["d".to_string()], 1.0_f64);
    transformer.fit(&df).await.unwrap();
    let transformed_df = transformer.transform(df).unwrap();
    let batches = transformed_df.collect().await.unwrap();
    let batch = &batches[0];
    let d_array = batch
        .column(0)
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap();

    let expected = vec![-1.0_f64, 0.0_f64, 1.0_f64, 2.0_f64];
    for i in 0..d_array.len() {
        assert_abs_diff_eq!(d_array.value(i), expected[i], epsilon = 1e-6);
    }
}

#[tokio::test]
async fn test_arcsin_transformer() {
    // Create a DataFrame with one column "x" holding values in [0, 1].
    let schema = Arc::new(Schema::new(vec![Field::new("x", DataType::Float64, false)]));
    let x: ArrayRef = Arc::new(Float64Array::from(vec![
        0.0_f64, 0.25_f64, 0.5_f64, 1.0_f64,
    ]));
    let batch = RecordBatch::try_new(schema.clone(), vec![x]).unwrap();
    let mem_table = MemTable::try_new(schema, vec![vec![batch]]).unwrap();
    let ctx = SessionContext::new();
    ctx.register_table("p", Arc::new(mem_table)).unwrap();
    let df = ctx.table("p").await.unwrap();

    let mut transformer = ArcsinTransformer::new(vec!["x".to_string()]);
    transformer.fit(&df).await.unwrap();
    let transformed_df = transformer.transform(df).unwrap();
    let batches = transformed_df.collect().await.unwrap();
    let batch = &batches[0];
    let x_array = batch
        .column(0)
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap();

    for i in 0..x_array.len() {
        let original_val: f64 = match i {
            0 => 0.0_f64,
            1 => 0.25_f64,
            2 => 0.5_f64,
            3 => 1.0_f64,
            _ => unreachable!(),
        };
        let expected = (original_val.sqrt()).asin();
        assert_abs_diff_eq!(x_array.value(i), expected, epsilon = 1e-6);
    }
}

/// -------- Error Condition Tests --------

#[tokio::test]
async fn test_log_transformer_invalid() {
    // Create a DataFrame where column "a" contains a negative value.
    let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Float64, false)]));
    let a: ArrayRef = Arc::new(Float64Array::from(vec![1.0_f64, -2.0_f64, 10.0_f64]));
    let batch = RecordBatch::try_new(schema.clone(), vec![a]).unwrap();
    let mem_table = MemTable::try_new(schema, vec![vec![batch]]).unwrap();
    let ctx = SessionContext::new();
    ctx.register_table("t", Arc::new(mem_table)).unwrap();
    let df = ctx.table("t").await.unwrap();

    let mut transformer = LogTransformer::new(vec!["a".to_string()]);
    let result = transformer.fit(&df).await;
    assert!(
        result.is_err(),
        "Expected error for LogTransformer with negative values"
    );
}

#[tokio::test]
async fn test_log_cp_transformer_invalid() {
    // Create a DataFrame where column "b" has a minimum such that (min + constant) <= 0.
    let schema = Arc::new(Schema::new(vec![Field::new("b", DataType::Float64, false)]));
    let b: ArrayRef = Arc::new(Float64Array::from(vec![-2.0_f64, -1.0_f64, 0.0_f64]));
    let batch = RecordBatch::try_new(schema.clone(), vec![b]).unwrap();
    let mem_table = MemTable::try_new(schema, vec![vec![batch]]).unwrap();
    let ctx = SessionContext::new();
    ctx.register_table("t", Arc::new(mem_table)).unwrap();
    let df = ctx.table("t").await.unwrap();

    let mut transformer = LogCpTransformer::new(vec!["b".to_string()], 1.0_f64);
    let result = transformer.fit(&df).await;
    assert!(
        result.is_err(),
        "Expected error for LogCpTransformer when (min + constant) <= 0"
    );
}

#[tokio::test]
async fn test_reciprocal_transformer_invalid() {
    // Create a DataFrame where column "c" contains a zero.
    let schema = Arc::new(Schema::new(vec![Field::new("c", DataType::Float64, false)]));
    let c: ArrayRef = Arc::new(Float64Array::from(vec![1.0_f64, 0.0_f64, 2.0_f64]));
    let batch = RecordBatch::try_new(schema.clone(), vec![c]).unwrap();
    let mem_table = MemTable::try_new(schema, vec![vec![batch]]).unwrap();
    let ctx = SessionContext::new();
    ctx.register_table("t", Arc::new(mem_table)).unwrap();
    let df = ctx.table("t").await.unwrap();

    let mut transformer = ReciprocalTransformer::new(vec!["c".to_string()]);
    let result = transformer.fit(&df).await;
    assert!(
        result.is_err(),
        "Expected error for ReciprocalTransformer with zero value"
    );
}

#[tokio::test]
async fn test_boxcox_transformer_invalid() {
    // Create a DataFrame where column "a" contains a non-positive value.
    let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Float64, false)]));
    let a: ArrayRef = Arc::new(Float64Array::from(vec![1.0_f64, 0.0_f64, 10.0_f64]));
    let batch = RecordBatch::try_new(schema.clone(), vec![a]).unwrap();
    let mem_table = MemTable::try_new(schema, vec![vec![batch]]).unwrap();
    let ctx = SessionContext::new();
    ctx.register_table("t", Arc::new(mem_table)).unwrap();
    let df = ctx.table("t").await.unwrap();

    let mut transformer = BoxCoxTransformer::new(vec!["a".to_string()], 2.0_f64);
    let result = transformer.fit(&df).await;
    assert!(
        result.is_err(),
        "Expected error for BoxCoxTransformer with non-positive values"
    );
}

#[tokio::test]
async fn test_arcsin_transformer_invalid() {
    // Create a DataFrame where column "x" has a value outside [0,1].
    let schema = Arc::new(Schema::new(vec![Field::new("x", DataType::Float64, false)]));
    let x: ArrayRef = Arc::new(Float64Array::from(vec![-0.1_f64, 0.5_f64, 1.2_f64]));
    let batch = RecordBatch::try_new(schema.clone(), vec![x]).unwrap();
    let mem_table = MemTable::try_new(schema, vec![vec![batch]]).unwrap();
    let ctx = SessionContext::new();
    ctx.register_table("p", Arc::new(mem_table)).unwrap();
    let df = ctx.table("p").await.unwrap();

    let mut transformer = ArcsinTransformer::new(vec!["x".to_string()]);
    let result = transformer.fit(&df).await;
    assert!(
        result.is_err(),
        "Expected error for ArcsinTransformer with values outside [0,1]"
    );
}
