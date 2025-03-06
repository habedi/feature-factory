use approx::assert_abs_diff_eq;
use arrow::array::{ArrayRef, Float64Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use datafusion::datasource::MemTable;
use datafusion::prelude::*;
use datafusion_expr::col;
use std::sync::Arc;

use feature_factory::transformers::feature_creation::{
    CyclicalFeatures, CyclicalMethod, MathFeatures, RelativeFeatures, RelativeOperation,
};

/// Create a DataFrame for MathFeatures tests with columns "x" and "y".
async fn create_math_dataframe() -> DataFrame {
    let schema = Arc::new(Schema::new(vec![
        Field::new("x", DataType::Float64, false),
        Field::new("y", DataType::Float64, false),
    ]));
    let x: ArrayRef = Arc::new(Float64Array::from(vec![1.0_f64, 2.0_f64, 3.0_f64]));
    let y: ArrayRef = Arc::new(Float64Array::from(vec![4.0_f64, 5.0_f64, 6.0_f64]));
    let batch = RecordBatch::try_new(schema.clone(), vec![x, y]).unwrap();
    let mem_table = MemTable::try_new(schema, vec![vec![batch]]).unwrap();
    let ctx = SessionContext::new();
    ctx.register_table("math", Arc::new(mem_table)).unwrap();
    ctx.table("math").await.unwrap()
}

/// Create a DataFrame for RelativeFeatures tests with columns "target" and "reference".
async fn create_relative_dataframe() -> DataFrame {
    let schema = Arc::new(Schema::new(vec![
        Field::new("target", DataType::Float64, false),
        Field::new("reference", DataType::Float64, false),
    ]));
    let target: ArrayRef = Arc::new(Float64Array::from(vec![10.0_f64, 20.0_f64, 30.0_f64]));
    let reference: ArrayRef = Arc::new(Float64Array::from(vec![2.0_f64, 4.0_f64, 5.0_f64]));
    let batch = RecordBatch::try_new(schema.clone(), vec![target, reference]).unwrap();
    let mem_table = MemTable::try_new(schema, vec![vec![batch]]).unwrap();
    let ctx = SessionContext::new();
    ctx.register_table("relative", Arc::new(mem_table)).unwrap();
    ctx.table("relative").await.unwrap()
}

/// Create a DataFrame for CyclicalFeatures tests with a column "time" (e.g., hours).
async fn create_cyclical_dataframe() -> DataFrame {
    let schema = Arc::new(Schema::new(vec![Field::new(
        "time",
        DataType::Float64,
        false,
    )]));
    // Example: hours 0, 6, 12, 18.
    let time: ArrayRef = Arc::new(Float64Array::from(vec![
        0.0_f64, 6.0_f64, 12.0_f64, 18.0_f64,
    ]));
    let batch = RecordBatch::try_new(schema.clone(), vec![time]).unwrap();
    let mem_table = MemTable::try_new(schema, vec![vec![batch]]).unwrap();
    let ctx = SessionContext::new();
    ctx.register_table("cyclical", Arc::new(mem_table)).unwrap();
    ctx.table("cyclical").await.unwrap()
}

/// --- Normal Operation Tests ---

#[tokio::test]
async fn test_math_features() {
    let df = create_math_dataframe().await;
    // Create new features: "sum" = x + y, "product" = x * y.
    let features = vec![
        ("sum".to_string(), col("x").add(col("y"))),
        ("product".to_string(), col("x").mul(col("y"))),
    ];
    let mut transformer = MathFeatures::new(features);
    transformer.fit(&df).await.unwrap();
    let transformed_df = transformer.transform(df).unwrap();
    let batches = transformed_df.collect().await.unwrap();
    let batch = &batches[0];
    // Expected new columns: "sum" and "product".
    let sum_array = batch
        .column(2)
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap();
    let product_array = batch
        .column(3)
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap();
    let expected_sum = vec![5.0_f64, 7.0_f64, 9.0_f64];
    let expected_product = vec![4.0_f64, 10.0_f64, 18.0_f64];
    for i in 0..sum_array.len() {
        assert_abs_diff_eq!(sum_array.value(i), expected_sum[i], epsilon = 1e-6);
        assert_abs_diff_eq!(product_array.value(i), expected_product[i], epsilon = 1e-6);
    }
}

#[tokio::test]
async fn test_relative_features() {
    let df = create_relative_dataframe().await;
    let features = vec![
        (
            "ratio".to_string(),
            "target".to_string(),
            "reference".to_string(),
            RelativeOperation::Ratio,
        ),
        (
            "difference".to_string(),
            "target".to_string(),
            "reference".to_string(),
            RelativeOperation::Difference,
        ),
        (
            "pct_change".to_string(),
            "target".to_string(),
            "reference".to_string(),
            RelativeOperation::PercentChange,
        ),
    ];
    let mut transformer = RelativeFeatures::new(features);
    transformer.fit(&df).await.unwrap();
    let transformed_df = transformer.transform(df).unwrap();
    let batches = transformed_df.collect().await.unwrap();
    let batch = &batches[0];
    let ratio_array = batch
        .column(2)
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap();
    let difference_array = batch
        .column(3)
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap();
    let pct_change_array = batch
        .column(4)
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap();
    let expected_ratio = vec![5.0_f64, 5.0_f64, 6.0_f64];
    let expected_difference = vec![8.0_f64, 16.0_f64, 25.0_f64];
    let expected_pct_change = vec![4.0_f64, 4.0_f64, 5.0_f64];
    for i in 0..ratio_array.len() {
        assert_abs_diff_eq!(ratio_array.value(i), expected_ratio[i], epsilon = 1e-6);
        assert_abs_diff_eq!(
            difference_array.value(i),
            expected_difference[i],
            epsilon = 1e-6
        );
        assert_abs_diff_eq!(
            pct_change_array.value(i),
            expected_pct_change[i],
            epsilon = 1e-6
        );
    }
}

#[tokio::test]
async fn test_cyclical_features() {
    let df = create_cyclical_dataframe().await;
    let features = vec![
        (
            "time_sin".to_string(),
            "time".to_string(),
            24.0_f64,
            CyclicalMethod::Sine,
        ),
        (
            "time_cos".to_string(),
            "time".to_string(),
            24.0_f64,
            CyclicalMethod::Cosine,
        ),
    ];
    let mut transformer = CyclicalFeatures::new(features);
    transformer.fit(&df).await.unwrap();
    let transformed_df = transformer.transform(df).unwrap();
    let batches = transformed_df.collect().await.unwrap();
    let batch = &batches[0];
    let time_array = batch
        .column(0)
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap();
    let time_sin_array = batch
        .column(1)
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap();
    let time_cos_array = batch
        .column(2)
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap();
    for i in 0..time_array.len() {
        let time_val = time_array.value(i);
        let base = 2.0_f64 * std::f64::consts::PI * time_val / 24.0;
        let expected_sin = base.sin();
        let expected_cos = base.cos();
        assert_abs_diff_eq!(time_sin_array.value(i), expected_sin, epsilon = 1e-6);
        assert_abs_diff_eq!(time_cos_array.value(i), expected_cos, epsilon = 1e-6);
    }
}

/// --- Additional Error Condition Tests ---

#[tokio::test]
#[should_panic(expected = "feature name cannot be empty")]
async fn test_math_features_empty_name() {
    // Expect a panic if MathFeatures is constructed with an empty feature name.
    let _transformer = MathFeatures::new(vec![("".to_string(), col("x").add(col("y")))]);
}

#[tokio::test]
async fn test_relative_features_missing_column() {
    let df = create_math_dataframe().await;
    // Create RelativeFeatures with a non-existent reference column.
    let features = vec![(
        "rel".to_string(),
        "x".to_string(),
        "nonexistent".to_string(),
        RelativeOperation::Ratio,
    )];
    let mut transformer = RelativeFeatures::new(features);
    let result = transformer.fit(&df).await;
    assert!(
        result.is_err(),
        "Expected error for missing reference column"
    );
}

#[tokio::test]
#[should_panic(expected = "period must be positive")]
async fn test_cyclical_features_invalid_period() {
    // This should panic because the period is 0.
    let _ = CyclicalFeatures::new(vec![(
        "cycle".to_string(),
        "time".to_string(),
        0.0_f64,
        CyclicalMethod::Sine,
    )]);
}

#[tokio::test]
async fn test_relative_features_empty_features() {
    // If no relative features are provided, transform should return the original DataFrame.
    let df = create_relative_dataframe().await;
    let mut transformer = RelativeFeatures::new(vec![]);
    transformer.fit(&df).await.unwrap();
    let transformed_df = transformer.transform(df).unwrap();
    let original_batches = create_relative_dataframe().await.collect().await.unwrap();
    let new_batches = transformed_df.collect().await.unwrap();
    assert_eq!(
        original_batches.len(),
        new_batches.len(),
        "Expected same number of batches"
    );
}
