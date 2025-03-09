use std::sync::Arc;

use arrow::array::{ArrayRef, Float64Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use datafusion::datasource::memory::MemTable;
use datafusion::prelude::*;
use feature_factory::exceptions::FeatureFactoryResult;
use feature_factory::transformers::feature_selection::{
    DropConstantFeatures, DropCorrelatedFeatures, DropDuplicateFeatures, DropFeatures,
    DropHighPSIFeatures, SelectByInformationValue, SelectBySingleFeaturePerformance,
    SelectByTargetMeanPerformance, SmartCorrelatedSelection, MRMR,
};

/// Create a simple DataFrame for non-target-based tests with four columns:
/// "a": constant [1.0, 1.0, 1.0, 1.0],
/// "b": increasing values [1.0, 2.0, 3.0, 4.0],
/// "c": duplicate of b, and
/// "d": slightly shifted values [1.1, 2.1, 3.1, 4.1].
async fn create_selection_df() -> DataFrame {
    let schema = Arc::new(Schema::new(vec![
        Field::new("a", DataType::Float64, true),
        Field::new("b", DataType::Float64, true),
        Field::new("c", DataType::Float64, true),
        Field::new("d", DataType::Float64, true),
    ]));
    let a = Float64Array::from(vec![Some(1.0); 4]);
    let b = Float64Array::from(vec![Some(1.0), Some(2.0), Some(3.0), Some(4.0)]);
    let c = Float64Array::from(vec![Some(1.0), Some(2.0), Some(3.0), Some(4.0)]);
    let d = Float64Array::from(vec![Some(1.1), Some(2.1), Some(3.1), Some(4.1)]);

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(a) as ArrayRef,
            Arc::new(b),
            Arc::new(c),
            Arc::new(d),
        ],
    )
    .unwrap();

    let mem_table = MemTable::try_new(schema, vec![vec![batch]]).unwrap();
    let ctx = SessionContext::new();
    ctx.register_table("t", Arc::new(mem_table)).unwrap();
    ctx.table("t").await.unwrap()
}

/// Create a DataFrame for target-based tests with columns:
/// - "target": binary [0, 1, 0, 1, 0, 1],
/// - "x": perfectly correlated with target,
/// - "y": constant, and
/// - "z": alternating values [10,20,10,20,10,20].
async fn create_target_df() -> DataFrame {
    let schema = Arc::new(Schema::new(vec![
        Field::new("target", DataType::Float64, true),
        Field::new("x", DataType::Float64, true),
        Field::new("y", DataType::Float64, true),
        Field::new("z", DataType::Float64, true),
    ]));
    let target = Float64Array::from(vec![
        Some(0.0),
        Some(1.0),
        Some(0.0),
        Some(1.0),
        Some(0.0),
        Some(1.0),
    ]);
    let x = Float64Array::from(vec![
        Some(0.0),
        Some(1.0),
        Some(0.0),
        Some(1.0),
        Some(0.0),
        Some(1.0),
    ]);
    let y = Float64Array::from(vec![Some(1.0); 6]);
    let z = Float64Array::from(vec![
        Some(10.0),
        Some(20.0),
        Some(10.0),
        Some(20.0),
        Some(10.0),
        Some(20.0),
    ]);

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(target) as ArrayRef,
            Arc::new(x),
            Arc::new(y),
            Arc::new(z),
        ],
    )
    .unwrap();
    let mem_table = MemTable::try_new(schema, vec![vec![batch]]).unwrap();
    let ctx = SessionContext::new();
    ctx.register_table("t", Arc::new(mem_table)).unwrap();
    ctx.table("t").await.unwrap()
}

/// Create DataFrames for PSI tests. Here, we create a reference DF with a non-degenerate distribution,
/// and a current DF where the distribution has shifted significantly.
async fn create_psi_dfs() -> (DataFrame, DataFrame) {
    // Create a schema with one column "x".
    let schema = Arc::new(Schema::new(vec![Field::new("x", DataType::Float64, true)]));
    // Reference: a non-uniform distribution.
    let ref_values = vec![
        Some(1.0),
        Some(1.0),
        Some(1.5),
        Some(1.5),
        Some(2.0),
        Some(2.0),
        Some(2.5),
        Some(2.5),
        Some(3.0),
        Some(3.0),
    ];
    // Current: all values shifted upward.
    let curr_values = vec![
        Some(4.0),
        Some(4.0),
        Some(4.0),
        Some(4.0),
        Some(4.0),
        Some(4.0),
        Some(4.0),
        Some(4.0),
        Some(4.0),
        Some(4.0),
    ];
    let ref_array = Float64Array::from(ref_values);
    let curr_array = Float64Array::from(curr_values);

    let ref_batch =
        RecordBatch::try_new(schema.clone(), vec![Arc::new(ref_array) as ArrayRef]).unwrap();
    let curr_batch =
        RecordBatch::try_new(schema.clone(), vec![Arc::new(curr_array) as ArrayRef]).unwrap();

    let ref_table = MemTable::try_new(schema.clone(), vec![vec![ref_batch]]).unwrap();
    let curr_table = MemTable::try_new(schema, vec![vec![curr_batch]]).unwrap();

    let ctx = SessionContext::new();
    ctx.register_table("ref", Arc::new(ref_table)).unwrap();
    ctx.register_table("curr", Arc::new(curr_table)).unwrap();
    let ref_df = ctx.table("ref").await.unwrap();
    let curr_df = ctx.table("curr").await.unwrap();
    (ref_df, curr_df)
}

#[tokio::test]
async fn test_drop_features() -> FeatureFactoryResult<()> {
    let df = create_selection_df().await;
    // Drop columns "a" and "c"
    let mut transformer = DropFeatures::new(vec!["a".to_string(), "c".to_string()]);
    transformer.fit(&df).await?;
    let transformed = transformer.transform(df)?;
    let schema = transformed.schema();
    assert!(schema.field_with_name(None, "a").is_err());
    assert!(schema.field_with_name(None, "c").is_err());
    assert!(schema.field_with_name(None, "b").is_ok());
    assert!(schema.field_with_name(None, "d").is_ok());
    Ok(())
}

#[tokio::test]
async fn test_drop_constant_features() -> FeatureFactoryResult<()> {
    let df = create_selection_df().await;
    // Column "a" is constant so it should be dropped.
    let mut transformer = DropConstantFeatures::new(0.1, 1);
    transformer.fit(&df).await?;
    let transformed = transformer.transform(df)?;
    let schema = transformed.schema();
    assert!(
        schema.field_with_name(None, "a").is_err(),
        "Column a should be dropped"
    );
    Ok(())
}

#[tokio::test]
async fn test_drop_duplicate_features() -> FeatureFactoryResult<()> {
    let df = create_selection_df().await;
    // Columns "b" and "c" are duplicates. Expect exactly one of them to be dropped.
    let mut transformer = DropDuplicateFeatures::new();
    transformer.fit(&df).await?;
    let transformed = transformer.transform(df)?;
    let schema = transformed.schema();
    let has_b = schema.field_with_name(None, "b").is_ok();
    let has_c = schema.field_with_name(None, "c").is_ok();
    // Assert that not both "b" and "c" are present simultaneously.
    assert!(!(has_b && has_c), "Both 'b' and 'c' should not be present");
    Ok(())
}

#[tokio::test]
async fn test_drop_correlated_features() -> FeatureFactoryResult<()> {
    let df = create_selection_df().await;
    // Columns "b" and "d" are highly correlated.
    let mut transformer = DropCorrelatedFeatures::new(0.95);
    transformer.fit(&df).await?;
    let transformed = transformer.transform(df)?;
    let schema = transformed.schema();
    let has_b = schema.field_with_name(None, "b").is_ok();
    let has_d = schema.field_with_name(None, "d").is_ok();
    // Assert that not both are present.
    assert!(!(has_b && has_d), "Both 'b' and 'd' should not be present");
    Ok(())
}

#[tokio::test]
async fn test_smart_correlated_selection() -> FeatureFactoryResult<()> {
    let df = create_selection_df().await;
    let mut transformer = SmartCorrelatedSelection::new(0.95);
    transformer.fit(&df).await?;
    let _transformed = transformer.transform(df).unwrap();
    // Ensure that some features were selected.
    assert!(
        !transformer.selected_features.is_empty(),
        "Some features should be selected"
    );
    Ok(())
}

#[tokio::test]
async fn test_drop_high_psi_features() -> FeatureFactoryResult<()> {
    let (ref_df, curr_df) = create_psi_dfs().await;
    // With the reference distribution skewed toward lower values and current all high,
    // PSI should be high so that the feature "x" is dropped.
    let mut transformer = DropHighPSIFeatures::new(ref_df, 0.1);
    transformer.fit(&curr_df).await?;
    let result = transformer.transform(curr_df);
    // We expect an error because the only feature ("x") is dropped.
    assert!(
        result.is_err(),
        "Expected error when all features are dropped"
    );
    if let Err(e) = result {
        // Instead of an exact match, check that the error message contains the expected substring.
        assert!(
            e.to_string()
                .contains("All features dropped by DropHighPSIFeatures."),
            "Unexpected error message: {}",
            e
        );
    }
    Ok(())
}

#[tokio::test]
async fn test_select_by_information_value() -> FeatureFactoryResult<()> {
    let df = create_target_df().await;
    // With a low threshold, expect feature "x" (which perfectly follows target) to be selected.
    let mut transformer = SelectByInformationValue::new("target".to_string(), 0.0);
    transformer.fit(&df).await?;
    let transformed = transformer.transform(df)?;
    let schema = transformed.schema();
    assert!(schema.field_with_name(None, "target").is_ok());
    assert!(schema.field_with_name(None, "x").is_ok());
    Ok(())
}

#[tokio::test]
async fn test_select_by_single_feature_performance() -> FeatureFactoryResult<()> {
    let df = create_target_df().await;
    // Feature "x" is perfectly correlated with target.
    let mut transformer = SelectBySingleFeaturePerformance::new("target".to_string(), 0.9);
    transformer.fit(&df).await?;
    let transformed = transformer.transform(df)?;
    let schema = transformed.schema();
    assert!(schema.field_with_name(None, "x").is_ok());
    Ok(())
}

#[tokio::test]
async fn test_select_by_target_mean_performance() -> FeatureFactoryResult<()> {
    let df = create_target_df().await;
    // For our DF, feature "z" splits target means.
    let mut transformer = SelectByTargetMeanPerformance::new("target".to_string(), 0.3);
    transformer.fit(&df).await?;
    let transformed = transformer.transform(df)?;
    let schema = transformed.schema();
    assert!(schema.field_with_name(None, "z").is_ok());
    Ok(())
}

#[tokio::test]
async fn test_mrmr() -> FeatureFactoryResult<()> {
    let df = create_target_df().await;
    // Feature "x" is perfectly correlated with target.
    let mut transformer = MRMR::new("target".to_string(), 0.9, 0.95);
    transformer.fit(&df).await?;
    let transformed = transformer.transform(df)?;
    let schema = transformed.schema();
    assert!(schema.field_with_name(None, "target").is_ok());
    let has_x = schema.field_with_name(None, "x").is_ok();
    let has_z = schema.field_with_name(None, "z").is_ok();
    assert!(has_x || has_z, "Either x or z should be selected");
    Ok(())
}
