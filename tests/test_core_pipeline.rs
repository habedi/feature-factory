use std::sync::Arc;

use arrow::array::{BooleanArray, Float64Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use datafusion::datasource::MemTable;
use datafusion::prelude::{DataFrame, SessionContext};
use feature_factory::exceptions::FeatureFactoryResult;
use feature_factory::make_pipeline;
// Import the pipeline! macro.
use feature_factory::pipeline::{Pipeline, Transformer};
use feature_factory::transformers::imputation::AddMissingIndicator;

// Import the numerical transformers.
use feature_factory::transformers::numerical::{ArcsinTransformer, LogTransformer};

#[tokio::test]
async fn test_pipeline_with_numerical_transformers() -> FeatureFactoryResult<()> {
    // Define schema with two columns:
    // - "x": used for LogTransformer (expects positive values)
    // - "y": used for ArcsinTransformer (expects values between 0 and 1)
    let schema = Arc::new(Schema::new(vec![
        Field::new("x", DataType::Float64, false),
        Field::new("y", DataType::Float64, false),
    ]));

    // Create sample data:
    // For column "x": [1.0, 2.0, 3.0] -> natural logarithm will be applied.
    // For column "y": [0.25, 0.5, 0.75] -> arcsine(sqrt(.)) transformation.
    let x_array = Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0]));
    let y_array = Arc::new(Float64Array::from(vec![0.25, 0.5, 0.75]));
    let batch = RecordBatch::try_new(schema.clone(), vec![x_array, y_array])?;

    // Create a MemTable with a single batch.
    let mem_table = MemTable::try_new(schema.clone(), vec![vec![batch]])?;
    let ctx = SessionContext::new();
    ctx.register_table("test_table", Arc::new(mem_table))?;

    // Obtain a DataFrame from the registered table.
    let df = ctx.table("test_table").await?;

    // Create numerical transformers:
    // - LogTransformer on column "x"
    // - ArcsinTransformer on column "y"
    let log_transformer = LogTransformer::new(vec!["x".to_string()]);
    let arcsin_transformer = ArcsinTransformer::new(vec!["y".to_string()]);

    // Build the pipeline with the two transformers.
    let mut pipeline = Pipeline::new(
        vec![
            (
                "log_x".to_string(),
                Box::new(log_transformer) as Box<dyn Transformer + Send + Sync>,
            ),
            (
                "arcsin_y".to_string(),
                Box::new(arcsin_transformer) as Box<dyn Transformer + Send + Sync>,
            ),
        ],
        false, // verbose off for testing
    );

    // Fit the pipeline (this may materialize data for parameter computation)
    // and obtain the transformed DataFrame.
    let transformed_df: DataFrame = pipeline.fit_transform(&df).await?;

    // Materialize the final DataFrame.
    let results = transformed_df.collect().await?;
    let result_batch = &results[0];

    // Extract transformed columns.
    let x_transformed = result_batch
        .column(result_batch.schema().index_of("x")?)
        .as_any()
        .downcast_ref::<Float64Array>()
        .expect("Failed to downcast column 'x'");
    let y_transformed = result_batch
        .column(result_batch.schema().index_of("y")?)
        .as_any()
        .downcast_ref::<Float64Array>()
        .expect("Failed to downcast column 'y'");

    // Expected values:
    // For LogTransformer: ln(1.0)=0.0, ln(2.0)≈0.693147, ln(3.0)≈1.098612
    let expected_x = [0.0, 0.693147, 1.098612];
    // For ArcsinTransformer: transformation = asin(sqrt(y))
    // sqrt(0.25)=0.5, asin(0.5)≈0.523599; sqrt(0.5)≈0.707107, asin(0.707107)≈0.785398;
    // sqrt(0.75)≈0.866025, asin(0.866025)≈1.047198.
    let expected_y = [0.523599, 0.785398, 1.047198];

    // Verify that the computed values are within a small tolerance of the expected values.
    for i in 0..expected_x.len() {
        let computed_x = x_transformed.value(i);
        let computed_y = y_transformed.value(i);
        assert!(
            (computed_x - expected_x[i]).abs() < 1e-5,
            "Mismatch in column 'x' at index {}: expected {}, got {}",
            i,
            expected_x[i],
            computed_x
        );
        assert!(
            (computed_y - expected_y[i]).abs() < 1e-5,
            "Mismatch in column 'y' at index {}: expected {}, got {}",
            i,
            expected_y[i],
            computed_y
        );
    }

    Ok(())
}

#[tokio::test]
async fn test_long_pipeline_with_multiple_transformers() -> FeatureFactoryResult<()> {
    // Define a schema with two columns:
    // - "x": used for LogTransformer (expects positive values)
    // - "y": used for ArcsinTransformer (expects values between 0 and 1)
    let schema = Arc::new(Schema::new(vec![
        Field::new("x", DataType::Float64, false),
        Field::new("y", DataType::Float64, false),
    ]));

    // Create sample data:
    // "x": [1.0, 2.0, 3.0] will be transformed by ln(x)
    // "y": [0.25, 0.5, 0.75] will be transformed by asin(sqrt(y))
    let x_array = Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0]));
    let y_array = Arc::new(Float64Array::from(vec![0.25, 0.5, 0.75]));
    let batch = RecordBatch::try_new(schema.clone(), vec![x_array, y_array])?;

    // Create a MemTable with a single batch.
    let mem_table = MemTable::try_new(schema.clone(), vec![vec![batch]])?;
    let ctx = SessionContext::new();
    ctx.register_table("test_table", Arc::new(mem_table))?;

    // Obtain a DataFrame from the registered table.
    let df = ctx.table("test_table").await?;

    // Build a longer pipeline using the pipeline! macro:
    // 1. Apply LogTransformer on "x"
    // 2. Apply ArcsinTransformer on "y"
    // 3. Add missing indicator columns for "x" and "y"
    let mut pipeline = make_pipeline!(
        false,
        ("log_x", LogTransformer::new(vec!["x".to_string()])),
        ("arcsin_y", ArcsinTransformer::new(vec!["y".to_string()])),
        (
            "add_missing",
            AddMissingIndicator::new(vec!["x".to_string(), "y".to_string()], None)
        )
    );

    // Fit and transform the DataFrame using the pipeline.
    let transformed_df: DataFrame = pipeline.fit_transform(&df).await?;
    let results = transformed_df.collect().await?;
    let batch = &results[0];

    // Extract transformed columns "x" and "y"
    let x_transformed = batch
        .column(batch.schema().index_of("x")?)
        .as_any()
        .downcast_ref::<Float64Array>()
        .expect("Failed to downcast column 'x'");
    let y_transformed = batch
        .column(batch.schema().index_of("y")?)
        .as_any()
        .downcast_ref::<Float64Array>()
        .expect("Failed to downcast column 'y'");

    // Expected values for "x": ln(1)=0.0, ln(2)≈0.693147, ln(3)≈1.098612.
    let expected_x = [0.0, 0.693147, 1.098612];
    // Expected values for "y": asin(sqrt(0.25))≈0.523599, asin(sqrt(0.5))≈0.785398, asin(sqrt(0.75))≈1.047198.
    let expected_y = [0.523599, 0.785398, 1.047198];

    for i in 0..expected_x.len() {
        let computed_x = x_transformed.value(i);
        let computed_y = y_transformed.value(i);
        assert!(
            (computed_x - expected_x[i]).abs() < 1e-5,
            "Mismatch in column 'x' at index {}: expected {}, got {}",
            i,
            expected_x[i],
            computed_x
        );
        assert!(
            (computed_y - expected_y[i]).abs() < 1e-5,
            "Mismatch in column 'y' at index {}: expected {}, got {}",
            i,
            expected_y[i],
            computed_y
        );
    }

    // Verify that the missing indicator columns exist and contain false values.
    let x_missing = batch
        .column(batch.schema().index_of("x_missing")?)
        .as_any()
        .downcast_ref::<BooleanArray>()
        .expect("Failed to downcast 'x_missing'");
    let y_missing = batch
        .column(batch.schema().index_of("y_missing")?)
        .as_any()
        .downcast_ref::<BooleanArray>()
        .expect("Failed to downcast 'y_missing'");

    for i in 0..x_missing.len() {
        assert!(
            !x_missing.value(i),
            "Expected 'x_missing' to be false at index {}, got true",
            i
        );
    }
    for i in 0..y_missing.len() {
        assert!(
            !y_missing.value(i),
            "Expected 'y_missing' to be false at index {}, got true",
            i
        );
    }

    Ok(())
}
