//! ## Common Types
//!
//! This module contains common type definitions and utility functions used across the library.

use crate::foundation::errors::{FeatureFactoryError, FeatureFactoryResult};
use datafusion::arrow::datatypes::DataType;
use datafusion::dataframe::DataFrame;

/// Validates that every column in `target_cols` exists in the DataFrame.
/// Returns an error if any target column is missing.
pub fn validate_columns(df: &DataFrame, target_cols: &[String]) -> FeatureFactoryResult<()> {
    let schema = df.schema();
    for col_name in target_cols {
        if schema.field_with_name(None, col_name).is_err() {
            return Err(FeatureFactoryError::MissingColumn(format!(
                "Column '{}' not found in DataFrame",
                col_name
            )));
        }
    }
    Ok(())
}

/// Validates that a column exists and is of Utf8 type.
pub fn validate_string_column(df: &DataFrame, col_name: &str) -> FeatureFactoryResult<()> {
    let field = df.schema().field_with_name(None, col_name).map_err(|_| {
        FeatureFactoryError::MissingColumn(format!("Column '{}' not found", col_name))
    })?;
    if field.data_type() != &DataType::Utf8 {
        return Err(FeatureFactoryError::InvalidParameter(format!(
            "Column '{}' must be of type Utf8, but found {:?}",
            col_name,
            field.data_type()
        )));
    }
    Ok(())
}

/// Validates that all columns in `cols` exist and are of Utf8 type.
pub fn validate_string_columns(df: &DataFrame, cols: &[String]) -> FeatureFactoryResult<()> {
    for col in cols {
        validate_string_column(df, col)?;
    }
    Ok(())
}

/// Validates that a column exists and is numeric (Float64 or Int64).
pub fn validate_numeric_column(df: &DataFrame, col_name: &str) -> FeatureFactoryResult<()> {
    let field = df.schema().field_with_name(None, col_name).map_err(|_| {
        FeatureFactoryError::MissingColumn(format!("Column '{}' not found", col_name))
    })?;
    match field.data_type() {
        DataType::Float64 | DataType::Int64 => Ok(()),
        dt => Err(FeatureFactoryError::InvalidParameter(format!(
            "Column '{}' must be numeric (Float64 or Int64), but found {:?}",
            col_name, dt
        ))),
    }
}

/// Helper function that checks if a DataFusion data type is numeric.
pub fn is_numeric(dt: &DataType) -> bool {
    matches!(dt, DataType::Float64 | DataType::Int64)
}

/// Sanitizes a category string so that it can be safely used as part of a column name.
/// Non-alphanumeric characters are replaced with underscores.
pub fn sanitize_category(cat: &str) -> String {
    cat.replace(|c: char| !c.is_alphanumeric(), "_")
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion::prelude::*;

    #[tokio::test]
    async fn test_validate_columns() {
        let ctx = SessionContext::new();
        let df = ctx.sql("SELECT 1 as a, 2 as b").await.unwrap();

        assert!(validate_columns(&df, &["a".to_string(), "b".to_string()]).is_ok());
        assert!(validate_columns(&df, &["c".to_string()]).is_err());
    }

    #[tokio::test]
    async fn test_validate_string_column() {
        let ctx = SessionContext::new();
        let df = ctx
            .sql("SELECT 'hello' as str_col, 42 as num_col")
            .await
            .unwrap();

        assert!(validate_string_column(&df, "str_col").is_ok());
        assert!(validate_string_column(&df, "num_col").is_err());
        assert!(validate_string_column(&df, "missing").is_err());
    }

    #[test]
    fn test_is_numeric() {
        assert!(is_numeric(&DataType::Float64));
        assert!(is_numeric(&DataType::Int64));
        assert!(!is_numeric(&DataType::Utf8));
        assert!(!is_numeric(&DataType::Boolean));
    }

    #[test]
    fn test_sanitize_category() {
        assert_eq!(sanitize_category("hello-world"), "hello_world");
        assert_eq!(sanitize_category("test@123"), "test_123");
        assert_eq!(sanitize_category("valid_name"), "valid_name");
    }
}
