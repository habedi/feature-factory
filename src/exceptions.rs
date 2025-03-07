//! # Custom Errors for Feature Factory
//!
//! This module defines custom error types for the Feature Factory library.
//! It uses the `thiserror` crate to derive the `Error` trait for custom error types.
//! The `FeatureFactoryError` enum defines different error variants that can be returned by the library.
//! The `FeatureFactoryResult` type alias is used to simplify error handling in the library.

use thiserror::Error;

/// Custom errors for the `feature_factory` library.
#[derive(Debug, Error)]
pub enum FeatureFactoryError {
    /// Wraps underlying I/O errors.
    #[error("I/O error: {0}")]
    IoError(#[from] std::io::Error),

    /// Wraps errors from DataFusion.
    #[error("DataFusion error: {0}")]
    DataFusionError(#[from] datafusion::error::DataFusionError),

    /// Wraps errors from Arrow.
    #[error("Arrow error: {0}")]
    ArrowError(#[from] arrow::error::ArrowError),

    /// Wraps errors from Parquet.
    #[error("Parquet error: {0}")]
    ParquetError(#[from] parquet::errors::ParquetError),

    /// Indicates that an invalid parameter was provided.
    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),

    /// Indicates that the data format is not supported.
    #[error("Unsupported format: {0}")]
    UnsupportedFormat(String),

    /// Indicates that a feature is not implemented.
    #[error("Not implemented: {0}")]
    NotImplemented(String),

    /// Indicates that the column does not exist in the DataFrame.
    #[error("Missing column: {0}")]
    MissingColumn(String),
}

/// Type alias for results returned by the `feature_factory` library.
pub type FeatureFactoryResult<T> = std::result::Result<T, FeatureFactoryError>;

#[cfg(test)]
mod tests {
    use super::*;
    use std::io;

    #[test]
    fn test_io_error() {
        // Create a simple I/O error.
        let io_err = io::Error::new(io::ErrorKind::Other, "test io error");
        // Convert the std::io::Error into FeatureFactoryError.
        let err: FeatureFactoryError = io_err.into();
        let err_msg = format!("{}", err);
        assert!(err_msg.contains("I/O error:"));
        assert!(err_msg.contains("test io error"));
    }

    #[test]
    fn test_datafusion_error() {
        // Construct a DataFusion error using one of its variants.
        // Using the DataFusionError::Plan variant as an example.
        let df_err = datafusion::error::DataFusionError::Plan("test plan error".into());
        let err: FeatureFactoryError = df_err.into();
        let err_msg = format!("{}", err);
        assert!(err_msg.contains("DataFusion error:"));
        assert!(err_msg.contains("test plan error"));
    }

    #[test]
    fn test_arrow_error() {
        // Construct an Arrow error using a common variant.
        let arrow_err = arrow::error::ArrowError::ComputeError("test compute error".into());
        let err: FeatureFactoryError = arrow_err.into();
        let err_msg = format!("{}", err);
        assert!(err_msg.contains("Arrow error:"));
        assert!(err_msg.contains("test compute error"));
    }

    #[test]
    fn test_parquet_error() {
        // Construct a Parquet error. Use a general error variant if available.
        // The parquet crate provides a general error variant.
        let parquet_err = parquet::errors::ParquetError::General("test parquet error".into());
        let err: FeatureFactoryError = parquet_err.into();
        let err_msg = format!("{}", err);
        assert!(err_msg.contains("Parquet error:"));
        assert!(err_msg.contains("test parquet error"));
    }

    #[test]
    fn test_invalid_parameter_error() {
        let err = FeatureFactoryError::InvalidParameter("bad param".into());
        let err_msg = format!("{}", err);
        assert!(err_msg.contains("Invalid parameter:"));
        assert!(err_msg.contains("bad param"));
    }

    #[test]
    fn test_unsupported_format_error() {
        let err = FeatureFactoryError::UnsupportedFormat("unknown format".into());
        let err_msg = format!("{}", err);
        assert!(err_msg.contains("Unsupported format:"));
        assert!(err_msg.contains("unknown format"));
    }

    #[test]
    fn test_not_implemented_error() {
        let err = FeatureFactoryError::NotImplemented("feature not implemented".into());
        let err_msg = format!("{}", err);
        assert!(err_msg.contains("Not implemented:"));
        assert!(err_msg.contains("feature not implemented"));
    }

    #[test]
    fn test_missing_column_error() {
        let err = FeatureFactoryError::MissingColumn("missing column".into());
        let err_msg = format!("{}", err);
        assert!(err_msg.contains("Missing column:"));
        assert!(err_msg.contains("missing column"));
    }
}
