#![allow(dead_code)]

use datafusion::dataframe::DataFrame;
use datafusion::prelude::{CsvReadOptions, SessionContext};
use std::path::Path;

// Path to the directory containing the datasets
pub const DATA_DIR: &str = "tests/testdata";

/// Loads data from a given path and automatically detects the format (CSV or Parquet).
pub async fn load_data(path: &str) -> Result<DataFrame, datafusion::error::DataFusionError> {
    // Create DataFusion execution context
    let ctx = SessionContext::new();

    // Detect file type and read accordingly
    let df = if Path::new(path)
        .extension()
        .map_or(false, |ext| ext == "parquet")
    {
        ctx.read_parquet(path, Default::default()).await?
    } else if Path::new(path)
        .extension()
        .map_or(false, |ext| ext == "csv")
    {
        ctx.read_csv(path, CsvReadOptions::new()).await?
    } else {
        return Err(datafusion::error::DataFusionError::Execution(
            "Unsupported file format. Please provide a CSV or Parquet file.".to_string(),
        ));
    };

    Ok(df)
}

fn main() {}
