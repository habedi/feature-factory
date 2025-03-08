//! ## Datetime Feature Transformers
//!
//! This module provides transformers to extract or calculate features from datetime data.
//!
//! ### Available Transformers
//!
//! - [`DatetimeFeatures`]: Extracts features such as year, month, day, hour, minute, second, and weekday from datetime columns.
//! - [`DatetimeSubtraction`]: Computes differences between datetime columns in specified units (seconds, minutes, hours, and days).
//!
//! Each transformer returns a new DataFrame with the added or modified columns.
//! Errors are returned as `FeatureFactoryError`, and successful transformations are wrapped in `FeatureFactoryResult`.

use crate::exceptions::{FeatureFactoryError, FeatureFactoryResult};
use crate::impl_transformer;
use datafusion::arrow::datatypes::DataType;
use datafusion::dataframe::DataFrame;
use datafusion_expr::{col, lit, Expr};
use datafusion_functions::datetime::{date_part, to_unixtime};
use std::ops::{Div, Sub};

/// Validates that a column exists and is of a datetime type (Timestamp, Date32, or Date64).
fn validate_datetime_column(df: &DataFrame, col_name: &str) -> FeatureFactoryResult<()> {
    let field = df.schema().field_with_name(None, col_name).map_err(|_| {
        FeatureFactoryError::MissingColumn(format!("Column '{}' not found", col_name))
    })?;
    match field.data_type() {
        DataType::Timestamp(_, _) | DataType::Date32 | DataType::Date64 => Ok(()),
        dt => Err(FeatureFactoryError::InvalidParameter(format!(
            "Column '{}' must be a datetime type (Timestamp, Date32, or Date64), but found {:?}",
            col_name, dt
        ))),
    }
}

/// Extracts features from datetime columns.
/// For each column in `self.columns`, it adds the following new features:
/// `<column>_year`, `<column>_month`, `<column>_day`, `<column>_hour`,
/// `<column>_minute`, `<column>_second`, and `<column>_weekday`.
pub struct DatetimeFeatures {
    pub columns: Vec<String>,
}

impl DatetimeFeatures {
    pub fn new(columns: Vec<String>) -> Self {
        Self { columns }
    }

    /// Stateless transformer: fit does nothing.
    pub async fn fit(&mut self, _df: &DataFrame) -> FeatureFactoryResult<()> {
        Ok(())
    }

    /// Transform validates that each target column exists and is a datetime type,
    /// then returns a new DataFrame with the additional extracted features.
    pub fn transform(&self, df: DataFrame) -> FeatureFactoryResult<DataFrame> {
        // Validate each target column in transform.
        for col_name in &self.columns {
            validate_datetime_column(&df, col_name)?;
        }
        // Retain all original columns.
        let mut exprs: Vec<Expr> = df.schema().fields().iter().map(|f| col(f.name())).collect();
        // Add new features.
        for col_name in &self.columns {
            let base = col(col_name);
            let year_expr = date_part()
                .call(vec![lit("year"), base.clone()])
                .alias(format!("{}_year", col_name));
            let month_expr = date_part()
                .call(vec![lit("month"), base.clone()])
                .alias(format!("{}_month", col_name));
            let day_expr = date_part()
                .call(vec![lit("day"), base.clone()])
                .alias(format!("{}_day", col_name));
            let hour_expr = date_part()
                .call(vec![lit("hour"), base.clone()])
                .alias(format!("{}_hour", col_name));
            let minute_expr = date_part()
                .call(vec![lit("minute"), base.clone()])
                .alias(format!("{}_minute", col_name));
            let second_expr = date_part()
                .call(vec![lit("second"), base.clone()])
                .alias(format!("{}_second", col_name));
            let weekday_expr = date_part()
                .call(vec![lit("dow"), base.clone()])
                .alias(format!("{}_weekday", col_name));
            exprs.push(year_expr);
            exprs.push(month_expr);
            exprs.push(day_expr);
            exprs.push(hour_expr);
            exprs.push(minute_expr);
            exprs.push(second_expr);
            exprs.push(weekday_expr);
        }
        df.select(exprs)
            .map_err(FeatureFactoryError::DataFusionError)
    }

    // This transformer is stateless.
    fn inherent_is_stateful(&self) -> bool {
        false
    }
}

/// Time units for datetime subtraction.
pub enum TimeUnit {
    Second,
    Minute,
    Hour,
    Day,
}

impl TimeUnit {
    pub fn as_str(&self) -> &'static str {
        match self {
            TimeUnit::Second => "second",
            TimeUnit::Minute => "minute",
            TimeUnit::Hour => "hour",
            TimeUnit::Day => "day",
        }
    }
}

/// Helper function to compute timestamp difference between two datetime expressions.
/// It converts both expressions to Unix time (in seconds) using `to_unixtime`,
/// subtracts them, and then converts the difference to the desired unit.
fn timestamp_diff_expr(left: Expr, right: Expr, unit: &str) -> Expr {
    let left_sec = to_unixtime().call(vec![left]);
    let right_sec = to_unixtime().call(vec![right]);
    let diff_in_seconds = left_sec.sub(right_sec);
    match unit {
        "second" => diff_in_seconds,
        "minute" => diff_in_seconds.div(lit(60.0)),
        "hour" => diff_in_seconds.div(lit(3600.0)),
        "day" => diff_in_seconds.div(lit(86400.0)),
        _ => diff_in_seconds,
    }
}

/// Computes time differences between two datetime columns.
/// `new_features` is a list of tuples: (new_feature_name, left_column, right_column, time_unit).
/// Transform validates that each left and right column exists and is of a datetime type.
pub struct DatetimeSubtraction {
    pub new_features: Vec<(String, String, String, TimeUnit)>,
}

impl DatetimeSubtraction {
    pub fn new(new_features: Vec<(String, String, String, TimeUnit)>) -> Self {
        Self { new_features }
    }

    /// Stateless transformer: fit does nothing.
    pub async fn fit(&mut self, _df: &DataFrame) -> FeatureFactoryResult<()> {
        Ok(())
    }

    /// Transform validates that each left and right column exists and is a datetime type,
    /// then adds a new column for each specified time difference.
    pub fn transform(&self, df: DataFrame) -> FeatureFactoryResult<DataFrame> {
        // Validate that each left and right column is a datetime type.
        for (_, left, right, _) in &self.new_features {
            // These calls now ensure that the column exists *and* is of a valid datetime type.
            validate_datetime_column(&df, left)?;
            validate_datetime_column(&df, right)?;
        }
        let mut exprs: Vec<Expr> = df.schema().fields().iter().map(|f| col(f.name())).collect();
        for (new_name, left, right, unit) in &self.new_features {
            let diff_expr =
                timestamp_diff_expr(col(left), col(right), unit.as_str()).alias(new_name);
            exprs.push(diff_expr);
        }
        df.select(exprs)
            .map_err(FeatureFactoryError::DataFusionError)
    }

    // This transformer is stateless.
    fn inherent_is_stateful(&self) -> bool {
        false
    }
}

// Implement the Transformer trait for the transformers in this module.
impl_transformer!(DatetimeFeatures);
impl_transformer!(DatetimeSubtraction);
