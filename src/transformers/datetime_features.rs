//! ## Transformers for extracting datetime-based features
//!
//! This module implements transformers for extracting features from datetime values.
//!
//! Currently, the following transformers are implemented:
//!
//! - **DatetimeFeatures:** Extract features from datetime variables (year, month, day, hour, minute, second, weekday).
//! - **DatetimeSubtraction:** Compute time differences between datetime variables using a specified time unit.
//!
//! Each transformer provides a constructor, an (async) `fit` method (if needed), and a `transform` method
//! that returns a new DataFrame with the new features added.
//! Errors are returned as `FeatureFactoryError` and results are wrapped in `FeatureFactoryResult`.

use crate::exceptions::{FeatureFactoryError, FeatureFactoryResult};
use datafusion::arrow::datatypes::DataType;
use datafusion::prelude::*;
use datafusion_expr::{col, lit, Expr};
use datafusion_functions::datetime::{date_part, to_unixtime};

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

    /// Validates that each specified datetime column exists and is of a valid datetime type.
    pub async fn fit(&mut self, df: &DataFrame) -> FeatureFactoryResult<()> {
        for col_name in &self.columns {
            validate_datetime_column(df, col_name)?;
        }
        Ok(())
    }

    /// Transforms the DataFrame by appending extracted datetime features.
    /// Returns a new DataFrame with original columns plus:
    /// `<column>_year`, `<column>_month`, `<column>_day`, `<column>_hour`,
    /// `<column>_minute`, `<column>_second`, `<column>_weekday`.
    pub fn transform(&self, df: DataFrame) -> FeatureFactoryResult<DataFrame> {
        // Retain all original columns.
        let mut exprs: Vec<Expr> = df.schema().fields().iter().map(|f| col(f.name())).collect();

        // Validate that each target column exists and is datetime.
        for col_name in &self.columns {
            // If validation fails, an error will be returned.
            validate_datetime_column(&df, col_name)?;
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
/// new_features is a list of tuples with the following format:
/// (new_feature_name, left_column, right_column, time_unit)
///
/// The transformer validates that both the left and right columns exist and are of a datetime type.
pub struct DatetimeSubtraction {
    pub new_features: Vec<(String, String, String, TimeUnit)>,
}

impl DatetimeSubtraction {
    pub fn new(new_features: Vec<(String, String, String, TimeUnit)>) -> Self {
        Self { new_features }
    }

    /// Validates that for each new feature, the left and right columns exist and are datetime types.
    pub async fn fit(&mut self, df: &DataFrame) -> FeatureFactoryResult<()> {
        for (_, left, right, _) in &self.new_features {
            validate_datetime_column(df, left)?;
            validate_datetime_column(df, right)?;
        }
        Ok(())
    }

    /// Transforms the DataFrame by computing time differences between datetime columns.
    /// For each new feature, the computed expression is:
    /// `timestamp_diff_expr(col(left), col(right), unit)` aliased as the new feature name.
    pub fn transform(&self, df: DataFrame) -> FeatureFactoryResult<DataFrame> {
        // Retain original columns.
        let mut exprs: Vec<Expr> = df.schema().fields().iter().map(|f| col(f.name())).collect();

        for (new_name, left, right, unit) in &self.new_features {
            // Validate that left and right columns exist.
            df.schema().field_with_name(None, left).map_err(|_| {
                FeatureFactoryError::MissingColumn(format!("Column '{}' not found", left))
            })?;
            df.schema().field_with_name(None, right).map_err(|_| {
                FeatureFactoryError::MissingColumn(format!("Column '{}' not found", right))
            })?;

            let diff_expr =
                timestamp_diff_expr(col(left), col(right), unit.as_str()).alias(new_name);
            exprs.push(diff_expr);
        }

        df.select(exprs)
            .map_err(FeatureFactoryError::DataFusionError)
    }
}
