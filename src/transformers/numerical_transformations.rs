//! ## Transformers for performing numerical transformations
//!
//! This module provides various transformers for transforming variables/features using mathematical functions.
//!
//! Currently, the following transformers are implemented:
//!
//! - **LogTransformer:** Apply natural logarithm transformation (requires positive values).
//! - **LogCpTransformer:** Apply logarithmic transformation with a constant (requires values + constant > 0).
//! - **ReciprocalTransformer:** Apply reciprocal transformation (requires non-zero values).
//! - **PowerTransformer:** Apply power transformation.
//! - **BoxCoxTransformer:** Apply Box–Cox transformation (requires positive values).
//! - **YeoJohnsonTransformer:** Apply Yeo–Johnson transformation.
//! - **ArcsinTransformer:** Apply arcsine transformation (typically for proportions).
//!
//! Each transformer provides a constructor, an (async) `fit` method (if needed), and a `transform` method
//! that returns a new DataFrame with the transformation applied.
//! Errors are returned as `FeatureFactoryError` and results are wrapped in `FeatureFactoryResult`.

use crate::exceptions::{FeatureFactoryError, FeatureFactoryResult};
use datafusion::functions_aggregate::approx_percentile_cont::approx_percentile_cont;
use datafusion::prelude::*;
use datafusion::scalar::ScalarValue;
use datafusion_expr::expr::Case;
use datafusion_expr::{col, lit, Expr};
use datafusion_functions::math;

/// Wrapper function wrapping math's natural logarithm UDF.
fn ln_expr(e: Expr) -> Expr {
    math::ln().call(vec![e])
}

/// Wrapper function wrapping math's power UDF.
fn power_expr(e: Expr, p: f64) -> Expr {
    math::power().call(vec![e, lit(p)])
}

/// Wrapper function wrapping math's square root UDF.
fn sqrt_expr(e: Expr) -> Expr {
    math::sqrt().call(vec![e])
}

/// Wrapper function wrapping math's arcsine UDF.
fn asin_expr(e: Expr) -> Expr {
    math::asin().call(vec![e])
}

/// Wrapper function to compute the minimum value in a numeric column using approximate percentiles (p=0).
async fn compute_min(df: &DataFrame, col_name: &str) -> FeatureFactoryResult<f64> {
    let min_df = df
        .clone()
        .aggregate(
            vec![],
            vec![approx_percentile_cont(col(col_name), lit(0.0), None).alias("min")],
        )
        .map_err(FeatureFactoryError::from)?;
    let batches = min_df.collect().await.map_err(FeatureFactoryError::from)?;
    if let Some(batch) = batches.first() {
        let array = batch.column(0);
        let scalar = ScalarValue::try_from_array(array, 0).map_err(FeatureFactoryError::from)?;
        if let ScalarValue::Float64(Some(val)) = scalar {
            Ok(val)
        } else {
            Err(FeatureFactoryError::DataFusionError(
                datafusion::error::DataFusionError::Plan(format!(
                    "Failed to compute min for column {}",
                    col_name
                )),
            ))
        }
    } else {
        Err(FeatureFactoryError::DataFusionError(
            datafusion::error::DataFusionError::Plan("No data found".to_string()),
        ))
    }
}

/// Helper function to compute the maximum value in a numeric column using approximate percentiles (p=1).
async fn compute_max(df: &DataFrame, col_name: &str) -> FeatureFactoryResult<f64> {
    let max_df = df
        .clone()
        .aggregate(
            vec![],
            vec![approx_percentile_cont(col(col_name), lit(1.0), None).alias("max")],
        )
        .map_err(FeatureFactoryError::from)?;
    let batches = max_df.collect().await.map_err(FeatureFactoryError::from)?;
    if let Some(batch) = batches.first() {
        let array = batch.column(0);
        let scalar = ScalarValue::try_from_array(array, 0).map_err(FeatureFactoryError::from)?;
        if let ScalarValue::Float64(Some(val)) = scalar {
            Ok(val)
        } else {
            Err(FeatureFactoryError::DataFusionError(
                datafusion::error::DataFusionError::Plan(format!(
                    "Failed to compute max for column {}",
                    col_name
                )),
            ))
        }
    } else {
        Err(FeatureFactoryError::DataFusionError(
            datafusion::error::DataFusionError::Plan("No data found".to_string()),
        ))
    }
}

/// Applies a natural logarithm transformation to the values in the columns.
/// Requires all values to be positive.
pub struct LogTransformer {
    pub columns: Vec<String>,
}

impl LogTransformer {
    pub fn new(columns: Vec<String>) -> Self {
        Self { columns }
    }

    /// Checks that each target column is numeric (Float64) and that all its values are positive.
    pub async fn fit(&mut self, df: &DataFrame) -> FeatureFactoryResult<()> {
        for col_name in &self.columns {
            let field = df.schema().field_with_name(None, col_name).map_err(|_| {
                FeatureFactoryError::MissingColumn(format!("Column '{}' not found", col_name))
            })?;
            if field.data_type() != &datafusion::arrow::datatypes::DataType::Float64 {
                return Err(FeatureFactoryError::InvalidParameter(format!(
                    "LogTransformer requires column '{}' to be Float64",
                    col_name
                )));
            }
            let min_val = compute_min(df, col_name).await?;
            if min_val <= 0.0 {
                return Err(FeatureFactoryError::InvalidParameter(format!(
                    "LogTransformer requires all values in column '{}' to be positive, found minimum {}",
                    col_name, min_val
                )));
            }
        }
        Ok(())
    }

    pub fn transform(&self, df: DataFrame) -> FeatureFactoryResult<DataFrame> {
        let exprs: Vec<Expr> = df
            .schema()
            .fields()
            .iter()
            .map(|field| {
                let name = field.name();
                if self.columns.contains(name) {
                    ln_expr(col(name)).alias(name)
                } else {
                    col(name)
                }
            })
            .collect();
        df.select(exprs).map_err(FeatureFactoryError::from)
    }
}

/// Applies a logarithmic transformation to the values in the columns.
/// Given a constant c, the transformation is log(x + c) where x is the original value (in the column).
/// It Requires all values in the columns to be positive.
pub struct LogCpTransformer {
    pub columns: Vec<String>,
    pub constant: f64,
}

impl LogCpTransformer {
    pub fn new(columns: Vec<String>, constant: f64) -> Self {
        Self { columns, constant }
    }

    /// Checks that for each target column, (min + constant) > 0.
    pub async fn fit(&mut self, df: &DataFrame) -> FeatureFactoryResult<()> {
        for col_name in &self.columns {
            let field = df.schema().field_with_name(None, col_name).map_err(|_| {
                FeatureFactoryError::MissingColumn(format!("Column '{}' not found", col_name))
            })?;
            if field.data_type() != &datafusion::arrow::datatypes::DataType::Float64 {
                return Err(FeatureFactoryError::InvalidParameter(format!(
                    "LogCpTransformer requires column '{}' to be Float64",
                    col_name
                )));
            }
            let min_val = compute_min(df, col_name).await?;
            if min_val + self.constant <= 0.0 {
                return Err(FeatureFactoryError::InvalidParameter(format!(
                    "LogCpTransformer requires (min + constant) > 0 for column '{}', but min {} + constant {} = {}",
                    col_name, min_val, self.constant, min_val + self.constant
                )));
            }
        }
        Ok(())
    }

    pub fn transform(&self, df: DataFrame) -> FeatureFactoryResult<DataFrame> {
        let exprs: Vec<Expr> = df
            .schema()
            .fields()
            .iter()
            .map(|field| {
                let name = field.name();
                if self.columns.contains(name) {
                    ln_expr(col(name).add(lit(self.constant))).alias(name)
                } else {
                    col(name)
                }
            })
            .collect();
        df.select(exprs).map_err(FeatureFactoryError::from)
    }
}

/// Applies a reciprocal transformation to the values in the columns (with the formula 1/x).
/// Requires all values to be non-zero.
pub struct ReciprocalTransformer {
    pub columns: Vec<String>,
}

impl ReciprocalTransformer {
    pub fn new(columns: Vec<String>) -> Self {
        Self { columns }
    }

    /// Checks that each target column is numeric and does not contain a zero.
    pub async fn fit(&mut self, df: &DataFrame) -> FeatureFactoryResult<()> {
        for col_name in &self.columns {
            let field = df.schema().field_with_name(None, col_name).map_err(|_| {
                FeatureFactoryError::MissingColumn(format!("Column '{}' not found", col_name))
            })?;
            if field.data_type() != &datafusion::arrow::datatypes::DataType::Float64 {
                return Err(FeatureFactoryError::InvalidParameter(format!(
                    "ReciprocalTransformer requires column '{}' to be Float64",
                    col_name
                )));
            }
            let min_val = compute_min(df, col_name).await?;
            let max_val = compute_max(df, col_name).await?;
            if min_val <= 0.0 && max_val >= 0.0 {
                return Err(FeatureFactoryError::InvalidParameter(format!(
                    "ReciprocalTransformer requires column '{}' to have no zero values (found range [{}, {}])",
                    col_name, min_val, max_val
                )));
            }
        }
        Ok(())
    }

    pub fn transform(&self, df: DataFrame) -> FeatureFactoryResult<DataFrame> {
        let exprs: Vec<Expr> = df
            .schema()
            .fields()
            .iter()
            .map(|field| {
                let name = field.name();
                if self.columns.contains(name) {
                    lit(1.0).div(col(name)).alias(name)
                } else {
                    col(name)
                }
            })
            .collect();
        df.select(exprs).map_err(FeatureFactoryError::from)
    }
}

/// Applies a power transformation to the values in the columns (with the formula x^power).
/// Requires all values to be numeric.
pub struct PowerTransformer {
    pub columns: Vec<String>,
    pub power: f64,
}

impl PowerTransformer {
    pub fn new(columns: Vec<String>, power: f64) -> Self {
        Self { columns, power }
    }

    /// Checks that each target column is numeric.
    pub async fn fit(&mut self, df: &DataFrame) -> FeatureFactoryResult<()> {
        for col_name in &self.columns {
            df.schema().field_with_name(None, col_name).map_err(|_| {
                FeatureFactoryError::MissingColumn(format!("Column '{}' not found", col_name))
            })?;
        }
        Ok(())
    }

    pub fn transform(&self, df: DataFrame) -> FeatureFactoryResult<DataFrame> {
        let exprs: Vec<Expr> = df
            .schema()
            .fields()
            .iter()
            .map(|field| {
                let name = field.name();
                if self.columns.contains(name) {
                    power_expr(col(name), self.power).alias(name)
                } else {
                    col(name)
                }
            })
            .collect();
        df.select(exprs).map_err(FeatureFactoryError::from)
    }
}

/// Applies a Box–Cox transformation to the values in the columns (with the formula (x^lambda - 1) / lambda).
/// Requires values to in the columns to be positive.
pub struct BoxCoxTransformer {
    pub columns: Vec<String>,
    pub lambda: f64,
}

impl BoxCoxTransformer {
    pub fn new(columns: Vec<String>, lambda: f64) -> Self {
        Self { columns, lambda }
    }

    /// Checks that each target column is numeric and that all its values are positive.
    pub async fn fit(&mut self, df: &DataFrame) -> FeatureFactoryResult<()> {
        for col_name in &self.columns {
            let field = df.schema().field_with_name(None, col_name).map_err(|_| {
                FeatureFactoryError::MissingColumn(format!("Column '{}' not found", col_name))
            })?;
            if field.data_type() != &datafusion::arrow::datatypes::DataType::Float64 {
                return Err(FeatureFactoryError::InvalidParameter(format!(
                    "BoxCoxTransformer requires column '{}' to be Float64",
                    col_name
                )));
            }
            let min_val = compute_min(df, col_name).await?;
            if min_val <= 0.0 {
                return Err(FeatureFactoryError::InvalidParameter(format!(
                    "BoxCoxTransformer requires all values in column '{}' to be positive, found min {}",
                    col_name, min_val
                )));
            }
        }
        Ok(())
    }

    pub fn transform(&self, df: DataFrame) -> FeatureFactoryResult<DataFrame> {
        let exprs: Vec<Expr> = df
            .schema()
            .fields()
            .iter()
            .map(|field| {
                let name = field.name();
                if self.columns.contains(name) {
                    let expr = if (self.lambda - 0.0).abs() > 1e-6 {
                        power_expr(col(name), self.lambda)
                            .sub(lit(1.0))
                            .div(lit(self.lambda))
                    } else {
                        ln_expr(col(name))
                    };
                    expr.alias(name)
                } else {
                    col(name)
                }
            })
            .collect();
        df.select(exprs).map_err(FeatureFactoryError::from)
    }
}

/// Applies a Yeo–Johnson transformation to values in the columns.
/// The transformation is defined as: (x^lambda - 1) / lambda for lambda != 0, and ln(x + 1) for lambda = 0.
/// Requires all values in the columns to be numeric.
pub struct YeoJohnsonTransformer {
    pub columns: Vec<String>,
    pub lambda: f64,
}

impl YeoJohnsonTransformer {
    pub fn new(columns: Vec<String>, lambda: f64) -> Self {
        Self { columns, lambda }
    }

    /// Checks that each target column is numeric.
    pub async fn fit(&mut self, df: &DataFrame) -> FeatureFactoryResult<()> {
        for col_name in &self.columns {
            df.schema().field_with_name(None, col_name).map_err(|_| {
                FeatureFactoryError::MissingColumn(format!("Column '{}' not found", col_name))
            })?;
        }
        Ok(())
    }

    pub fn transform(&self, df: DataFrame) -> FeatureFactoryResult<DataFrame> {
        let exprs: Vec<Expr> = df
            .schema()
            .fields()
            .iter()
            .map(|field| {
                let name = field.name();
                if self.columns.contains(name) {
                    let pos = if (self.lambda - 0.0).abs() > 1e-6 {
                        power_expr(col(name).add(lit(1.0)), self.lambda)
                            .sub(lit(1.0))
                            .div(lit(self.lambda))
                    } else {
                        ln_expr(col(name).add(lit(1.0)))
                    };
                    let neg = if (self.lambda - 2.0).abs() > 1e-6 {
                        power_expr(lit(1.0).sub(col(name)), 2.0 - self.lambda)
                            .sub(lit(1.0))
                            .div(lit(2.0 - self.lambda))
                            .neg()
                    } else {
                        ln_expr(lit(1.0).sub(col(name))).neg()
                    };
                    let case_expr = Expr::Case(Case {
                        expr: None,
                        when_then_expr: vec![(Box::new(col(name).gt_eq(lit(0.0))), Box::new(pos))],
                        else_expr: Some(Box::new(neg)),
                    });
                    case_expr.alias(name)
                } else {
                    col(name)
                }
            })
            .collect();
        df.select(exprs).map_err(FeatureFactoryError::from)
    }
}

/// Applies an arcsine transformation defined as asin(sqrt(x)) to the values (i.e., x) in the columns.
/// Requires all values in the columns to be between 0 and 1.
pub struct ArcsinTransformer {
    pub columns: Vec<String>,
}

impl ArcsinTransformer {
    pub fn new(columns: Vec<String>) -> Self {
        Self { columns }
    }

    /// Checks that each target column is numeric and that its values are between 0 and 1.
    pub async fn fit(&mut self, df: &DataFrame) -> FeatureFactoryResult<()> {
        for col_name in &self.columns {
            let field = df.schema().field_with_name(None, col_name).map_err(|_| {
                FeatureFactoryError::MissingColumn(format!("Column '{}' not found", col_name))
            })?;
            if field.data_type() != &datafusion::arrow::datatypes::DataType::Float64 {
                return Err(FeatureFactoryError::InvalidParameter(format!(
                    "ArcsinTransformer requires column '{}' to be Float64",
                    col_name
                )));
            }
            let min_val = compute_min(df, col_name).await?;
            let max_val = compute_max(df, col_name).await?;
            if min_val < 0.0 || max_val > 1.0 {
                return Err(FeatureFactoryError::InvalidParameter(format!(
                    "ArcsinTransformer requires all values in column '{}' to be between 0 and 1, found range [{}, {}]",
                    col_name, min_val, max_val
                )));
            }
        }
        Ok(())
    }

    pub fn transform(&self, df: DataFrame) -> FeatureFactoryResult<DataFrame> {
        let exprs: Vec<Expr> = df
            .schema()
            .fields()
            .iter()
            .map(|field| {
                let name = field.name();
                if self.columns.contains(name) {
                    asin_expr(sqrt_expr(col(name))).alias(name)
                } else {
                    col(name)
                }
            })
            .collect();
        df.select(exprs).map_err(FeatureFactoryError::from)
    }
}
