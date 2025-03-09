//! ## Numerical Transformation Transformers
//!
//! This module provides transformers for applying mathematical transformations to numerical features.
//!
//! ### Available Transformers
//!
//! - [`LogTransformer`]: Applies the natural logarithm transformation (requires positive values).
//! - [`LogCpTransformer`]: Applies a logarithmic transformation with a constant (requires values + constant > 0).
//! - [`ReciprocalTransformer`]: Applies the reciprocal transformation (requires non-zero values).
//! - [`PowerTransformer`]: Applies a power transformation with a specified exponent.
//! - [`BoxCoxTransformer`]: Applies the Box-Cox transformation (requires positive values).
//! - [`YeoJohnsonTransformer`]: Applies the Yeo-Johnson transformation (supports all real numbers).
//! - [`ArcsinTransformer`]: Applies the arcsine transformation (commonly used for proportions).
//!
//! Each transformer returns a new DataFrame with transformed features.
//! Errors are returned as [`FeatureFactoryError`], and results are wrapped in [`FeatureFactoryResult`].

use crate::exceptions::{FeatureFactoryError, FeatureFactoryResult};
use crate::impl_transformer;
use datafusion::dataframe::DataFrame;
use datafusion::functions_aggregate::approx_percentile_cont::approx_percentile_cont;
use datafusion::scalar::ScalarValue;
use datafusion_expr::{col, lit, Expr};
use datafusion_functions::math;
use std::ops::{Add, Div, Neg, Sub};

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

/// Helper function to compute the minimum value in a numeric column using approximate percentiles (p=0).
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

/// Applies natural logarithm transformation to the values in the columns.
/// Needs all values to be positive.
pub struct LogTransformer {
    pub columns: Vec<String>,
}

impl LogTransformer {
    pub fn new(columns: Vec<String>) -> Self {
        Self { columns }
    }

    /// Stateless transformer: fit does nothing.
    pub async fn fit(&mut self, _df: &DataFrame) -> FeatureFactoryResult<()> {
        Ok(())
    }

    /// Validates that each target column exists, is Float64, and that the minimum value is > 0.
    fn validate(&self, df: &DataFrame) -> FeatureFactoryResult<()> {
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
            // Compute min value.
            let min_val = futures::executor::block_on(compute_min(df, col_name))?;
            if min_val <= 0.0 {
                return Err(FeatureFactoryError::InvalidParameter(format!(
                    "LogTransformer requires all values in column '{}' to be positive, found min {}",
                    col_name, min_val
                )));
            }
        }
        Ok(())
    }

    pub fn transform(&self, df: DataFrame) -> FeatureFactoryResult<DataFrame> {
        self.validate(&df)?;
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

    fn inherent_is_stateful(&self) -> bool {
        false
    }
}

/// Applies logarithmic transformation with a constant to the values in the columns.
/// Transformation: log(x + constant). Requires (min + constant) > 0.
pub struct LogCpTransformer {
    pub columns: Vec<String>,
    pub constant: f64,
}

impl LogCpTransformer {
    pub fn new(columns: Vec<String>, constant: f64) -> Self {
        Self { columns, constant }
    }

    /// Stateless transformer: fit does nothing.
    pub async fn fit(&mut self, _df: &DataFrame) -> FeatureFactoryResult<()> {
        Ok(())
    }

    /// Validates that each target column exists, is Float64, and that (min + constant) > 0.
    fn validate(&self, df: &DataFrame) -> FeatureFactoryResult<()> {
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
            let min_val = futures::executor::block_on(compute_min(df, col_name))?;
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
        self.validate(&df)?;
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

    fn inherent_is_stateful(&self) -> bool {
        false
    }
}

/// Applies reciprocal transformation (1/x) to the values in the columns.
/// Requires that no value is zero.
pub struct ReciprocalTransformer {
    pub columns: Vec<String>,
}

impl ReciprocalTransformer {
    pub fn new(columns: Vec<String>) -> Self {
        Self { columns }
    }

    /// Stateless transformer: fit does nothing.
    pub async fn fit(&mut self, _df: &DataFrame) -> FeatureFactoryResult<()> {
        Ok(())
    }

    /// Validates that each target column exists, is Float64, and that no value is zero.
    fn validate(&self, df: &DataFrame) -> FeatureFactoryResult<()> {
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
            let min_val = futures::executor::block_on(compute_min(df, col_name))?;
            let max_val = futures::executor::block_on(compute_max(df, col_name))?;
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
        self.validate(&df)?;
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

    fn inherent_is_stateful(&self) -> bool {
        false
    }
}

/// Applies power transformation to the values in the columns (x^power).
pub struct PowerTransformer {
    pub columns: Vec<String>,
    pub power: f64,
}

impl PowerTransformer {
    pub fn new(columns: Vec<String>, power: f64) -> Self {
        Self { columns, power }
    }

    /// Stateless transformer: fit does nothing.
    pub async fn fit(&mut self, _df: &DataFrame) -> FeatureFactoryResult<()> {
        Ok(())
    }

    /// Validates that each target column exists.
    fn validate(&self, df: &DataFrame) -> FeatureFactoryResult<()> {
        for col_name in &self.columns {
            df.schema().field_with_name(None, col_name).map_err(|_| {
                FeatureFactoryError::MissingColumn(format!("Column '{}' not found", col_name))
            })?;
        }
        Ok(())
    }

    pub fn transform(&self, df: DataFrame) -> FeatureFactoryResult<DataFrame> {
        self.validate(&df)?;
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

    fn inherent_is_stateful(&self) -> bool {
        false
    }
}

/// Applies Box–Cox transformation to the values in the columns.
/// Transformation: (x^lambda - 1) / lambda for lambda != 0, else ln(x)
/// Needs all values to be positive.
pub struct BoxCoxTransformer {
    pub columns: Vec<String>,
    pub lambda: f64,
}

impl BoxCoxTransformer {
    pub fn new(columns: Vec<String>, lambda: f64) -> Self {
        Self { columns, lambda }
    }

    /// Stateless transformer: fit does nothing.
    pub async fn fit(&mut self, _df: &DataFrame) -> FeatureFactoryResult<()> {
        Ok(())
    }

    /// Validates that each target column exists, is Float64, and that all values are positive.
    fn validate(&self, df: &DataFrame) -> FeatureFactoryResult<()> {
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
            let min_val = futures::executor::block_on(compute_min(df, col_name))?;
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
        self.validate(&df)?;
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

    fn inherent_is_stateful(&self) -> bool {
        false
    }
}

/// Applies Yeo–Johnson transformation to the values in the columns.
/// For x >= 0: ( (x + 1)^lambda - 1) / lambda for lambda != 0, else ln(x + 1)
/// and for x < 0: -((1 - x)^(2 - lambda) - 1) / (2 - lambda) for lambda != 2, else -ln(1 - x)
pub struct YeoJohnsonTransformer {
    pub columns: Vec<String>,
    pub lambda: f64,
}

impl YeoJohnsonTransformer {
    pub fn new(columns: Vec<String>, lambda: f64) -> Self {
        Self { columns, lambda }
    }

    /// Stateless transformer: fit does nothing.
    pub async fn fit(&mut self, _df: &DataFrame) -> FeatureFactoryResult<()> {
        Ok(())
    }

    /// Validates that each target column exists.
    fn validate(&self, df: &DataFrame) -> FeatureFactoryResult<()> {
        for col_name in &self.columns {
            df.schema().field_with_name(None, col_name).map_err(|_| {
                FeatureFactoryError::MissingColumn(format!("Column '{}' not found", col_name))
            })?;
        }
        Ok(())
    }

    pub fn transform(&self, df: DataFrame) -> FeatureFactoryResult<DataFrame> {
        self.validate(&df)?;
        let exprs: Vec<Expr> = df
            .schema()
            .fields()
            .iter()
            .map(|field| {
                let name = field.name();
                if self.columns.contains(name) {
                    let pos_expr = if (self.lambda - 0.0).abs() > 1e-6 {
                        power_expr(col(name).add(lit(1.0)), self.lambda)
                            .sub(lit(1.0))
                            .div(lit(self.lambda))
                    } else {
                        ln_expr(col(name).add(lit(1.0)))
                    };
                    let neg_expr = if (self.lambda - 2.0).abs() > 1e-6 {
                        power_expr(lit(1.0).sub(col(name)), 2.0 - self.lambda)
                            .sub(lit(1.0))
                            .div(lit(2.0 - self.lambda))
                            .neg()
                    } else {
                        ln_expr(lit(1.0).sub(col(name))).neg()
                    };
                    let case_expr = Expr::Case(datafusion_expr::expr::Case {
                        expr: None,
                        when_then_expr: vec![(
                            Box::new(col(name).gt_eq(lit(0.0))),
                            Box::new(pos_expr),
                        )],
                        else_expr: Some(Box::new(neg_expr)),
                    });
                    case_expr.alias(name)
                } else {
                    col(name)
                }
            })
            .collect();
        df.select(exprs).map_err(FeatureFactoryError::from)
    }

    fn inherent_is_stateful(&self) -> bool {
        false
    }
}

/// Applies an arcsine transformation defined as asin(sqrt(x)) to the values in the columns.
/// Needs all values to be between 0 and 1.
pub struct ArcsinTransformer {
    pub columns: Vec<String>,
}

impl ArcsinTransformer {
    pub fn new(columns: Vec<String>) -> Self {
        Self { columns }
    }

    /// Stateless transformer: fit does nothing.
    pub async fn fit(&mut self, _df: &DataFrame) -> FeatureFactoryResult<()> {
        Ok(())
    }

    /// Validates that each target column exists, is Float64, and that all values are between 0 and 1.
    fn validate(&self, df: &DataFrame) -> FeatureFactoryResult<()> {
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
            let min_val = futures::executor::block_on(compute_min(df, col_name))?;
            let max_val = futures::executor::block_on(compute_max(df, col_name))?;
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
        self.validate(&df)?;
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

    fn inherent_is_stateful(&self) -> bool {
        false
    }
}

// Implement the Transformer trait for the transformers in this module.
impl_transformer!(LogTransformer);
impl_transformer!(LogCpTransformer);
impl_transformer!(ReciprocalTransformer);
impl_transformer!(PowerTransformer);
impl_transformer!(BoxCoxTransformer);
impl_transformer!(YeoJohnsonTransformer);
impl_transformer!(ArcsinTransformer);
