//! # Feature Factory Imputation Transformers
//!
//! This module implements various imputation transforms for a DataFusion‑based
//! feature engineering library. It supports imputation strategies for numeric
//! and categorical data as well as operations for adding missing value indicators
//! and dropping rows with missing data.
//!
//! ## Imputation Strategies
//!
//! - **MeanMedianImputer**: Imputes numeric columns using the mean (median is not implemented in lazy mode).
//! - **ArbitraryNumberImputer**: Imputes numeric columns using a fixed arbitrary number.
//! - **EndTailImputer**: Imputes numeric columns using a percentile value computed from the data (e.g. for tail imputation).
//! - **CategoricalImputer**: Imputes categorical columns using the mode (or a provided default).
//! - **AddMissingIndicator**: Adds additional Boolean indicator columns for missing values.
//! - **DropMissingData**: Filters out rows that contain any missing data.
//!
//! Each transform returns a new DataFrame with the applied imputation logic.
//! Errors from underlying DataFusion operations are wrapped into a `FeatureFactoryError` from the `exceptions` module.

use crate::exceptions::{FeatureFactoryError, FeatureFactoryResult};
use datafusion::functions_aggregate::expr_fn::{approx_percentile_cont, avg, count};
use datafusion::logical_expr::{col, lit, not, Case as DFCase, Expr};
use datafusion::prelude::*;
use datafusion::scalar::ScalarValue;
use std::collections::HashMap;

/// Constructs an expression equivalent to SQL COALESCE(col, fallback).
/// This is implemented as a CASE expression: if `col` is not null then return it, otherwise return `fallback`.
fn coalesce_expr_for(name: &str, fallback: Expr) -> Expr {
    Expr::Case(DFCase {
        expr: None,
        when_then_expr: vec![(Box::new(not(col(name).is_null())), Box::new(col(name)))],
        else_expr: Some(Box::new(fallback)),
    })
}

/// Generic helper to apply a mapping to a set of target columns.
/// For each field in the DataFrame, if its name is in `target_cols` and a mapping is available via `get_fallback`,
/// then the column is replaced by a CASE–WHEN expression; otherwise, the original column is retained.
/// The `default_fn` closure produces a default expression for a given column name.
fn apply_imputation<F>(
    df: DataFrame,
    target_cols: &[String],
    get_fallback: F,
) -> FeatureFactoryResult<DataFrame>
where
    F: Fn(&str) -> Option<Expr>,
{
    let exprs: Vec<Expr> = df
        .schema()
        .fields()
        .iter()
        .map(|field| {
            let name = field.name();
            if target_cols.contains(name) {
                if let Some(fallback_expr) = get_fallback(name) {
                    coalesce_expr_for(name, fallback_expr).alias(name)
                } else {
                    col(name)
                }
            } else {
                col(name)
            }
        })
        .collect();
    df.select(exprs).map_err(FeatureFactoryError::from)
}

/// ------------------------- MeanMedianImputer -------------------------
#[derive(Debug, Clone, Copy)]
pub enum ImputeStrategy {
    Mean,
    Median, // Not implemented in lazy mode.
}

/// Imputer for numeric columns that computes a statistic (currently mean) and replaces nulls.
/// When the strategy is `Mean`, the imputer aggregates the data to compute the average value for each target column
/// and uses that as the fallback value.
pub struct MeanMedianImputer {
    pub columns: Vec<String>,
    pub strategy: ImputeStrategy,
    pub impute_values: HashMap<String, f64>,
}

impl MeanMedianImputer {
    /// Create a new imputer for the given columns and strategy.
    pub fn new(columns: Vec<String>, strategy: ImputeStrategy) -> Self {
        Self {
            columns,
            strategy,
            impute_values: HashMap::new(),
        }
    }

    /// For each target column, compute the mean value via an aggregate query.
    pub async fn fit(&mut self, df: &DataFrame) -> FeatureFactoryResult<()> {
        for col_name in &self.columns {
            match self.strategy {
                ImputeStrategy::Mean => {
                    let agg_df = df
                        .clone()
                        .aggregate(vec![], vec![avg(col(col_name)).alias("avg")])
                        .map_err(FeatureFactoryError::from)?;
                    let batches = agg_df.collect().await.map_err(FeatureFactoryError::from)?;
                    if let Some(batch) = batches.first() {
                        if batch.num_rows() > 0 {
                            let array = batch.column(0);
                            let scalar = ScalarValue::try_from_array(array, 0)
                                .map_err(FeatureFactoryError::from)?;
                            if let ScalarValue::Float64(Some(avg_val)) = scalar {
                                self.impute_values.insert(col_name.clone(), avg_val);
                            } else {
                                return Err(FeatureFactoryError::DataFusionError(
                                    datafusion::error::DataFusionError::Plan(format!(
                                        "Failed to compute average for column {}",
                                        col_name
                                    )),
                                ));
                            }
                        }
                    }
                }
                ImputeStrategy::Median => {
                    return Err(FeatureFactoryError::NotImplemented(
                        "Median imputation not implemented in DF mode".to_string(),
                    ));
                }
            }
        }
        Ok(())
    }

    /// Returns a new DataFrame where, for each target column, nulls are replaced with the computed mean.
    pub fn transform(&self, df: DataFrame) -> FeatureFactoryResult<DataFrame> {
        apply_imputation(df, &self.columns, |name| {
            self.impute_values.get(name).map(|&v| lit(v))
        })
    }
}

/// ------------------------- ArbitraryNumberImputer -------------------------
pub struct ArbitraryNumberImputer {
    pub columns: Vec<String>,
    pub number: f64,
}

impl ArbitraryNumberImputer {
    /// Create a new arbitrary number imputer for the given columns.
    pub fn new(columns: Vec<String>, number: f64) -> Self {
        Self { columns, number }
    }

    /// Returns a new DataFrame where, for each target column, nulls are replaced with the fixed number.
    pub fn transform(&self, df: DataFrame) -> FeatureFactoryResult<DataFrame> {
        apply_imputation(df, &self.columns, |_| Some(lit(self.number)))
    }
}

/// ------------------------- EndTailImputer -------------------------
pub struct EndTailImputer {
    pub columns: Vec<String>,
    pub percentile: f64,
    pub impute_values: HashMap<String, f64>,
}

impl EndTailImputer {
    /// Create a new end-tail imputer for the given columns and percentile.
    pub fn new(columns: Vec<String>, percentile: f64) -> Self {
        Self {
            columns,
            percentile,
            impute_values: HashMap::new(),
        }
    }

    /// For each target column, compute the approximate percentile value via an aggregate query.
    pub async fn fit(&mut self, df: &DataFrame) -> FeatureFactoryResult<()> {
        for col_name in &self.columns {
            let agg_df = df
                .clone()
                .aggregate(
                    vec![],
                    vec![
                        approx_percentile_cont(col(col_name), lit(self.percentile), None)
                            .alias("perc"),
                    ],
                )
                .map_err(FeatureFactoryError::from)?;
            let batches = agg_df.collect().await.map_err(FeatureFactoryError::from)?;
            if let Some(batch) = batches.first() {
                let array = batch.column(0);
                let scalar =
                    ScalarValue::try_from_array(array, 0).map_err(FeatureFactoryError::from)?;
                if let ScalarValue::Float64(Some(val)) = scalar {
                    self.impute_values.insert(col_name.clone(), val);
                } else {
                    return Err(FeatureFactoryError::DataFusionError(
                        datafusion::error::DataFusionError::Plan(format!(
                            "Failed to compute percentile for column {}",
                            col_name
                        )),
                    ));
                }
            }
        }
        Ok(())
    }

    /// Returns a new DataFrame where, for each target column, nulls are replaced with the computed percentile.
    pub fn transform(&self, df: DataFrame) -> FeatureFactoryResult<DataFrame> {
        apply_imputation(df, &self.columns, |name| {
            self.impute_values.get(name).map(|&v| lit(v))
        })
    }
}

/// ------------------------- CategoricalImputer -------------------------
pub struct CategoricalImputer {
    pub columns: Vec<String>,
    pub default: Option<String>,
    pub impute_values: HashMap<String, String>,
}

impl CategoricalImputer {
    /// Create a new categorical imputer for the given columns and optional default.
    pub fn new(columns: Vec<String>, default: Option<String>) -> Self {
        Self {
            columns,
            default,
            impute_values: HashMap::new(),
        }
    }

    /// For each target column, if no default is provided, compute the mode via grouping and counting.
    pub async fn fit(&mut self, df: &DataFrame) -> FeatureFactoryResult<()> {
        if self.default.is_some() {
            return Ok(());
        }
        for col_name in &self.columns {
            let grouped = df
                .clone()
                .aggregate(vec![col(col_name)], vec![count(col(col_name)).alias("cnt")])
                .map_err(FeatureFactoryError::from)?
                .sort(vec![col("cnt").sort(false, false)])
                .map_err(FeatureFactoryError::from)?
                .limit(0, Some(1))
                .map_err(FeatureFactoryError::from)?;
            let batches = grouped.collect().await.map_err(FeatureFactoryError::from)?;
            if let Some(batch) = batches.first() {
                let array = batch.column(0);
                let scalar =
                    ScalarValue::try_from_array(array, 0).map_err(FeatureFactoryError::from)?;
                if let ScalarValue::Utf8(Some(mode_val)) = scalar {
                    self.impute_values.insert(col_name.clone(), mode_val);
                } else {
                    return Err(FeatureFactoryError::DataFusionError(
                        datafusion::error::DataFusionError::Plan(format!(
                            "Failed to compute mode for column {}",
                            col_name
                        )),
                    ));
                }
            }
        }
        Ok(())
    }

    /// Returns a new DataFrame where, for each target column, nulls are replaced with the computed mode (or the provided default).
    pub fn transform(&self, df: DataFrame) -> FeatureFactoryResult<DataFrame> {
        apply_imputation(df, &self.columns, |name| {
            if let Some(default_val) = &self.default {
                Some(lit(default_val.clone()))
            } else {
                self.impute_values
                    .get(name)
                    .map(|mode_val| lit(mode_val.clone()))
            }
        })
    }
}

/// ------------------------- AddMissingIndicator -------------------------
pub struct AddMissingIndicator {
    pub columns: Vec<String>,
    pub suffix: String,
}

impl AddMissingIndicator {
    /// Create a new missing indicator transformer for the given columns.
    pub fn new(columns: Vec<String>, suffix: Option<String>) -> Self {
        Self {
            columns,
            suffix: suffix.unwrap_or_else(|| "_missing".to_string()),
        }
    }

    /// Returns a new DataFrame with additional Boolean indicator columns for missing values.
    pub fn transform(&self, df: DataFrame) -> FeatureFactoryResult<DataFrame> {
        let mut exprs = vec![];
        for field in df.schema().fields() {
            let name = field.name();
            exprs.push(col(name));
            if self.columns.contains(name) {
                exprs.push(
                    col(name)
                        .is_null()
                        .alias(format!("{}{}", name, self.suffix)),
                );
            }
        }
        df.select(exprs).map_err(FeatureFactoryError::from)
    }
}

/// ------------------------- DropMissingData -------------------------
pub struct DropMissingData;

impl DropMissingData {
    /// Create a new drop-missing-data transformer.
    pub fn new() -> Self {
        Self {}
    }

    /// Returns a new DataFrame that excludes rows with any null values.
    pub fn transform(&self, df: DataFrame) -> FeatureFactoryResult<DataFrame> {
        let predicates: Vec<Expr> = df
            .schema()
            .fields()
            .iter()
            .map(|field| col(field.name()).is_not_null())
            .collect();
        let combined = predicates
            .into_iter()
            .reduce(|acc, expr| acc.and(expr))
            .unwrap();
        df.filter(combined).map_err(FeatureFactoryError::from)
    }
}

impl Default for DropMissingData {
    fn default() -> Self {
        Self::new()
    }
}
