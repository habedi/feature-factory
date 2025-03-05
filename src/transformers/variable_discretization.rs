//! # Variable Discretization Transformers
//!
//! This module provides various discretization (binning) strategies to transform a continuous
//! variable into a categorical one. The supported discretizers are:
//!
//! - **ArbitraryDiscretizer:** Discretizes based on user‑defined intervals.
//! - **EqualFrequencyDiscretizer:** Discretizes a column into bins containing roughly equal numbers of records.
//! - **EqualWidthDiscretizer:** Discretizes a column into bins of equal width.
//! - **GeometricWidthDiscretizer:** Discretizes a column into bins whose boundaries follow a geometric progression.
//!
//! Each discretizer exposes a similar API with a constructor, an asynchronous `fit` method to learn
//! the bin boundaries (if applicable), and a `transform` method that applies the discretization
//! to a DataFrame. Errors from underlying DataFusion operations are wrapped into a `FeatureFactoryError`
//! and returned as a `FeatureFactoryResult` (defined in the `exceptions` module).

use crate::exceptions::{FeatureFactoryError, FeatureFactoryResult};
use datafusion::functions_aggregate::expr_fn::approx_percentile_cont;
use datafusion::logical_expr::{col, lit, Case as DFCase, Expr};
use datafusion::prelude::*;
use datafusion::scalar::ScalarValue;
use std::collections::HashMap;

/// Helper to build a CASE expression for intervals.
/// For each interval in `intervals` (tuple: lower, upper, label), it generates a condition:
/// For the first (n-1) intervals, the condition is:
/// `WHEN col >= lower AND col < upper THEN label`
/// For the last interval, the condition is:
/// `WHEN col >= lower AND col <= upper THEN label`
/// If none match, returns NULL.
fn build_interval_case_expr(col_name: &str, intervals: &[(f64, f64, String)]) -> Expr {
    let n = intervals.len();
    let when_then_expr = intervals
        .iter()
        .enumerate()
        .map(|(i, (lower, upper, label))| {
            let condition = if i == n - 1 {
                col(col_name)
                    .gt_eq(lit(*lower))
                    .and(col(col_name).lt_eq(lit(*upper)))
            } else {
                col(col_name)
                    .gt_eq(lit(*lower))
                    .and(col(col_name).lt(lit(*upper)))
            };
            (Box::new(condition), Box::new(lit(label.clone())))
        })
        .collect::<Vec<_>>();
    Expr::Case(DFCase {
        expr: None,
        when_then_expr,
        else_expr: Some(Box::new(lit(ScalarValue::Utf8(None)))),
    })
}

/// Generic helper that applies an interval mapping to each target column in a DataFrame.
/// For each column in `target_cols`, if a mapping exists in `mapping` then a CASE expression
/// is built; otherwise, the original column is retained.
fn apply_interval_mapping(
    df: DataFrame,
    target_cols: &[String],
    mapping: &HashMap<String, Vec<(f64, f64, String)>>,
) -> FeatureFactoryResult<DataFrame> {
    let exprs: Vec<Expr> = df
        .schema()
        .fields()
        .iter()
        .map(|field| {
            let name = field.name();
            if target_cols.contains(name) {
                if let Some(intervals) = mapping.get(name) {
                    build_interval_case_expr(name, intervals).alias(name)
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

/// Helper function to compute the min and max of a column using approximate percentiles.
/// It uses p=0 for min and p=1 for max.
async fn compute_min_max(df: &DataFrame, col_name: &str) -> FeatureFactoryResult<(f64, f64)> {
    // Compute min
    let min_df = df
        .clone()
        .aggregate(
            vec![],
            vec![approx_percentile_cont(col(col_name), lit(0.0), None).alias("min")],
        )
        .map_err(FeatureFactoryError::from)?;
    let min_batches = min_df.collect().await.map_err(FeatureFactoryError::from)?;
    let min_val = if let Some(batch) = min_batches.first() {
        let array = batch.column(0);
        let scalar = ScalarValue::try_from_array(array, 0).map_err(FeatureFactoryError::from)?;
        if let ScalarValue::Float64(Some(val)) = scalar {
            val
        } else {
            return Err(FeatureFactoryError::DataFusionError(
                datafusion::error::DataFusionError::Plan(format!(
                    "Failed to compute min for column {}",
                    col_name
                )),
            ));
        }
    } else {
        return Err(FeatureFactoryError::DataFusionError(
            datafusion::error::DataFusionError::Plan("No data found".to_string()),
        ));
    };

    // Compute max
    let max_df = df
        .clone()
        .aggregate(
            vec![],
            vec![approx_percentile_cont(col(col_name), lit(1.0), None).alias("max")],
        )
        .map_err(FeatureFactoryError::from)?;
    let max_batches = max_df.collect().await.map_err(FeatureFactoryError::from)?;
    let max_val = if let Some(batch) = max_batches.first() {
        let array = batch.column(0);
        let scalar = ScalarValue::try_from_array(array, 0).map_err(FeatureFactoryError::from)?;
        if let ScalarValue::Float64(Some(val)) = scalar {
            val
        } else {
            return Err(FeatureFactoryError::DataFusionError(
                datafusion::error::DataFusionError::Plan(format!(
                    "Failed to compute max for column {}",
                    col_name
                )),
            ));
        }
    } else {
        return Err(FeatureFactoryError::DataFusionError(
            datafusion::error::DataFusionError::Plan("No data found".to_string()),
        ));
    };

    Ok((min_val, max_val))
}

// ─────────────────────────────────────────────────────────────────────────────
// ArbitraryDiscretizer
// ─────────────────────────────────────────────────────────────────────────────

/// Discretizer based on user‑defined intervals.
/// The user supplies a mapping from column name to a vector of intervals.
/// Each interval is defined as (lower bound, upper bound, label).
pub struct ArbitraryDiscretizer {
    pub columns: Vec<String>,
    pub intervals: HashMap<String, Vec<(f64, f64, String)>>,
}

impl ArbitraryDiscretizer {
    /// Create a new ArbitraryDiscretizer.
    pub fn new(columns: Vec<String>, intervals: HashMap<String, Vec<(f64, f64, String)>>) -> Self {
        Self { columns, intervals }
    }

    /// For arbitrary discretization, no fitting is required.
    pub async fn fit(&mut self, _df: &DataFrame) -> FeatureFactoryResult<()> {
        Ok(())
    }

    /// Transform the DataFrame by applying user‑defined intervals.
    pub fn transform(&self, df: DataFrame) -> FeatureFactoryResult<DataFrame> {
        let mut exprs = vec![];
        for field in df.schema().fields() {
            let name = field.name();
            if self.columns.contains(name) {
                if let Some(intervals) = self.intervals.get(name) {
                    exprs.push(build_interval_case_expr(name, intervals).alias(name));
                } else {
                    exprs.push(col(name));
                }
            } else {
                exprs.push(col(name));
            }
        }
        df.select(exprs).map_err(FeatureFactoryError::from)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// EqualFrequencyDiscretizer
// ─────────────────────────────────────────────────────────────────────────────

/// Discretizer that divides a column into bins of equal frequency.
pub struct EqualFrequencyDiscretizer {
    pub columns: Vec<String>,
    pub bins: usize,
    /// Mapping from column name to vector of intervals: (lower, upper, label)
    pub mapping: HashMap<String, Vec<(f64, f64, String)>>,
}

impl EqualFrequencyDiscretizer {
    /// Create a new EqualFrequencyDiscretizer with the specified number of bins.
    pub fn new(columns: Vec<String>, bins: usize) -> Self {
        Self {
            columns,
            bins,
            mapping: HashMap::new(),
        }
    }

    /// Fit the discretizer by computing quantile boundaries for each column.
    pub async fn fit(&mut self, df: &DataFrame) -> FeatureFactoryResult<()> {
        for col_name in &self.columns {
            let mut boundaries = Vec::with_capacity(self.bins + 1);
            // Compute quantiles at p = 0, 1/bins, …, 1.0.
            for i in 0..=self.bins {
                let p = i as f64 / self.bins as f64;
                let agg_df = df
                    .clone()
                    .aggregate(
                        vec![],
                        vec![approx_percentile_cont(col(col_name), lit(p), None).alias("q")],
                    )
                    .map_err(FeatureFactoryError::from)?;
                let batches = agg_df.collect().await.map_err(FeatureFactoryError::from)?;
                if let Some(batch) = batches.first() {
                    let array = batch.column(0);
                    let scalar =
                        ScalarValue::try_from_array(array, 0).map_err(FeatureFactoryError::from)?;
                    if let ScalarValue::Float64(Some(val)) = scalar {
                        boundaries.push(val);
                    } else {
                        return Err(FeatureFactoryError::DataFusionError(
                            datafusion::error::DataFusionError::Plan(format!(
                                "Failed to compute quantile {} for column {}",
                                p, col_name
                            )),
                        ));
                    }
                }
            }
            // Build intervals as pairs of consecutive boundaries.
            let intervals = boundaries
                .windows(2)
                .map(|pair| {
                    let lower = pair[0];
                    let upper = pair[1];
                    let label = format!("[{:.2}, {:.2})", lower, upper);
                    (lower, upper, label)
                })
                .collect::<Vec<_>>();
            self.mapping.insert(col_name.clone(), intervals);
        }
        Ok(())
    }

    /// Transform the DataFrame by applying the equal-frequency binning.
    pub fn transform(&self, df: DataFrame) -> FeatureFactoryResult<DataFrame> {
        apply_interval_mapping(df, &self.columns, &self.mapping)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// EqualWidthDiscretizer
// ─────────────────────────────────────────────────────────────────────────────

/// Discretizer that divides a column into bins of equal width.
pub struct EqualWidthDiscretizer {
    pub columns: Vec<String>,
    pub bins: usize,
    pub mapping: HashMap<String, Vec<(f64, f64, String)>>,
}

impl EqualWidthDiscretizer {
    /// Create a new EqualWidthDiscretizer with the specified number of bins.
    pub fn new(columns: Vec<String>, bins: usize) -> Self {
        Self {
            columns,
            bins,
            mapping: HashMap::new(),
        }
    }

    /// Fit the discretizer by computing min and max and then equal-width intervals.
    pub async fn fit(&mut self, df: &DataFrame) -> FeatureFactoryResult<()> {
        for col_name in &self.columns {
            let (min_val, max_val) = compute_min_max(df, col_name).await?;
            let width = (max_val - min_val) / self.bins as f64;
            let intervals = (0..self.bins)
                .map(|i| {
                    let lower = min_val + i as f64 * width;
                    let upper = if i == self.bins - 1 {
                        max_val
                    } else {
                        min_val + (i as f64 + 1.0) * width
                    };
                    let label = format!("[{:.2}, {:.2})", lower, upper);
                    (lower, upper, label)
                })
                .collect::<Vec<_>>();
            self.mapping.insert(col_name.clone(), intervals);
        }
        Ok(())
    }

    /// Transform the DataFrame by applying equal-width binning.
    pub fn transform(&self, df: DataFrame) -> FeatureFactoryResult<DataFrame> {
        apply_interval_mapping(df, &self.columns, &self.mapping)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// DecisionTreeDiscretizer
// ─────────────────────────────────────────────────────────────────────────────

/// Discretizer that uses decision tree predictions to discretize a column.
/// Not implemented.
pub struct DecisionTreeDiscretizer {
    pub columns: Vec<String>,
}

impl DecisionTreeDiscretizer {
    /// Create a new DecisionTreeDiscretizer.
    pub fn new(columns: Vec<String>) -> Self {
        Self { columns }
    }

    /// Not implemented.
    pub async fn fit(&mut self, _df: &DataFrame) -> FeatureFactoryResult<()> {
        Err(FeatureFactoryError::NotImplemented(
            "DecisionTreeDiscretizer is not implemented".to_string(),
        ))
    }

    /// Not implemented.
    pub fn transform(&self, _df: DataFrame) -> FeatureFactoryResult<DataFrame> {
        Err(FeatureFactoryError::NotImplemented(
            "DecisionTreeDiscretizer is not implemented".to_string(),
        ))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// GeometricWidthDiscretizer
// ─────────────────────────────────────────────────────────────────────────────

/// Discretizer that divides a column into bins whose boundaries follow a geometric progression.
/// The column values must be positive.
pub struct GeometricWidthDiscretizer {
    pub columns: Vec<String>,
    pub bins: usize,
    pub mapping: HashMap<String, Vec<(f64, f64, String)>>,
}

impl GeometricWidthDiscretizer {
    /// Create a new GeometricWidthDiscretizer with the specified number of bins.
    pub fn new(columns: Vec<String>, bins: usize) -> Self {
        Self {
            columns,
            bins,
            mapping: HashMap::new(),
        }
    }

    /// Fit the discretizer by computing min and max and then generating geometric intervals.
    /// Returns an error if any column has non-positive values.
    pub async fn fit(&mut self, df: &DataFrame) -> FeatureFactoryResult<()> {
        for col_name in &self.columns {
            let (min_val, max_val) = compute_min_max(df, col_name).await?;
            if min_val <= 0.0 {
                return Err(FeatureFactoryError::DataFusionError(
                    datafusion::error::DataFusionError::Plan(format!(
                        "Column {} has non-positive values, cannot apply geometric discretization",
                        col_name
                    )),
                ));
            }
            let ratio = (max_val / min_val).powf(1.0 / self.bins as f64);
            let intervals = (0..self.bins)
                .map(|i| {
                    let lower = min_val * ratio.powi(i as i32);
                    let upper = if i == self.bins - 1 {
                        max_val
                    } else {
                        min_val * ratio.powi((i + 1) as i32)
                    };
                    let label = format!("[{:.2}, {:.2})", lower, upper);
                    (lower, upper, label)
                })
                .collect::<Vec<_>>();
            self.mapping.insert(col_name.clone(), intervals);
        }
        Ok(())
    }

    /// Transform the DataFrame by applying geometric-width binning.
    pub fn transform(&self, df: DataFrame) -> FeatureFactoryResult<DataFrame> {
        apply_interval_mapping(df, &self.columns, &self.mapping)
    }
}
