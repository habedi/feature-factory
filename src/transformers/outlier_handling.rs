//! ## Transformers for handling outliers
//!
//! This module provides several transformers for dealing with outliers.
//!
//! Currently, the following transformers are implemented:
//!
//! - **ArbitraryOutlierCapper:** Cap outliers at user‑defined lower and upper bounds.
//! - **Winsorizer:** Cap outliers based on percentile thresholds.
//! - **OutlierTrimmer:** Remove rows with outlier values based on percentile thresholds.
//!
//! Capping outliers is a common technique to prevent extreme values from skewing the distribution of a dataset.
//! It is done by setting a maximum and minimum value for values that exceed a certain threshold or fall below a certain threshold (e.g., 1st and 99th percentiles).
//! Each transformer returns a new DataFrame with the applied encodings to the specified columns.
//! Errors are returned as `FeatureFactoryError` and results are wrapped in `FeatureFactoryResult`.

use crate::exceptions::{FeatureFactoryError, FeatureFactoryResult};
use datafusion::functions_aggregate::expr_fn::approx_percentile_cont;
use datafusion::logical_expr::{col, lit, Case as DFCase, Expr};
use datafusion::prelude::*;
use datafusion::scalar::ScalarValue;
use std::collections::HashMap;

/// Helper function to build a CASE expression for capping values.
/// Depending on the provided lower and upper bounds, this function returns a CASE expression:
/// - If both bounds are provided:
///   `CASE WHEN col < lower THEN lower WHEN col > upper THEN upper ELSE col END`
/// - If only one bound is provided, the corresponding condition is applied.
/// - If no bounds are provided, returns the original column.
fn cap_expr_for(col_name: &str, lower: Option<f64>, upper: Option<f64>) -> Expr {
    let base = col(col_name);
    match (lower, upper) {
        (Some(l), Some(u)) => Expr::Case(DFCase {
            expr: None,
            when_then_expr: vec![
                (Box::new(base.clone().lt(lit(l))), Box::new(lit(l))),
                (Box::new(base.clone().gt(lit(u))), Box::new(lit(u))),
            ],
            else_expr: Some(Box::new(base)),
        }),
        (Some(l), None) => Expr::Case(DFCase {
            expr: None,
            when_then_expr: vec![(Box::new(base.clone().lt(lit(l))), Box::new(lit(l)))],
            else_expr: Some(Box::new(base)),
        }),
        (None, Some(u)) => Expr::Case(DFCase {
            expr: None,
            when_then_expr: vec![(Box::new(base.clone().gt(lit(u))), Box::new(lit(u)))],
            else_expr: Some(Box::new(base)),
        }),
        (None, None) => base,
    }
}

/// Helper function to compute percentile thresholds for a given column.
/// Returns a tuple (lower_threshold, upper_threshold) based on the provided percentile levels.
/// The percentile values must be between 0 and 1 and lower_percentile must be less than upper_percentile.
async fn compute_percentiles_for_column(
    df: &DataFrame,
    col_name: &str,
    lower_percentile: f64,
    upper_percentile: f64,
) -> FeatureFactoryResult<(f64, f64)> {
    // Validate percentile inputs.
    if !(0.0..=1.0).contains(&lower_percentile) {
        return Err(FeatureFactoryError::InvalidParameter(format!(
            "lower_percentile {} must be between 0 and 1",
            lower_percentile
        )));
    }
    if !(0.0..=1.0).contains(&upper_percentile) {
        return Err(FeatureFactoryError::InvalidParameter(format!(
            "upper_percentile {} must be between 0 and 1",
            upper_percentile
        )));
    }
    if lower_percentile >= upper_percentile {
        return Err(FeatureFactoryError::InvalidParameter(format!(
            "lower_percentile {} must be less than upper_percentile {}",
            lower_percentile, upper_percentile
        )));
    }

    let lower_df = df
        .clone()
        .aggregate(
            vec![],
            vec![approx_percentile_cont(col(col_name), lit(lower_percentile), None).alias("lower")],
        )
        .map_err(FeatureFactoryError::DataFusionError)?;
    let lower_batches = lower_df
        .collect()
        .await
        .map_err(FeatureFactoryError::DataFusionError)?;
    let lower = if let Some(batch) = lower_batches.first() {
        let array = batch.column(0);
        let scalar = ScalarValue::try_from_array(array, 0).map_err(|e| {
            FeatureFactoryError::DataFusionError(datafusion::error::DataFusionError::Plan(format!(
                "Error converting lower percentile for column {}: {}",
                col_name, e
            )))
        })?;
        if let ScalarValue::Float64(Some(val)) = scalar {
            val
        } else {
            return Err(FeatureFactoryError::DataFusionError(
                datafusion::error::DataFusionError::Plan(format!(
                    "Failed to compute lower percentile for column {}",
                    col_name
                )),
            ));
        }
    } else {
        return Err(FeatureFactoryError::DataFusionError(
            datafusion::error::DataFusionError::Plan(format!(
                "No data found when computing lower percentile for column {}",
                col_name
            )),
        ));
    };

    let upper_df = df
        .clone()
        .aggregate(
            vec![],
            vec![approx_percentile_cont(col(col_name), lit(upper_percentile), None).alias("upper")],
        )
        .map_err(FeatureFactoryError::DataFusionError)?;
    let upper_batches = upper_df
        .collect()
        .await
        .map_err(FeatureFactoryError::DataFusionError)?;
    let upper = if let Some(batch) = upper_batches.first() {
        let array = batch.column(0);
        let scalar = ScalarValue::try_from_array(array, 0).map_err(|e| {
            FeatureFactoryError::DataFusionError(datafusion::error::DataFusionError::Plan(format!(
                "Error converting upper percentile for column {}: {}",
                col_name, e
            )))
        })?;
        if let ScalarValue::Float64(Some(val)) = scalar {
            val
        } else {
            return Err(FeatureFactoryError::DataFusionError(
                datafusion::error::DataFusionError::Plan(format!(
                    "Failed to compute upper percentile for column {}",
                    col_name
                )),
            ));
        }
    } else {
        return Err(FeatureFactoryError::DataFusionError(
            datafusion::error::DataFusionError::Plan(format!(
                "No data found when computing upper percentile for column {}",
                col_name
            )),
        ));
    };

    Ok((lower, upper))
}

/// Caps outliers by applying user‑defined lower and upper bounds.
pub struct ArbitraryOutlierCapper {
    pub columns: Vec<String>,
    pub lower_caps: HashMap<String, f64>,
    pub upper_caps: HashMap<String, f64>,
}

impl ArbitraryOutlierCapper {
    /// Create a new ArbitraryOutlierCapper.
    pub fn new(
        columns: Vec<String>,
        lower_caps: HashMap<String, f64>,
        upper_caps: HashMap<String, f64>,
    ) -> Self {
        Self {
            columns,
            lower_caps,
            upper_caps,
        }
    }

    /// This transformer is stateless, so fit does nothing.
    pub async fn fit(&mut self, _df: &DataFrame) -> FeatureFactoryResult<()> {
        Ok(())
    }

    /// Transform the DataFrame by capping each target column at the user‑defined bounds.
    pub fn transform(&self, df: DataFrame) -> FeatureFactoryResult<DataFrame> {
        let exprs: Vec<Expr> = df
            .schema()
            .fields()
            .iter()
            .map(|field| {
                let name = field.name();
                if self.columns.contains(&name.to_string()) {
                    let lower = self.lower_caps.get(name).cloned();
                    let upper = self.upper_caps.get(name).cloned();
                    cap_expr_for(name, lower, upper).alias(name)
                } else {
                    col(name)
                }
            })
            .collect();
        df.select(exprs)
            .map_err(FeatureFactoryError::DataFusionError)
    }
}

/// Caps outliers based on percentile thresholds.
/// The percentile values must be between 0 and 1 and lower_percentile must be less than upper_percentile.
pub struct Winsorizer {
    pub columns: Vec<String>,
    pub lower_percentile: f64,
    pub upper_percentile: f64,
    pub thresholds: HashMap<String, (f64, f64)>,
}

impl Winsorizer {
    /// Create a new Winsorizer.
    pub fn new(columns: Vec<String>, lower_percentile: f64, upper_percentile: f64) -> Self {
        Self {
            columns,
            lower_percentile,
            upper_percentile,
            thresholds: HashMap::new(),
        }
    }

    /// Fit the winsorizer by computing percentile thresholds for each target column.
    pub async fn fit(&mut self, df: &DataFrame) -> FeatureFactoryResult<()> {
        // Validate percentile inputs.
        if self.lower_percentile < 0.0 || self.lower_percentile > 1.0 {
            return Err(FeatureFactoryError::InvalidParameter(format!(
                "lower_percentile {} must be between 0 and 1",
                self.lower_percentile
            )));
        }
        if self.upper_percentile < 0.0 || self.upper_percentile > 1.0 {
            return Err(FeatureFactoryError::InvalidParameter(format!(
                "upper_percentile {} must be between 0 and 1",
                self.upper_percentile
            )));
        }
        if self.lower_percentile >= self.upper_percentile {
            return Err(FeatureFactoryError::InvalidParameter(format!(
                "lower_percentile {} must be less than upper_percentile {}",
                self.lower_percentile, self.upper_percentile
            )));
        }

        for col_name in &self.columns {
            let (lower, upper) = compute_percentiles_for_column(
                df,
                col_name,
                self.lower_percentile,
                self.upper_percentile,
            )
            .await?;
            self.thresholds.insert(col_name.clone(), (lower, upper));
        }
        Ok(())
    }

    /// Returns a new DataFrame where each target column is capped using the computed thresholds.
    pub fn transform(&self, df: DataFrame) -> FeatureFactoryResult<DataFrame> {
        let exprs: Vec<Expr> = df
            .schema()
            .fields()
            .iter()
            .map(|field| {
                let name = field.name();
                if self.columns.contains(&name.to_string()) {
                    if let Some(&(lower, upper)) = self.thresholds.get(name) {
                        cap_expr_for(name, Some(lower), Some(upper)).alias(name)
                    } else {
                        col(name)
                    }
                } else {
                    col(name)
                }
            })
            .collect();
        df.select(exprs)
            .map_err(FeatureFactoryError::DataFusionError)
    }
}

/// Removes rows with outlier values based on percentile thresholds.
/// The percentile values must be between 0 and 1 and lower_percentile must be less than upper_percentile.
pub struct OutlierTrimmer {
    pub columns: Vec<String>,
    pub lower_percentile: f64,
    pub upper_percentile: f64,
    pub thresholds: HashMap<String, (f64, f64)>,
}

impl OutlierTrimmer {
    /// Create a new OutlierTrimmer.
    pub fn new(columns: Vec<String>, lower_percentile: f64, upper_percentile: f64) -> Self {
        Self {
            columns,
            lower_percentile,
            upper_percentile,
            thresholds: HashMap::new(),
        }
    }

    /// Fit the trimmer by computing percentile thresholds for each target column.
    pub async fn fit(&mut self, df: &DataFrame) -> FeatureFactoryResult<()> {
        // Validate percentile inputs.
        if self.lower_percentile < 0.0 || self.lower_percentile > 1.0 {
            return Err(FeatureFactoryError::InvalidParameter(format!(
                "lower_percentile {} must be between 0 and 1",
                self.lower_percentile
            )));
        }
        if self.upper_percentile < 0.0 || self.upper_percentile > 1.0 {
            return Err(FeatureFactoryError::InvalidParameter(format!(
                "upper_percentile {} must be between 0 and 1",
                self.upper_percentile
            )));
        }
        if self.lower_percentile >= self.upper_percentile {
            return Err(FeatureFactoryError::InvalidParameter(format!(
                "lower_percentile {} must be less than upper_percentile {}",
                self.lower_percentile, self.upper_percentile
            )));
        }

        for col_name in &self.columns {
            let (lower, upper) = compute_percentiles_for_column(
                df,
                col_name,
                self.lower_percentile,
                self.upper_percentile,
            )
            .await?;
            self.thresholds.insert(col_name.clone(), (lower, upper));
        }
        Ok(())
    }

    /// Returns a new DataFrame with rows dropped if any target column has a value outside the computed thresholds.
    pub fn transform(&self, df: DataFrame) -> FeatureFactoryResult<DataFrame> {
        let predicates: Vec<Expr> = df
            .schema()
            .fields()
            .iter()
            .filter_map(|field| {
                let name = field.name();
                if self.columns.contains(&name.to_string()) {
                    if let Some(&(lower, upper)) = self.thresholds.get(name) {
                        Some(col(name).gt_eq(lit(lower)).and(col(name).lt_eq(lit(upper))))
                    } else {
                        None
                    }
                } else {
                    None
                }
            })
            .collect();
        if predicates.is_empty() {
            return Ok(df);
        }
        let combined = predicates
            .into_iter()
            .reduce(|acc, expr| acc.and(expr))
            .ok_or_else(|| {
                FeatureFactoryError::DataFusionError(datafusion::error::DataFusionError::Plan(
                    "Failed to combine predicates".into(),
                ))
            })?;
        df.filter(combined)
            .map_err(FeatureFactoryError::DataFusionError)
    }
}
