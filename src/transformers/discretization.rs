//! ## Continuous Variable Discretization Transformers
//!
//! This module provides transformers to convert continuous variables into categorical ones by binning them into discrete intervals.
//!
//! ### Available Transformers
//!
//! - [`ArbitraryDiscretizer`]: Discretizes based on user-defined intervals.
//! - [`EqualFrequencyDiscretizer`]: Splits a column into bins containing approximately equal numbers of values.
//! - [`EqualWidthDiscretizer`]: Divides a column into bins of equal width.
//! - [`GeometricWidthDiscretizer`]: Uses a geometric progression to determine bin boundaries.
//!
//! Each transformer returns a new DataFrame with the transformed columns.
//! Errors are returned as [`FeatureFactoryError`], and results are wrapped in [`FeatureFactoryResult`].

use crate::exceptions::{FeatureFactoryError, FeatureFactoryResult};
use crate::impl_transformer;
use datafusion::dataframe::DataFrame;
use datafusion::functions_aggregate::expr_fn::approx_percentile_cont;
use datafusion::logical_expr::{col, lit, Case as DFCase, Expr};
use datafusion::scalar::ScalarValue;
use std::collections::HashMap;

/// Validates that a column exists and is numeric (Float64 or Int64).
fn validate_numeric_column(df: &DataFrame, col_name: &str) -> FeatureFactoryResult<()> {
    let field = df.schema().field_with_name(None, col_name).map_err(|_| {
        FeatureFactoryError::MissingColumn(format!("Column '{}' not found", col_name))
    })?;
    match field.data_type() {
        datafusion::arrow::datatypes::DataType::Float64
        | datafusion::arrow::datatypes::DataType::Int64 => Ok(()),
        dt => Err(FeatureFactoryError::InvalidParameter(format!(
            "Column '{}' must be numeric (Float64 or Int64), but found {:?}",
            col_name, dt
        ))),
    }
}

/// Helper function to build a CASE expression for intervals.
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

/// Generic helper function that applies an interval mapping to each target column in a DataFrame.
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
    validate_numeric_column(df, col_name)?;
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

/// Discretizes a column into arbitrary intervals defined by the user.
pub struct ArbitraryDiscretizer {
    pub columns: Vec<String>,
    pub intervals: HashMap<String, Vec<(f64, f64, String)>>,
}

impl ArbitraryDiscretizer {
    pub fn new(columns: Vec<String>, intervals: HashMap<String, Vec<(f64, f64, String)>>) -> Self {
        Self { columns, intervals }
    }

    /// Stateless transformer: fit does nothing.
    pub async fn fit(&mut self, _df: &DataFrame) -> FeatureFactoryResult<()> {
        Ok(())
    }

    /// Transform validates that each target column is numeric and that the intervals are valid,
    /// then applies the interval mapping.
    pub fn transform(&self, df: DataFrame) -> FeatureFactoryResult<DataFrame> {
        for col_name in &self.columns {
            validate_numeric_column(&df, col_name)?;
        }
        for (col, intervals) in &self.intervals {
            for (lower, upper, _) in intervals {
                if lower >= upper {
                    return Err(FeatureFactoryError::InvalidParameter(format!(
                        "For column '{}', lower bound {} is not less than upper bound {}",
                        col, lower, upper
                    )));
                }
            }
        }
        apply_interval_mapping(df, &self.columns, &self.intervals)
    }

    // This transformer is stateless.
    fn inherent_is_stateful(&self) -> bool {
        false
    }
}

/// Splits a column into bins containing approximately equal numbers of values from the column.
pub struct EqualFrequencyDiscretizer {
    pub columns: Vec<String>,
    pub bins: usize,
    pub mapping: HashMap<String, Vec<(f64, f64, String)>>,
    fitted: bool,
}

impl EqualFrequencyDiscretizer {
    pub fn new(columns: Vec<String>, bins: usize) -> Self {
        Self {
            columns,
            bins,
            mapping: HashMap::new(),
            fitted: false,
        }
    }

    /// Fit computes equal-frequency intervals and stores the mapping.
    pub async fn fit(&mut self, df: &DataFrame) -> FeatureFactoryResult<()> {
        if self.bins < 1 {
            return Err(FeatureFactoryError::InvalidParameter(
                "Number of bins must be at least 1".to_string(),
            ));
        }
        for col_name in &self.columns {
            validate_numeric_column(df, col_name)?;
            let mut boundaries = Vec::with_capacity(self.bins + 1);
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
                                "Failed to compute percentile for column {}",
                                col_name
                            )),
                        ));
                    }
                }
            }
            if let (Some(first), Some(last)) = (boundaries.first(), boundaries.last()) {
                if (first - last).abs() < 1e-6 {
                    return Err(FeatureFactoryError::InvalidParameter(format!(
                        "Column {} appears to be constant; cannot discretize into equal-frequency bins",
                        col_name
                    )));
                }
            }
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
        self.fitted = true;
        Ok(())
    }

    /// Transform applies the equal-frequency discretization.
    pub fn transform(&self, df: DataFrame) -> FeatureFactoryResult<DataFrame> {
        if !self.fitted {
            return Err(FeatureFactoryError::FitNotCalled);
        }
        apply_interval_mapping(df, &self.columns, &self.mapping)
    }

    // This transformer is stateful.
    fn inherent_is_stateful(&self) -> bool {
        true
    }
}

/// Splits a column into bins of equal width.
pub struct EqualWidthDiscretizer {
    pub columns: Vec<String>,
    pub bins: usize,
    pub mapping: HashMap<String, Vec<(f64, f64, String)>>,
    fitted: bool,
}

impl EqualWidthDiscretizer {
    pub fn new(columns: Vec<String>, bins: usize) -> Self {
        Self {
            columns,
            bins,
            mapping: HashMap::new(),
            fitted: false,
        }
    }

    /// Fit computes the min and max and then builds equal-width intervals.
    pub async fn fit(&mut self, df: &DataFrame) -> FeatureFactoryResult<()> {
        if self.bins < 1 {
            return Err(FeatureFactoryError::InvalidParameter(
                "Number of bins must be at least 1".to_string(),
            ));
        }
        for col_name in &self.columns {
            validate_numeric_column(df, col_name)?;
            let (min_val, max_val) = compute_min_max(df, col_name).await?;
            if (max_val - min_val).abs() < 1e-6 {
                return Err(FeatureFactoryError::InvalidParameter(format!(
                    "Column {} is constant (min == max), cannot discretize into equal-width bins",
                    col_name
                )));
            }
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
        self.fitted = true;
        Ok(())
    }

    /// Transform applies the equal-width discretization.
    pub fn transform(&self, df: DataFrame) -> FeatureFactoryResult<DataFrame> {
        if !self.fitted {
            return Err(FeatureFactoryError::FitNotCalled);
        }
        apply_interval_mapping(df, &self.columns, &self.mapping)
    }

    // This transformer is stateful.
    fn inherent_is_stateful(&self) -> bool {
        true
    }
}

/// Uses a geometric progression to determine bin boundaries.
pub struct GeometricWidthDiscretizer {
    pub columns: Vec<String>,
    pub bins: usize,
    pub mapping: HashMap<String, Vec<(f64, f64, String)>>,
    fitted: bool,
}

impl GeometricWidthDiscretizer {
    pub fn new(columns: Vec<String>, bins: usize) -> Self {
        Self {
            columns,
            bins,
            mapping: HashMap::new(),
            fitted: false,
        }
    }

    /// Fit computes min and max and then generates geometric intervals.
    /// Returns an error if any column has non-positive values.
    pub async fn fit(&mut self, df: &DataFrame) -> FeatureFactoryResult<()> {
        if self.bins < 1 {
            return Err(FeatureFactoryError::InvalidParameter(
                "Number of bins must be at least 1".to_string(),
            ));
        }
        for col_name in &self.columns {
            validate_numeric_column(df, col_name)?;
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
        self.fitted = true;
        Ok(())
    }

    /// Transform applies the geometric-width discretization.
    pub fn transform(&self, df: DataFrame) -> FeatureFactoryResult<DataFrame> {
        if !self.fitted {
            return Err(FeatureFactoryError::FitNotCalled);
        }
        apply_interval_mapping(df, &self.columns, &self.mapping)
    }

    // This transformer is stateful.
    fn inherent_is_stateful(&self) -> bool {
        true
    }
}

// Implement the Transformer trait for the transformers in this module.
impl_transformer!(ArbitraryDiscretizer);
impl_transformer!(EqualFrequencyDiscretizer);
impl_transformer!(EqualWidthDiscretizer);
impl_transformer!(GeometricWidthDiscretizer);
