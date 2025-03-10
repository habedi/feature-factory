//! ## Categorical Encoding Transformers
//!
//! This module provides transformers (or encoders) that convert categorical features into numeric values.
//!
//! ### Available Transformers
//!
//! - [`OneHotEncoder`]: Expands each categorical column into multiple binary columns, one per distinct category.
//! - [`CountFrequencyEncoder`]: Replaces each category with its count (or frequency).
//! - [`OrdinalEncoder`]: Replaces each category with an ordinal (ordered integer) value.
//! - [`MeanEncoder`]: Replaces each category with the mean of a target variable.
//! - [`WoEEncoder`]: Replaces each category with its weight of evidence, calculated as the logarithm of the ratio of probabilities of “good” outcomes to “bad” outcomes.
//! - [`RareLabelEncoder`]: Groups infrequent categories into a single “rare” label.
//!
//! Each transformer returns a new DataFrame with the applied encodings.
//! Errors are returned as `FeatureFactoryError`, and results are wrapped in `FeatureFactoryResult`.

use crate::exceptions::{FeatureFactoryError, FeatureFactoryResult};
use crate::impl_transformer;
use arrow::array::Array;
use arrow::datatypes::DataType;
use datafusion::dataframe::DataFrame;
use datafusion::functions_aggregate::expr_fn::{avg, count};
use datafusion::logical_expr::{col, lit, Case as DFCase, Expr};
use std::collections::HashMap;

/// Validates that a column exists and is of Utf8 type.
fn validate_string_column(df: &DataFrame, col_name: &str) -> FeatureFactoryResult<()> {
    let field = df.schema().field_with_name(None, col_name).map_err(|_| {
        FeatureFactoryError::MissingColumn(format!("Column '{}' not found", col_name))
    })?;
    if field.data_type() != &DataType::Utf8 {
        return Err(FeatureFactoryError::InvalidParameter(format!(
            "Column '{}' must be of type Utf8, but found {:?}",
            col_name,
            field.data_type()
        )));
    }
    Ok(())
}

/// Validates that all columns in `cols` exist and are of Utf8 type.
fn validate_string_columns(df: &DataFrame, cols: &[String]) -> FeatureFactoryResult<()> {
    for col in cols {
        validate_string_column(df, col)?;
    }
    Ok(())
}

/// Validates that a column exists and is numeric (Float64 or Int64).
fn validate_numeric_column(df: &DataFrame, col_name: &str) -> FeatureFactoryResult<()> {
    let field = df.schema().field_with_name(None, col_name).map_err(|_| {
        FeatureFactoryError::MissingColumn(format!("Column '{}' not found", col_name))
    })?;
    match field.data_type() {
        DataType::Float64 | DataType::Int64 => Ok(()),
        dt => Err(FeatureFactoryError::InvalidParameter(format!(
            "Column '{}' must be numeric (Float64 or Int64), but found {:?}",
            col_name, dt
        ))),
    }
}

/// Sanitizes a category string so that it can be safely used as part of a column name.
/// Non-alphanumeric characters are replaced with underscores.
fn sanitize_category(cat: &str) -> String {
    cat.replace(|c: char| !c.is_alphanumeric(), "_")
}

/// Helper function to build a CASE WHEN expression given a mapping from category strings to values.
/// For each pair, the expression generated is:
/// `WHEN <col> = lit(<category>) THEN lit(<encoded_value>)`
/// If provided, `default` is used as the ELSE branch; otherwise, the original column is returned.
fn build_case_expr<T: Clone + 'static + datafusion::logical_expr::Literal>(
    col_name: &str,
    mapping: &[(String, T)],
    default: Option<Expr>,
) -> Expr {
    let when_then_expr = mapping
        .iter()
        .map(|(cat, val)| {
            (
                Box::new(col(col_name).eq(lit(cat.clone()))),
                Box::new(lit(val.clone())),
            )
        })
        .collect();
    Expr::Case(DFCase {
        expr: None,
        when_then_expr,
        else_expr: default.map(Box::new),
    })
}

/// Extract distinct string values for a given column from a DataFrame.
async fn extract_distinct_values(
    df: &DataFrame,
    col_name: &str,
) -> FeatureFactoryResult<Vec<String>> {
    // Validate that the column is of string type.
    validate_string_column(df, col_name)?;
    let distinct_df = df.clone().select(vec![col(col_name)])?.distinct()?;
    let batches = distinct_df
        .collect()
        .await
        .map_err(FeatureFactoryError::from)?;
    let mut values = Vec::new();
    for batch in batches {
        let array = batch
            .column(0)
            .as_any()
            .downcast_ref::<datafusion::arrow::array::StringArray>()
            .ok_or_else(|| {
                FeatureFactoryError::DataFusionError(datafusion::error::DataFusionError::Plan(
                    format!("Expected Utf8 array for column {}", col_name),
                ))
            })?;
        for i in 0..array.len() {
            if !array.is_null(i) {
                values.push(array.value(i).to_string());
            }
        }
    }
    Ok(values)
}

/// Extract a mapping (category -> count) for a given column by aggregating counts.
async fn extract_count_mapping(
    df: &DataFrame,
    col_name: &str,
) -> FeatureFactoryResult<HashMap<String, i64>> {
    validate_string_column(df, col_name)?;
    let grouped = df
        .clone()
        .aggregate(vec![col(col_name)], vec![count(col(col_name)).alias("cnt")])
        .map_err(FeatureFactoryError::from)?;
    let batches = grouped.collect().await.map_err(FeatureFactoryError::from)?;
    let mut map = HashMap::new();
    for batch in batches {
        let cat_array = batch
            .column(0)
            .as_any()
            .downcast_ref::<datafusion::arrow::array::StringArray>()
            .ok_or_else(|| {
                FeatureFactoryError::DataFusionError(datafusion::error::DataFusionError::Plan(
                    format!("Expected Utf8 array for column {}", col_name),
                ))
            })?;
        let count_array = batch
            .column(1)
            .as_any()
            .downcast_ref::<datafusion::arrow::array::Int64Array>()
            .ok_or_else(|| {
                FeatureFactoryError::DataFusionError(datafusion::error::DataFusionError::Plan(
                    "Expected Int64 array".into(),
                ))
            })?;
        for i in 0..batch.num_rows() {
            if !cat_array.is_null(i) {
                map.insert(cat_array.value(i).to_string(), count_array.value(i));
            }
        }
    }
    Ok(map)
}

/// Generic helper function to apply a mapping to each target column in a DataFrame.
/// For each field, if the column is in `target_cols` and a mapping is available via `mapping_fn`,
/// then the function replaces the column with a CASE–WHEN expression; otherwise, the original
/// column is retained. The `default_fn` closure produces a default expression for a given column name.
fn apply_mapping<T: Clone + 'static + datafusion::logical_expr::Literal>(
    df: DataFrame,
    target_cols: &[String],
    mapping_fn: impl Fn(&str) -> Option<Vec<(String, T)>>,
    default_fn: impl Fn(&str) -> Option<Expr>,
) -> FeatureFactoryResult<DataFrame> {
    let exprs: Vec<Expr> = df
        .schema()
        .fields()
        .iter()
        .map(|field| {
            let name = field.name();
            if target_cols.contains(name) {
                if let Some(map) = mapping_fn(name) {
                    build_case_expr(name, &map, default_fn(name)).alias(name)
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

/// Expands each categorical column into multiple binary columns, one per distinct category.
pub struct OneHotEncoder {
    pub columns: Vec<String>,
    /// Mapping from column name to list of distinct category values.
    pub categories: HashMap<String, Vec<String>>,
    fitted: bool,
}

impl OneHotEncoder {
    /// Create a new OneHotEncoder for the specified columns.
    pub fn new(columns: Vec<String>) -> Self {
        Self {
            columns,
            categories: HashMap::new(),
            fitted: false,
        }
    }

    /// Fit computes and stores distinct category values.
    pub async fn fit(&mut self, df: &DataFrame) -> FeatureFactoryResult<()> {
        validate_string_columns(df, &self.columns)?;
        for col_name in &self.columns {
            let values = extract_distinct_values(df, col_name).await?;
            self.categories.insert(col_name.clone(), values);
        }
        self.fitted = true;
        Ok(())
    }

    /// Transform applies one-hot encoding and returns a new DataFrame.
    pub fn transform(&self, df: DataFrame) -> FeatureFactoryResult<DataFrame> {
        if !self.fitted {
            return Err(FeatureFactoryError::FitNotCalled);
        }
        let mut exprs = vec![];
        for field in df.schema().fields() {
            exprs.push(col(field.name()));
        }
        for col_name in &self.columns {
            if let Some(cats) = self.categories.get(col_name) {
                for cat in cats {
                    let safe_cat = sanitize_category(cat);
                    let new_col_name = format!("{}_{}", col_name, safe_cat);
                    let case_expr = Expr::Case(DFCase {
                        expr: None,
                        when_then_expr: vec![(
                            Box::new(col(col_name).eq(lit(cat.clone()))),
                            Box::new(lit(1_i32)),
                        )],
                        else_expr: Some(Box::new(lit(0_i32))),
                    })
                    .alias(new_col_name);
                    exprs.push(case_expr);
                }
            }
        }
        df.select(exprs).map_err(FeatureFactoryError::from)
    }

    // This transformer is stateful.
    fn inherent_is_stateful(&self) -> bool {
        true
    }
}

/// Replaces each category in a column with its frequency.
pub struct CountFrequencyEncoder {
    pub columns: Vec<String>,
    /// Mapping from column to (category -> count)
    pub mapping: HashMap<String, HashMap<String, i64>>,
    fitted: bool,
}

impl CountFrequencyEncoder {
    /// Create a new CountFrequencyEncoder for the specified columns.
    pub fn new(columns: Vec<String>) -> Self {
        Self {
            columns,
            mapping: HashMap::new(),
            fitted: false,
        }
    }

    /// Fit computes counts for each category.
    pub async fn fit(&mut self, df: &DataFrame) -> FeatureFactoryResult<()> {
        validate_string_columns(df, &self.columns)?;
        for col_name in &self.columns {
            let map = extract_count_mapping(df, col_name).await?;
            self.mapping.insert(col_name.clone(), map);
        }
        self.fitted = true;
        Ok(())
    }

    /// Transform replaces each category with its count.
    pub fn transform(&self, df: DataFrame) -> FeatureFactoryResult<DataFrame> {
        if !self.fitted {
            return Err(FeatureFactoryError::FitNotCalled);
        }
        apply_mapping(
            df,
            &self.columns,
            |name| {
                self.mapping.get(name).map(|m| {
                    m.iter()
                        .map(|(k, &v)| (k.clone(), v))
                        .collect::<Vec<(String, i64)>>()
                })
            },
            |_| Some(lit(0_i64)),
        )
    }

    // This transformer is stateful.
    fn inherent_is_stateful(&self) -> bool {
        true
    }
}

/// Replaces each category with an ordinal (ordered integer) value.
/// Categories are sorted alphabetically and assigned increasing integers starting at 0.
pub struct OrdinalEncoder {
    pub columns: Vec<String>,
    /// Mapping from column to (category -> ordinal index)
    pub mapping: HashMap<String, HashMap<String, i64>>,
    fitted: bool,
}

impl OrdinalEncoder {
    /// Create a new OrdinalEncoder for the specified columns.
    pub fn new(columns: Vec<String>) -> Self {
        Self {
            columns,
            mapping: HashMap::new(),
            fitted: false,
        }
    }

    /// Fit computes the ordinal mapping.
    pub async fn fit(&mut self, df: &DataFrame) -> FeatureFactoryResult<()> {
        validate_string_columns(df, &self.columns)?;
        for col_name in &self.columns {
            let mut values = extract_distinct_values(df, col_name).await?;
            values.sort();
            let mapping = values
                .into_iter()
                .enumerate()
                .map(|(i, cat)| (cat, i as i64))
                .collect();
            self.mapping.insert(col_name.clone(), mapping);
        }
        self.fitted = true;
        Ok(())
    }

    /// Transform replaces each category with its ordinal index.
    pub fn transform(&self, df: DataFrame) -> FeatureFactoryResult<DataFrame> {
        if !self.fitted {
            return Err(FeatureFactoryError::FitNotCalled);
        }
        apply_mapping(
            df,
            &self.columns,
            |name| {
                self.mapping.get(name).map(|m| {
                    m.iter()
                        .map(|(k, &v)| (k.clone(), v))
                        .collect::<Vec<(String, i64)>>()
                })
            },
            |_| Some(lit(0_i64)),
        )
    }

    // This transformer is stateful.
    fn inherent_is_stateful(&self) -> bool {
        true
    }
}

/// Replaces each category with the mean of a target variable.
pub struct MeanEncoder {
    pub columns: Vec<String>,
    pub target: String,
    /// Mapping from column to (category -> mean)
    pub mapping: HashMap<String, HashMap<String, f64>>,
    fitted: bool,
}

impl MeanEncoder {
    /// Create a new MeanEncoder for the specified columns and target.
    pub fn new(columns: Vec<String>, target: String) -> Self {
        Self {
            columns,
            target,
            mapping: HashMap::new(),
            fitted: false,
        }
    }

    /// Fit computes the mean for each category.
    pub async fn fit(&mut self, df: &DataFrame) -> FeatureFactoryResult<()> {
        validate_string_columns(df, &self.columns)?;
        validate_numeric_column(df, &self.target)?;
        for col_name in &self.columns {
            let agg_df = df
                .clone()
                .aggregate(
                    vec![col(col_name)],
                    vec![avg(col(&self.target)).alias("mean")],
                )
                .map_err(FeatureFactoryError::from)?;
            let batches = agg_df.collect().await.map_err(FeatureFactoryError::from)?;
            let mut map = HashMap::new();
            for batch in batches {
                let cat_array = batch
                    .column(0)
                    .as_any()
                    .downcast_ref::<datafusion::arrow::array::StringArray>()
                    .ok_or_else(|| {
                        FeatureFactoryError::DataFusionError(
                            datafusion::error::DataFusionError::Plan(format!(
                                "Expected Utf8 array for column {}",
                                col_name
                            )),
                        )
                    })?;
                let mean_array = batch
                    .column(1)
                    .as_any()
                    .downcast_ref::<datafusion::arrow::array::Float64Array>()
                    .ok_or_else(|| {
                        FeatureFactoryError::DataFusionError(
                            datafusion::error::DataFusionError::Plan(
                                "Expected Float64 array".into(),
                            ),
                        )
                    })?;
                for i in 0..batch.num_rows() {
                    if !cat_array.is_null(i) {
                        map.insert(cat_array.value(i).to_string(), mean_array.value(i));
                    }
                }
            }
            self.mapping.insert(col_name.clone(), map);
        }
        self.fitted = true;
        Ok(())
    }

    /// Transform replaces each category with its mean.
    pub fn transform(&self, df: DataFrame) -> FeatureFactoryResult<DataFrame> {
        if !self.fitted {
            return Err(FeatureFactoryError::FitNotCalled);
        }
        apply_mapping(
            df,
            &self.columns,
            |name| {
                self.mapping.get(name).map(|m| {
                    m.iter()
                        .map(|(k, &v)| (k.clone(), v))
                        .collect::<Vec<(String, f64)>>()
                })
            },
            |_| Some(lit(0.0_f64)),
        )
    }

    // This transformer is stateful.
    fn inherent_is_stateful(&self) -> bool {
        true
    }
}

/// Replaces each category with its weight of evidence (WoE).
/// WoE is computed as ln((good_rate)/(bad_rate)), assuming a binary target.
pub struct WoEEncoder {
    pub columns: Vec<String>,
    pub target: String,
    /// Mapping from column to (category -> WoE)
    pub mapping: HashMap<String, HashMap<String, f64>>,
    fitted: bool,
}

impl WoEEncoder {
    /// Create a new WoEEncoder for the specified columns and target.
    pub fn new(columns: Vec<String>, target: String) -> Self {
        Self {
            columns,
            target,
            mapping: HashMap::new(),
            fitted: false,
        }
    }

    /// Fit computes the WoE for each category.
    pub async fn fit(&mut self, df: &DataFrame) -> FeatureFactoryResult<()> {
        validate_string_columns(df, &self.columns)?;
        validate_numeric_column(df, &self.target)?;
        let overall_df = df
            .clone()
            .aggregate(vec![], vec![count(col(&self.target)).alias("total")])
            .map_err(FeatureFactoryError::from)?;
        let overall_batches = overall_df
            .collect()
            .await
            .map_err(FeatureFactoryError::from)?;
        let _total = if let Some(batch) = overall_batches.first() {
            let total_array = batch
                .column(0)
                .as_any()
                .downcast_ref::<datafusion::arrow::array::Int64Array>()
                .ok_or_else(|| {
                    FeatureFactoryError::DataFusionError(datafusion::error::DataFusionError::Plan(
                        "Expected Int64 array".into(),
                    ))
                })?;
            total_array.value(0) as f64
        } else {
            return Err(FeatureFactoryError::DataFusionError(
                datafusion::error::DataFusionError::Plan("No data found".into()),
            ));
        };

        for col_name in &self.columns {
            let grouped = df
                .clone()
                .aggregate(
                    vec![col(col_name), col(&self.target)],
                    vec![count(lit(1)).alias("cnt")],
                )
                .map_err(FeatureFactoryError::from)?;
            let batches = grouped.collect().await.map_err(FeatureFactoryError::from)?;
            let mut cat_counts: HashMap<String, (f64, f64)> = HashMap::new(); // (good, bad)
            for batch in batches {
                let cat_array = batch
                    .column(0)
                    .as_any()
                    .downcast_ref::<datafusion::arrow::array::StringArray>()
                    .ok_or_else(|| {
                        FeatureFactoryError::DataFusionError(
                            datafusion::error::DataFusionError::Plan(format!(
                                "Expected Utf8 array for column {}",
                                col_name
                            )),
                        )
                    })?;
                let target_array = batch
                    .column(1)
                    .as_any()
                    .downcast_ref::<datafusion::arrow::array::Int64Array>()
                    .ok_or_else(|| {
                        FeatureFactoryError::DataFusionError(
                            datafusion::error::DataFusionError::Plan("Expected Int64 array".into()),
                        )
                    })?;
                let count_array = batch
                    .column(2)
                    .as_any()
                    .downcast_ref::<datafusion::arrow::array::Int64Array>()
                    .ok_or_else(|| {
                        FeatureFactoryError::DataFusionError(
                            datafusion::error::DataFusionError::Plan("Expected Int64 array".into()),
                        )
                    })?;
                for i in 0..batch.num_rows() {
                    if !cat_array.is_null(i) {
                        let cat = cat_array.value(i).to_string();
                        let target_val = target_array.value(i);
                        let cnt = count_array.value(i) as f64;
                        let entry = cat_counts.entry(cat).or_insert((0.0, 0.0));
                        if target_val == 1 {
                            entry.0 += cnt;
                        } else {
                            entry.1 += cnt;
                        }
                    }
                }
            }
            let mut mapping = HashMap::new();
            for (cat, (good, bad)) in cat_counts {
                let woe = ((good + 1e-6) / (bad + 1e-6)).ln();
                mapping.insert(cat, woe);
            }
            self.mapping.insert(col_name.clone(), mapping);
        }
        self.fitted = true;
        Ok(())
    }

    /// Transform replaces each category with its computed WoE.
    pub fn transform(&self, df: DataFrame) -> FeatureFactoryResult<DataFrame> {
        if !self.fitted {
            return Err(FeatureFactoryError::FitNotCalled);
        }
        apply_mapping(
            df,
            &self.columns,
            |name| {
                self.mapping.get(name).map(|m| {
                    m.iter()
                        .map(|(k, &v)| (k.clone(), v))
                        .collect::<Vec<(String, f64)>>()
                })
            },
            |_| Some(lit(0.0_f64)),
        )
    }

    // This transformer is stateful.
    fn inherent_is_stateful(&self) -> bool {
        true
    }
}

/// Groups infrequent categories into a single “rare” label.
pub struct RareLabelEncoder {
    pub columns: Vec<String>,
    pub threshold: f64, // frequency threshold (between 0 and 1)
    /// Mapping from column to (category -> encoded label)
    pub mapping: HashMap<String, HashMap<String, String>>,
    fitted: bool,
}

impl RareLabelEncoder {
    /// Create a new RareLabelEncoder for the specified columns and threshold.
    pub fn new(columns: Vec<String>, threshold: f64) -> Self {
        Self {
            columns,
            threshold,
            mapping: HashMap::new(),
            fitted: false,
        }
    }

    /// Fit computes frequencies and marks infrequent categories as "rare".
    pub async fn fit(&mut self, df: &DataFrame) -> FeatureFactoryResult<()> {
        if self.threshold < 0.0 || self.threshold > 1.0 {
            return Err(FeatureFactoryError::InvalidParameter(format!(
                "Threshold {} must be between 0 and 1",
                self.threshold
            )));
        }
        validate_string_columns(df, &self.columns)?;
        let total_df = df
            .clone()
            .aggregate(vec![], vec![count(lit(1)).alias("total")])
            .map_err(FeatureFactoryError::from)?;
        let total_batches = total_df
            .collect()
            .await
            .map_err(FeatureFactoryError::from)?;
        let total = if let Some(batch) = total_batches.first() {
            let total_array = batch
                .column(0)
                .as_any()
                .downcast_ref::<datafusion::arrow::array::Int64Array>()
                .ok_or_else(|| {
                    FeatureFactoryError::DataFusionError(datafusion::error::DataFusionError::Plan(
                        "Expected Int64 array".into(),
                    ))
                })?;
            total_array.value(0) as f64
        } else {
            return Err(FeatureFactoryError::DataFusionError(
                datafusion::error::DataFusionError::Plan("No data found".into()),
            ));
        };

        for col_name in &self.columns {
            let grouped = df
                .clone()
                .aggregate(vec![col(col_name)], vec![count(col(col_name)).alias("cnt")])
                .map_err(FeatureFactoryError::from)?;
            let batches = grouped.collect().await.map_err(FeatureFactoryError::from)?;
            let mut map = HashMap::new();
            for batch in batches {
                let cat_array = batch
                    .column(0)
                    .as_any()
                    .downcast_ref::<datafusion::arrow::array::StringArray>()
                    .ok_or_else(|| {
                        FeatureFactoryError::DataFusionError(
                            datafusion::error::DataFusionError::Plan(format!(
                                "Expected Utf8 array for column {}",
                                col_name
                            )),
                        )
                    })?;
                let cnt_array = batch
                    .column(1)
                    .as_any()
                    .downcast_ref::<datafusion::arrow::array::Int64Array>()
                    .ok_or_else(|| {
                        FeatureFactoryError::DataFusionError(
                            datafusion::error::DataFusionError::Plan("Expected Int64 array".into()),
                        )
                    })?;
                for i in 0..batch.num_rows() {
                    if !cat_array.is_null(i) {
                        let cat = cat_array.value(i).to_string();
                        let cnt = cnt_array.value(i) as f64;
                        let freq = cnt / total;
                        let encoded = if freq < self.threshold {
                            "rare".to_string()
                        } else {
                            cat.clone()
                        };
                        map.insert(cat, encoded);
                    }
                }
            }
            self.mapping.insert(col_name.clone(), map);
        }
        self.fitted = true;
        Ok(())
    }

    /// Transform replaces each category with its encoded label.
    pub fn transform(&self, df: DataFrame) -> FeatureFactoryResult<DataFrame> {
        if !self.fitted {
            return Err(FeatureFactoryError::FitNotCalled);
        }
        apply_mapping(
            df,
            &self.columns,
            |name| {
                self.mapping.get(name).map(|m| {
                    m.iter()
                        .map(|(k, v)| (k.clone(), v.clone()))
                        .collect::<Vec<(String, String)>>()
                })
            },
            |name| Some(col(name)),
        )
    }

    // This transformer is stateful.
    fn inherent_is_stateful(&self) -> bool {
        true
    }
}

// Implement the Transformer trait for the transformers in this module.
impl_transformer!(OneHotEncoder);
impl_transformer!(CountFrequencyEncoder);
impl_transformer!(OrdinalEncoder);
impl_transformer!(MeanEncoder);
impl_transformer!(WoEEncoder);
impl_transformer!(RareLabelEncoder);
