//! ## Feature Selection Transformers
//!
//! This module provides transformers for selecting the most relevant features based on some criteria.
//!
//! ### Available Transformers
//!
//! - [`DropFeatures`]: Removes specific features from the dataset.
//! - [`DropConstantFeatures`]: Eliminates constant and quasi-constant features.
//! - [`DropDuplicateFeatures`]: Removes duplicate columns.
//! - [`DropCorrelatedFeatures`]: Drops highly correlated features to reduce redundancy.
//! - [`SmartCorrelatedSelection`]: Retains the best feature from correlated groups based on relevance.
//! - [`DropHighPSIFeatures`]: Discards features with a high Population Stability Index (PSI).
//! - [`SelectByInformationValue`]: Selects features based on Information Value (IV) for binary classification tasks.
//! - [`SelectBySingleFeaturePerformance`]: Chooses features based on absolute correlation with a binary target.
//! - [`SelectByTargetMeanPerformance`]: Selects features based on variations in target mean across bins.
//! - [`MRMR`]: Uses Maximum Relevance Minimum Redundancy (MRMR) algorithm for feature selection.
//!
//! ### Assumptions
//!
//! - The DataFrame is fully materialized (`collect()`) for computing statistics.
//! - Numeric columns are expected to be of Arrow’s `Float64` type.
//! - Target-dependent methods assume a binary target column (values `0` and `1`).
//!
//! Each transformer returns a new DataFrame with the selected features.
//! Errors are returned as [`FeatureFactoryError`], and results are wrapped in [`FeatureFactoryResult`].

use crate::exceptions::{FeatureFactoryError, FeatureFactoryResult};
use crate::impl_transformer;
use datafusion::arrow::array::{as_primitive_array, Array, StringArray};
use datafusion::arrow::datatypes::{DataType, Float64Type};
use datafusion::dataframe::DataFrame;
use datafusion::logical_expr::{col, Expr};
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

/// Helper function that checks if a DataFusion data type is numeric (only handling Float64 here).
fn is_numeric(dt: &DataType) -> bool {
    matches!(dt, DataType::Float64)
}

/// Removes the specified columns from the DataFrame.
pub struct DropFeatures {
    pub features: Vec<String>,
}

impl DropFeatures {
    pub fn new(features: Vec<String>) -> Self {
        Self { features }
    }

    pub async fn fit(&mut self, _df: &DataFrame) -> FeatureFactoryResult<()> {
        Ok(())
    }

    pub fn transform(&self, df: DataFrame) -> FeatureFactoryResult<DataFrame> {
        let available_exprs: Vec<Expr> = df
            .schema()
            .fields()
            .iter()
            .filter_map(|field| {
                if !self.features.contains(field.name()) {
                    Some(col(field.name()))
                } else {
                    None
                }
            })
            .collect();

        if available_exprs.is_empty() {
            return Err(FeatureFactoryError::InvalidParameter(
                "Dropping these features would result in an empty DataFrame.".to_string(),
            ));
        }
        df.select(available_exprs)
            .map_err(FeatureFactoryError::from)
    }

    fn inherent_is_stateful(&self) -> bool {
        false
    }
}

/// Removes features that are constant or nearly constant (where variance is below a threshold).
pub struct DropConstantFeatures {
    pub numeric_threshold: f64,
    pub categorical_threshold: usize,
    pub drop_columns: HashSet<String>,
    fitted: bool,
}

impl DropConstantFeatures {
    pub fn new(numeric_threshold: f64, categorical_threshold: usize) -> Self {
        Self {
            numeric_threshold,
            categorical_threshold,
            drop_columns: HashSet::new(),
            fitted: false,
        }
    }

    pub async fn fit(&mut self, df: &DataFrame) -> FeatureFactoryResult<()> {
        let schema = df.schema();
        let batches = df.clone().collect().await?;
        if batches.is_empty() {
            return Err(FeatureFactoryError::InvalidParameter(
                "DataFrame is empty.".to_string(),
            ));
        }
        let batch = &batches[0];

        for field in schema.fields() {
            let name = field.name();
            if is_numeric(field.data_type()) {
                let array =
                    as_primitive_array::<Float64Type>(batch.column_by_name(name).ok_or_else(
                        || FeatureFactoryError::MissingColumn(format!("Column {} not found", name)),
                    )?);
                let n = array.len() as f64;
                let sum: f64 = array.iter().flatten().par_bridge().sum();
                let mean = sum / n;
                let sum_sq: f64 = array.iter().flatten().par_bridge().map(|v| v * v).sum();
                let variance = sum_sq / n - mean * mean;
                if variance < self.numeric_threshold {
                    self.drop_columns.insert(name.to_string());
                }
            } else {
                let string_array = batch
                    .column_by_name(name)
                    .ok_or_else(|| {
                        FeatureFactoryError::MissingColumn(format!("Column {} not found", name))
                    })?
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .ok_or_else(|| {
                        FeatureFactoryError::DataFusionError(
                            datafusion::error::DataFusionError::Plan(format!(
                                "Expected Utf8 array for column {}",
                                name
                            )),
                        )
                    })?;
                let mut distinct = HashSet::new();
                for i in 0..string_array.len() {
                    if !string_array.is_null(i) {
                        distinct.insert(string_array.value(i).to_string());
                    }
                }
                if distinct.len() <= self.categorical_threshold {
                    self.drop_columns.insert(name.to_string());
                }
            }
        }
        self.fitted = true;
        Ok(())
    }

    pub fn transform(&self, df: DataFrame) -> FeatureFactoryResult<DataFrame> {
        if !self.fitted {
            return Err(FeatureFactoryError::FitNotCalled);
        }
        let keep_exprs: Vec<Expr> = df
            .schema()
            .fields()
            .iter()
            .filter_map(|field| {
                if !self.drop_columns.contains(field.name()) {
                    Some(col(field.name()))
                } else {
                    None
                }
            })
            .collect();

        if keep_exprs.is_empty() {
            return Err(FeatureFactoryError::InvalidParameter(
                "All features were dropped by DropConstantFeatures.".to_string(),
            ));
        }
        df.select(keep_exprs).map_err(FeatureFactoryError::from)
    }

    fn inherent_is_stateful(&self) -> bool {
        true
    }
}

/// Removes duplicate features by comparing values in each column.
pub struct DropDuplicateFeatures {
    pub drop_columns: HashSet<String>,
    fitted: bool,
}

impl Default for DropDuplicateFeatures {
    fn default() -> Self {
        Self::new()
    }
}

impl DropDuplicateFeatures {
    pub fn new() -> Self {
        Self {
            drop_columns: HashSet::new(),
            fitted: false,
        }
    }

    pub async fn fit(&mut self, df: &DataFrame) -> FeatureFactoryResult<()> {
        let batches = df.clone().collect().await?;
        if batches.is_empty() {
            return Err(FeatureFactoryError::InvalidParameter(
                "Empty DataFrame".to_string(),
            ));
        }
        let batch = &batches[0];
        let schema = batch.schema();
        let mut seen: Vec<(String, Arc<dyn Array>)> = Vec::new();
        for field in schema.fields() {
            let name = field.name().clone();
            let array = batch.column_by_name(&name).unwrap();
            let mut is_duplicate = false;
            for (_seen_name, seen_array) in &seen {
                if array == seen_array {
                    self.drop_columns.insert(name.clone());
                    is_duplicate = true;
                    break;
                }
            }
            if !is_duplicate {
                seen.push((name, array.clone()));
            }
        }
        self.fitted = true;
        Ok(())
    }

    pub fn transform(&self, df: DataFrame) -> FeatureFactoryResult<DataFrame> {
        if !self.fitted {
            return Err(FeatureFactoryError::FitNotCalled);
        }
        let keep_exprs: Vec<Expr> = df
            .schema()
            .fields()
            .iter()
            .filter_map(|field| {
                if !self.drop_columns.contains(field.name()) {
                    Some(col(field.name()))
                } else {
                    None
                }
            })
            .collect();
        if keep_exprs.is_empty() {
            return Err(FeatureFactoryError::InvalidParameter(
                "All features were dropped by DropDuplicateFeatures.".to_string(),
            ));
        }
        df.select(keep_exprs).map_err(FeatureFactoryError::from)
    }

    fn inherent_is_stateful(&self) -> bool {
        true
    }
}

/// Removes one feature from each highly correlated pair (using Pearson correlation).
pub struct DropCorrelatedFeatures {
    pub threshold: f64,
    pub drop_columns: HashSet<String>,
    fitted: bool,
}

impl DropCorrelatedFeatures {
    pub fn new(threshold: f64) -> Self {
        Self {
            threshold,
            drop_columns: HashSet::new(),
            fitted: false,
        }
    }

    pub async fn fit(&mut self, df: &DataFrame) -> FeatureFactoryResult<()> {
        let batches = df.clone().collect().await?;
        if batches.is_empty() {
            return Err(FeatureFactoryError::InvalidParameter(
                "Empty DataFrame".to_string(),
            ));
        }
        let batch = &batches[0];
        let schema = df.schema();
        let numeric_fields: Vec<_> = schema
            .fields()
            .iter()
            .filter(|f| is_numeric(f.data_type()))
            .collect();
        let mut data: HashMap<String, Vec<f64>> = HashMap::new();
        for field in &numeric_fields {
            let name = field.name();
            let array = as_primitive_array::<Float64Type>(batch.column_by_name(name).unwrap());
            let vec: Vec<f64> = array.iter().flatten().collect();
            data.insert(name.to_string(), vec);
        }
        let mut to_drop = HashSet::new();
        let names: Vec<_> = data.keys().cloned().collect();
        for i in 0..names.len() {
            for j in (i + 1)..names.len() {
                let x = &data[&names[i]];
                let y = &data[&names[j]];
                if x.len() != y.len() || x.is_empty() {
                    continue;
                }
                let n_f = x.len() as f64;
                let mean_x = x.iter().sum::<f64>() / n_f;
                let mean_y = y.iter().sum::<f64>() / n_f;
                let cov: f64 = x
                    .iter()
                    .zip(y.iter())
                    .map(|(a, b)| (a - mean_x) * (b - mean_y))
                    .sum();
                let var_x: f64 = x.iter().map(|a| (a - mean_x).powi(2)).sum();
                let var_y: f64 = y.iter().map(|b| (b - mean_y).powi(2)).sum();
                if var_x == 0.0 || var_y == 0.0 {
                    continue;
                }
                let corr = cov / ((var_x).sqrt() * (var_y).sqrt());
                if corr.abs() > self.threshold {
                    if var_x < var_y {
                        to_drop.insert(names[i].clone());
                    } else {
                        to_drop.insert(names[j].clone());
                    }
                }
            }
        }
        self.drop_columns = to_drop;
        self.fitted = true;
        Ok(())
    }

    pub fn transform(&self, df: DataFrame) -> FeatureFactoryResult<DataFrame> {
        if !self.fitted {
            return Err(FeatureFactoryError::FitNotCalled);
        }
        let keep_exprs: Vec<Expr> = df
            .schema()
            .fields()
            .iter()
            .filter_map(|f| {
                if !self.drop_columns.contains(f.name()) {
                    Some(col(f.name()))
                } else {
                    None
                }
            })
            .collect();
        if keep_exprs.is_empty() {
            return Err(FeatureFactoryError::InvalidParameter(
                "All features were dropped by DropCorrelatedFeatures.".to_string(),
            ));
        }
        df.select(keep_exprs).map_err(FeatureFactoryError::from)
    }

    fn inherent_is_stateful(&self) -> bool {
        true
    }
}

/// Groups correlated features and keeps the one with the highest variance from each group.
pub struct SmartCorrelatedSelection {
    pub threshold: f64,
    pub selected_features: HashSet<String>,
    fitted: bool,
}

impl SmartCorrelatedSelection {
    pub fn new(threshold: f64) -> Self {
        Self {
            threshold,
            selected_features: HashSet::new(),
            fitted: false,
        }
    }

    pub async fn fit(&mut self, df: &DataFrame) -> FeatureFactoryResult<()> {
        let batches = df.clone().collect().await?;
        if batches.is_empty() {
            return Err(FeatureFactoryError::InvalidParameter(
                "Empty DataFrame".to_string(),
            ));
        }
        let batch = &batches[0];
        let schema = df.schema();
        let numeric_fields: Vec<_> = schema
            .fields()
            .iter()
            .filter(|f| is_numeric(f.data_type()))
            .collect();
        let mut stats: Vec<(String, f64, Vec<f64>)> = Vec::new();
        for field in &numeric_fields {
            let name = field.name();
            let array = as_primitive_array::<Float64Type>(batch.column_by_name(name).unwrap());
            let vec: Vec<f64> = array.iter().flatten().collect();
            let n = vec.len() as f64;
            let mean = vec.iter().sum::<f64>() / n;
            let var = vec.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
            stats.push((name.to_string(), var, vec));
        }
        let mut candidates: HashSet<String> =
            stats.iter().map(|(name, _, _)| name.clone()).collect();
        let mut selected: Vec<String> = Vec::<String>::new();
        for i in 0..stats.len() {
            for j in (i + 1)..stats.len() {
                let (ref name_i, var_i, ref x) = stats[i];
                let (ref name_j, var_j, ref y) = stats[j];
                if !candidates.contains(name_i) || !candidates.contains(name_j) {
                    continue;
                }
                if x.len() != y.len() || x.is_empty() {
                    continue;
                }
                let n_f = x.len() as f64;
                let mean_i = x.iter().sum::<f64>() / n_f;
                let mean_j = y.iter().sum::<f64>() / n_f;
                let cov: f64 = x
                    .iter()
                    .zip(y.iter())
                    .map(|(a, b)| (a - mean_i) * (b - mean_j))
                    .sum();
                let sxx: f64 = x.iter().map(|a| (a - mean_i).powi(2)).sum();
                let syy: f64 = y.iter().map(|b| (b - mean_j).powi(2)).sum();
                if sxx == 0.0 || syy == 0.0 {
                    continue;
                }
                let corr = cov / (sxx.sqrt() * syy.sqrt());
                if corr.abs() > self.threshold {
                    if var_i < var_j {
                        candidates.remove(name_i);
                    } else {
                        candidates.remove(name_j);
                    }
                }
            }
        }
        selected.extend(candidates.into_iter());
        self.selected_features = selected.into_iter().collect();
        self.fitted = true;
        Ok(())
    }

    pub fn transform(&self, df: DataFrame) -> FeatureFactoryResult<DataFrame> {
        if !self.fitted {
            return Err(FeatureFactoryError::FitNotCalled);
        }
        let keep_exprs: Vec<Expr> = df
            .schema()
            .fields()
            .iter()
            .filter_map(|f| {
                if is_numeric(f.data_type()) {
                    if self.selected_features.contains(f.name()) {
                        Some(col(f.name()))
                    } else {
                        None
                    }
                } else {
                    Some(col(f.name()))
                }
            })
            .collect();
        if keep_exprs.is_empty() {
            return Err(FeatureFactoryError::InvalidParameter(
                "No features selected by SmartCorrelatedSelection.".to_string(),
            ));
        }
        df.select(keep_exprs).map_err(FeatureFactoryError::from)
    }

    fn inherent_is_stateful(&self) -> bool {
        true
    }
}

/// Drops features that their Population Stability Index (PSI) is larger than a threshold.
pub struct DropHighPSIFeatures {
    pub reference: DataFrame,
    pub psi_threshold: f64,
    pub drop_columns: HashSet<String>,
    fitted: bool,
}

impl DropHighPSIFeatures {
    pub fn new(reference: DataFrame, psi_threshold: f64) -> Self {
        Self {
            reference,
            psi_threshold,
            drop_columns: HashSet::new(),
            fitted: false,
        }
    }

    fn compute_psi(ref_vals: &[f64], curr_vals: &[f64], bins: &[f64]) -> f64 {
        let mut psi = 0.0;
        let total_ref = ref_vals.len() as f64;
        let total_curr = curr_vals.len() as f64;
        for i in 0..bins.len() - 1 {
            let lower = bins[i];
            let upper = bins[i + 1];
            let count_ref = ref_vals
                .par_iter()
                .filter(|v| **v >= lower && **v < upper)
                .count() as f64;
            let count_curr = curr_vals
                .par_iter()
                .filter(|v| **v >= lower && **v < upper)
                .count() as f64;
            let pct_ref = (count_ref / total_ref).max(0.0001);
            let pct_curr = (count_curr / total_curr).max(0.0001);
            psi += (pct_ref - pct_curr) * (pct_ref / pct_curr).ln();
        }
        psi
    }

    pub async fn fit(&mut self, df: &DataFrame) -> FeatureFactoryResult<()> {
        let ref_batches = self.reference.clone().collect().await?;
        let curr_batches = df.clone().collect().await?;
        if ref_batches.is_empty() || curr_batches.is_empty() {
            return Err(FeatureFactoryError::InvalidParameter(
                "Empty DataFrame".to_string(),
            ));
        }
        let ref_batch = &ref_batches[0];
        let curr_batch = &curr_batches[0];
        let schema = df.schema();
        for field in schema.fields() {
            if is_numeric(field.data_type()) {
                let name = field.name();
                let ref_array =
                    as_primitive_array::<Float64Type>(ref_batch.column_by_name(name).ok_or_else(
                        || FeatureFactoryError::MissingColumn(format!("Column {} missing", name)),
                    )?);
                let curr_array =
                    as_primitive_array::<Float64Type>(curr_batch.column_by_name(name).ok_or_else(
                        || FeatureFactoryError::MissingColumn(format!("Column {} missing", name)),
                    )?);
                let ref_vals: Vec<f64> = ref_array.iter().flatten().par_bridge().collect();
                let curr_vals: Vec<f64> = curr_array.iter().flatten().par_bridge().collect();
                let mut sorted = ref_vals.clone();
                sorted.par_sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
                let mut bins = Vec::new();
                for i in 0..11 {
                    let idx = ((sorted.len() - 1) as f64 * i as f64 / 10.0).round() as usize;
                    bins.push(sorted[idx]);
                }
                let psi = Self::compute_psi(&ref_vals, &curr_vals, &bins);
                if psi > self.psi_threshold {
                    self.drop_columns.insert(name.to_string());
                }
            }
        }
        self.fitted = true;
        Ok(())
    }

    pub fn transform(&self, df: DataFrame) -> FeatureFactoryResult<DataFrame> {
        if !self.fitted {
            return Err(FeatureFactoryError::FitNotCalled);
        }
        let keep_exprs: Vec<Expr> = df
            .schema()
            .fields()
            .iter()
            .filter_map(|f| {
                if !self.drop_columns.contains(f.name()) {
                    Some(col(f.name()))
                } else {
                    None
                }
            })
            .collect();
        if keep_exprs.is_empty() {
            return Err(FeatureFactoryError::InvalidParameter(
                "All features dropped by DropHighPSIFeatures.".to_string(),
            ));
        }
        df.select(keep_exprs).map_err(FeatureFactoryError::from)
    }

    fn inherent_is_stateful(&self) -> bool {
        true
    }
}

/// Computes Information Value (IV) for each feature relative to a binary target and selects the best.
pub struct SelectByInformationValue {
    pub target: String,
    pub iv_threshold: f64,
    pub selected_features: HashSet<String>,
    fitted: bool,
}

impl SelectByInformationValue {
    pub fn new(target: String, iv_threshold: f64) -> Self {
        Self {
            target,
            iv_threshold,
            selected_features: HashSet::new(),
            fitted: false,
        }
    }

    pub async fn fit(&mut self, df: &DataFrame) -> FeatureFactoryResult<()> {
        let batches = df.clone().collect().await?;
        if batches.is_empty() {
            return Err(FeatureFactoryError::InvalidParameter(
                "Empty DataFrame".to_string(),
            ));
        }
        let batch = &batches[0];
        let schema = df.schema();
        let target_array =
            as_primitive_array::<Float64Type>(batch.column_by_name(&self.target).ok_or_else(
                || FeatureFactoryError::MissingColumn(format!("Target {} missing", self.target)),
            )?);
        let target_vals: Vec<f64> = target_array.iter().flatten().par_bridge().collect();
        let total_good = target_vals.iter().filter(|&&v| v == 1.0).count() as f64;
        let total_bad = target_vals.iter().filter(|&&v| v == 0.0).count() as f64;
        let mut selected = HashSet::new();
        for field in schema.fields() {
            let name = field.name();
            if name == &self.target {
                continue;
            }
            let col_array = batch.column_by_name(name).ok_or_else(|| {
                FeatureFactoryError::MissingColumn(format!("Column {} missing", name))
            })?;
            let mut iv = 0.0;
            if is_numeric(field.data_type()) {
                let array = as_primitive_array::<Float64Type>(col_array);
                let mut vals: Vec<f64> = array.iter().flatten().par_bridge().collect();
                if vals.is_empty() {
                    continue;
                }
                vals.par_sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
                let mut bins = Vec::new();
                for i in 0..11 {
                    let idx = ((vals.len() - 1) as f64 * i as f64 / 10.0).round() as usize;
                    bins.push(vals[idx]);
                }
                for i in 0..bins.len() - 1 {
                    let lower = bins[i];
                    let upper = bins[i + 1];
                    let mut good = 0.0;
                    let mut bad = 0.0;
                    for (j, v_opt) in array.iter().enumerate() {
                        if let Some(v) = v_opt {
                            if v >= lower && v < upper {
                                if target_vals[j] == 1.0 {
                                    good += 1.0;
                                } else {
                                    bad += 1.0;
                                }
                            }
                        }
                    }
                    let pct_good = (good / total_good).max(0.0001);
                    let pct_bad = (bad / total_bad).max(0.0001);
                    iv += (pct_good - pct_bad) * (pct_good / pct_bad).ln();
                }
            } else {
                let string_array = col_array
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .ok_or_else(|| {
                        FeatureFactoryError::DataFusionError(
                            datafusion::error::DataFusionError::Plan(format!(
                                "Expected Utf8 array for column {}",
                                name
                            )),
                        )
                    })?;
                let mut counts: HashMap<String, (f64, f64)> = HashMap::new();
                for (j, v_opt) in string_array.iter().enumerate() {
                    if let Some(v) = v_opt {
                        let key = v.to_string();
                        let entry = counts.entry(key).or_insert((0.0, 0.0));
                        if target_vals[j] == 1.0 {
                            entry.0 += 1.0;
                        } else {
                            entry.1 += 1.0;
                        }
                    }
                }
                for (_k, (good, bad)) in counts.iter() {
                    let pct_good = (*good / total_good).max(0.0001);
                    let pct_bad = (*bad / total_bad).max(0.0001);
                    iv += (pct_good - pct_bad) * (pct_good / pct_bad).ln();
                }
            }
            if iv >= self.iv_threshold {
                selected.insert(name.to_string());
            }
        }
        self.selected_features = selected;
        self.fitted = true;
        Ok(())
    }

    pub fn transform(&self, df: DataFrame) -> FeatureFactoryResult<DataFrame> {
        if !self.fitted {
            return Err(FeatureFactoryError::FitNotCalled);
        }
        let keep_exprs: Vec<Expr> = df
            .schema()
            .fields()
            .iter()
            .filter_map(|f| {
                if f.name() == &self.target || self.selected_features.contains(f.name()) {
                    Some(col(f.name()))
                } else {
                    None
                }
            })
            .collect();
        if keep_exprs.is_empty() {
            return Err(FeatureFactoryError::InvalidParameter(
                "No features passed the IV threshold.".to_string(),
            ));
        }
        df.select(keep_exprs).map_err(FeatureFactoryError::from)
    }

    fn inherent_is_stateful(&self) -> bool {
        true
    }
}

/// Selects numeric features based on absolute correlation with a binary target.
/// Note: To preserve order between feature and target values, this transformer uses sequential iterators so it may be relatively slow.
pub struct SelectBySingleFeaturePerformance {
    pub target: String,
    pub correlation_threshold: f64,
    pub selected_features: HashSet<String>,
    fitted: bool,
}

impl SelectBySingleFeaturePerformance {
    pub fn new(target: String, correlation_threshold: f64) -> Self {
        Self {
            target,
            correlation_threshold,
            selected_features: HashSet::new(),
            fitted: false,
        }
    }

    pub async fn fit(&mut self, df: &DataFrame) -> FeatureFactoryResult<()> {
        let batches = df.clone().collect().await?;
        if batches.is_empty() {
            return Err(FeatureFactoryError::InvalidParameter(
                "Empty DataFrame".to_string(),
            ));
        }
        let batch = &batches[0];
        let target_array =
            as_primitive_array::<Float64Type>(batch.column_by_name(&self.target).ok_or_else(
                || FeatureFactoryError::MissingColumn(format!("Target {} missing", self.target)),
            )?);
        // Use sequential iteration to preserve order.
        let target_vals: Vec<f64> = target_array.iter().flatten().collect();
        let mut selected = HashSet::new();
        for field in df.schema().fields() {
            let name = field.name();
            if name == &self.target || !is_numeric(field.data_type()) {
                continue;
            }
            let array = as_primitive_array::<Float64Type>(batch.column_by_name(name).unwrap());
            let x: Vec<f64> = array.iter().flatten().collect();
            if x.len() != target_vals.len() || x.is_empty() {
                continue;
            }
            let n = x.len() as f64;
            let mean_x = x.iter().sum::<f64>() / n;
            let mean_y = target_vals.iter().sum::<f64>() / n;
            let cov: f64 = x
                .iter()
                .zip(target_vals.iter())
                .map(|(a, b)| (a - mean_x) * (b - mean_y))
                .sum();
            let var_x: f64 = x.iter().map(|a| (a - mean_x).powi(2)).sum();
            let var_y: f64 = target_vals.iter().map(|b| (b - mean_y).powi(2)).sum();
            if var_x == 0.0 || var_y == 0.0 {
                continue;
            }
            let corr = cov / (var_x.sqrt() * var_y.sqrt());
            if corr.abs() >= self.correlation_threshold {
                selected.insert(name.to_string());
            }
        }
        self.selected_features = selected;
        self.fitted = true;
        Ok(())
    }

    pub fn transform(&self, df: DataFrame) -> FeatureFactoryResult<DataFrame> {
        if !self.fitted {
            return Err(FeatureFactoryError::FitNotCalled);
        }
        let mut keep_exprs: Vec<Expr> = vec![col(&self.target)];
        for field in df.schema().fields() {
            if self.selected_features.contains(field.name()) {
                keep_exprs.push(col(field.name()));
            }
        }
        if keep_exprs.is_empty() {
            return Err(FeatureFactoryError::InvalidParameter(
                "No features passed single feature performance selection.".to_string(),
            ));
        }
        df.select(keep_exprs).map_err(FeatureFactoryError::from)
    }

    fn inherent_is_stateful(&self) -> bool {
        true
    }
}

/// Selects features based on the difference in target mean across different bins.
pub struct SelectByTargetMeanPerformance {
    pub target: String,
    pub mean_diff_threshold: f64,
    pub selected_features: HashSet<String>,
    fitted: bool,
}

impl SelectByTargetMeanPerformance {
    pub fn new(target: String, mean_diff_threshold: f64) -> Self {
        Self {
            target,
            mean_diff_threshold,
            selected_features: HashSet::new(),
            fitted: false,
        }
    }

    pub async fn fit(&mut self, df: &DataFrame) -> FeatureFactoryResult<()> {
        let batches = df.clone().collect().await?;
        if batches.is_empty() {
            return Err(FeatureFactoryError::InvalidParameter(
                "Empty DataFrame".to_string(),
            ));
        }
        let batch = &batches[0];
        let target_array =
            as_primitive_array::<Float64Type>(batch.column_by_name(&self.target).ok_or_else(
                || FeatureFactoryError::MissingColumn(format!("Target {} missing", self.target)),
            )?);
        let target_vals: Vec<f64> = target_array.iter().flatten().collect();
        let mut selected = HashSet::new();
        for field in df.schema().fields() {
            let name = field.name();
            if name == &self.target || !is_numeric(field.data_type()) {
                continue;
            }
            let array = as_primitive_array::<Float64Type>(batch.column_by_name(name).unwrap());
            let mut vals: Vec<f64> = array.iter().flatten().collect();
            if vals.is_empty() {
                continue;
            }
            vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let median = vals[vals.len() / 2];
            let mut group1 = Vec::new();
            let mut group2 = Vec::new();
            for (j, v_opt) in array.iter().enumerate() {
                if let Some(val) = v_opt {
                    if val < median {
                        group1.push(target_vals[j]);
                    } else {
                        group2.push(target_vals[j]);
                    }
                }
            }
            let mean1 = if !group1.is_empty() {
                group1.iter().sum::<f64>() / group1.len() as f64
            } else {
                0.0
            };
            let mean2 = if !group2.is_empty() {
                group2.iter().sum::<f64>() / group2.len() as f64
            } else {
                0.0
            };
            if (mean1 - mean2).abs() >= self.mean_diff_threshold {
                selected.insert(name.to_string());
            }
        }
        self.selected_features = selected;
        self.fitted = true;
        Ok(())
    }

    pub fn transform(&self, df: DataFrame) -> FeatureFactoryResult<DataFrame> {
        if !self.fitted {
            return Err(FeatureFactoryError::FitNotCalled);
        }
        let mut keep_exprs: Vec<Expr> = vec![col(&self.target)];
        for field in df.schema().fields() {
            if self.selected_features.contains(field.name()) {
                keep_exprs.push(col(field.name()));
            }
        }
        if keep_exprs.is_empty() {
            return Err(FeatureFactoryError::InvalidParameter(
                "No features selected by target mean performance.".to_string(),
            ));
        }
        df.select(keep_exprs).map_err(FeatureFactoryError::from)
    }

    fn inherent_is_stateful(&self) -> bool {
        true
    }
}

/// Selects features using MRMR algorithm based on feature-target relevance and redundancy.
pub struct MRMR {
    pub target: String,
    pub relevance_threshold: f64,
    pub redundancy_threshold: f64,
    pub selected_features: HashSet<String>,
    fitted: bool,
}

impl MRMR {
    pub fn new(target: String, relevance_threshold: f64, redundancy_threshold: f64) -> Self {
        Self {
            target,
            relevance_threshold,
            redundancy_threshold,
            selected_features: HashSet::new(),
            fitted: false,
        }
    }

    pub async fn fit(&mut self, df: &DataFrame) -> FeatureFactoryResult<()> {
        let batches = df.clone().collect().await?;
        if batches.is_empty() {
            return Err(FeatureFactoryError::InvalidParameter(
                "Empty DataFrame".to_string(),
            ));
        }
        let batch = &batches[0];
        let target_array =
            as_primitive_array::<Float64Type>(batch.column_by_name(&self.target).ok_or_else(
                || FeatureFactoryError::MissingColumn(format!("Target {} missing", self.target)),
            )?);
        let target_vals: Vec<f64> = target_array.iter().flatten().collect();
        let schema = df.schema();
        let mut candidates = Vec::new();
        for field in schema.fields() {
            let name = field.name();
            if name == &self.target || !is_numeric(field.data_type()) {
                continue;
            }
            let array = as_primitive_array::<Float64Type>(batch.column_by_name(name).unwrap());
            let x: Vec<f64> = array.iter().flatten().collect();
            if x.len() != target_vals.len() || x.is_empty() {
                continue;
            }
            let n = x.len() as f64;
            let mean_x = x.iter().sum::<f64>() / n;
            let mean_y = target_vals.iter().sum::<f64>() / n;
            let cov: f64 = x
                .iter()
                .zip(target_vals.iter())
                .map(|(a, b)| (a - mean_x) * (b - mean_y))
                .sum();
            let var_x: f64 = x.iter().map(|a| (a - mean_x).powi(2)).sum();
            let var_y: f64 = target_vals.iter().map(|b| (b - mean_y).powi(2)).sum();
            if var_x == 0.0 || var_y == 0.0 {
                continue;
            }
            let corr = cov / (var_x.sqrt() * var_y.sqrt());
            if corr.abs() >= self.relevance_threshold {
                candidates.push((name.to_string(), corr.abs()));
            }
        }
        let mut selected = Vec::<String>::new();
        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        for (feat, _) in candidates {
            let mut redundant = false;
            for sel in &selected {
                let array_feat =
                    as_primitive_array::<Float64Type>(batch.column_by_name(&feat).unwrap());
                let array_sel =
                    as_primitive_array::<Float64Type>(batch.column_by_name(sel).unwrap());
                let x: Vec<f64> = array_feat.iter().flatten().collect();
                let y: Vec<f64> = array_sel.iter().flatten().collect();
                if x.len() != y.len() || x.is_empty() {
                    continue;
                }
                let n = x.len() as f64;
                let mean_x = x.iter().sum::<f64>() / n;
                let mean_y = y.iter().sum::<f64>() / n;
                let cov: f64 = x
                    .iter()
                    .zip(y.iter())
                    .map(|(a, b)| (a - mean_x) * (b - mean_y))
                    .sum();
                let var_x: f64 = x.iter().map(|a| (a - mean_x).powi(2)).sum();
                let var_y: f64 = y.iter().map(|b| (b - mean_y).powi(2)).sum();
                if var_x == 0.0 || var_y == 0.0 {
                    continue;
                }
                let corr = cov / (var_x.sqrt() * var_y.sqrt());
                if corr.abs() > self.redundancy_threshold {
                    redundant = true;
                    break;
                }
            }
            if !redundant {
                selected.push(feat);
            }
        }
        self.selected_features = selected.into_iter().collect();
        self.fitted = true;
        Ok(())
    }

    pub fn transform(&self, df: DataFrame) -> FeatureFactoryResult<DataFrame> {
        if !self.fitted {
            return Err(FeatureFactoryError::FitNotCalled);
        }
        let mut keep_exprs: Vec<Expr> = vec![col(&self.target)];
        for field in df.schema().fields() {
            if self.selected_features.contains(field.name()) {
                keep_exprs.push(col(field.name()));
            }
        }
        if keep_exprs.is_empty() {
            return Err(FeatureFactoryError::InvalidParameter(
                "No features selected by MRMR.".to_string(),
            ));
        }
        df.select(keep_exprs).map_err(FeatureFactoryError::from)
    }

    fn inherent_is_stateful(&self) -> bool {
        true
    }
}

// Implement the Transformer trait for all the above feature selection transformers.
impl_transformer!(DropFeatures);
impl_transformer!(DropConstantFeatures);
impl_transformer!(DropDuplicateFeatures);
impl_transformer!(DropCorrelatedFeatures);
impl_transformer!(SmartCorrelatedSelection);
impl_transformer!(DropHighPSIFeatures);
impl_transformer!(SelectByInformationValue);
impl_transformer!(SelectBySingleFeaturePerformance);
impl_transformer!(SelectByTargetMeanPerformance);
impl_transformer!(MRMR);
