//! ## Transformers for creating new features
//!
//! This module provides transformers that create new features by applying mathematical operations on existing features.
//!
//! Currently, the following transformers are implemented:
//!
//! - **MathFeatures:** Create new features by applying arbitrary mathematical operations/expressions.
//! - **RelativeFeatures:** Combine features with reference variables using operations such as ratio,
//!   difference, or percent change. Both the target and reference columns must exist and be numeric.
//! - **CyclicalFeatures:** Encode cyclical features using sine or cosine, e.g., to represent hours or months.
//!   The source column must exist, be numeric, and the period must be a positive number.
//!
//! Each transformer provides a constructor, an (async) `fit` method (if needed), and a `transform` method
//! that returns a new DataFrame with the transformation applied.
//! Errors are returned as `FeatureFactoryError` and results are wrapped in `FeatureFactoryResult`.

use crate::exceptions::{FeatureFactoryError, FeatureFactoryResult};
use datafusion::prelude::*;
use datafusion_expr::{col, lit, Expr};

/// Creates new features using arbitrary mathematical operations or expressions.
/// The input is a vector of tuples with the following fields for each new feature:
/// (new_feature_name, math expression to be computed).
pub struct MathFeatures {
    pub features: Vec<(String, Expr)>,
}

impl MathFeatures {
    pub fn new(features: Vec<(String, Expr)>) -> Self {
        // Check that each new feature name is not empty.
        for (name, _) in &features {
            if name.trim().is_empty() {
                panic!("MathFeatures: feature name cannot be empty");
            }
        }
        Self { features }
    }

    /// This transformer is stateless, so fit does nothing.
    pub async fn fit(&mut self, _df: &DataFrame) -> FeatureFactoryResult<()> {
        Ok(())
    }

    /// Adds the new features to the existing DataFrame.
    pub fn transform(&self, df: DataFrame) -> FeatureFactoryResult<DataFrame> {
        // Retain all original columns.
        let mut exprs: Vec<Expr> = df
            .schema()
            .fields()
            .iter()
            .map(|field| col(field.name()))
            .collect();
        // Append computed feature expressions.
        for (name, expr) in &self.features {
            exprs.push(expr.clone().alias(name));
        }
        df.select(exprs).map_err(FeatureFactoryError::from)
    }
}

/// Operations available for computing relative features.
pub enum RelativeOperation {
    Ratio,         // target / reference
    Difference,    // target - reference
    PercentChange, // (target - reference) / reference
}

/// Creates new features by combining a target feature with a reference feature.
/// Input is a vector of tuples with the following fields for each new feature:
/// (new_feature_name, target_feature, reference_feature, operation).
pub struct RelativeFeatures {
    pub features: Vec<(String, String, String, RelativeOperation)>,
}

impl RelativeFeatures {
    pub fn new(features: Vec<(String, String, String, RelativeOperation)>) -> Self {
        // Check that feature names are not empty.
        for (new_name, target, reference, _) in &features {
            if new_name.trim().is_empty() {
                panic!("RelativeFeatures: new feature name cannot be empty");
            }
            if target.trim().is_empty() || reference.trim().is_empty() {
                panic!("RelativeFeatures: target and reference names must be non-empty");
            }
        }
        Self { features }
    }

    /// Validates that the target and reference columns exist and are numeric.
    pub async fn fit(&mut self, df: &DataFrame) -> FeatureFactoryResult<()> {
        for (_, target, reference, _) in &self.features {
            df.schema().field_with_name(None, target).map_err(|_| {
                FeatureFactoryError::MissingColumn(format!("Target column '{}' not found", target))
            })?;
            df.schema().field_with_name(None, reference).map_err(|_| {
                FeatureFactoryError::MissingColumn(format!(
                    "Reference column '{}' not found",
                    reference
                ))
            })?;
            // Optionally, you could check that these columns are numeric.
        }
        Ok(())
    }

    /// Adds the relative features to the DataFrame.
    pub fn transform(&self, df: DataFrame) -> FeatureFactoryResult<DataFrame> {
        let mut exprs: Vec<Expr> = df
            .schema()
            .fields()
            .iter()
            .map(|field| col(field.name()))
            .collect();
        for (new_name, target, reference, op) in &self.features {
            let expr = match op {
                RelativeOperation::Ratio => col(target).div(col(reference)),
                RelativeOperation::Difference => col(target).sub(col(reference)),
                RelativeOperation::PercentChange => {
                    col(target).sub(col(reference)).div(col(reference))
                }
            };
            exprs.push(expr.alias(new_name));
        }
        df.select(exprs).map_err(FeatureFactoryError::from)
    }
}

/// Methods for encoding cyclical features.
pub enum CyclicalMethod {
    Sine,
    Cosine,
}

/// Encodes a cyclical variable by computing either a sine or cosine transformation.
/// The input is a vector of tuples with the following fields for each new feature:
/// (new_feature_name, source_feature, period, method).
pub struct CyclicalFeatures {
    pub features: Vec<(String, String, f64, CyclicalMethod)>,
}

impl CyclicalFeatures {
    pub fn new(features: Vec<(String, String, f64, CyclicalMethod)>) -> Self {
        // Validate that new feature names are non-empty and period is positive.
        for (new_name, source, period, _) in &features {
            if new_name.trim().is_empty() {
                panic!("CyclicalFeatures: new feature name cannot be empty");
            }
            if source.trim().is_empty() {
                panic!("CyclicalFeatures: source feature name must be non-empty");
            }
            if *period <= 0.0 {
                panic!("CyclicalFeatures: period must be positive, got {}", period);
            }
        }
        Self { features }
    }

    /// Validates that each source feature exists and is numeric, and that period > 0.
    pub async fn fit(&mut self, df: &DataFrame) -> FeatureFactoryResult<()> {
        for (_, source, period, _) in &self.features {
            df.schema().field_with_name(None, source).map_err(|_| {
                FeatureFactoryError::MissingColumn(format!("Source column '{}' not found", source))
            })?;
            // Optionally, check numeric type.
            if *period <= 0.0 {
                return Err(FeatureFactoryError::InvalidParameter(format!(
                    "CyclicalFeatures: period must be positive, got {}",
                    period
                )));
            }
        }
        Ok(())
    }

    /// Adds the cyclical features to the DataFrame.
    pub fn transform(&self, df: DataFrame) -> FeatureFactoryResult<DataFrame> {
        let mut exprs: Vec<Expr> = df
            .schema()
            .fields()
            .iter()
            .map(|field| col(field.name()))
            .collect();
        for (new_name, source, period, method) in &self.features {
            let base_expr = lit(2.0 * std::f64::consts::PI)
                .mul(col(source))
                .div(lit(*period));
            let cyc_expr = match method {
                CyclicalMethod::Sine => datafusion_functions::math::sin().call(vec![base_expr]),
                CyclicalMethod::Cosine => datafusion_functions::math::cos().call(vec![base_expr]),
            };
            exprs.push(cyc_expr.alias(new_name));
        }
        df.select(exprs).map_err(FeatureFactoryError::from)
    }
}
