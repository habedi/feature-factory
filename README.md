<div align="center">
  <picture>
    <source media="(prefers-color-scheme: light)" srcset="logo.png">
    <source media="(prefers-color-scheme: dark)" srcset="logo.png">
    <img alt="template-rust-project logo" src="logo.png" height="30%" width="30%">
  </picture>
</div>
<br>

## Feature Factory

[![Tests](https://img.shields.io/github/actions/workflow/status/habedi/feature-factory/tests.yml?label=tests&style=flat&labelColor=555555&logo=github)](https://github.com/habedi/feature-factory/actions/workflows/tests.yml)
[![Lints](https://img.shields.io/github/actions/workflow/status/habedi/feature-factory/lints.yml?label=lints&style=flat&labelColor=555555&logo=github)](https://github.com/habedi/feature-factory/actions/workflows/lints.yml)
[![Code Coverage](https://img.shields.io/codecov/c/github/habedi/feature-factory?style=flat&labelColor=555555&logo=codecov)](https://codecov.io/gh/habedi/feature-factory)
[![CodeFactor](https://img.shields.io/codefactor/grade/github/habedi/feature-factory?style=flat&labelColor=555555&logo=codefactor)](https://www.codefactor.io/repository/github/habedi/feature-factory)
[![Crates.io](https://img.shields.io/crates/v/feature-factory.svg?style=flat&color=fc8d62&logo=rust)](https://crates.io/crates/feature-factory)
[![Docs.rs](https://img.shields.io/badge/docs.rs-feature--factory-66c2a5?style=flat&labelColor=655555&logo=docs.rs)](https://docs.rs/feature-factory)
[![Downloads](https://img.shields.io/crates/d/feature-factory?style=flat&logo=rust)](https://crates.io/crates/feature-factory)
[![MSRV](https://img.shields.io/badge/MSRV-1.83.0-orange?style=flat&logo=rust&label=msrv)](https://github.com/rust-lang/rust/releases/tag/1.83.0)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-007ec6?style=flat&labelColor=555555&logo=open-source-initiative)](https://github.com/habedi/feature-factory)
[![Status: Alpha](https://img.shields.io/badge/status-alpha-pink.svg)](https://github.com/habedi/feature-factory)

Feature Factory is a high-performance feature engineering library for Rust.
It is built on top of [Apache DataFusion](https://datafusion.apache.org/) and uses it internally for fast
in-memory data processing.
Feature Factory is inspired by the [Feature engine](https://feature-engine.trainindata.com/en/latest/) Python library,
and provides a wide range of components for common feature engineering tasks like imputation, encoding,
discretization, and selecting the best features.

Feature Factory aims to be feature-rich and provide an API similar to
[Scikit-learn](https://scikit-learn.org/stable/) with performance benefits of Rust.
Its components (referred to as transformers) follow a
[fit-transform paradigm](https://scikit-learn.org/stable/data_transforms.html)
where a transformer is first fitted to the data if needed and then used to transform the data.

> [!IMPORTANT]
> Feature Factory is currently in the early stage of development.
> APIs are unstable and may change without notice.
> Inconsistencies in documentation are expected and not all features are implemented yet.
> It is not thoroughly tested, benchmarked, or optimized for performance yet.
> Bug reports, feature requests, and contributions are welcome!

### Features

- **High Performance** - Feature Factory uses Apache DataFusion as the backend data processing engine to implement the
  transformers.
- **Scikit-learn API** - Feature Factory provides a Scikit-learn-like API for feature engineering, which most data
  scientists are familiar with.
- **Pipeline API** - Feature Factory provides a pipeline API that allows users to chain multiple transformers together
  to create a feature engineering pipeline.
- **Large Collection of Transformers** - Currently, Feature Factory provides the following transformers that can be used
  in a feature engineering pipeline:

| **Category**                                                                     | **Transformers**                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   | Status |
|----------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------|
| [**Imputation**](src/transformers/imputation.rs)                                 | - `MeanMedianImputer`: Replace missing values with mean (or median).  <br>- `ArbitraryNumberImputer`: Replace missing values with an arbitrary number.  <br>- `EndTailImputer`: Replace missing values with values at distribution tails.  <br>- `CategoricalImputer`: Replace missing values with an arbitrary string or most frequent category.  <br>- `AddMissingIndicator`: Add a binary indicator for missing values.  <br>- `DropMissingData`: Remove rows with missing values.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              | Tested |
| [**Categorical Encoding**](src/transformers/categorical_encoding.rs)             | - `OneHotEncoder`: Perform one-hot encoding.  <br>- `CountFrequencyEncoder`: Replace categories with their frequencies.  <br>- `OrdinalEncoder`: Replace categories with ordered numbers.  <br>- `MeanEncoder`: Replace categories with target mean.  <br>- `WoEEncoder`: Replace categories with the weight of evidence.  <br>- `RareLabelEncoder`: Group infrequent categories.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | Tested |
| [**Variable Discretization**](src/transformers/variable_discretization.rs)       | - `ArbitraryDiscretizer`: Discretize based on user-defined intervals.  <br>- `EqualFrequencyDiscretizer`: Discretize into equal-frequency bins.  <br>- `EqualWidthDiscretizer`: Discretize into equal-width bins.  <br>- `GeometricWidthDiscretizer`: Discretize into geometric intervals.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         | Tested |
| [**Outlier Handling**](src/transformers/outlier_handling.rs)                     | - `ArbitraryOutlierCapper`: Cap outliers at user-defined bounds.  <br>- `Winsorizer`: Cap outliers using percentile thresholds.  <br>- `OutlierTrimmer`: Remove outliers from the dataset.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         | Tested |
| [**Numerical Transformations**](src/transformers/numerical_transformations.rs)   | - `LogTransformer`: Apply logarithmic transformation.  <br>- `LogCpTransformer`: Apply log transformation with a constant.  <br>- `ReciprocalTransformer`: Apply reciprocal transformation.  <br>- `PowerTransformer`: Apply power transformation.  <br>- `BoxCoxTransformer`: Apply Box-Cox transformation.  <br>- `YeoJohnsonTransformer`: Apply Yeo-Johnson transformation.  <br>- `ArcsinTransformer`: Apply arcsin transformation.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            | Tested |
| [**Feature Creation**](src/transformers/feature_creation.rs)                     | - `MathFeatures`: Create new features with mathematical operations.  <br>- `RelativeFeatures`: Combine features with reference features.  <br>- `CyclicalFeatures`: Encode cyclical features using sine or cosine.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 | Tested |
| [**Datetime Features**](src/transformers/datetime_features.rs)                   | - `DatetimeFeatures`: Extract features from datetime values.  <br>- `DatetimeSubtraction`: Compute time differences between datetime values.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       | Tested |
| [**Scaling & Normalization**](src/transformers/scaling_and_normalization.rs)     | - `MeanNormalizationScaler`: Scale features using mean normalization.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |        |
| [**Feature Selection**](src/transformers/feature_selection.rs)                   | - `DropFeatures`: Drop specific features.  <br>- `DropConstantFeatures`: Remove constant and quasi-constant features.  <br>- `DropDuplicateFeatures`: Remove duplicate features.  <br>- `DropCorrelatedFeatures`: Remove highly correlated features.  <br>- `SmartCorrelatedSelection`: Select the best features from correlated groups.  <br>- `DropHighPSIFeatures`: Drop features based on Population Stability Index (PSI).  <br>- `SelectByInformationValue`: Select features based on information value.  <br>- `SelectByShuffling`: Select features by evaluating performance after shuffling.  <br>- `SelectBySingleFeaturePerformance`: Select features based on univariate estimators.  <br>- `SelectByTargetMeanPerformance`: Select features based on target mean encoding.  <br>- `RecursiveFeatureElimination`: Recursively eliminate features based on model performance.  <br>- `RecursiveFeatureAddition`: Recursively add features based on model performance.  <br>- `ProbeFeatureSelection`: Select features by comparing them to random variables.  <br>- `MRMR`: Select features using Maximum Relevance Minimum Redundancy. |        |
| [**Additional Transformations**](src/transformers/additional_transformations.rs) | - `PolynomialFeatures`: Generate polynomial and interaction features.  <br>- `SplineTransformer`: Generate spline-based features for non-linear transformations.  <br>- `Binarizer`: Convert numerical features into binary indicators based on a threshold.  <br>- `FeatureHasher`: Convert categorical features into a numerical matrix using hashing.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |        | 

> [!NOTE]
> Status shows whether the module is `Tested` and `Benchmarked`.
> Empty status means the module is not tested and benchmarked yet.

### Installation

```bash
cargo add feature-factory
```

*Feature Factory requires Rust 1.83 or later.*

### Documentation

You can find the latest API documentation on [docs.rs/feature-factory](https://docs.rs/feature-factory).

### Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for details on how to make a contribution.

### Logo

The mascot of this project is named "Weldon the Penguin".
He is a Rustacean penguin who loves to swim in the sea and play video games—and is always ready to help you with your
data.

The logo was created using Gimp, ComfyUI, and a Flux Schnell v2 model.

### Licensing

Feature-factory is available under the terms of either of these licenses:

* MIT License ([LICENSE-MIT](LICENSE-MIT))
* Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
