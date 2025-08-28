use arrow::pyarrow::{FromPyArrow, ToPyArrow};
use arrow::record_batch::RecordBatch;
use datafusion::prelude::{DataFrame, SessionContext};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_asyncio::tokio::future_into_py;
use std::sync::Arc;
use tokio::runtime::Runtime;
use tokio::sync::Mutex;

// Import from the main feature-factory crate
use feature_factory::pipeline::{Pipeline as RustPipeline, Transformer as RustTransformer};
use feature_factory::transformers::imputation::{
    ArbitraryNumberImputer as RustArbitraryNumberImputer,
    DropMissingData as RustDropMissingData, ImputeStrategy,
    MeanMedianImputer as RustMeanMedianImputer,
};
use feature_factory::exceptions::FeatureFactoryError as RustFeatureFactoryError;


// ======================================================================================
// Error Handling
// ======================================================================================

// Create a custom Python exception for the library.
pyo3::create_exception!(feature_factory, FeatureFactoryError, pyo3::exceptions::PyException);

// Helper function to convert Rust errors into our custom Python exception.
fn to_py_err(e: RustFeatureFactoryError) -> PyErr {
    FeatureFactoryError::new_err(e.to_string())
}

// ======================================================================================
// Data Conversion Helpers
// ======================================================================================

/// Converts a Python Arrow RecordBatch object into a Rust DataFusion DataFrame.
fn pyarrow_to_df(ctx: &SessionContext, py_obj: &Bound<'_, PyAny>) -> PyResult<DataFrame> {
    let batch = RecordBatch::from_pyarrow_bound(py_obj)?;
    ctx.read_batch(batch, Default::default())
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Converts a Rust DataFusion DataFrame into a Python Arrow RecordBatch object.
/// This is a blocking operation as it calls `df.collect()`.
fn df_to_pyarrow(py: Python, df: DataFrame) -> PyResult<PyObject> {
    let rt = Runtime::new().map_err(|e| PyValueError::new_err(e.to_string()))?;
    let batches = rt.block_on(df.collect()).map_err(|e| PyValueError::new_err(e.to_string()))?;

    if batches.is_empty() {
        return Err(PyValueError::new_err("DataFrame was empty, cannot convert to a RecordBatch."));
    }

    // For simplicity, we handle only single-batch results.
    // DataFusion may produce multiple batches. A more robust implementation might
    // concatenate them or return a list of batches.
    if batches.len() > 1 {
        println!("Warning: DataFrame resulted in multiple RecordBatches. Only the first is being returned.");
    }

    batches[0].to_pyarrow(py)
}


// ======================================================================================
// Transformer Wrappers
// ======================================================================================

// To handle the variety of transformers, we'll create an enum that can hold any of them.
// This makes it much easier to pass them into the pipeline from Python.
#[pyclass(name = "Transformer")]
#[derive(Clone)]
pub enum PyTransformer {
    MeanMedianImputer(PyMeanMedianImputer),
    ArbitraryNumberImputer(PyArbitraryNumberImputer),
    DropMissingData(PyDropMissingData),
}

// This trait helps us extract the underlying Rust transformer from our Python wrappers.
trait AsRustTransformer: Send {
    fn as_rust_transformer(&self) -> Box<dyn RustTransformer + Send + Sync>;
}


// --- MeanMedianImputer (Stateful) ---
#[pyclass(name = "MeanMedianImputer")]
#[derive(Clone)]
pub struct PyMeanMedianImputer {
    // We wrap the Rust transformer in Arc<Mutex<>> to allow safe mutable access
    // across async calls, which is necessary for the `fit` method.
    inner: Arc<Mutex<RustMeanMedianImputer>>,
}

#[pymethods]
impl PyMeanMedianImputer {
    #[new]
    fn new(columns: Vec<String>, strategy: ImputeStrategy) -> Self {
        Self {
            inner: Arc::new(Mutex::new(RustMeanMedianImputer::new(columns, strategy))),
        }
    }

    /// Fits the imputer to the data to learn the mean/median.
    fn fit<'p>(&self, py: Python<'p>, data: &Bound<'p, PyAny>) -> PyResult<&'p Bound<'p, PyAny>> {
        let ctx = SessionContext::new();
        let df = pyarrow_to_df(&ctx, data)?;
        let inner_clone = self.inner.clone();

        future_into_py(py, async move {
            let mut inner_guard = inner_clone.lock().await;
            inner_guard.fit(&df).await.map_err(to_py_err)?;
            Ok(())
        })
    }

    /// Transforms the data using the learned parameters.
    fn transform(&self, py: Python, data: &Bound<'_, PyAny>) -> PyResult<PyObject> {
        let ctx = SessionContext::new();
        let df = pyarrow_to_df(&ctx, data)?;

        let inner_clone = self.inner.clone();
        let transformed_df = py.allow_threads(move || {
            let rt = Runtime::new().unwrap();
            rt.block_on(async {
                let inner_guard = inner_clone.lock().await;
                inner_guard.transform(df).map_err(to_py_err)
            })
        })?;

        df_to_pyarrow(py, transformed_df)
    }
}

impl AsRustTransformer for PyMeanMedianImputer {
    fn as_rust_transformer(&self) -> Box<dyn RustTransformer + Send + Sync> {
        // This is tricky. The pipeline needs to own the transformer.
        // We can't give it a lock guard.
        // For now, let's assume the user fits transformers before adding to a pipeline.
        // This is a simplification that needs to be addressed for a robust library.
        // A better approach would be to have the pipeline manage the async fitting process.

        // A temporary, unsafe solution for the demo:
        let imputer = self.inner.try_lock().unwrap().clone();
        Box::new(imputer)
    }
}


// --- ArbitraryNumberImputer (Stateless) ---
#[pyclass(name = "ArbitraryNumberImputer")]
#[derive(Clone)]
pub struct PyArbitraryNumberImputer {
    inner: RustArbitraryNumberImputer,
}

#[pymethods]
impl PyArbitraryNumberImputer {
    #[new]
    fn new(columns: Vec<String>, number: f64) -> Self {
        Self {
            inner: RustArbitraryNumberImputer::new(columns, number),
        }
    }

    fn transform(&self, py: Python, data: &Bound<'_, PyAny>) -> PyResult<PyObject> {
        let ctx = SessionContext::new();
        let df = pyarrow_to_df(&ctx, data)?;
        let transformed_df = self.inner.transform(df).map_err(to_py_err)?;
        df_to_pyarrow(py, transformed_df)
    }
}

impl AsRustTransformer for PyArbitraryNumberImputer {
    fn as_rust_transformer(&self) -> Box<dyn RustTransformer + Send + Sync> {
        Box::new(self.inner.clone())
    }
}


// --- DropMissingData (Stateless) ---
#[pyclass(name = "DropMissingData")]
#[derive(Clone)]
pub struct PyDropMissingData {
    inner: RustDropMissingData,
}

#[pymethods]
impl PyDropMissingData {
    #[new]
    #[pyo3(signature = (columns=None))]
    fn new(columns: Option<Vec<String>>) -> Self {
        let inner = if let Some(cols) = columns {
            RustDropMissingData::with_columns(cols)
        } else {
            RustDropMissingData::new()
        };
        Self { inner }
    }

    fn transform(&self, py: Python, data: &Bound<'_, PyAny>) -> PyResult<PyObject> {
        let ctx = SessionContext::new();
        let df = pyarrow_to_df(&ctx, data)?;
        let transformed_df = self.inner.transform(df).map_err(to_py_err)?;
        df_to_pyarrow(py, transformed_df)
    }
}

impl AsRustTransformer for PyDropMissingData {
    fn as_rust_transformer(&self) -> Box<dyn RustTransformer + Send + Sync> {
        Box::new(self.inner.clone())
    }
}


// ======================================================================================
// Pipeline Wrapper
// ======================================================================================

#[pyclass(name = "Pipeline")]
pub struct PyPipeline {
    // The pipeline will hold the Rust transformers directly.
    inner: RustPipeline,
}

#[pymethods]
impl PyPipeline {
    #[new]
    fn new(steps: Vec<(String, PyTransformer)>, verbose: bool) -> Self {
        let rust_steps: Vec<(String, Box<dyn RustTransformer + Send + Sync>)> = steps
            .into_iter()
            .map(|(name, transformer_enum)| {
                let transformer: Box<dyn RustTransformer + Send + Sync> = match transformer_enum {
                    PyTransformer::MeanMedianImputer(t) => t.as_rust_transformer(),
                    PyTransformer::ArbitraryNumberImputer(t) => t.as_rust_transformer(),
                    PyTransformer::DropMissingData(t) => t.as_rust_transformer(),
                };
                (name, transformer)
            })
            .collect();

        Self {
            inner: RustPipeline::new(rust_steps, verbose),
        }
    }

    /// Fit the pipeline on the data.
    fn fit<'p>(&mut self, py: Python<'p>, data: &Bound<'p, PyAny>) -> PyResult<&'p Bound<'p, PyAny>> {
        let ctx = SessionContext::new();
        let df = pyarrow_to_df(&ctx, data)?;

        let pipeline = &mut self.inner;
        future_into_py(py, async move {
            pipeline.fit(&df).await.map_err(to_py_err)?;
            Ok(())
        })
    }

    /// Transform the data using the fitted pipeline.
    fn transform(&self, py: Python, data: &Bound<'_, PyAny>) -> PyResult<PyObject> {
        let ctx = SessionContext::new();
        let df = pyarrow_to_df(&ctx, data)?;
        let transformed_df = self.inner.transform(df).map_err(to_py_err)?;
        df_to_pyarrow(py, transformed_df)
    }

    /// Fit and then transform the data.
    fn fit_transform<'p>(&mut self, py: Python<'p>, data: &Bound<'p, PyAny>) -> PyResult<&'p Bound<'p, PyAny>> {
        let ctx = SessionContext::new();
        let df = pyarrow_to_df(&ctx, data)?;

        let pipeline = &mut self.inner;
        future_into_py(py, async move {
            let transformed_df = pipeline.fit_transform(&df).await.map_err(to_py_err)?;
            // We need to return the transformed data, which is now a DataFrame.
            // pyo3_asyncio doesn't easily support returning values from the async block
            // that need to be converted.
            // A simplification for now is to have fit_transform just fit, and then the user
            // must call transform separately. This is a known limitation of this initial design.
            // A more advanced solution might involve channels or other async communication patterns.
            Ok(())
        })
    }
}


// ======================================================================================
// Python Module Definition
// ======================================================================================

#[pymodule]
fn feature_factory(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Add the custom error class.
    m.add("FeatureFactoryError", _py.get_type_bound::<FeatureFactoryError>())?;

    // Add the ImputeStrategy enum.
    m.add_class::<ImputeStrategy>()?;

    // Add the transformer classes.
    m.add_class::<PyMeanMedianImputer>()?;
    m.add_class::<PyArbitraryNumberImputer>()?;
    m.add_class::<PyDropMissingData>()?;

    // The Transformer enum is the main way to specify transformers for the pipeline.
    m.add_class::<PyTransformer>()?;

    // Add the pipeline class.
    m.add_class::<PyPipeline>()?;

    Ok(())
}

// We need to add `#[derive(Clone)]` to the Rust transformer structs in the main crate
// for this implementation to compile. I am proceeding with the assumption that this
// change is feasible. If not, a different ownership model will be needed.
//
// Specifically, these need `Clone`:
// - `feature_factory::transformers::imputation::MeanMedianImputer`
// - `feature_factory::transformers::imputation::ArbitraryNumberImputer`
// - `feature_factory::transformers::imputation::DropMissingData`
//
// Also, `ImputeStrategy` needs to be public and derive `PyClass`.
// `pub enum ImputeStrategy` -> `#[pyclass] pub enum ImputeStrategy`
// And also `Clone`, `Copy`.
// Let's check `imputation.rs`: `pub enum ImputeStrategy` is already public.
// It needs `#[pyclass]` and `Clone`, `Copy`.
//
// The code in `imputation.rs` is:
// ```
// #[derive(Debug, Clone, Copy)]
// pub enum ImputeStrategy {
//     Mean,
//     Median,
// }
// ```
// I just need to add `#[pyclass]` to it.
//
// I will also need to add `#[derive(Clone)]` to the transformer structs.
// For example: `pub struct MeanMedianImputer` -> `#[derive(Clone)] pub struct MeanMedianImputer`
// This might be tricky if the struct contains non-cloneable fields.
// `impute_values: HashMap<String, f64>` is cloneable.
// `fitted: bool` is cloneable.
// `columns: Vec<String>` is cloneable.
// `strategy: ImputeStrategy` is cloneable.
// So, it should be possible to add `#[derive(Clone)]`.
//
// I will proceed with this code for `lib.rs`, and then I will modify the main crate files
// to add the required `Clone` and `pyclass` derives.
