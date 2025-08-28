use std::sync::{Arc, OnceLock};

use arrow::pyarrow::{FromPyArrow, ToPyArrow};
use arrow::record_batch::RecordBatch;
use datafusion::prelude::{DataFrame, SessionContext};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use tokio::runtime::Runtime;
use tokio::sync::Mutex;

// feature-factory crate imports
use ::feature_factory::exceptions::FeatureFactoryError as RustFeatureFactoryError;
use ::feature_factory::pipeline::{Pipeline as RustPipeline, Transformer as RustTransformer};
use ::feature_factory::transformers::imputation::{
    ArbitraryNumberImputer as RustArbitraryNumberImputer,
    DropMissingData as RustDropMissingData,
    ImputeStrategy,
    MeanMedianImputer as RustMeanMedianImputer,
};

// ======================================================================================
// Global Tokio runtime (single instance to avoid overhead).
// ======================================================================================

static RUNTIME: OnceLock<Runtime> = OnceLock::new();

fn runtime() -> &'static Runtime {
    RUNTIME.get_or_init(|| Runtime::new().expect("Failed to create Tokio runtime"))
}

// ======================================================================================
// Error Handling
// ======================================================================================

pyo3::create_exception!(feature_factory, FeatureFactoryError, pyo3::exceptions::PyException);

fn to_py_err(e: RustFeatureFactoryError) -> PyErr {
    FeatureFactoryError::new_err(e.to_string())
}

// ======================================================================================
// Data Conversion Helpers
// ======================================================================================

fn pyarrow_to_df(ctx: &SessionContext, obj: &Bound<'_, PyAny>) -> PyResult<DataFrame> {
    let batch = RecordBatch::from_pyarrow_bound(obj)?;
    ctx.read_batch(batch)
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

fn df_to_pyarrow(py: Python<'_>, df: DataFrame) -> PyResult<PyObject> {
    let batches = runtime()
        .block_on(df.collect())
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    if batches.is_empty() {
        return Err(PyValueError::new_err("Empty DataFrame result"));
    }
    if batches.len() > 1 {
        eprintln!(
            "feature_factory: multiple RecordBatches produced; returning first only ({} total)",
            batches.len()
        );
    }
    batches[0].to_pyarrow(py)
}

// ======================================================================================
// Transformer Wrapper Enum
// ======================================================================================

#[pyclass(name = "Transformer")]
#[derive(Clone)]
pub enum PyTransformer {
    MeanMedianImputer(PyMeanMedianImputer),
    ArbitraryNumberImputer(PyArbitraryNumberImputer),
    DropMissingData(PyDropMissingData),
}

trait AsRustTransformer: Send + Sync {
    fn as_rust_transformer(&self) -> Box<dyn RustTransformer + Send + Sync>;
}

// ======================================================================================
// MeanMedianImputer (stateful)
// ======================================================================================

#[pyclass(name = "MeanMedianImputer")]
#[derive(Clone)]
pub struct PyMeanMedianImputer {
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

    fn fit(&self, data: &Bound<'_, PyAny>) -> PyResult<()> {
        let ctx = SessionContext::new();
        let df = pyarrow_to_df(&ctx, data)?;
        let inner = self.inner.clone();
        runtime()
            .block_on(async {
                let mut guard = inner.lock().await;
                guard.fit(&df).await
            })
            .map_err(to_py_err)?;
        Ok(())
    }

    fn transform(&self, py: Python<'_>, data: &Bound<'_, PyAny>) -> PyResult<PyObject> {
        let ctx = SessionContext::new();
        let df = pyarrow_to_df(&ctx, data)?;
        let inner = self.inner.clone();
        let out_df = py.allow_threads(move || {
            runtime()
                .block_on(async {
                    let guard = inner.lock().await;
                    guard.transform(df)
                })
                .map_err(to_py_err)
        })?;
        df_to_pyarrow(py, out_df)
    }
}

impl AsRustTransformer for PyMeanMedianImputer {
    fn as_rust_transformer(&self) -> Box<dyn RustTransformer + Send + Sync> {
        let cloned = runtime().block_on(async {
            let guard = self.inner.lock().await;
            guard.clone()
        });
        Box::new(cloned)
    }
}

// ======================================================================================
// ArbitraryNumberImputer (stateless)
// ======================================================================================

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

    fn transform(&self, py: Python<'_>, data: &Bound<'_, PyAny>) -> PyResult<PyObject> {
        let ctx = SessionContext::new();
        let df = pyarrow_to_df(&ctx, data)?;
        let out_df = self.inner.transform(df).map_err(to_py_err)?;
        df_to_pyarrow(py, out_df)
    }
}

impl AsRustTransformer for PyArbitraryNumberImputer {
    fn as_rust_transformer(&self) -> Box<dyn RustTransformer + Send + Sync> {
        Box::new(self.inner.clone())
    }
}

// ======================================================================================
// DropMissingData (stateless)
// ======================================================================================

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
        let inner = match columns {
            Some(cols) => RustDropMissingData::with_columns(cols),
            None => RustDropMissingData::new(),
        };
        Self { inner }
    }

    fn transform(&self, py: Python<'_>, data: &Bound<'_, PyAny>) -> PyResult<PyObject> {
        let ctx = SessionContext::new();
        let df = pyarrow_to_df(&ctx, data)?;
        let out_df = self.inner.transform(df).map_err(to_py_err)?;
        df_to_pyarrow(py, out_df)
    }
}

impl AsRustTransformer for PyDropMissingData {
    fn as_rust_transformer(&self) -> Box<dyn RustTransformer + Send + Sync> {
        Box::new(self.inner.clone())
    }
}

// ======================================================================================
// Pipeline
// ======================================================================================

#[pyclass(name = "Pipeline")]
pub struct PyPipeline {
    inner: RustPipeline,
}

#[pymethods]
impl PyPipeline {
    #[new]
    fn new(steps: Vec<(String, PyTransformer)>, verbose: bool) -> Self {
        let rust_steps: Vec<(String, Box<dyn RustTransformer + Send + Sync>)> = steps
            .into_iter()
            .map(|(name, t)| {
                let boxed: Box<dyn RustTransformer + Send + Sync> = match t {
                    PyTransformer::MeanMedianImputer(v) => v.as_rust_transformer(),
                    PyTransformer::ArbitraryNumberImputer(v) => v.as_rust_transformer(),
                    PyTransformer::DropMissingData(v) => v.as_rust_transformer(),
                };
                (name, boxed)
            })
            .collect();
        Self {
            inner: RustPipeline::new(rust_steps, verbose),
        }
    }

    fn fit(&mut self, data: &Bound<'_, PyAny>) -> PyResult<()> {
        let ctx = SessionContext::new();
        let df = pyarrow_to_df(&ctx, data)?;
        runtime()
            .block_on(self.inner.fit(&df))
            .map_err(to_py_err)?;
        Ok(())
    }

    fn transform(&self, py: Python<'_>, data: &Bound<'_, PyAny>) -> PyResult<PyObject> {
        let ctx = SessionContext::new();
        let df = pyarrow_to_df(&ctx, data)?;
        let out_df = self.inner.transform(df).map_err(to_py_err)?;
        df_to_pyarrow(py, out_df)
    }

    fn fit_transform(&mut self, py: Python<'_>, data: &Bound<'_, PyAny>) -> PyResult<PyObject> {
        let ctx = SessionContext::new();
        let df = pyarrow_to_df(&ctx, data)?;
        let out_df = runtime()
            .block_on(self.inner.fit_transform(&df))
            .map_err(to_py_err)?;
        df_to_pyarrow(py, out_df)
    }
}

// ======================================================================================
// Module
// ======================================================================================

#[pymodule]
fn feature_factory(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("FeatureFactoryError", py.get_type::<FeatureFactoryError>())?;
    m.add_class::<ImputeStrategy>()?;
    m.add_class::<PyMeanMedianImputer>()?;
    m.add_class::<PyArbitraryNumberImputer>()?;
    m.add_class::<PyDropMissingData>()?;
    m.add_class::<PyTransformer>()?;
    m.add_class::<PyPipeline>()?;
    Ok(())
}
