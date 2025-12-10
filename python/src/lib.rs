mod runtime;
mod errors;
mod conversion;
mod transformers;

use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;

use ::feature_factory::Pipeline as RustPipeline;
use ::feature_factory::Transformer as RustTransformer;

use crate::transformers::{
    AsRustTransformer,
    PyArbitraryNumberImputer,
    PyDropMissingData,
    PyMeanMedianImputer,
    PyOneHotEncoder,
    PyCountFrequencyEncoder,
    PyOrdinalEncoder,
    PyMeanEncoder,
    PyWoEEncoder,
    PyRareLabelEncoder,
    PyDatetimeFeatures,
    PyDatetimeSubtraction,
    PyArbitraryDiscretizer,
    PyEqualFrequencyDiscretizer,
    PyEqualWidthDiscretizer,
    PyGeometricWidthDiscretizer,
    PyLogTransformer,
    PyLogCpTransformer,
    PyReciprocalTransformer,
    PyPowerTransformer,
    PyBoxCoxTransformer,
    PyYeoJohnsonTransformer,
    PyArcsinTransformer,
    PyArbitraryOutlierCapper,
    PyWinsorizer,
    PyOutlierTrimmer,
    PyRelativeFeatures,
    PyCyclicalFeatures,
    PyEndTailImputer,
    PyCategoricalImputer,
    PyAddMissingIndicator,
};
use crate::conversion::{df_to_pyarrow, pyarrow_to_df};
use crate::errors::{to_py_err, FeatureFactoryError};
use crate::runtime::runtime;
use datafusion::prelude::{DataFrame, SessionContext};

// ======================================================================================
// Module
// ======================================================================================

#[pymodule]
fn feature_factory(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("FeatureFactoryError", py.get_type::<FeatureFactoryError>())?;
    // Imputation
    m.add_class::<PyMeanMedianImputer>()?;
    m.add_class::<PyArbitraryNumberImputer>()?;
    m.add_class::<PyDropMissingData>()?;
    m.add_class::<PyEndTailImputer>()?;
    m.add_class::<PyCategoricalImputer>()?;
    m.add_class::<PyAddMissingIndicator>()?;
    // Categorical
    m.add_class::<PyOneHotEncoder>()?;
    m.add_class::<PyCountFrequencyEncoder>()?;
    m.add_class::<PyOrdinalEncoder>()?;
    m.add_class::<PyMeanEncoder>()?;
    m.add_class::<PyWoEEncoder>()?;
    m.add_class::<PyRareLabelEncoder>()?;
    // Datetime
    m.add_class::<PyDatetimeFeatures>()?;
    m.add_class::<PyDatetimeSubtraction>()?;
    // Discretization
    m.add_class::<PyArbitraryDiscretizer>()?;
    m.add_class::<PyEqualFrequencyDiscretizer>()?;
    m.add_class::<PyEqualWidthDiscretizer>()?;
    m.add_class::<PyGeometricWidthDiscretizer>()?;
    // Numerical
    m.add_class::<PyLogTransformer>()?;
    m.add_class::<PyLogCpTransformer>()?;
    m.add_class::<PyReciprocalTransformer>()?;
    m.add_class::<PyPowerTransformer>()?;
    m.add_class::<PyBoxCoxTransformer>()?;
    m.add_class::<PyYeoJohnsonTransformer>()?;
    m.add_class::<PyArcsinTransformer>()?;
    // Outliers
    m.add_class::<PyArbitraryOutlierCapper>()?;
    m.add_class::<PyWinsorizer>()?;
    m.add_class::<PyOutlierTrimmer>()?;
    // Feature creation
    m.add_class::<PyRelativeFeatures>()?;
    m.add_class::<PyCyclicalFeatures>()?;

    m.add_class::<PyPipeline>()?;
    Ok(())
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
    fn new(steps: Vec<(String, PyObject)>, verbose: bool) -> PyResult<Self> {
        let rust_steps: PyResult<Vec<(String, Box<dyn RustTransformer + Send + Sync>)>> =
            Python::with_gil(|py| {
                steps
                    .into_iter()
                    .map(|(name, obj)| {
                        let any = obj.bind(py);
                        macro_rules! try_downcast {
                            ($t:ty) => {
                                if let Ok(cell) = any.downcast::<$t>() {
                                    let inst = cell.borrow().clone();
                                    return Ok((name, inst.as_rust_transformer()));
                                }
                            };
                        }
                        // Try each known transformer type
                        try_downcast!(PyMeanMedianImputer);
                        try_downcast!(PyArbitraryNumberImputer);
                        try_downcast!(PyDropMissingData);
                        try_downcast!(PyEndTailImputer);
                        try_downcast!(PyCategoricalImputer);
                        try_downcast!(PyAddMissingIndicator);
                        try_downcast!(PyOneHotEncoder);
                        try_downcast!(PyCountFrequencyEncoder);
                        try_downcast!(PyOrdinalEncoder);
                        try_downcast!(PyMeanEncoder);
                        try_downcast!(PyWoEEncoder);
                        try_downcast!(PyRareLabelEncoder);
                        try_downcast!(PyDatetimeFeatures);
                        try_downcast!(PyDatetimeSubtraction);
                        try_downcast!(PyArbitraryDiscretizer);
                        try_downcast!(PyEqualFrequencyDiscretizer);
                        try_downcast!(PyEqualWidthDiscretizer);
                        try_downcast!(PyGeometricWidthDiscretizer);
                        try_downcast!(PyLogTransformer);
                        try_downcast!(PyLogCpTransformer);
                        try_downcast!(PyReciprocalTransformer);
                        try_downcast!(PyPowerTransformer);
                        try_downcast!(PyBoxCoxTransformer);
                        try_downcast!(PyYeoJohnsonTransformer);
                        try_downcast!(PyArcsinTransformer);
                        try_downcast!(PyArbitraryOutlierCapper);
                        try_downcast!(PyWinsorizer);
                        try_downcast!(PyOutlierTrimmer);
                        try_downcast!(PyRelativeFeatures);
                        try_downcast!(PyCyclicalFeatures);

                        Err(PyTypeError::new_err(format!(
                            "Unsupported transformer type for step '{}'",
                            name
                        )))
                    })
                    .collect()
            });
        let rust_steps = rust_steps?;
        Ok(Self { inner: RustPipeline::new(rust_steps, verbose) })
    }

    fn fit(&mut self, data: &Bound<'_, PyAny>) -> PyResult<()> {
        let ctx = SessionContext::new();
        let df = pyarrow_to_df(&ctx, data)?;
        runtime().block_on(self.inner.fit(&df)).map_err(to_py_err)?;
        Ok(())
    }

    fn transform(&self, py: Python<'_>, data: &Bound<'_, PyAny>) -> PyResult<PyObject> {
        let ctx = SessionContext::new();
        let df = pyarrow_to_df(&ctx, data)?;
        let out_df: DataFrame = self.inner.transform(df).map_err(to_py_err)?;
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
