use arrow::pyarrow::{FromPyArrow, ToPyArrow};
use arrow::record_batch::RecordBatch;
use datafusion::prelude::{DataFrame, SessionContext};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::runtime::runtime;

pub fn pyarrow_to_df(ctx: &SessionContext, obj: &Bound<'_, PyAny>) -> PyResult<DataFrame> {
    let batch = RecordBatch::from_pyarrow_bound(obj)?;
    ctx.read_batch(batch)
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

pub fn df_to_pyarrow(py: Python<'_>, df: DataFrame) -> PyResult<PyObject> {
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

