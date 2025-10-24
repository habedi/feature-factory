use pyo3::prelude::*;

pyo3::create_exception!(feature_factory, FeatureFactoryError, pyo3::exceptions::PyException);

use ::feature_factory::FeatureFactoryError as RustFeatureFactoryError;

pub fn to_py_err(e: RustFeatureFactoryError) -> PyErr {
    FeatureFactoryError::new_err(e.to_string())
}

