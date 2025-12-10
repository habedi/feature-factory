use std::collections::HashMap;
use std::sync::Arc;

use datafusion::prelude::SessionContext;
use pyo3::prelude::*;
use tokio::sync::Mutex;

use ::feature_factory::Transformer as RustTransformer;
use ::feature_factory::transformers::imputation::{
    ArbitraryNumberImputer as RustArbitraryNumberImputer,
    DropMissingData as RustDropMissingData,
    ImputeStrategy,
    MeanMedianImputer as RustMeanMedianImputer,
    EndTailImputer as RustEndTailImputer,
    CategoricalImputer as RustCategoricalImputer,
    AddMissingIndicator as RustAddMissingIndicator,
};
use ::feature_factory::transformers::categorical as rust_cat;
use ::feature_factory::transformers::datetime as rust_dt;
use ::feature_factory::transformers::discretization as rust_disc;
use ::feature_factory::transformers::numerical as rust_num;
use ::feature_factory::transformers::outliers as rust_out;
use ::feature_factory::transformers::feature_creation as rust_fc;

use crate::conversion::{df_to_pyarrow, pyarrow_to_df};
use crate::errors::to_py_err;
use crate::runtime::runtime;

pub trait AsRustTransformer: Send + Sync {
    fn as_rust_transformer(&self) -> Box<dyn RustTransformer + Send + Sync>;
}

#[pyclass(name = "MeanMedianImputer")]
#[derive(Clone)]
pub struct PyMeanMedianImputer {
    pub(crate) inner: Arc<Mutex<RustMeanMedianImputer>>,
}

#[pymethods]
impl PyMeanMedianImputer {
    #[new]
    fn new(columns: Vec<String>, strategy: String) -> PyResult<Self> {
        let strat = match strategy.to_lowercase().as_str() {
            "mean" => ImputeStrategy::Mean,
            "median" => ImputeStrategy::Median,
            other => return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Invalid strategy '{}'. Use 'mean' or 'median'",
                other
            ))),
        };
        Ok(Self { inner: Arc::new(Mutex::new(RustMeanMedianImputer::new(columns, strat))) })
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

#[pyclass(name = "ArbitraryNumberImputer")]
#[derive(Clone)]
pub struct PyArbitraryNumberImputer {
    pub(crate) inner: RustArbitraryNumberImputer,
}

#[pymethods]
impl PyArbitraryNumberImputer {
    #[new]
    fn new(columns: Vec<String>, number: f64) -> Self {
        Self { inner: RustArbitraryNumberImputer::new(columns, number) }
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

#[pyclass(name = "DropMissingData")]
#[derive(Clone)]
pub struct PyDropMissingData {
    pub(crate) inner: RustDropMissingData,
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

#[pyclass(name = "EndTailImputer")]
#[derive(Clone)]
pub struct PyEndTailImputer {
    inner: Arc<Mutex<RustEndTailImputer>>,
}

#[pymethods]
impl PyEndTailImputer {
    #[new]
    fn new(columns: Vec<String>, percentile: f64) -> Self {
        Self { inner: Arc::new(Mutex::new(RustEndTailImputer::new(columns, percentile))) }
    }
    fn fit(&self, data: &Bound<'_, PyAny>) -> PyResult<()> {
        let ctx = SessionContext::new();
        let df = pyarrow_to_df(&ctx, data)?;
        let inner = self.inner.clone();
        runtime().block_on(async { inner.lock().await.fit(&df).await }).map_err(to_py_err)
    }
    fn transform(&self, py: Python<'_>, data: &Bound<'_, PyAny>) -> PyResult<PyObject> {
        let ctx = SessionContext::new();
        let df = pyarrow_to_df(&ctx, data)?;
        let inner = self.inner.clone();
        let out = runtime().block_on(async {
            let guard = inner.lock().await;
            guard.transform(df)
        }).map_err(to_py_err)?;
        df_to_pyarrow(py, out)
    }
}

impl AsRustTransformer for PyEndTailImputer {
    fn as_rust_transformer(&self) -> Box<dyn RustTransformer + Send + Sync> {
        let cloned = runtime().block_on(async {
            let guard = self.inner.lock().await;
            guard.clone()
        });
        Box::new(cloned)
    }
}

#[pyclass(name = "CategoricalImputer")]
#[derive(Clone)]
pub struct PyCategoricalImputer {
    inner: Arc<Mutex<RustCategoricalImputer>>,
}

#[pymethods]
impl PyCategoricalImputer {
    #[new]
    #[pyo3(signature = (columns, default=None))]
    fn new(columns: Vec<String>, default: Option<String>) -> Self {
        Self { inner: Arc::new(Mutex::new(RustCategoricalImputer::new(columns, default))) }
    }
    fn fit(&self, data: &Bound<'_, PyAny>) -> PyResult<()> {
        let ctx = SessionContext::new();
        let df = pyarrow_to_df(&ctx, data)?;
        let inner = self.inner.clone();
        runtime().block_on(async { inner.lock().await.fit(&df).await }).map_err(to_py_err)
    }
    fn transform(&self, py: Python<'_>, data: &Bound<'_, PyAny>) -> PyResult<PyObject> {
        let ctx = SessionContext::new();
        let df = pyarrow_to_df(&ctx, data)?;
        let inner = self.inner.clone();
        let out = runtime().block_on(async {
            let guard = inner.lock().await;
            guard.transform(df)
        }).map_err(to_py_err)?;
        df_to_pyarrow(py, out)
    }
}

impl AsRustTransformer for PyCategoricalImputer {
    fn as_rust_transformer(&self) -> Box<dyn RustTransformer + Send + Sync> {
        let cloned = runtime().block_on(async {
            let guard = self.inner.lock().await;
            guard.clone()
        });
        Box::new(cloned)
    }
}

#[pyclass(name = "AddMissingIndicator")]
#[derive(Clone)]
pub struct PyAddMissingIndicator {
    inner: RustAddMissingIndicator,
}

#[pymethods]
impl PyAddMissingIndicator {
    #[new]
    #[pyo3(signature = (columns, suffix=None))]
    fn new(columns: Vec<String>, suffix: Option<String>) -> Self {
        Self { inner: RustAddMissingIndicator::new(columns, suffix) }
    }
    fn transform(&self, py: Python<'_>, data: &Bound<'_, PyAny>) -> PyResult<PyObject> {
        let ctx = SessionContext::new();
        let df = pyarrow_to_df(&ctx, data)?;
        let out = self.inner.transform(df).map_err(to_py_err)?;
        df_to_pyarrow(py, out)
    }
}

impl AsRustTransformer for PyAddMissingIndicator {
    fn as_rust_transformer(&self) -> Box<dyn RustTransformer + Send + Sync> {
        Box::new(self.inner.clone())
    }
}

// ========================= Categorical =========================

#[pyclass(name = "OneHotEncoder")]
#[derive(Clone)]
pub struct PyOneHotEncoder {
    inner: Arc<Mutex<rust_cat::OneHotEncoder>>,
}

#[pymethods]
impl PyOneHotEncoder {
    #[new]
    fn new(columns: Vec<String>) -> Self {
        Self { inner: Arc::new(Mutex::new(rust_cat::OneHotEncoder::new(columns))) }
    }
    fn fit(&self, data: &Bound<'_, PyAny>) -> PyResult<()> {
        let ctx = SessionContext::new();
        let df = pyarrow_to_df(&ctx, data)?;
        let inner = self.inner.clone();
        runtime().block_on(async { inner.lock().await.fit(&df).await }).map_err(to_py_err)
    }
    fn transform(&self, py: Python<'_>, data: &Bound<'_, PyAny>) -> PyResult<PyObject> {
        let ctx = SessionContext::new();
        let df = pyarrow_to_df(&ctx, data)?;
        let inner = self.inner.clone();
        let out = runtime().block_on(async { inner.lock().await.transform(df) }).map_err(to_py_err)?;
        df_to_pyarrow(py, out)
    }
}

impl AsRustTransformer for PyOneHotEncoder {
    fn as_rust_transformer(&self) -> Box<dyn RustTransformer + Send + Sync> {
        let cloned = runtime().block_on(async {
            let guard = self.inner.lock().await;
            (*guard).clone()
        });
        Box::new(cloned)
    }
}

#[pyclass(name = "CountFrequencyEncoder")]
#[derive(Clone)]
pub struct PyCountFrequencyEncoder { inner: Arc<Mutex<rust_cat::CountFrequencyEncoder>> }
#[pymethods]
impl PyCountFrequencyEncoder {
    #[new]
    fn new(columns: Vec<String>) -> Self { Self { inner: Arc::new(Mutex::new(rust_cat::CountFrequencyEncoder::new(columns))) } }
    fn fit(&self, data: &Bound<'_, PyAny>) -> PyResult<()> { let ctx=SessionContext::new(); let df=pyarrow_to_df(&ctx, data)?; runtime().block_on(async{self.inner.lock().await.fit(&df).await}).map_err(to_py_err) }
    fn transform(&self, py: Python<'_>, data: &Bound<'_, PyAny>) -> PyResult<PyObject> { let ctx=SessionContext::new(); let df=pyarrow_to_df(&ctx, data)?; let out=runtime().block_on(async{self.inner.lock().await.transform(df)}).map_err(to_py_err)?; df_to_pyarrow(py, out) }
}
impl AsRustTransformer for PyCountFrequencyEncoder { fn as_rust_transformer(&self)->Box<dyn RustTransformer+Send+Sync>{ let c=runtime().block_on(async{(*self.inner.lock().await).clone()}); Box::new(c) } }

#[pyclass(name = "OrdinalEncoder")]
#[derive(Clone)]
pub struct PyOrdinalEncoder { inner: Arc<Mutex<rust_cat::OrdinalEncoder>> }
#[pymethods]
impl PyOrdinalEncoder {
    #[new] fn new(columns: Vec<String>)->Self{ Self{ inner: Arc::new(Mutex::new(rust_cat::OrdinalEncoder::new(columns)))}}
    fn fit(&self, data:&Bound<'_,PyAny>)->PyResult<()>{ let ctx=SessionContext::new(); let df=pyarrow_to_df(&ctx,data)?; runtime().block_on(async{self.inner.lock().await.fit(&df).await}).map_err(to_py_err) }
    fn transform(&self, py: Python<'_>, data:&Bound<'_,PyAny>)->PyResult<PyObject>{ let ctx=SessionContext::new(); let df=pyarrow_to_df(&ctx,data)?; let out=runtime().block_on(async{self.inner.lock().await.transform(df)}).map_err(to_py_err)?; df_to_pyarrow(py,out) }
}
impl AsRustTransformer for PyOrdinalEncoder{ fn as_rust_transformer(&self)->Box<dyn RustTransformer+Send+Sync>{ let c=runtime().block_on(async{(*self.inner.lock().await).clone()}); Box::new(c)} }

#[pyclass(name = "MeanEncoder")]
#[derive(Clone)]
pub struct PyMeanEncoder { inner: Arc<Mutex<rust_cat::MeanEncoder>> }
#[pymethods]
impl PyMeanEncoder {
    #[new] fn new(columns: Vec<String>, target: String)->Self{ Self{ inner: Arc::new(Mutex::new(rust_cat::MeanEncoder::new(columns, target)))}}
    fn fit(&self, data:&Bound<'_,PyAny>)->PyResult<()>{ let ctx=SessionContext::new(); let df=pyarrow_to_df(&ctx,data)?; runtime().block_on(async{self.inner.lock().await.fit(&df).await}).map_err(to_py_err) }
    fn transform(&self, py: Python<'_>, data:&Bound<'_,PyAny>)->PyResult<PyObject>{ let ctx=SessionContext::new(); let df=pyarrow_to_df(&ctx,data)?; let out=runtime().block_on(async{self.inner.lock().await.transform(df)}).map_err(to_py_err)?; df_to_pyarrow(py,out) }
}
impl AsRustTransformer for PyMeanEncoder{ fn as_rust_transformer(&self)->Box<dyn RustTransformer+Send+Sync>{ let c=runtime().block_on(async{(*self.inner.lock().await).clone()}); Box::new(c)} }

#[pyclass(name = "WoEEncoder")]
#[derive(Clone)]
pub struct PyWoEEncoder { inner: Arc<Mutex<rust_cat::WoEEncoder>> }
#[pymethods]
impl PyWoEEncoder {
    #[new] fn new(columns: Vec<String>, target: String)->Self{ Self{ inner: Arc::new(Mutex::new(rust_cat::WoEEncoder::new(columns, target)))}}
    fn fit(&self, data:&Bound<'_,PyAny>)->PyResult<()>{ let ctx=SessionContext::new(); let df=pyarrow_to_df(&ctx,data)?; runtime().block_on(async{self.inner.lock().await.fit(&df).await}).map_err(to_py_err) }
    fn transform(&self, py: Python<'_>, data:&Bound<'_,PyAny>)->PyResult<PyObject>{ let ctx=SessionContext::new(); let df=pyarrow_to_df(&ctx,data)?; let out=runtime().block_on(async{self.inner.lock().await.transform(df)}).map_err(to_py_err)?; df_to_pyarrow(py,out) }
}
impl AsRustTransformer for PyWoEEncoder{ fn as_rust_transformer(&self)->Box<dyn RustTransformer+Send+Sync>{ let c=runtime().block_on(async{(*self.inner.lock().await).clone()}); Box::new(c)} }

#[pyclass(name = "RareLabelEncoder")]
#[derive(Clone)]
pub struct PyRareLabelEncoder { inner: Arc<Mutex<rust_cat::RareLabelEncoder>> }
#[pymethods]
impl PyRareLabelEncoder {
    #[new] fn new(columns: Vec<String>, threshold: f64)->Self{ Self{ inner: Arc::new(Mutex::new(rust_cat::RareLabelEncoder::new(columns, threshold)))}}
    fn fit(&self, data:&Bound<'_,PyAny>)->PyResult<()>{ let ctx=SessionContext::new(); let df=pyarrow_to_df(&ctx,data)?; runtime().block_on(async{self.inner.lock().await.fit(&df).await}).map_err(to_py_err) }
    fn transform(&self, py: Python<'_>, data:&Bound<'_,PyAny>)->PyResult<PyObject>{ let ctx=SessionContext::new(); let df=pyarrow_to_df(&ctx,data)?; let out=runtime().block_on(async{self.inner.lock().await.transform(df)}).map_err(to_py_err)?; df_to_pyarrow(py,out) }
}
impl AsRustTransformer for PyRareLabelEncoder{ fn as_rust_transformer(&self)->Box<dyn RustTransformer+Send+Sync>{ let c=runtime().block_on(async{(*self.inner.lock().await).clone()}); Box::new(c)} }

// ========================= Datetime =========================

#[pyclass(name = "DatetimeFeatures")]
#[derive(Clone)]
pub struct PyDatetimeFeatures { inner: rust_dt::DatetimeFeatures }
#[pymethods]
impl PyDatetimeFeatures{
    #[new] fn new(columns: Vec<String>)->Self{ Self{ inner: rust_dt::DatetimeFeatures::new(columns) } }
    fn transform(&self, py: Python<'_>, data:&Bound<'_,PyAny>)->PyResult<PyObject>{ let ctx=SessionContext::new(); let df=pyarrow_to_df(&ctx,data)?; let out=self.inner.transform(df).map_err(to_py_err)?; df_to_pyarrow(py,out)}
}
impl AsRustTransformer for PyDatetimeFeatures{ fn as_rust_transformer(&self)->Box<dyn RustTransformer+Send+Sync>{ Box::new(self.inner.clone()) } }

#[pyclass(name = "DatetimeSubtraction")]
#[derive(Clone)]
pub struct PyDatetimeSubtraction { inner: rust_dt::DatetimeSubtraction }

fn parse_unit(u: &str)->PyResult<rust_dt::TimeUnit>{ match u.to_lowercase().as_str(){"second"=>Ok(rust_dt::TimeUnit::Second),"minute"=>Ok(rust_dt::TimeUnit::Minute),"hour"=>Ok(rust_dt::TimeUnit::Hour),"day"=>Ok(rust_dt::TimeUnit::Day),_=>Err(pyo3::exceptions::PyValueError::new_err("Invalid time unit"))}}

#[pymethods]
impl PyDatetimeSubtraction{
    #[new]
    fn new(new_features: Vec<(String,String,String,String)>)->PyResult<Self>{
        let feats: Vec<(String, String, String, rust_dt::TimeUnit)> = new_features
            .into_iter()
            .map(|(n,l,r,u)| Ok((n, l, r, parse_unit(&u)? )) )
            .collect::<PyResult<_>>()?;
        Ok(Self{ inner: rust_dt::DatetimeSubtraction::new(feats) })
    }
    fn transform(&self, py: Python<'_>, data:&Bound<'_,PyAny>)->PyResult<PyObject>{ let ctx=SessionContext::new(); let df=pyarrow_to_df(&ctx,data)?; let out=self.inner.transform(df).map_err(to_py_err)?; df_to_pyarrow(py,out)}
}
impl AsRustTransformer for PyDatetimeSubtraction{ fn as_rust_transformer(&self)->Box<dyn RustTransformer+Send+Sync>{ Box::new(self.inner.clone()) } }

// ========================= Discretization =========================

#[pyclass(name = "ArbitraryDiscretizer")]
#[derive(Clone)]
pub struct PyArbitraryDiscretizer { inner: rust_disc::ArbitraryDiscretizer }
#[pymethods]
impl PyArbitraryDiscretizer{
    #[new] fn new(columns: Vec<String>, intervals: HashMap<String, Vec<(f64,f64,String)>>)->Self{ Self{ inner: rust_disc::ArbitraryDiscretizer::new(columns, intervals) } }
    fn transform(&self, py: Python<'_>, data:&Bound<'_,PyAny>)->PyResult<PyObject>{ let ctx=SessionContext::new(); let df=pyarrow_to_df(&ctx,data)?; let out=self.inner.transform(df).map_err(to_py_err)?; df_to_pyarrow(py,out)}
}
impl AsRustTransformer for PyArbitraryDiscretizer{ fn as_rust_transformer(&self)->Box<dyn RustTransformer+Send+Sync>{ Box::new(self.inner.clone()) } }

#[pyclass(name = "EqualFrequencyDiscretizer")]
#[derive(Clone)]
pub struct PyEqualFrequencyDiscretizer { inner: Arc<Mutex<rust_disc::EqualFrequencyDiscretizer>> }
#[pymethods]
impl PyEqualFrequencyDiscretizer{
    #[new] fn new(columns: Vec<String>, bins: usize)->Self{ Self{ inner: Arc::new(Mutex::new(rust_disc::EqualFrequencyDiscretizer::new(columns, bins))) } }
    fn fit(&self, data:&Bound<'_,PyAny>)->PyResult<()>{ let ctx=SessionContext::new(); let df=pyarrow_to_df(&ctx,data)?; runtime().block_on(async{ self.inner.lock().await.fit(&df).await }).map_err(to_py_err) }
    fn transform(&self, py: Python<'_>, data:&Bound<'_,PyAny>)->PyResult<PyObject>{ let ctx=SessionContext::new(); let df=pyarrow_to_df(&ctx,data)?; let out=runtime().block_on(async{ self.inner.lock().await.transform(df) }).map_err(to_py_err)?; df_to_pyarrow(py,out)}
}
impl AsRustTransformer for PyEqualFrequencyDiscretizer{ fn as_rust_transformer(&self)->Box<dyn RustTransformer+Send+Sync>{ let c=runtime().block_on(async{ (*self.inner.lock().await).clone()}); Box::new(c)} }

#[pyclass(name = "EqualWidthDiscretizer")]
#[derive(Clone)]
pub struct PyEqualWidthDiscretizer { inner: Arc<Mutex<rust_disc::EqualWidthDiscretizer>> }
#[pymethods]
impl PyEqualWidthDiscretizer{
    #[new] fn new(columns: Vec<String>, bins: usize)->Self{ Self{ inner: Arc::new(Mutex::new(rust_disc::EqualWidthDiscretizer::new(columns, bins))) } }
    fn fit(&self, data:&Bound<'_,PyAny>)->PyResult<()>{ let ctx=SessionContext::new(); let df=pyarrow_to_df(&ctx,data)?; runtime().block_on(async{ self.inner.lock().await.fit(&df).await }).map_err(to_py_err) }
    fn transform(&self, py: Python<'_>, data:&Bound<'_,PyAny>)->PyResult<PyObject>{ let ctx=SessionContext::new(); let df=pyarrow_to_df(&ctx,data)?; let out=runtime().block_on(async{ self.inner.lock().await.transform(df) }).map_err(to_py_err)?; df_to_pyarrow(py,out)}
}
impl AsRustTransformer for PyEqualWidthDiscretizer{ fn as_rust_transformer(&self)->Box<dyn RustTransformer+Send+Sync>{ let c=runtime().block_on(async{ (*self.inner.lock().await).clone()}); Box::new(c)} }

#[pyclass(name = "GeometricWidthDiscretizer")]
#[derive(Clone)]
pub struct PyGeometricWidthDiscretizer { inner: Arc<Mutex<rust_disc::GeometricWidthDiscretizer>> }
#[pymethods]
impl PyGeometricWidthDiscretizer{
    #[new] fn new(columns: Vec<String>, bins: usize)->Self{ Self{ inner: Arc::new(Mutex::new(rust_disc::GeometricWidthDiscretizer::new(columns, bins))) } }
    fn fit(&self, data:&Bound<'_,PyAny>)->PyResult<()>{ let ctx=SessionContext::new(); let df=pyarrow_to_df(&ctx,data)?; runtime().block_on(async{ self.inner.lock().await.fit(&df).await }).map_err(to_py_err) }
    fn transform(&self, py: Python<'_>, data:&Bound<'_,PyAny>)->PyResult<PyObject>{ let ctx=SessionContext::new(); let df=pyarrow_to_df(&ctx,data)?; let out=runtime().block_on(async{ self.inner.lock().await.transform(df) }).map_err(to_py_err)?; df_to_pyarrow(py,out)}
}
impl AsRustTransformer for PyGeometricWidthDiscretizer{ fn as_rust_transformer(&self)->Box<dyn RustTransformer+Send+Sync>{ let c=runtime().block_on(async{ (*self.inner.lock().await).clone()}); Box::new(c)} }

// ========================= Numerical =========================

#[pyclass(name = "LogTransformer")]
#[derive(Clone)]
pub struct PyLogTransformer { inner: rust_num::LogTransformer }
#[pymethods]
impl PyLogTransformer{ #[new] fn new(columns: Vec<String>)->Self{ Self{ inner: rust_num::LogTransformer::new(columns)} } fn transform(&self, py: Python<'_>, data:&Bound<'_,PyAny>)->PyResult<PyObject>{ let ctx=SessionContext::new(); let df=pyarrow_to_df(&ctx,data)?; let out=self.inner.transform(df).map_err(to_py_err)?; df_to_pyarrow(py,out)} }
impl AsRustTransformer for PyLogTransformer{ fn as_rust_transformer(&self)->Box<dyn RustTransformer+Send+Sync>{ Box::new(self.inner.clone()) } }

#[pyclass(name = "LogCpTransformer")]
#[derive(Clone)]
pub struct PyLogCpTransformer { inner: rust_num::LogCpTransformer }
#[pymethods]
impl PyLogCpTransformer{ #[new] fn new(columns: Vec<String>, constant: f64)->Self{ Self{ inner: rust_num::LogCpTransformer::new(columns, constant)} } fn transform(&self, py: Python<'_>, data:&Bound<'_,PyAny>)->PyResult<PyObject>{ let ctx=SessionContext::new(); let df=pyarrow_to_df(&ctx,data)?; let out=self.inner.transform(df).map_err(to_py_err)?; df_to_pyarrow(py,out)} }
impl AsRustTransformer for PyLogCpTransformer{ fn as_rust_transformer(&self)->Box<dyn RustTransformer+Send+Sync>{ Box::new(self.inner.clone()) } }

#[pyclass(name = "ReciprocalTransformer")]
#[derive(Clone)]
pub struct PyReciprocalTransformer { inner: rust_num::ReciprocalTransformer }
#[pymethods]
impl PyReciprocalTransformer{ #[new] fn new(columns: Vec<String>)->Self{ Self{ inner: rust_num::ReciprocalTransformer::new(columns)} } fn transform(&self, py: Python<'_>, data:&Bound<'_,PyAny>)->PyResult<PyObject>{ let ctx=SessionContext::new(); let df=pyarrow_to_df(&ctx,data)?; let out=self.inner.transform(df).map_err(to_py_err)?; df_to_pyarrow(py,out)} }
impl AsRustTransformer for PyReciprocalTransformer{ fn as_rust_transformer(&self)->Box<dyn RustTransformer+Send+Sync>{ Box::new(self.inner.clone()) } }

#[pyclass(name = "PowerTransformer")]
#[derive(Clone)]
pub struct PyPowerTransformer { inner: rust_num::PowerTransformer }
#[pymethods]
impl PyPowerTransformer{ #[new] fn new(columns: Vec<String>, power: f64)->Self{ Self{ inner: rust_num::PowerTransformer::new(columns, power)} } fn transform(&self, py: Python<'_>, data:&Bound<'_,PyAny>)->PyResult<PyObject>{ let ctx=SessionContext::new(); let df=pyarrow_to_df(&ctx,data)?; let out=self.inner.transform(df).map_err(to_py_err)?; df_to_pyarrow(py,out)} }
impl AsRustTransformer for PyPowerTransformer{ fn as_rust_transformer(&self)->Box<dyn RustTransformer+Send+Sync>{ Box::new(self.inner.clone()) } }

#[pyclass(name = "BoxCoxTransformer")]
#[derive(Clone)]
pub struct PyBoxCoxTransformer { inner: rust_num::BoxCoxTransformer }
#[pymethods]
impl PyBoxCoxTransformer{ #[new] fn new(columns: Vec<String>, lambda: f64)->Self{ Self{ inner: rust_num::BoxCoxTransformer::new(columns, lambda)} } fn transform(&self, py: Python<'_>, data:&Bound<'_,PyAny>)->PyResult<PyObject>{ let ctx=SessionContext::new(); let df=pyarrow_to_df(&ctx,data)?; let out=self.inner.transform(df).map_err(to_py_err)?; df_to_pyarrow(py,out)} }
impl AsRustTransformer for PyBoxCoxTransformer{ fn as_rust_transformer(&self)->Box<dyn RustTransformer+Send+Sync>{ Box::new(self.inner.clone()) } }

#[pyclass(name = "YeoJohnsonTransformer")]
#[derive(Clone)]
pub struct PyYeoJohnsonTransformer { inner: rust_num::YeoJohnsonTransformer }
#[pymethods]
impl PyYeoJohnsonTransformer{ #[new] fn new(columns: Vec<String>, lambda: f64)->Self{ Self{ inner: rust_num::YeoJohnsonTransformer::new(columns, lambda)} } fn transform(&self, py: Python<'_>, data:&Bound<'_,PyAny>)->PyResult<PyObject>{ let ctx=SessionContext::new(); let df=pyarrow_to_df(&ctx,data)?; let out=self.inner.transform(df).map_err(to_py_err)?; df_to_pyarrow(py,out)} }
impl AsRustTransformer for PyYeoJohnsonTransformer{ fn as_rust_transformer(&self)->Box<dyn RustTransformer+Send+Sync>{ Box::new(self.inner.clone()) } }

#[pyclass(name = "ArcsinTransformer")]
#[derive(Clone)]
pub struct PyArcsinTransformer { inner: rust_num::ArcsinTransformer }
#[pymethods]
impl PyArcsinTransformer{ #[new] fn new(columns: Vec<String>)->Self{ Self{ inner: rust_num::ArcsinTransformer::new(columns)} } fn transform(&self, py: Python<'_>, data:&Bound<'_,PyAny>)->PyResult<PyObject>{ let ctx=SessionContext::new(); let df=pyarrow_to_df(&ctx,data)?; let out=self.inner.transform(df).map_err(to_py_err)?; df_to_pyarrow(py,out)} }
impl AsRustTransformer for PyArcsinTransformer{ fn as_rust_transformer(&self)->Box<dyn RustTransformer+Send+Sync>{ Box::new(self.inner.clone()) } }

// ========================= Outliers =========================

#[pyclass(name = "ArbitraryOutlierCapper")]
#[derive(Clone)]
pub struct PyArbitraryOutlierCapper { inner: rust_out::ArbitraryOutlierCapper }
#[pymethods]
impl PyArbitraryOutlierCapper{ #[new] fn new(columns: Vec<String>, lower_caps: HashMap<String,f64>, upper_caps: HashMap<String,f64>)->Self{ Self{ inner: rust_out::ArbitraryOutlierCapper::new(columns, lower_caps, upper_caps)} } fn transform(&self, py: Python<'_>, data:&Bound<'_,PyAny>)->PyResult<PyObject>{ let ctx=SessionContext::new(); let df=pyarrow_to_df(&ctx,data)?; let out=self.inner.transform(df).map_err(to_py_err)?; df_to_pyarrow(py,out)} }
impl AsRustTransformer for PyArbitraryOutlierCapper{ fn as_rust_transformer(&self)->Box<dyn RustTransformer+Send+Sync>{ Box::new(self.inner.clone()) } }

#[pyclass(name = "Winsorizer")]
#[derive(Clone)]
pub struct PyWinsorizer { inner: Arc<Mutex<rust_out::Winsorizer>> }
#[pymethods]
impl PyWinsorizer{ #[new] fn new(columns: Vec<String>, lower_percentile: f64, upper_percentile: f64)->Self{ Self{ inner: Arc::new(Mutex::new(rust_out::Winsorizer::new(columns, lower_percentile, upper_percentile))) } } fn fit(&self, data:&Bound<'_,PyAny>)->PyResult<()>{ let ctx=SessionContext::new(); let df=pyarrow_to_df(&ctx,data)?; runtime().block_on(async{ self.inner.lock().await.fit(&df).await }).map_err(to_py_err) } fn transform(&self, py: Python<'_>, data:&Bound<'_,PyAny>)->PyResult<PyObject>{ let ctx=SessionContext::new(); let df=pyarrow_to_df(&ctx,data)?; let out=runtime().block_on(async{ self.inner.lock().await.transform(df) }).map_err(to_py_err)?; df_to_pyarrow(py,out)} }
impl AsRustTransformer for PyWinsorizer{ fn as_rust_transformer(&self)->Box<dyn RustTransformer+Send+Sync>{ let c=runtime().block_on(async{ (*self.inner.lock().await).clone()}); Box::new(c)} }

#[pyclass(name = "OutlierTrimmer")]
#[derive(Clone)]
pub struct PyOutlierTrimmer { inner: Arc<Mutex<rust_out::OutlierTrimmer>> }
#[pymethods]
impl PyOutlierTrimmer{ #[new] fn new(columns: Vec<String>, lower_percentile: f64, upper_percentile: f64)->Self{ Self{ inner: Arc::new(Mutex::new(rust_out::OutlierTrimmer::new(columns, lower_percentile, upper_percentile))) } } fn fit(&self, data:&Bound<'_,PyAny>)->PyResult<()>{ let ctx=SessionContext::new(); let df=pyarrow_to_df(&ctx,data)?; runtime().block_on(async{ self.inner.lock().await.fit(&df).await }).map_err(to_py_err) } fn transform(&self, py: Python<'_>, data:&Bound<'_,PyAny>)->PyResult<PyObject>{ let ctx=SessionContext::new(); let df=pyarrow_to_df(&ctx,data)?; let out=runtime().block_on(async{ self.inner.lock().await.transform(df) }).map_err(to_py_err)?; df_to_pyarrow(py,out) } }
impl AsRustTransformer for PyOutlierTrimmer{ fn as_rust_transformer(&self)->Box<dyn RustTransformer+Send+Sync>{ let c=runtime().block_on(async{ (*self.inner.lock().await).clone()}); Box::new(c)} }

// ========================= Feature Creation =========================

#[pyclass(name = "RelativeFeatures")]
#[derive(Clone)]
pub struct PyRelativeFeatures { inner: rust_fc::RelativeFeatures }

fn parse_relop(op: &str)->PyResult<rust_fc::RelativeOperation>{ match op.to_lowercase().as_str(){"ratio"=>Ok(rust_fc::RelativeOperation::Ratio),"difference"=>Ok(rust_fc::RelativeOperation::Difference),"percent_change"=>Ok(rust_fc::RelativeOperation::PercentChange),_=>Err(pyo3::exceptions::PyValueError::new_err("Invalid RelativeOperation"))}}

#[pymethods]
impl PyRelativeFeatures{
    #[new]
    fn new(features: Vec<(String,String,String,String)>)->PyResult<Self>{
        let feats: Vec<(String, String, String, rust_fc::RelativeOperation)> = features
            .into_iter()
            .map(|(n,t,r,o)| Ok((n, t, r, parse_relop(&o)?)))
            .collect::<PyResult<_>>()?;
        Ok(Self{ inner: rust_fc::RelativeFeatures::new(feats) })
    }
    fn transform(&self, py: Python<'_>, data:&Bound<'_,PyAny>)->PyResult<PyObject>{ let ctx=SessionContext::new(); let df=pyarrow_to_df(&ctx,data)?; let out=self.inner.transform(df).map_err(to_py_err)?; df_to_pyarrow(py,out)}
}
impl AsRustTransformer for PyRelativeFeatures{ fn as_rust_transformer(&self)->Box<dyn RustTransformer+Send+Sync>{ Box::new(self.inner.clone()) } }

#[pyclass(name = "CyclicalFeatures")]
#[derive(Clone)]
pub struct PyCyclicalFeatures { inner: rust_fc::CyclicalFeatures }

fn parse_cyc(m: &str)->PyResult<rust_fc::CyclicalMethod>{ match m.to_lowercase().as_str(){"sine"=>Ok(rust_fc::CyclicalMethod::Sine),"cosine"=>Ok(rust_fc::CyclicalMethod::Cosine),_=>Err(pyo3::exceptions::PyValueError::new_err("Invalid CyclicalMethod"))}}

#[pymethods]
impl PyCyclicalFeatures{
    #[new]
    fn new(features: Vec<(String,String,f64,String)>)->PyResult<Self>{
        let feats: Vec<(String, String, f64, rust_fc::CyclicalMethod)> = features
            .into_iter()
            .map(|(n,s,p,m)| Ok((n, s, p, parse_cyc(&m)? )) )
            .collect::<PyResult<_>>()?;
        Ok(Self{ inner: rust_fc::CyclicalFeatures::new(feats) })
    }
    fn transform(&self, py: Python<'_>, data:&Bound<'_,PyAny>)->PyResult<PyObject>{ let ctx=SessionContext::new(); let df=pyarrow_to_df(&ctx,data)?; let out=self.inner.transform(df).map_err(to_py_err)?; df_to_pyarrow(py,out)}
}
impl AsRustTransformer for PyCyclicalFeatures{ fn as_rust_transformer(&self)->Box<dyn RustTransformer+Send+Sync>{ Box::new(self.inner.clone()) } }
