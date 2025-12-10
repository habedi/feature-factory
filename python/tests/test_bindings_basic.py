import pyarrow as pa
import pyarrow.compute as pc
import pytest

import feature_factory as ff


def make_batch():
    arr_a = pa.array([1.0, None, 3.0], type=pa.float64())
    arr_b = pa.array([None, 2.0, None], type=pa.float64())
    return pa.record_batch([arr_a, arr_b], names=["a", "b"])


def test_arbitrary_number_imputer_transform():
    rb = make_batch()
    t = ff.ArbitraryNumberImputer(["a", "b"], 0.0)
    out = t.transform(rb)
    assert isinstance(out, pa.RecordBatch)
    # no nulls after imputation
    assert pc.sum(pc.is_null(out.column(0))).as_py() == 0
    assert pc.sum(pc.is_null(out.column(1))).as_py() == 0


def test_mean_imputer_fit_transform():
    rb = make_batch()
    t = ff.MeanMedianImputer(["a"], "mean")
    t.fit(rb)
    out = t.transform(rb)
    assert isinstance(out, pa.RecordBatch)


def test_pipeline_fit_transform():
    rb = make_batch()
    steps = [
        ("arb", ff.ArbitraryNumberImputer(["b"], -1.0)),
        ("drop", ff.DropMissingData(None)),
    ]
    pipe = ff.Pipeline(steps, verbose=False)
    out = pipe.fit_transform(rb)
    assert isinstance(out, pa.RecordBatch)
    # after drop-missing, there should be no nulls remaining
    for i in range(out.num_columns):
        assert pc.sum(pc.is_null(out.column(i))).as_py() == 0
