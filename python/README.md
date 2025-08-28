# Feature Factory for Python

This package provides Python bindings for the [Feature Factory](https://github.com/habedi/feature-factory) Rust library, a high-performance, batteries-included toolkit for feature engineering.

The Python bindings are built using [PyO3](https://pyo3.rs/) and [Maturin](https://www.maturin.rs/), and they interoperate with Python's data science ecosystem through [Apache Arrow](https://arrow.apache.org/docs/python/).

## Installation

You can install the library from PyPI:

```bash
pip install feature-factory
```

## Quickstart

Here's a quick example of how to use a transformer to impute missing values in an Arrow `RecordBatch`.

### Working with Transformers

The `fit` methods on transformers are `async`, so you'll need to use them in an `async` function with `await`.

```python
import asyncio
import pyarrow as pa
from feature_factory import MeanMedianImputer, ImputeStrategy

async def main():
    # 1. Create some sample data with a null value
    data = [
        pa.array([1.0, 2.0, None, 4.0, 5.0], type=pa.float64()),
        pa.array([10, 20, 30, 40, 50], type=pa.int64())
    ]
    batch = pa.RecordBatch.from_arrays(data, names=['numeric_col', 'other_col'])

    print("--- Original Data ---")
    print(batch)

    # 2. Initialize a transformer
    # We'll use the Mean strategy to fill the null value
    imputer, _ = MeanMedianImputer(
        columns=['numeric_col'],
        strategy=ImputeStrategy.Mean
    )

    # 3. Fit the transformer to the data
    # The `fit` method is async
    await imputer.fit(batch)

    # 4. Transform the data
    transformed_batch = imputer.transform(batch)

    print("\n--- Transformed Data ---")
    print(transformed_batch)
    # Expected: The null in 'numeric_col' is replaced by the mean (3.0)

if __name__ == "__main__":
    asyncio.run(main())
```

### Using a Pipeline

You can chain multiple transformers together into a `Pipeline`.

```python
import asyncio
import pyarrow as pa
from feature_factory import Pipeline, DropMissingData, ArbitraryNumberImputer, Transformer

async def main():
    # 1. Create data with missing values in two columns
    data = [
        pa.array([1.0, 2.0, None, 4.0, 5.0], type=pa.float64()),
        pa.array([10.0, None, 30.0, None, 50.0], type=pa.float64())
    ]
    batch = pa.RecordBatch.from_arrays(data, names=['col_a', 'col_b'])

    print("--- Original Data ---")
    print(batch)

    # 2. Define the pipeline steps
    # Note: The constructor for each transformer returns a tuple,
    # and we pass the second element to the pipeline.
    steps = [
        ("impute_b", ArbitraryNumberImputer(columns=['col_b'], number=-1.0)[1]),
        ("drop_missing_a", DropMissingData(columns=['col_a'])[1])
    ]

    # 3. Create and fit the pipeline
    pipeline = Pipeline(steps, verbose=True)

    # The pipeline's fit method is also async
    await pipeline.fit(batch)

    # 4. Transform the data
    transformed_batch = pipeline.transform(batch)

    print("\n--- Transformed Data ---")
    print(transformed_batch)
    # Expected:
    # - Nulls in 'col_b' are replaced with -1.0
    # - The row with a null in 'col_a' is dropped

if __name__ == "__main__":
    asyncio.run(main())
```

## Development

To build the Python bindings from source, you'll need a Rust toolchain and Python 3.10+.

1.  Clone the repository.
2.  Install `maturin`: `pip install maturin`
3.  Build and install the package in a virtual environment: `maturin develop`
