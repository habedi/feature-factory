## Datasets for Testing

This directory contains the datasets used for the tests in the [`tests`](../) directory.

### Downloading the Datasets

Run the following command to download the datasets used for testing:

```shell
bash download_datasets.sh
```

### Checking the Datasets

To check the datasets after downloading, run the following command:

```shell
duckdb -init check_datasets.sql -no-stdin
```

You need to have the `duckdb` binary installed on your system to run the above command.
Check the [DuckDB installation guide](https://duckdb.org/docs/installation) for more information.
