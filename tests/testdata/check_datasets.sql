-- Description: This script is used to check the datasets that are available in the testdata directory.
-- Run using DuckDB CLI (in current directory): duckdb -init check_datasets.sql -no-stdin

-- Query the Wine Quality CSV file using csv_scan:
SELECT * FROM read_csv('winequality-red.csv') LIMIT 5;

-- Query the NYC Yellow Taxi Parquet file using parquet_scan:
SELECT * FROM read_parquet('yellow_tripdata_2019-01.parquet') LIMIT 5;

-- Query the NYC Green Taxi Parquet file using parquet_scan:
SELECT * FROM read_parquet('green_tripdata_2019-01.parquet') LIMIT 5;

