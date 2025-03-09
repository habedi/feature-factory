#!/bin/bash
set -euo pipefail

# Directory for test data (relative to this script)
TESTDATA_DIR="$(dirname "$0")"
echo "Using test data directory: $TESTDATA_DIR"

# Create the directory if it doesn't exist
mkdir -p "$TESTDATA_DIR"

echo "Downloading Wine Quality Dataset (red wine)..."
wget -c -O "$TESTDATA_DIR/winequality-red.csv" "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"

echo "Downloading NYC Yellow Taxi Trip Data (January 2019, Parquet)..."
wget -c -O "$TESTDATA_DIR/yellow_tripdata_2019-01.parquet" "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2019-01.parquet"

echo "Downloading NYC Green Taxi Trip Data (January 2019, Parquet)..."
wget -c -O "$TESTDATA_DIR/green_tripdata_2019-01.parquet" "https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_2019-01.parquet"

echo "Download complete. Test data saved to $TESTDATA_DIR"
