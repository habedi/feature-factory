#!/bin/bash
set -euo pipefail

# Directory for test data (relative to this script)
TESTDATA_DIR="$(dirname "$0")"
echo "Using test data directory: $TESTDATA_DIR"

SUBDIR="feature-factory-datasets"
HF_DATASET="habedi/$SUBDIR"

# Create the path if it doesn't exist
mkdir -p "$TESTDATA_DIR/$SUBDIR"

# Download the datasets from the Hugging Face Hub
huggingface-cli download $HF_DATASET --repo-type dataset --local-dir "$TESTDATA_DIR/$SUBDIR"

echo "Download complete. Test data saved to $TESTDATA_DIR"
