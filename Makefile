# Variables
REPO_URL = github.com/habedi/rustfe
BINARY_NAME := $(or $(PROJ_BINARY), $(notdir $(REPO_URL)))
BINARY := target/release/$(BINARY_NAME)
PATH := /snap/bin:$(PATH)
DEBUG_FEATURE_FACTORY := 0
RUST_BACKTRACE := 1
ASSET_DIR := assets
TEST_DATA_DIR := tests/testdata
SHELL := /bin/bash

# Default target
.DEFAULT_GOAL := help

.PHONY: help
help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

.PHONY: format
format: ## Format Rust files
	@echo "Formatting Rust files..."
	@cargo fmt

.PHONY: test
test: format ## Run the tests
	@echo "Running tests..."
	@DEBUG_FEATURE_FACTORY=$(DEBUG_FEATURE_FACTORY) RUST_BACKTRACE=$(RUST_BACKTRACE) cargo test -- --nocapture

.PHONY: coverage
coverage: format ## Generate test coverage report
	@echo "Generating test coverage report..."
	@DEBUG_FEATURE_FACTORY=$(DEBUG_FEATURE_FACTORY) cargo tarpaulin --out Xml --out Html

.PHONY: build
build: format ## Build the binary for the current platform
	@echo "Building the project..."
	@DEBUG_FEATURE_FACTORY=$(DEBUG_FEATURE_FACTORY) cargo build --release

.PHONY: run
run: build ## Build and run the binary
	@echo "Running the $(BINARY) binary..."
	@DEBUG_FEATURE_FACTORY=$(DEBUG_FEATURE_FACTORY) ./$(BINARY)

.PHONY: clean
clean: ## Remove generated and temporary files
	@echo "Cleaning up..."
	@cargo clean
	@rm -f $(ASSET_DIR)/*.svg && echo "Removed SVG files; might want to run 'make figs' to regenerate them."

.PHONY: install-snap
install-snap: ## Install a few dependencies using Snapcraft
	@echo "Installing the snap package..."
	@sudo apt-get update
	@sudo apt-get install -y snapd graphviz wget
	@sudo snap refresh
	@sudo snap install rustup --classic

.PHONY: install-deps
install-deps: install-snap ## Install development dependencies
	@echo "Installing dependencies..."
	@rustup component add rustfmt clippy
	@cargo install cargo-tarpaulin
	@cargo install cargo-audit
	@cargo install cargo-nextest

.PHONY: lint
lint: format ## Run the linters
	@echo "Linting Rust files..."
	@DEBUG_FEATURE_FACTORY=$(DEBUG_FEATURE_FACTORY) cargo clippy -- -D warnings

.PHONY: publish
publish: ## Publish the package to crates.io (requires CARGO_REGISTRY_TOKEN to be set)
	@echo "Publishing the package to Cargo registry..."
	@cargo publish --token $(CARGO_REGISTRY_TOKEN)

.PHONY: bench
bench: ## Run the benchmarks
	@echo "Running benchmarks..."
	@DEBUG_FEATURE_FACTORY=$(DEBUG_FEATURE_FACTORY) cargo bench

.PHONY: audit
audit: ## Run security audit on Rust dependencies
	@echo "Running security audit..."
	@cargo audit

.PHONY: doc
doc: format ## Generate the documentation
	@echo "Generating documentation..."
	@cargo doc --no-deps --document-private-items

.PHONE: figs
figs: ## Generate the figures in the assets directory
	@echo "Generating figures..."
	@$(SHELL) $(ASSET_DIR)/make_figures.sh $(ASSET_DIR)

.PHONY: fix-lint
fix_lint: ## Fix the linter warnings
	@echo "Fixing linter warnings..."
	@cargo clippy --fix --allow-dirty --allow-staged

.PHONY: testdata
testdata: ## Download the datasets used in tests
	@echo "Downloading test data..."
	@$(SHELL) $(TEST_DATA_DIR)/download_datasets.sh $(TEST_DATA_DIR)

.PHONY: nextest
nextest: ## Run tests using nextest
	@echo "Running tests using nextest..."
	@DEBUG_FEATURE_FACTORY=$(DEBUG_FEATURE_FACTORY) RUST_BACKTRACE=$(RUST_BACKTRACE) cargo nextest run
