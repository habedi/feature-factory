# Variables
PROJ_REPO := github.com/CogitatorTech/feature-factory
BINARY_NAME := $(or $(PROJ_BINARY), $(notdir $(PROJ_REPO)))
BINARY = :target/release/$(BINARY_NAME)
PATH := /snap/bin:$(PATH)
RUST_BACKTRACE := 0
RUST_LOG        := info
WHEEL_DIR       := dist
PYTHON_DIR     := python
PY_DEP_MNGR     := uv
DEBUG_FEATURE_FACTORY := 0
WHEEL_FILE      := $(shell ls $(PYTHON_DIR)/$(WHEEL_DIR)/feature_factory-*.whl 2>/dev/null | head -n 1)

# Default target
.DEFAULT_GOAL := help

.PHONY: help
help: ## Show the help messages for all targets
	@echo "Usage: make <target>"
	@echo ""
	@echo "Targets:"
	@grep -E '^[a-zA-Z_-]+:.*## .*$$' Makefile | \
	awk 'BEGIN {FS = ":.*## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

.PHONY: format
format: ## Format Rust files
	@echo "Formatting Rust files..."
	@cargo fmt

.PHONY: test
test: format ## Run the tests
	@echo "Running tests..."
	@RUST_BACKTRACE=$(RUST_BACKTRACE) cargo test -- --nocapture

.PHONY: coverage
coverage: format ## Generate test coverage report
	@echo "Generating test coverage report..."
	@cargo tarpaulin --out Xml --out Html

.PHONY: build
build: format ## Build the binary for the current platform
	@echo "Building the project..."
	@cargo build --release

.PHONY: run
run: build ## Build and run the binary
	@echo "Running the $(BINARY) binary..."
	@./$(BINARY)

.PHONY: run-examples
run-examples: build ## Run the Rust examples
	@echo "Running Rust examples..."
	@cargo run --example basic_usage

.PHONY: clean
clean: ## Remove generated and temporary files
	@echo "Cleaning up..."
	@cargo clean
	@rm -rf $(WHEEL_DIR) dist/ $(PYTHON_DIR)/$(WHEEL_DIR) $(PYTHON_DIR)/*.so $(PYTHON_DIR)/target

.PHONY: install-snap
install-snap: ## Install a few dependencies using Snapcraft
	@echo "Installing the snap package..."
	@sudo apt-get update
	@sudo apt-get install -y snapd
	@sudo snap refresh
	@sudo snap install rustup --classic

.PHONY: install-deps
install-deps: install-snap ## Install development dependencies
	@echo "Installing dependencies..."
	@rustup component add rustfmt clippy
	@cargo install --locked cargo-tarpaulin --version 0.31.4
	@cargo install --locked cargo-audit --version 0.21.0
	@cargo install --locked cargo-careful
	@cargo install --locked cargo-nextest --version 0.9.97-b.2
	@sudo apt-get install -y python3-pip
	@pip install $(PY_DEP_MNGR)

.PHONY: lint
lint: format ## Run linters on Rust files
	@echo "Linting Rust files..."
	@cargo clippy -- -D warnings -D clippy::unwrap_used -D clippy::expect_used

.PHONY: publish
publish: ## Publish the package to crates.io (requires CARGO_REGISTRY_TOKEN to be set)
	@echo "Publishing the package to Cargo registry..."
	@cargo publish --token $(CARGO_REGISTRY_TOKEN)

.PHONY: bench
bench: ## Run benchmarks
	@echo "Running benchmarks..."
	@cargo bench

.PHONY: audit
audit: ## Run security audit on Rust dependencies
	@echo "Running security audit..."
	@cargo audit

.PHONY: nextest
nextest: ## Run tests using nextest
	@echo "Running tests using nextest..."
	@RUST_BACKTRACE=$(RUST_BACKTRACE) cargo nextest run

.PHONY: docs
docs: format ## Generate the documentation
	@echo "Generating documentation..."
	@cargo doc --no-deps --document-private-items

.PHONY: fix-lint
fix-lint: ## Fix the linter warnings
	@echo "Fixing linter warnings..."
	@cargo clippy --fix --allow-dirty --allow-staged --all-targets --workspace --all-features \
	-- -D warnings -D clippy::unwrap_used -D clippy::expect_used

.PHONY: careful
careful: ## Run security checks on Rust code
	@echo "Running security checks..."
	@DEBUG_FEATURE_FACTORY=$(DEBUG_FEATURE_FACTORY) RUST_BACKTRACE=$(RUST_BACKTRACE) cargo careful run

########################################################################################
## Python targets
########################################################################################

.PHONY: develop-py
develop-py: ## Build and install feature-factory in the current Python environment
	@echo "Building and installing feature-factory..."
	# Note: Maturin does not work when CONDA_PREFIX and VIRTUAL_ENV are both set
	@bash -c "source .venv/bin/activate && unset CONDA_PREFIX && maturin develop --manifest-path $(PYTHON_DIR)/Cargo.toml"

.PHONY: wheel
wheel: ## Build the wheel file for feature-factory
	@echo "Building the feature-factory wheel..."
	@maturin build --release --out $(WHEEL_DIR) --manifest-path $(PYTHON_DIR)/Cargo.toml

.PHONY: wheel-manylinux
wheel-manylinux: ## Build the manylinux wheel file for feature-factory (using Zig)
	@echo "Building the manylinux feature-factory wheel..."
	@maturin build --release --out $(WHEEL_DIR) --manifest-path $(PYTHON_DIR)/Cargo.toml --zig

.PHONY: test-py
test-py: develop-py ## Run Python tests
	@echo "Running Python tests..."
	@bash -c "source .venv/bin/activate && pytest"

.PHONY: publish-py
publish-py: wheel-manylinux ## Publish the feature-factory wheel to PyPI (requires PYPI_TOKEN to be set)
	@echo "Publishing feature-factory to PyPI..."
	@if [ -z "$(WHEEL_FILE)" ]; then \
	   echo "Error: No wheel file found. Please run 'make wheel' first."; \
	   exit 1; \
	fi
	@echo "Found wheel file: $(WHEEL_FILE)"
	@twine upload -u __token__ -p $(PYPI_TOKEN) $(WHEEL_FILE)

########################################################################################
## Additional targets
########################################################################################

.PHONY: setup-hooks
setup-hooks: ## Install Git hooks (pre-commit and pre-push)
	@echo "Installing Git hooks..."
	@pre-commit install --hook-type pre-commit
	@pre-commit install --hook-type pre-push
	@pre-commit install-hooks

.PHONY: test-hooks
test-hooks: ## Test Git hooks on all files
	@echo "Testing Git hooks..."
	@pre-commit run --all-files
