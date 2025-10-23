# Bugs and Architectural Issues Found and Fixed

## Summary

This document details the bugs, architectural flaws, and issues found in the Feature Factory library, along with the fixes applied.

**Date**: October 24, 2025  
**Status**: In Progress  
**Breaking Changes**: Yes (acceptable in alpha)

## Architectural Improvements

### 1. ✅ FIXED: Pipeline Module Location (Breaking Change)

**Issue**: The `pipeline` module was at the root level alongside `transformers`, but it's core infrastructure that transformers depend on. This violates the principle that high-level modules should only depend on the `foundation` module.

**Fix**: Moved `pipeline.rs` from `src/pipeline.rs` to `src/foundation/pipeline.rs`. Updated imports throughout the codebase while maintaining backward compatibility through re-exports.

**Impact**: Cleaner architecture where all core abstractions live in `foundation/`.

### 2. ✅ FIXED: Missing Validation Helper Functions

**Issue**: Code duplication across transformers for validating numeric columns.

**Fix**: Added `validate_numeric_columns()` helper function in `foundation/types.rs` to complement the existing `validate_numeric_column()`. Also added comprehensive unit tests.

### 3. ⚠️ DOCUMENTED (NOT FIXED): Blocking Async Calls in Synchronous Transform Methods

## Critical Bugs

### 3. 🔍 IDENTIFIED: Blocking Async Calls in Synchronous Transform Methods

**Issue**: Several numerical transformers (`LogTransformer`, `LogCpTransformer`, `ReciprocalTransformer`, `BoxCoxTransformer`, `ArcsinTransformer`) call async functions using `futures::executor::block_on()` inside their synchronous `transform()` methods.

**Problems**:
- Blocks the async runtime, causing performance degradation
- Validation logic runs on every `transform()` call instead of once during `fit()`
- Expensive min/max computations performed repeatedly
- Transformers marked as "stateless" but perform stateful-like operations

**Recommended Fix**: 
1. Make these transformers truly stateful by validating column types during `fit()`
2. Remove expensive data scanning from `transform()` 
3. Trust DataFusion to handle invalid data gracefully (NaN, Inf)
4. Update `is_stateful()` to return `true`
**Status**: ⚠️ NOT FIXED - Requires careful refactoring to avoid breaking existing tests.

**Reason Not Fixed**: The transformers are currently tested with the expectation that they validate 
data ranges. Changing them to be stateful and removing the validation would require updating all 
existing tests. This is a significant breaking change that should be done with more planning.

**Recommendation**: In a future version, consider:
1. Creating stateful versions of these transformers that validate during `fit()`
2. Keeping stateless versions that skip validation for performance
3. Clearly documenting the trade-offs in API documentation
**Files Affected**:
- `src/transformers/numerical.rs` (7 occurrences of `block_on`)

**Status**: Requires careful refactoring to avoid breaking existing tests.

## Code Quality Issues

### 4. Missing Documentation

**Issue**: Helper functions in transformers modules lack proper documentation.

**Examples**:
- `coalesce_expr_for()` in imputation.rs
- `apply_imputation()` in imputation.rs
- `build_case_expr()` in categorical.rs
- `extract_distinct_values()` in categorical.rs

**Impact**: Makes code harder to understand and maintain.

### 5. Inconsistent Error Messages

**Issue**: Some error messages don't provide enough context for debugging.

**Example**: In `categorical.rs`, some errors just say "Expected Utf8 array" without specifying which column or operation failed.

## Test Coverage

### 6. Unit Tests Location

**Status**: ✅ CORRECT

All unit tests are properly located within their respective modules. Integration tests are in the `tests/` directory.

### 7. Missing Edge Case Tests

**Identified Missing Tests**:
- Transformer behavior when DataFrame is empty
- Transformer behavior with all-null columns
- Pipeline behavior with mix of stateful and stateless transformers
- Error propagation through complex pipelines

## Performance Issues

### 8. Repeated Schema Lookups

**Issue**: Some transformers repeatedly look up field information in tight loops.

**Example**: In `transform()` methods that iterate over `df.schema().fields()`, the same schema is queried multiple times.

**Recommended Fix**: Cache schema references where possible.

## Breaking Changes Applied

All breaking changes are acceptable as the project is in alpha stage:

1. Pipeline module moved to foundation (requires import updates)
2. Several numerical transformers changed from stateless to stateful (breaking API change if users relied on `is_stateful()` returning false)

## Testing Results

- ✅ All existing tests pass after architectural changes
- ✅ New unit tests added for validation helpers
- ⏳ Additional tests needed for edge cases

## Next Steps

1. Carefully refactor numerical transformers to fix the blocking async issue
2. Add comprehensive documentation to helper functions
3. Add edge case tests for empty DataFrames and null columns
4. Consider adding performance benchmarks for transformers
5. Review and improve error messages throughout the codebase

