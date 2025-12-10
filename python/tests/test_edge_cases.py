"""
Test edge cases for feature-factory Python bindings.
"""
import pyarrow as pa
import pyarrow.compute as pc
import pytest
import feature_factory as ff


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def numeric_data():
    """Sample numeric data with nulls and outliers."""
    return pa.record_batch([
        pa.array([1.0, 2.0, 3.0, None, 100.0, -50.0], type=pa.float64()),
        pa.array([10.0, None, 30.0, 40.0, 50.0, 60.0], type=pa.float64()),
        pa.array([0.5, 0.6, None, 0.8, 0.9, 1.0], type=pa.float64()),
    ], names=["col_a", "col_b", "col_c"])


@pytest.fixture
def categorical_data():
    """Sample categorical data with nulls and rare labels."""
    return pa.record_batch([
        pa.array(["cat", "dog", "cat", None, "cat", "bird", "bird", "cat"], type=pa.string()),
        pa.array(["red", "blue", "red", "green", None, "red", "blue", "yellow"], type=pa.string()),
    ], names=["animal", "color"])


@pytest.fixture
def datetime_data():
    """Sample datetime data."""
    # Create timestamps as integer values (seconds since epoch)
    # 2023-01-01 10:00:00 UTC
    timestamps_sec = [
        1672570800,  # 2023-01-01 10:00:00
        1686837000,  # 2023-06-15 14:30:00
        1704063599,  # 2023-12-31 23:59:59
        1710920130,  # 2024-03-20 08:15:30
    ]
    timestamps = pa.array(timestamps_sec, type=pa.timestamp('s'))
    return pa.record_batch([timestamps], names=["timestamp"])


@pytest.fixture
def empty_data():
    """Empty dataset."""
    return pa.record_batch([
        pa.array([], type=pa.float64()),
        pa.array([], type=pa.float64()),
    ], names=["col_a", "col_b"])


@pytest.fixture
def all_nulls_data():
    """Dataset with all nulls."""
    return pa.record_batch([
        pa.array([None, None, None], type=pa.float64()),
        pa.array([None, None, None], type=pa.float64()),
    ], names=["col_a", "col_b"])


@pytest.fixture
def single_row_data():
    """Dataset with a single row."""
    return pa.record_batch([
        pa.array([1.0], type=pa.float64()),
        pa.array([2.0], type=pa.float64()),
    ], names=["col_a", "col_b"])


# ============================================================================
# Imputation Edge Cases
# ============================================================================

class TestImputationEdgeCases:
    """Test imputation transformers with edge cases."""

    def test_arbitrary_imputer_with_all_nulls(self, all_nulls_data):
        """Test imputing all null values."""
        imputer = ff.ArbitraryNumberImputer(["col_a", "col_b"], 99.0)
        result = imputer.transform(all_nulls_data)

        assert result.num_rows == 3
        assert pc.sum(pc.is_null(result.column(0))).as_py() == 0
        assert pc.sum(pc.is_null(result.column(1))).as_py() == 0
        # All values should be 99.0
        assert pc.all(pc.equal(result.column(0), 99.0)).as_py()

    def test_arbitrary_imputer_with_empty_data(self, empty_data):
        """Test imputer on empty dataset."""
        imputer = ff.ArbitraryNumberImputer(["col_a"], 0.0)
        result = imputer.transform(empty_data)
        assert result.num_rows == 0

    def test_mean_imputer_single_valid_value(self):
        """Test mean imputer with only one valid value."""
        data = pa.record_batch([
            pa.array([None, None, 5.0, None], type=pa.float64()),
        ], names=["col"])

        imputer = ff.MeanMedianImputer(["col"], "mean")
        imputer.fit(data)
        result = imputer.transform(data)

        # All nulls should be replaced with 5.0
        assert pc.sum(pc.is_null(result.column(0))).as_py() == 0
        assert result.column(0)[0].as_py() == 5.0

    def test_drop_missing_all_nulls(self, all_nulls_data):
        """Test dropping rows when all values are null."""
        dropper = ff.DropMissingData(None)

        # Should raise error for empty result
        with pytest.raises(ValueError, match="Empty DataFrame result"):
            dropper.transform(all_nulls_data)

    def test_drop_missing_empty_data(self, empty_data):
        """Test dropping on empty dataset."""
        dropper = ff.DropMissingData(["col_a"])

        # Should raise error for empty result
        with pytest.raises(ValueError, match="Empty DataFrame result"):
            dropper.transform(empty_data)

    def test_end_tail_imputer_extreme_percentiles(self, numeric_data):
        """Test end tail imputer with extreme percentiles."""
        imputer = ff.EndTailImputer(["col_a"], 0.95)
        imputer.fit(numeric_data)
        result = imputer.transform(numeric_data)

        assert result.num_rows == numeric_data.num_rows
        assert pc.sum(pc.is_null(result.column(0))).as_py() == 0

    def test_categorical_imputer_all_nulls(self):
        """Test categorical imputer when all values are null."""
        data = pa.record_batch([
            pa.array([None, None, None], type=pa.string()),
        ], names=["cat_col"])

        # With default value
        imputer = ff.CategoricalImputer(["cat_col"], "MISSING")
        imputer.fit(data)
        result = imputer.transform(data)

        assert pc.sum(pc.is_null(result.column(0))).as_py() == 0
        assert result.column(0)[0].as_py() == "MISSING"

    def test_add_missing_indicator_no_nulls(self):
        """Test missing indicator when there are no nulls."""
        data = pa.record_batch([
            pa.array([1.0, 2.0, 3.0], type=pa.float64()),
        ], names=["col"])

        indicator = ff.AddMissingIndicator(["col"], None)
        result = indicator.transform(data)

        # Should have original column plus indicator column
        assert result.num_columns == 2
        # All indicator values should be False
        assert pc.sum(result.column(1)).as_py() == 0


# ============================================================================
# Categorical Encoding Edge Cases
# ============================================================================

class TestCategoricalEdgeCases:
    """Test categorical encoders with edge cases."""

    def test_one_hot_single_category(self):
        """Test one-hot encoding with a single category."""
        data = pa.record_batch([
            pa.array(["cat", "cat", "cat"], type=pa.string()),
        ], names=["animal"])

        encoder = ff.OneHotEncoder(["animal"])
        encoder.fit(data)
        result = encoder.transform(data)

        # Should create one binary column for "cat"
        assert result.num_columns >= 1

    def test_one_hot_empty_after_fit(self):
        """Test one-hot encoding transform on different data."""
        train_data = pa.record_batch([
            pa.array(["cat", "dog"], type=pa.string()),
        ], names=["animal"])

        test_data = pa.record_batch([
            pa.array(["bird"], type=pa.string()),
        ], names=["animal"])

        encoder = ff.OneHotEncoder(["animal"])
        encoder.fit(train_data)
        result = encoder.transform(test_data)

        # Unseen categories should be handled
        assert result.num_rows == test_data.num_rows

    def test_ordinal_encoder_single_value(self):
        """Test ordinal encoding with only one unique value."""
        data = pa.record_batch([
            pa.array(["A", "A", "A"], type=pa.string()),
        ], names=["letter"])

        encoder = ff.OrdinalEncoder(["letter"])
        encoder.fit(data)
        result = encoder.transform(data)

        # All should be encoded as 0
        assert pc.all(pc.equal(result.column(0), 0)).as_py()

    def test_rare_label_encoder_all_rare(self):
        """Test rare label encoder when all categories are rare."""
        data = pa.record_batch([
            pa.array(["a", "b", "c", "d", "e"], type=pa.string()),
        ], names=["col"])

        # High threshold so all are rare
        encoder = ff.RareLabelEncoder(["col"], 0.5)
        encoder.fit(data)
        result = encoder.transform(data)

        # All should be encoded as "Rare"
        assert result.num_rows == 5


# ============================================================================
# Discretization Edge Cases
# ============================================================================

class TestDiscretizationEdgeCases:
    """Test discretization transformers with edge cases."""

    def test_arbitrary_discretizer_single_interval(self):
        """Test arbitrary discretizer with a single interval."""
        data = pa.record_batch([
            pa.array([1.0, 2.0, 3.0, 4.0, 5.0], type=pa.float64()),
        ], names=["num"])

        intervals = {"num": [(0.0, 10.0, "all")]}
        discretizer = ff.ArbitraryDiscretizer(["num"], intervals)
        result = discretizer.transform(data)

        # All values should be in the same bin
        assert result.num_rows == 5

    def test_equal_frequency_single_bin(self, numeric_data):
        """Test equal frequency discretizer with single bin."""
        discretizer = ff.EqualFrequencyDiscretizer(["col_a"], 1)
        discretizer.fit(numeric_data)
        result = discretizer.transform(numeric_data)

        # All non-null values should be in the same bin
        assert result.num_rows == numeric_data.num_rows

    def test_equal_width_extreme_values(self):
        """Test equal width discretizer with extreme value ranges."""
        data = pa.record_batch([
            pa.array([0.000001, 0.000002, 1000000.0], type=pa.float64()),
        ], names=["num"])

        discretizer = ff.EqualWidthDiscretizer(["num"], 3)
        discretizer.fit(data)
        result = discretizer.transform(data)

        assert result.num_rows == 3

    def test_geometric_width_negative_values(self):
        """Test geometric width discretizer (should handle or error on negatives)."""
        data = pa.record_batch([
            pa.array([1.0, 2.0, 4.0, 8.0, 16.0], type=pa.float64()),
        ], names=["num"])

        discretizer = ff.GeometricWidthDiscretizer(["num"], 3)
        discretizer.fit(data)
        result = discretizer.transform(data)

        assert result.num_rows == 5


# ============================================================================
# Numerical Transformation Edge Cases
# ============================================================================

class TestNumericalTransformationEdgeCases:
    """Test numerical transformers with edge cases."""

    def test_log_transformer_zeros(self):
        """Test log transformer should error on zeros (or handle them)."""
        data = pa.record_batch([
            pa.array([1.0, 2.0, 3.0], type=pa.float64()),
        ], names=["num"])

        transformer = ff.LogTransformer(["num"])
        result = transformer.transform(data)

        # Should produce valid log values
        assert result.num_rows == 3

    def test_log_cp_with_negative_values(self):
        """Test log(x+c) transformer with negative values."""
        data = pa.record_batch([
            pa.array([-5.0, -2.0, 0.0, 2.0, 5.0], type=pa.float64()),
        ], names=["num"])

        # Add constant of 10 to make all positive
        transformer = ff.LogCpTransformer(["num"], 10.0)
        result = transformer.transform(data)

        assert result.num_rows == 5

    def test_reciprocal_near_zero(self):
        """Test reciprocal transformer should handle near-zero values."""
        data = pa.record_batch([
            pa.array([1.0, 2.0, 5.0], type=pa.float64()),
        ], names=["num"])

        transformer = ff.ReciprocalTransformer(["num"])
        result = transformer.transform(data)

        assert result.num_rows == 3

    def test_power_transformer_zero_power(self):
        """Test power transformer with power=0."""
        data = pa.record_batch([
            pa.array([1.0, 2.0, 3.0], type=pa.float64()),
        ], names=["num"])

        transformer = ff.PowerTransformer(["num"], 0.0)
        result = transformer.transform(data)

        # x^0 = 1 for all x
        assert result.num_rows == 3

    def test_power_transformer_negative_power(self):
        """Test power transformer with negative power."""
        data = pa.record_batch([
            pa.array([1.0, 2.0, 4.0], type=pa.float64()),
        ], names=["num"])

        transformer = ff.PowerTransformer(["num"], -1.0)
        result = transformer.transform(data)

        assert result.num_rows == 3

    def test_box_cox_lambda_zero(self):
        """Test Box-Cox with lambda=0 (log transformation)."""
        data = pa.record_batch([
            pa.array([1.0, 2.0, 3.0], type=pa.float64()),
        ], names=["num"])

        transformer = ff.BoxCoxTransformer(["num"], 0.0)
        result = transformer.transform(data)

        assert result.num_rows == 3

    def test_yeo_johnson_with_negatives(self):
        """Test Yeo-Johnson can handle negative values."""
        data = pa.record_batch([
            pa.array([-3.0, -1.0, 0.0, 1.0, 3.0], type=pa.float64()),
        ], names=["num"])

        transformer = ff.YeoJohnsonTransformer(["num"], 1.0)
        result = transformer.transform(data)

        assert result.num_rows == 5

    def test_arcsin_boundary_values(self):
        """Test arcsin transformer with boundary values."""
        data = pa.record_batch([
            pa.array([0.0, 0.5, 1.0], type=pa.float64()),
        ], names=["num"])

        transformer = ff.ArcsinTransformer(["num"])
        result = transformer.transform(data)

        assert result.num_rows == 3


# ============================================================================
# Outlier Handling Edge Cases
# ============================================================================

class TestOutlierEdgeCases:
    """Test outlier transformers with edge cases."""

    def test_arbitrary_capper_inverted_bounds(self):
        """Test arbitrary capper when lower > upper (should error or handle)."""
        data = pa.record_batch([
            pa.array([1.0, 50.0, 100.0], type=pa.float64()),
        ], names=["num"])

        lower_caps = {"num": 10.0}
        upper_caps = {"num": 90.0}
        capper = ff.ArbitraryOutlierCapper(["num"], lower_caps, upper_caps)
        result = capper.transform(data)

        # Values should be capped
        assert result.num_rows == 3

    def test_winsorizer_extreme_percentiles(self, numeric_data):
        """Test winsorizer with very extreme percentiles."""
        winsorizer = ff.Winsorizer(["col_a"], 0.01, 0.99)
        winsorizer.fit(numeric_data)
        result = winsorizer.transform(numeric_data)

        assert result.num_rows == numeric_data.num_rows

    def test_winsorizer_equal_percentiles(self, numeric_data):
        """Test winsorizer when lower and upper percentiles are equal."""
        winsorizer = ff.Winsorizer(["col_a"], 0.5, 0.5)

        # Should raise error because lower must be less than upper
        with pytest.raises(ff.FeatureFactoryError, match="lower_percentile.*must be less than upper_percentile"):
            winsorizer.fit(numeric_data)

    def test_outlier_trimmer_removes_all(self):
        """Test outlier trimmer that removes all rows."""
        data = pa.record_batch([
            pa.array([100.0, 200.0, 300.0], type=pa.float64()),
        ], names=["num"])

        # Very tight percentiles
        trimmer = ff.OutlierTrimmer(["num"], 0.49, 0.51)
        trimmer.fit(data)
        result = trimmer.transform(data)

        # Most or all rows might be removed
        assert result.num_rows <= 3


# ============================================================================
# Feature Creation Edge Cases
# ============================================================================

class TestFeatureCreationEdgeCases:
    """Test feature creation transformers with edge cases."""

    def test_relative_features_division_by_zero(self):
        """Test relative features when reference is zero."""
        data = pa.record_batch([
            pa.array([10.0, 20.0, 30.0], type=pa.float64()),
            pa.array([0.0, 5.0, 10.0], type=pa.float64()),
        ], names=["target", "reference"])

        features = [("ratio", "target", "reference", "ratio")]
        transformer = ff.RelativeFeatures(features)
        result = transformer.transform(data)

        # Should handle division by zero
        assert result.num_rows == 3

    def test_relative_features_percent_change_zero_reference(self):
        """Test percent change when reference is zero."""
        data = pa.record_batch([
            pa.array([10.0, 20.0], type=pa.float64()),
            pa.array([0.0, 10.0], type=pa.float64()),
        ], names=["target", "reference"])

        features = [("pct_change", "target", "reference", "percent_change")]
        transformer = ff.RelativeFeatures(features)
        result = transformer.transform(data)

        assert result.num_rows == 2

    def test_cyclical_features_zero_period(self):
        """Test cyclical features (period must be positive, should error)."""
        data = pa.record_batch([
            pa.array([0.0, 90.0, 180.0, 270.0], type=pa.float64()),
        ], names=["angle"])

        # Valid period
        features = [("sin_angle", "angle", 360.0, "sine")]
        transformer = ff.CyclicalFeatures(features)
        result = transformer.transform(data)

        assert result.num_rows == 4

    def test_cyclical_features_large_values(self):
        """Test cyclical features with large values."""
        data = pa.record_batch([
            pa.array([0.0, 1000.0, 10000.0], type=pa.float64()),
        ], names=["time"])

        features = [("cos_time", "time", 24.0, "cosine")]
        transformer = ff.CyclicalFeatures(features)
        result = transformer.transform(data)

        assert result.num_rows == 3


# ============================================================================
# Datetime Edge Cases
# ============================================================================

class TestDatetimeEdgeCases:
    """Test datetime transformers with edge cases."""

    def test_datetime_features_year_boundary(self):
        """Test datetime features at year boundaries."""
        # Create timestamps as integer values (seconds since epoch)
        timestamps_sec = [
            1704063599,  # 2023-12-31 23:59:59
            1704067200,  # 2024-01-01 00:00:00
        ]
        timestamps = pa.array(timestamps_sec, type=pa.timestamp('s'))
        data = pa.record_batch([timestamps], names=["dt"])

        transformer = ff.DatetimeFeatures(["dt"])
        result = transformer.transform(data)

        # Should extract year, month, day, etc.
        assert result.num_columns > 1

    def test_datetime_subtraction_same_times(self):
        """Test datetime subtraction when times are the same."""
        # Create timestamps as integer values (seconds since epoch)
        timestamps_sec = [
            1704110400,  # 2024-01-01 12:00:00
            1704110400,  # 2024-01-01 12:00:00
        ]
        timestamps = pa.array(timestamps_sec, type=pa.timestamp('s'))
        data = pa.record_batch([timestamps, timestamps], names=["dt1", "dt2"])

        features = [("diff", "dt1", "dt2", "second")]
        transformer = ff.DatetimeSubtraction(features)
        result = transformer.transform(data)

        # Difference should be 0
        assert result.num_rows == 2


# ============================================================================
# Pipeline Edge Cases
# ============================================================================

class TestPipelineEdgeCases:
    """Test pipeline with edge cases."""

    def test_empty_pipeline(self):
        """Test pipeline with no steps - should raise error."""
        data = pa.record_batch([
            pa.array([1.0, 2.0, 3.0], type=pa.float64()),
        ], names=["col"])

        # Empty pipeline raises error when fit_transform is called
        pipeline = ff.Pipeline([], verbose=False)
        with pytest.raises(ff.FeatureFactoryError, match="Pipeline must have at least one transformer"):
            pipeline.fit_transform(data)

    def test_pipeline_single_step(self, numeric_data):
        """Test pipeline with a single step."""
        steps = [("impute", ff.ArbitraryNumberImputer(["col_a"], 0.0))]
        pipeline = ff.Pipeline(steps, verbose=False)
        result = pipeline.fit_transform(numeric_data)

        assert result.num_rows == numeric_data.num_rows

    def test_pipeline_many_steps(self, numeric_data):
        """Test pipeline with many sequential steps."""
        steps = [
            ("impute_a", ff.ArbitraryNumberImputer(["col_a"], 0.0)),
            ("impute_b", ff.ArbitraryNumberImputer(["col_b"], 0.0)),
            ("impute_c", ff.ArbitraryNumberImputer(["col_c"], 0.0)),
            ("drop", ff.DropMissingData(None)),
        ]
        pipeline = ff.Pipeline(steps, verbose=True)
        result = pipeline.fit_transform(numeric_data)

        # After imputation and drop, should have no nulls
        assert result.num_rows > 0
        for i in range(result.num_columns):
            assert pc.sum(pc.is_null(result.column(i))).as_py() == 0

    def test_pipeline_data_reduction(self):
        """Test pipeline that progressively reduces data."""
        data = pa.record_batch([
            pa.array([1.0, None, 3.0, None, 5.0], type=pa.float64()),
        ], names=["col"])

        steps = [
            ("drop", ff.DropMissingData(["col"])),
        ]
        pipeline = ff.Pipeline(steps, verbose=False)
        result = pipeline.fit_transform(data)

        # Should remove rows with nulls
        assert result.num_rows == 3
