"""Unit tests for AGG_FUNCTIONS and _STRICT_AGG_FUNCTIONS."""

from __future__ import annotations

import numpy as np
import pytest

from trackpull.aggregate import AGG_FUNCTIONS, _STRICT_AGG_FUNCTIONS


class TestAggFunctionsScalar:
    """Aggregation over a 1-D (N,) array → scalar result."""

    def test_mean(self):
        x = np.array([1.0, 2.0, 3.0])
        assert AGG_FUNCTIONS["mean"](x) == pytest.approx(2.0)

    def test_std(self):
        x = np.array([1.0, 1.0, 1.0])
        assert AGG_FUNCTIONS["std"](x) == pytest.approx(0.0)

    def test_min(self):
        assert AGG_FUNCTIONS["min"](np.array([3.0, 1.0, 2.0])) == pytest.approx(1.0)

    def test_max(self):
        assert AGG_FUNCTIONS["max"](np.array([3.0, 1.0, 2.0])) == pytest.approx(3.0)

    def test_median(self):
        assert AGG_FUNCTIONS["median"](np.array([1.0, 3.0, 2.0])) == pytest.approx(2.0)

    def test_first(self):
        assert AGG_FUNCTIONS["first"](np.array([5.0, 6.0, 7.0])) == pytest.approx(5.0)

    def test_first_empty(self):
        result = AGG_FUNCTIONS["first"](np.array([]))
        assert np.isnan(result)


class TestAggFunctionsTimeSeries:
    """Aggregation over a 2-D (N, T) array → curve (T,)."""

    def test_mean_shape(self):
        x = np.array([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]])
        result = AGG_FUNCTIONS["mean"](x)
        assert result.shape == (3,)
        np.testing.assert_allclose(result, [2.0, 3.0, 4.0])

    def test_std_shape(self):
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = AGG_FUNCTIONS["std"](x)
        assert result.shape == (2,)

    def test_min_shape(self):
        x = np.array([[1.0, 5.0], [3.0, 2.0]])
        result = AGG_FUNCTIONS["min"](x)
        np.testing.assert_allclose(result, [1.0, 2.0])


class TestNaNHandling:
    """NaN-ignoring vs strict behaviour."""

    def test_nanmean_ignores_nan(self):
        x = np.array([1.0, np.nan, 3.0])
        assert AGG_FUNCTIONS["mean"](x) == pytest.approx(2.0)

    def test_strict_mean_propagates_nan(self):
        x = np.array([1.0, np.nan, 3.0])
        assert np.isnan(_STRICT_AGG_FUNCTIONS["mean"](x))

    def test_nanmean_2d_ignores_nan(self):
        x = np.array([[1.0, np.nan], [3.0, 4.0]])
        result = AGG_FUNCTIONS["mean"](x)
        assert result[0] == pytest.approx(2.0)
        assert result[1] == pytest.approx(4.0)
