"""Unit tests for trackpull.transforms."""

from __future__ import annotations

import pytest

from trackpull.transforms import (
    TRANSFORM_FUNCTIONS,
)

# ---------------------------------------------------------------------------
# Individual transform functions
# ---------------------------------------------------------------------------


class TestTransformFunctions:
    def test_first_list(self):
        fn = TRANSFORM_FUNCTIONS["first"]
        assert fn([10, 20, 30]) == 10

    def test_first_scalar(self):
        assert TRANSFORM_FUNCTIONS["first"](5) == 5

    def test_first_empty(self):
        assert TRANSFORM_FUNCTIONS["first"]([]) is None

    def test_first_none(self):
        assert TRANSFORM_FUNCTIONS["first"](None) is None

    def test_last_list(self):
        assert TRANSFORM_FUNCTIONS["last"]([10, 20, 30]) == 30

    def test_last_empty(self):
        assert TRANSFORM_FUNCTIONS["last"]([]) is None

    def test_max_list(self):
        assert TRANSFORM_FUNCTIONS["max"]([1, 5, 3]) == 5

    def test_min_list(self):
        assert TRANSFORM_FUNCTIONS["min"]([1, 5, 3]) == 1

    def test_sum_list(self):
        assert TRANSFORM_FUNCTIONS["sum"]([1, 2, 3]) == 6

    def test_len_list(self):
        assert TRANSFORM_FUNCTIONS["len"]([1, 2, 3]) == 3

    def test_len_scalar(self):
        assert TRANSFORM_FUNCTIONS["len"](42) == 1

    def test_mean_list(self):
        assert TRANSFORM_FUNCTIONS["mean"]([0.0, 2.0]) == pytest.approx(1.0)

    def test_mean_scalar(self):
        assert TRANSFORM_FUNCTIONS["mean"](3.0) == pytest.approx(3.0)

    def test_str(self):
        assert TRANSFORM_FUNCTIONS["str"]([1, 2]) == "[1, 2]"


class TestParameterisedTransforms:
    def test_index_positive(self):
        from trackpull.transforms import _make_index_transform

        fn = _make_index_transform(2)
        assert fn([10, 20, 30, 40]) == 30

    def test_index_negative(self):
        from trackpull.transforms import _make_index_transform

        fn = _make_index_transform(-1)
        assert fn([10, 20, 30]) == 30

    def test_index_empty_list(self):
        from trackpull.transforms import _make_index_transform

        fn = _make_index_transform(0)
        assert fn([]) is None

    def test_index_scalar_zero(self):
        from trackpull.transforms import _make_index_transform

        fn = _make_index_transform(0)
        assert fn(99) == 99

    def test_index_scalar_non_zero_raises(self):
        from trackpull.transforms import _make_index_transform

        fn = _make_index_transform(2)
        with pytest.raises(IndexError):
            fn(99)


class TestResolveTransform:
    def test_known_name(self):
        from trackpull.transforms import _resolve_transform

        fn = _resolve_transform("first")
        assert fn([1, 2]) == 1

    def test_parameterised(self):
        from trackpull.transforms import _resolve_transform

        fn = _resolve_transform("index:1")
        assert fn([10, 20, 30]) == 20

    def test_unknown_raises(self):
        from trackpull.transforms import _resolve_transform

        with pytest.raises(ValueError, match="Unknown transform"):
            _resolve_transform("nonexistent")
