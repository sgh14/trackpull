"""Unit tests for HDF5Store and InMemoryStore."""

from __future__ import annotations

import numpy as np
import pytest

from trackpull.store import HDF5Store

# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def store(tmp_path):
    return HDF5Store(tmp_path / "test.h5")


# ---------------------------------------------------------------------------
# write_field / read_field round-trip
# ---------------------------------------------------------------------------


class TestWriteRead:
    def test_float_array(self, store):
        store.write_field("grp", "x", np.array([1.0, 2.0, 3.0]))
        np.testing.assert_allclose(store.read_field("grp", "x"), [1.0, 2.0, 3.0])

    def test_2d_array(self, store):
        store.write_field("grp", "mat", np.array([[1.0, 2.0], [3.0, 4.0]]))
        np.testing.assert_allclose(
            store.read_field("grp", "mat"), [[1.0, 2.0], [3.0, 4.0]]
        )

    def test_scalar(self, store):
        store.write_field("grp", "v", np.float64(3.14))
        assert float(store.read_field("grp", "v")) == pytest.approx(3.14)

    def test_string_array(self, store):
        store.write_field("grp", "ids", np.array(["r1", "r2", "r3"], dtype=object))
        assert list(store.read_field("grp", "ids")) == ["r1", "r2", "r3"]

    def test_write_field_overwrites(self, store):
        store.write_field("grp", "x", np.array([1.0]))
        store.write_field("grp", "x", np.array([99.0]))
        assert float(store.read_field("grp", "x").flat[0]) == pytest.approx(99.0)

    def test_clear_group_removes_stale_fields(self, store):
        store.write_field("grp", "old", np.array([1.0]))
        store.clear_group("grp")
        store.write_field("grp", "new", np.array([2.0]))
        assert store.list_fields("grp") == ["new"]

    def test_clear_group_nonexistent_is_noop(self, store):
        store.clear_group("nonexistent")  # should not raise


# ---------------------------------------------------------------------------
# list_fields
# ---------------------------------------------------------------------------


class TestListFields:
    def test_returns_field_names(self, store):
        store.write_field("grp", "a", np.array([1.0]))
        store.write_field("grp", "b", np.array([2.0]))
        fields = store.list_fields("grp")
        assert set(fields) == {"a", "b"}

    def test_unknown_group_raises(self, store):
        store.write_field("grp", "x", np.array([1.0]))
        with pytest.raises(KeyError):
            store.list_fields("nonexistent")


# ---------------------------------------------------------------------------
# read_field
# ---------------------------------------------------------------------------


class TestReadColumn:
    def test_returns_correct_array(self, store):
        store.write_field("grp", "x", np.array([10.0, 20.0]))
        store.write_field("grp", "y", np.array([1.0, 2.0]))
        col = store.read_field("grp", "x")
        np.testing.assert_allclose(col, [10.0, 20.0])

    def test_unknown_field_raises(self, store):
        store.write_field("grp", "x", np.array([1.0]))
        with pytest.raises(KeyError):
            store.read_field("grp", "nonexistent")

    def test_unknown_group_raises(self, store):
        store.write_field("grp", "x", np.array([1.0]))
        with pytest.raises(KeyError):
            store.read_field("nonexistent", "x")


# ---------------------------------------------------------------------------
# run cache
# ---------------------------------------------------------------------------


class TestRunCache:
    def test_bulk_read_configs_and_summaries_preserve_order(self, store):
        store.write_run_cache(
            "r2",
            config={"width": 128},
            summary={"energy": -2.0},
            history={"energy_step": [-0.8, -2.0]},
        )
        store.write_run_cache(
            "r1",
            config={"width": 64},
            summary={"energy": -1.0},
            history={"energy_step": [-0.5, -1.0]},
        )

        configs = store.read_run_cache_configs(["r1", "r2"])
        summaries = store.read_run_cache_summaries(["r1", "r2"])

        assert configs == [{"width": 64}, {"width": 128}]
        assert summaries == [{"energy": -1.0}, {"energy": -2.0}]

    def test_bulk_read_history_fields_returns_lists_in_order(self, store):
        store.write_run_cache(
            "r1",
            config={},
            summary={},
            history={"energy_step": [-0.5, -1.0]},
        )
        store.write_run_cache(
            "r2",
            config={},
            summary={},
            history={"energy_step": [-0.8, -2.0]},
        )

        values = store.read_run_cache_history_fields(["r2", "r1"], "energy_step")

        assert values == [[-0.8, -2.0], [-0.5, -1.0]]

    def test_bulk_read_history_fields_returns_empty_list_for_missing_field(self, store):
        store.write_run_cache("r1", config={}, summary={}, history={})
        store.write_run_cache(
            "r2",
            config={},
            summary={},
            history={"energy_step": [-0.8, -2.0]},
        )

        values = store.read_run_cache_history_fields(["r1", "r2"], "energy_step")

        assert values == [[], [-0.8, -2.0]]
