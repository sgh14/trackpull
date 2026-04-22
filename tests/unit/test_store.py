"""Unit tests for HDF5Store and InMemoryStore."""

from __future__ import annotations

import numpy as np
import pytest

from trackpull.store import HDF5Store, InMemoryStore, POINTS_GROUP


# ---------------------------------------------------------------------------
# Parametrise over both backends
# ---------------------------------------------------------------------------


@pytest.fixture(params=["memory", "hdf5"])
def store(request, tmp_path):
    if request.param == "memory":
        return InMemoryStore()
    return HDF5Store(tmp_path / "test.h5")


# ---------------------------------------------------------------------------
# write / read round-trip
# ---------------------------------------------------------------------------


class TestWriteRead:
    def test_float_array(self, store):
        data = {"x": np.array([1.0, 2.0, 3.0])}
        store.write("grp", data)
        result = store.read("grp")
        np.testing.assert_allclose(result["x"], data["x"])

    def test_2d_array(self, store):
        data = {"mat": np.array([[1.0, 2.0], [3.0, 4.0]])}
        store.write("grp", data)
        result = store.read("grp")
        np.testing.assert_allclose(result["mat"], data["mat"])

    def test_scalar(self, store):
        data = {"v": np.float64(3.14)}
        store.write("grp", data)
        result = store.read("grp")
        assert float(result["v"]) == pytest.approx(3.14)

    def test_string_array(self, store):
        data = {"ids": np.array(["r1", "r2", "r3"], dtype=object)}
        store.write("grp", data)
        result = store.read("grp")
        assert list(result["ids"]) == ["r1", "r2", "r3"]

    def test_overwrites_existing_group(self, store):
        store.write("grp", {"x": np.array([1.0])})
        store.write("grp", {"x": np.array([99.0])})
        result = store.read("grp")
        assert float(result["x"].flat[0]) == pytest.approx(99.0)

    def test_read_unknown_group_raises(self, store):
        with pytest.raises(KeyError):
            store.read("nonexistent")


# ---------------------------------------------------------------------------
# list_fields
# ---------------------------------------------------------------------------


class TestListFields:
    def test_returns_field_names(self, store):
        store.write("grp", {"a": np.array([1.0]), "b": np.array([2.0])})
        fields = store.list_fields("grp")
        assert set(fields) == {"a", "b"}

    def test_unknown_group_raises(self, store):
        with pytest.raises(KeyError):
            store.list_fields("nonexistent")


# ---------------------------------------------------------------------------
# read_column
# ---------------------------------------------------------------------------


class TestReadColumn:
    def test_returns_correct_array(self, store):
        store.write("grp", {"x": np.array([10.0, 20.0]), "y": np.array([1.0, 2.0])})
        col = store.read_column("grp", "x")
        np.testing.assert_allclose(col, [10.0, 20.0])

    def test_unknown_field_raises(self, store):
        store.write("grp", {"x": np.array([1.0])})
        with pytest.raises(KeyError):
            store.read_column("grp", "nonexistent")

    def test_unknown_group_raises(self, store):
        with pytest.raises(KeyError):
            store.read_column("nonexistent", "x")


# ---------------------------------------------------------------------------
# open_writer (streaming history)
# ---------------------------------------------------------------------------


class TestOpenWriter:
    def test_scalar_only(self, store):
        N = 3
        with store.open_writer(N, history_fields=[]) as writer:
            writer.write_scalars(
                {
                    "run_id": np.array(["r1", "r2", "r3"], dtype=object),
                    "energy": np.array([-1.0, -2.0, -3.0]),
                }
            )
        fields = store.list_fields(POINTS_GROUP)
        assert "run_id" in fields
        assert "energy" in fields
        energy = store.read_column(POINTS_GROUP, "energy")
        np.testing.assert_allclose(energy, [-1.0, -2.0, -3.0])

    def test_history_shape(self, store):
        N, T = 3, 4
        with store.open_writer(N, history_fields=["loss"]) as writer:
            writer.ensure_history_capacity(T)
            for i in range(N):
                writer.write_history_row("loss", i, np.arange(T, dtype=float))
        mat = store.read_column(POINTS_GROUP, "loss")
        assert mat.shape == (N, T)

    def test_history_nan_fill(self, store):
        """Runs with shorter history should be NaN-padded to max T."""
        N = 2
        with store.open_writer(N, history_fields=["loss"]) as writer:
            # Run 0: T=3, run 1: T=2 → row 1 should be NaN at column 2
            writer.ensure_history_capacity(3)
            writer.write_history_row("loss", 0, np.array([1.0, 2.0, 3.0]))
            writer.ensure_history_capacity(2)
            writer.write_history_row("loss", 1, np.array([4.0, 5.0]))
        mat = store.read_column(POINTS_GROUP, "loss")
        assert mat.shape == (2, 3)
        np.testing.assert_allclose(mat[0], [1.0, 2.0, 3.0])
        np.testing.assert_allclose(mat[1, :2], [4.0, 5.0])
        assert np.isnan(mat[1, 2])

    def test_open_writer_overwrites_previous(self, store):
        """Calling open_writer twice should replace the points/ group."""
        with store.open_writer(1, []) as writer:
            writer.write_scalars({"x": np.array([1.0])})
        with store.open_writer(1, []) as writer:
            writer.write_scalars({"x": np.array([99.0])})
        col = store.read_column(POINTS_GROUP, "x")
        assert float(col[0]) == pytest.approx(99.0)
