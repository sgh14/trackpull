"""Integration tests — full export → aggregate pipeline.

Uses MockSource + InMemoryStore (no disk, no W&B).
Also runs against HDF5Store to verify the full on-disk path.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from trackpull.aggregate import AggregateConfig, aggregate
from trackpull.export import ExportConfig, export
from trackpull.store import POINTS_GROUP, STATISTICS_GROUP


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_export(source, store, history=False):
    cfg = ExportConfig(
        config_fields=["model.width", "seed"],
        summary_fields=["energy", "variance"],
        history_fields=["energy_step"] if history else [],
    )
    export(cfg, source, store)
    return cfg


def _run_aggregate(store):
    cfg = AggregateConfig(
        group_by=["model.width"],
        aggregations={"energy": ["mean", "std"], "variance": ["mean"]},
        nan_policy="warn",
    )
    aggregate(cfg, store)
    return cfg


# ---------------------------------------------------------------------------
# Export step
# ---------------------------------------------------------------------------


class TestExport:
    def test_points_group_created(self, mock_source, memory_store):
        _run_export(mock_source, memory_store)
        fields = memory_store.list_fields(POINTS_GROUP)
        assert "run_id" in fields
        assert "model.width" in fields
        assert "energy" in fields

    def test_run_ids_correct(self, mock_source, memory_store):
        _run_export(mock_source, memory_store)
        ids = memory_store.read_column(POINTS_GROUP, "run_id")
        assert set(ids) == {"r1", "r2", "r3"}

    def test_scalar_values_correct(self, mock_source, memory_store):
        _run_export(mock_source, memory_store)
        widths = memory_store.read_column(POINTS_GROUP, "model.width")
        assert set(widths) == {64.0, 128.0}

    def test_history_shape(self, mock_source, memory_store):
        _run_export(mock_source, memory_store, history=True)
        mat = memory_store.read_column(POINTS_GROUP, "energy_step")
        assert mat.shape == (3, 3)  # 3 runs × 3 steps

    def test_history_values(self, mock_source, memory_store):
        _run_export(mock_source, memory_store, history=True)
        mat = memory_store.read_column(POINTS_GROUP, "energy_step")
        # r1 last step: -1.2
        r1_idx = list(
            memory_store.read_column(POINTS_GROUP, "run_id")
        ).index("r1")
        assert mat[r1_idx, -1] == pytest.approx(-1.2)

    def test_hdf5_store(self, mock_source, hdf5_store):
        _run_export(mock_source, hdf5_store)
        fields = hdf5_store.list_fields(POINTS_GROUP)
        assert "energy" in fields

    def test_transforms_applied(self, memory_store):
        from trackpull.source import RunRecord

        class _Src:
            def fetch(self):
                return iter(
                    [
                        RunRecord(
                            id="x",
                            config={"hidden_dims": [64, 32]},
                            summary={"energy": -1.0},
                        )
                    ]
                )

        cfg = ExportConfig(
            config_fields=["hidden_dims"],
            summary_fields=["energy"],
            transforms={"hidden_dims": "first"},
        )
        export(cfg, _Src(), memory_store)
        col = memory_store.read_column(POINTS_GROUP, "hidden_dims")
        assert float(col[0]) == pytest.approx(64.0)


# ---------------------------------------------------------------------------
# Aggregate step
# ---------------------------------------------------------------------------


class TestAggregate:
    def test_statistics_group_created(self, mock_source, memory_store):
        _run_export(mock_source, memory_store)
        _run_aggregate(memory_store)
        fields = memory_store.list_fields(STATISTICS_GROUP)
        assert "mean_energy" in fields
        assert "std_energy" in fields
        assert "mean_variance" in fields

    def test_group_by_values(self, mock_source, memory_store):
        _run_export(mock_source, memory_store)
        _run_aggregate(memory_store)
        widths = memory_store.read_column(STATISTICS_GROUP, "model.width")
        assert set(widths) == {64.0, 128.0}

    def test_mean_correct(self, mock_source, memory_store):
        _run_export(mock_source, memory_store)
        _run_aggregate(memory_store)
        widths = memory_store.read_column(STATISTICS_GROUP, "model.width")
        means = memory_store.read_column(STATISTICS_GROUP, "mean_energy")
        w64_idx = int(np.where(widths == 64.0)[0][0])
        # r1=-1.2, r2=-1.5 → mean=-1.35
        assert means[w64_idx] == pytest.approx(-1.35)

    def test_single_group_no_group_by(self, mock_source, memory_store):
        _run_export(mock_source, memory_store)
        cfg = AggregateConfig(
            group_by=[],
            aggregations={"energy": ["mean"]},
        )
        aggregate(cfg, memory_store)
        mean_energy = memory_store.read_column(STATISTICS_GROUP, "mean_energy")
        # mean of -1.2, -1.5, -2.0 — single group, shape (1,)
        assert float(mean_energy.flat[0]) == pytest.approx((-1.2 - 1.5 - 2.0) / 3)

    def test_history_aggregate_shape(self, mock_source, memory_store):
        _run_export(mock_source, memory_store, history=True)
        cfg = AggregateConfig(
            group_by=["model.width"],
            aggregations={"energy_step": ["mean"]},
        )
        aggregate(cfg, memory_store)
        mean_curve = memory_store.read_column(STATISTICS_GROUP, "mean_energy_step")
        # 2 groups × 3 timesteps
        assert mean_curve.shape == (2, 3)

    def test_unknown_group_by_raises(self, mock_source, memory_store):
        _run_export(mock_source, memory_store)
        cfg = AggregateConfig(
            group_by=["nonexistent_field"],
            aggregations={"energy": ["mean"]},
        )
        with pytest.raises(ValueError, match="not found in points"):
            aggregate(cfg, memory_store)

    def test_unknown_agg_function_raises(self, mock_source, memory_store):
        _run_export(mock_source, memory_store)
        cfg = AggregateConfig(
            group_by=["model.width"],
            aggregations={"energy": ["bogus"]},
        )
        with pytest.raises(ValueError, match="Unknown aggregation"):
            aggregate(cfg, memory_store)


# ---------------------------------------------------------------------------
# NaN policy
# ---------------------------------------------------------------------------


class TestNanPolicy:
    def _make_source_with_nan(self):
        from trackpull.source import RunRecord

        class _Src:
            def fetch(self):
                return iter(
                    [
                        RunRecord(
                            id="a",
                            config={"width": 64},
                            summary={"energy": None},  # → NaN
                        ),
                        RunRecord(
                            id="b",
                            config={"width": 64},
                            summary={"energy": -1.0},
                        ),
                    ]
                )

        return _Src()

    def test_warn_policy_logs_warning(self, memory_store, caplog):
        src = self._make_source_with_nan()
        export(ExportConfig(["width"], ["energy"]), src, memory_store)
        cfg = AggregateConfig(
            group_by=["width"],
            aggregations={"energy": ["mean"]},
            nan_policy="warn",
        )
        with caplog.at_level(logging.WARNING, logger="trackpull.aggregate"):
            aggregate(cfg, memory_store)
        assert "NaN" in caplog.text

    def test_ignore_policy_no_warning(self, memory_store, caplog):
        src = self._make_source_with_nan()
        export(ExportConfig(["width"], ["energy"]), src, memory_store)
        cfg = AggregateConfig(
            group_by=["width"],
            aggregations={"energy": ["mean"]},
            nan_policy="ignore",
        )
        with caplog.at_level(logging.WARNING, logger="trackpull.aggregate"):
            aggregate(cfg, memory_store)
        assert "NaN inputs" not in caplog.text

    def test_raise_policy_raises(self, memory_store):
        src = self._make_source_with_nan()
        export(ExportConfig(["width"], ["energy"]), src, memory_store)
        cfg = AggregateConfig(
            group_by=["width"],
            aggregations={"energy": ["mean"]},
            nan_policy="raise",
        )
        with pytest.raises(ValueError, match="NaN inputs"):
            aggregate(cfg, memory_store)

    def test_invalid_nan_policy_raises(self, memory_store):
        cfg = AggregateConfig(nan_policy="invalid")
        with pytest.raises(ValueError, match="Invalid nan_policy"):
            aggregate(cfg, memory_store)
