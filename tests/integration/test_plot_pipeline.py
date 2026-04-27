"""Integration tests — plot_timeseries and plot_trend against a real HDF5 store.

Uses the shared mock data (model.width groups 64 and 128) populated by a
quick export + aggregate step.  Rendering is headless via matplotlib Agg.
"""

from __future__ import annotations

import os

import matplotlib
import pytest

matplotlib.use("Agg")

from trackpull.aggregate import AggregateConfig, aggregate
from trackpull.export import ExportConfig, export
from trackpull.plot import (
    AxisConfig,
    BandConfig,
    InputConfig,
    LineConfig,
    MasterPlotConfig,
    SelectConfig,
    TimeseriesPlotConfig,
    TrendPlotConfig,
    plot_timeseries,
    plot_trend,
)
from trackpull.store import STATISTICS_GROUP

# ---------------------------------------------------------------------------
# Fixture — HDF5 store with statistics/ populated
# ---------------------------------------------------------------------------


def _populate_store(mock_source, store):
    """Run export + aggregate to fill points/ and statistics/."""
    exp_cfg = ExportConfig(
        config_fields=["model.width"],
        summary_fields=["energy", "variance"],
        history_fields=["energy_step"],
    )
    export(exp_cfg, mock_source, store)

    agg_cfg = AggregateConfig(
        group_by=["model.width"],
        aggregations={
            "energy": ["mean", "std"],
            "variance": ["mean", "std"],
            "energy_step": ["mean", "std"],
        },
    )
    aggregate(agg_cfg, store)
    return store


@pytest.fixture
def stats_store(mock_source, hdf5_store):
    """HDF5Store with points/ and statistics/ both populated."""
    return _populate_store(mock_source, hdf5_store)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _master(stats_store, tmp_path) -> MasterPlotConfig:
    return MasterPlotConfig(
        inputs=[InputConfig(path=str(stats_store.path), label="test")],
        filter={},
        select=None,
        output_dir=str(tmp_path / "plots"),
        output_formats=["png"],
    )


def _assert_output(tmp_path, name):
    path = tmp_path / "plots" / f"{name}.png"
    assert path.exists(), f"Expected output file {path}"
    assert os.path.getsize(path) > 0, f"Output file {path} is empty"


# ---------------------------------------------------------------------------
# statistics/ sanity checks (verify fixture before testing plots)
# ---------------------------------------------------------------------------


class TestStatsFixture:
    def test_group_key_present(self, stats_store):
        fields = stats_store.list_fields(STATISTICS_GROUP)
        assert "model.width" in fields

    def test_scalar_mean_present(self, stats_store):
        fields = stats_store.list_fields(STATISTICS_GROUP)
        assert "mean_energy" in fields

    def test_history_mean_shape(self, stats_store):
        arr = stats_store.read_field(STATISTICS_GROUP, "mean_energy_step")
        assert arr.shape == (2, 3)  # 2 groups × 3 steps

    def test_two_width_groups(self, stats_store):
        widths = stats_store.read_field(STATISTICS_GROUP, "model.width")
        assert set(widths.tolist()) == {64.0, 128.0}


# ---------------------------------------------------------------------------
# plot_timeseries
# ---------------------------------------------------------------------------


class TestPlotTimeseries:
    def test_output_file_created(self, stats_store, tmp_path):
        master = _master(stats_store, tmp_path)
        plot_cfg = TimeseriesPlotConfig(
            output_name="ts_no_x",
            y_axis=AxisConfig(field="energy_step", label="Energy step"),
        )
        plot_timeseries(master, plot_cfg)
        _assert_output(tmp_path, "ts_no_x")

    def test_with_x_field(self, stats_store, tmp_path):
        master = _master(stats_store, tmp_path)
        plot_cfg = TimeseriesPlotConfig(
            output_name="ts_with_x",
            x_axis=AxisConfig(field="energy_step", label="Step", clip_to_shortest=True),
            y_axis=AxisConfig(field="energy_step", label="Energy step"),
        )
        plot_timeseries(master, plot_cfg)
        _assert_output(tmp_path, "ts_with_x")

    def test_band_disabled(self, stats_store, tmp_path):
        master = _master(stats_store, tmp_path)
        plot_cfg = TimeseriesPlotConfig(
            output_name="ts_no_band",
            y_axis=AxisConfig(field="energy_step"),
            band=BandConfig(enabled=False),
        )
        plot_timeseries(master, plot_cfg)
        _assert_output(tmp_path, "ts_no_band")

    def test_color_by_field(self, stats_store, tmp_path):
        """color_by a group key produces one curve per unique value."""
        master = _master(stats_store, tmp_path)
        plot_cfg = TimeseriesPlotConfig(
            output_name="ts_color_by_field",
            color_by="model.width",
            y_axis=AxisConfig(field="energy_step", label="Energy step"),
        )
        plot_timeseries(master, plot_cfg)
        _assert_output(tmp_path, "ts_color_by_field")

    def test_with_filter(self, stats_store, tmp_path):
        master = MasterPlotConfig(
            inputs=[InputConfig(path=str(stats_store.path), label="test")],
            filter={"model.width": 64.0},
            output_dir=str(tmp_path / "plots"),
            output_formats=["png"],
        )
        plot_cfg = TimeseriesPlotConfig(
            output_name="ts_filtered",
            y_axis=AxisConfig(field="energy_step"),
        )
        plot_timeseries(master, plot_cfg)
        _assert_output(tmp_path, "ts_filtered")

    def test_with_select(self, stats_store, tmp_path):
        master = MasterPlotConfig(
            inputs=[InputConfig(path=str(stats_store.path), label="test")],
            filter={},
            select=SelectConfig(by="mean_energy", criterion="min"),
            output_dir=str(tmp_path / "plots"),
            output_formats=["png"],
        )
        plot_cfg = TimeseriesPlotConfig(
            output_name="ts_selected",
            y_axis=AxisConfig(field="energy_step"),
        )
        plot_timeseries(master, plot_cfg)
        _assert_output(tmp_path, "ts_selected")

    def test_multiple_formats(self, stats_store, tmp_path):
        master = MasterPlotConfig(
            inputs=[InputConfig(path=str(stats_store.path), label="test")],
            filter={},
            output_dir=str(tmp_path / "plots"),
            output_formats=["png", "pdf"],
        )
        plot_cfg = TimeseriesPlotConfig(
            output_name="ts_multi_fmt",
            y_axis=AxisConfig(field="energy_step"),
        )
        plot_timeseries(master, plot_cfg)
        assert (tmp_path / "plots" / "ts_multi_fmt.png").exists()
        assert (tmp_path / "plots" / "ts_multi_fmt.pdf").exists()


# ---------------------------------------------------------------------------
# plot_trend
# ---------------------------------------------------------------------------


class TestPlotTrend:
    def test_output_file_created(self, stats_store, tmp_path):
        master = _master(stats_store, tmp_path)
        plot_cfg = TrendPlotConfig(
            output_name="trend_basic",
            x_axis=AxisConfig(field="model.width", label="Width"),
            y_axis=AxisConfig(field="energy", label="Energy"),
        )
        plot_trend(master, plot_cfg)
        _assert_output(tmp_path, "trend_basic")

    def test_log_scale_x(self, stats_store, tmp_path):
        master = _master(stats_store, tmp_path)
        plot_cfg = TrendPlotConfig(
            output_name="trend_log_x",
            x_axis=AxisConfig(field="model.width", scale="log", log_base=2),
            y_axis=AxisConfig(field="energy"),
        )
        plot_trend(master, plot_cfg)
        _assert_output(tmp_path, "trend_log_x")

    def test_auto_scale_y(self, stats_store, tmp_path):
        master = _master(stats_store, tmp_path)
        plot_cfg = TrendPlotConfig(
            output_name="trend_auto_y",
            x_axis=AxisConfig(field="model.width"),
            y_axis=AxisConfig(field="energy", scale="auto"),
        )
        plot_trend(master, plot_cfg)
        _assert_output(tmp_path, "trend_auto_y")

    def test_color_by_field(self, stats_store, tmp_path):
        """color_by a non-group field falls back gracefully (single group)."""
        master = _master(stats_store, tmp_path)
        plot_cfg = TrendPlotConfig(
            output_name="trend_color_by",
            color_by="label",
            x_axis=AxisConfig(field="model.width"),
            y_axis=AxisConfig(field="energy"),
        )
        plot_trend(master, plot_cfg)
        _assert_output(tmp_path, "trend_color_by")

    def test_with_filter(self, stats_store, tmp_path):
        master = MasterPlotConfig(
            inputs=[InputConfig(path=str(stats_store.path), label="test")],
            filter={"model.width": 64.0},
            output_dir=str(tmp_path / "plots"),
            output_formats=["png"],
        )
        plot_cfg = TrendPlotConfig(
            output_name="trend_filtered",
            x_axis=AxisConfig(field="model.width"),
            y_axis=AxisConfig(field="energy"),
        )
        plot_trend(master, plot_cfg)
        _assert_output(tmp_path, "trend_filtered")

    def test_band_disabled(self, stats_store, tmp_path):
        master = _master(stats_store, tmp_path)
        plot_cfg = TrendPlotConfig(
            output_name="trend_no_band",
            x_axis=AxisConfig(field="model.width"),
            y_axis=AxisConfig(field="energy"),
            band=BandConfig(enabled=False),
        )
        plot_trend(master, plot_cfg)
        _assert_output(tmp_path, "trend_no_band")

    def test_with_marker(self, stats_store, tmp_path):
        master = _master(stats_store, tmp_path)
        plot_cfg = TrendPlotConfig(
            output_name="trend_marker",
            x_axis=AxisConfig(field="model.width"),
            y_axis=AxisConfig(field="energy"),
            line=LineConfig(linewidth=2, marker="o"),
        )
        plot_trend(master, plot_cfg)
        _assert_output(tmp_path, "trend_marker")
