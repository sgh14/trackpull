"""Hydra CLI entry point for trackplot.

Usage::

    # Single plot (type specified in child YAML)
    trackplot --config-dir=conf --config-name=example_plot "plot=timeseries_energy"

    # Hydra multirun sweep over filter combinations
    trackplot --config-dir=conf --config-name=example_plot -m

YAML schema
-----------
::

        # conf/example_plot.yaml
    defaults:
      - _self_
      - plot: null   # specify via CLI or Hydra sweeper

    plot:
      inputs:
        - path: results/plain.h5
          label: "Plain"
        - path: results/sr.h5
          label: "SR"
      filter: {}              # Hydra sweeper appends keys here
      select:
        by: mean_Energy_RelError_Final
        criterion: min
      figure:
        style: conf/plots/styles/publication.mplstyle
      output:
        dir: report/figures
        formats: [pdf, png, svg]

    # conf/plot/timeseries_energy.yaml
    type: timeseries
    output:
      name: energy_vs_time
    color_by: label           # 'label' = input file label (default)
    axes:
      x:
        field: Training_Time
        label: "Training Time (s)"
      y:
        field: Energy_Mean_Exact
        label: "Energy"
    line:
      linewidth: 2
    band:
      enabled: true
      alpha: 0.2

    # conf/plot/trend_energy_vs_width.yaml
    type: trend
    output:
      name: energy_vs_width
    color_by: [vstate.n_samples, graph.length]
    axes:
      x:
        field: model.hidden_dims
        label: "Model Width"
        scale: log
        log_base: 2
      y:
        field: variational_energy
        label: "Energy"
        scale: auto
    line:
      linewidth: 2
      marker: o
    band:
      enabled: true
      alpha: 0.2
    reference_lines:
      horizontal:
        - field: exact_energy
          label: "Exact energy"
"""

from __future__ import annotations

import logging
import sys

import hydra
from omegaconf import DictConfig, OmegaConf

from trackpull.plot import (
    AxisConfig,
    BandConfig,
    InputConfig,
    LineConfig,
    MasterPlotConfig,
    ReferenceLineConfig,
    SelectConfig,
    TimeseriesPlotConfig,
    TrendPlotConfig,
    _flatten_dict,
    plot_timeseries,
    plot_trend,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DictConfig -> dataclass converters  (one per config concept)
# ---------------------------------------------------------------------------


def _input_from_cfg(inp: DictConfig) -> InputConfig:
    return InputConfig(
        path=inp.path,
        label=inp.get("label"),
        label_template=inp.get("label_template"),
    )


def _select_from_cfg(plot: DictConfig) -> SelectConfig | None:
    sel = plot.get("select")
    if not sel:
        return None
    return SelectConfig(
        by=sel.by,
        criterion=sel.get("criterion", "min"),
    )


def _axis_from_cfg(cfg: DictConfig | None) -> AxisConfig | None:
    if cfg is None:
        return None
    return AxisConfig(
        field=cfg.get("field"),
        label=cfg.get("label", ""),
        scale=cfg.get("scale", "linear"),
        log_base=cfg.get("log_base", 10),
        ticks=cfg.get("ticks", "auto"),
        tick_rotation=cfg.get("tick_rotation", 0),
        clip_to_shortest=cfg.get("clip_to_shortest", False),
    )


def _line_from_cfg(cfg: DictConfig | None) -> LineConfig:
    if cfg is None:
        return LineConfig()
    return LineConfig(
        linewidth=cfg.get("linewidth", 1.5),
        marker=cfg.get("marker"),
    )


def _band_from_cfg(cfg: DictConfig | None) -> BandConfig:
    if cfg is None:
        return BandConfig()
    return BandConfig(
        enabled=cfg.get("enabled", True),
        alpha=cfg.get("alpha", 0.2),
    )


def _ref_lines_from_cfg(cfg: DictConfig | None) -> list[ReferenceLineConfig]:
    if cfg is None or not cfg.get("horizontal"):
        return []
    return [
        ReferenceLineConfig(field=r.field, label=r.get("label", ""))
        for r in cfg.horizontal
    ]


def _master_from_cfg(cfg: DictConfig) -> MasterPlotConfig:
    plot = cfg.plot
    raw_filter = plot.get("filter") or {}
    if not isinstance(raw_filter, dict):
        raw_filter = OmegaConf.to_container(raw_filter, resolve=True)
    return MasterPlotConfig(
        inputs=[_input_from_cfg(inp) for inp in plot.inputs],
        filter=_flatten_dict(raw_filter),
        select=_select_from_cfg(plot),
        figure_style=(plot.get("figure") or {}).get("style"),
        output_dir=(plot.get("output") or {}).get("dir", "."),
        output_formats=list((plot.get("output") or {}).get("formats", ["pdf"])),
    )


def _color_by_from_cfg(plot: DictConfig) -> str | list[str]:
    """Normalize color_by from YAML into str or list[str]."""
    color_by = plot.get("color_by", "label")
    if isinstance(color_by, str):
        return color_by

    raw = OmegaConf.to_container(color_by, resolve=True)
    if isinstance(raw, list):
        return [str(value) for value in raw]
    return "label"


def _timeseries_from_cfg(cfg: DictConfig) -> TimeseriesPlotConfig:
    plot = cfg.plot
    axes = plot.get("axes") or {}
    return TimeseriesPlotConfig(
        output_name=(plot.get("output") or {}).get("name", "plot"),
        color_by=_color_by_from_cfg(plot),
        x_axis=_axis_from_cfg(axes.get("x")),
        y_axis=_axis_from_cfg(axes.get("y")) or AxisConfig(),
        line=_line_from_cfg(plot.get("line")),
        band=_band_from_cfg(plot.get("band")),
    )


def _trend_from_cfg(cfg: DictConfig) -> TrendPlotConfig:
    plot = cfg.plot
    axes = plot.get("axes") or {}
    return TrendPlotConfig(
        output_name=(plot.get("output") or {}).get("name", "plot"),
        color_by=_color_by_from_cfg(plot),
        x_axis=_axis_from_cfg(axes.get("x")) or AxisConfig(),
        y_axis=_axis_from_cfg(axes.get("y")) or AxisConfig(),
        line=_line_from_cfg(plot.get("line")),
        band=_band_from_cfg(plot.get("band")),
        reference_lines=_ref_lines_from_cfg(plot.get("reference_lines")),
    )


# ---------------------------------------------------------------------------
# Hydra entry point
# ---------------------------------------------------------------------------


@hydra.main(version_base=None, config_path=None, config_name=None)
def _run(cfg: DictConfig) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s - %(message)s",
        datefmt="%H:%M:%S",
    )
    logger.info("Configuration:\n%s", OmegaConf.to_yaml(cfg))

    plot_type = (cfg.get("plot") or {}).get("type")
    if not plot_type:
        logger.error("'plot.type' is required (timeseries or trend).")
        sys.exit(1)

    master_cfg = _master_from_cfg(cfg)

    if plot_type == "timeseries":
        plot_timeseries(master_cfg, _timeseries_from_cfg(cfg))
    elif plot_type == "trend":
        plot_trend(master_cfg, _trend_from_cfg(cfg))
    else:
        logger.error("Unknown plot type %r. Expected: timeseries, trend.", plot_type)
        sys.exit(1)

    logger.info("Done -> %s/", master_cfg.output_dir)


def main() -> None:
    """Console script entry point."""
    _run()
