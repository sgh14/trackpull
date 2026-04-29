"""Hydra CLI entry point for trackpull.

Requires the ``[hydra]`` optional extra::

    pip install trackpull[hydra,wandb]

Usage::

    # Run full pipeline (export + aggregate)
    trackpull --config-dir=conf -cn example

    # Hydra multirun sweep
    trackpull --config-dir=conf -cn example \\
        "source.filters.tags=[v1]","source.filters.tags=[v2]" -m

YAML schema
-----------
::

    # conf/example.yaml
    # @package _global_

    source:
      project: me/my-project     # required
      entity: null               # optional; null = wandb default
      filters:
        tags: [v2]               # optional

    export:
      config_fields: [model.width, model.depth]
      summary_fields: [energy, variance]
      history_fields: [energy_step]          # optional
      transforms:
        model.hidden_dims: first             # optional

    aggregate:
      group_by: [model.width]
      aggregations:
        energy: [mean, std]
        variance: [mean]
      nan_policy: warn                       # optional; default: warn

    output: results/my_experiment.h5         # required
"""

from __future__ import annotations

import logging
import sys

import hydra
from omegaconf import DictConfig, OmegaConf

from trackpull.aggregate import AggregateConfig, aggregate
from trackpull.export import ExportConfig, export
from trackpull.source import WandbSource
from trackpull.store import HDF5Store

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DictConfig -> dataclass converters
# ---------------------------------------------------------------------------


def _source_from_cfg(cfg: DictConfig) -> WandbSource:
    filters_raw = cfg.source.get("filters") or {}
    filters = dict(OmegaConf.to_container(filters_raw, resolve=True))
    return WandbSource(
        project=cfg.source.project,
        entity=cfg.source.get("entity"),
        filters=filters or None,
    )


def _export_config_from_cfg(cfg: DictConfig) -> ExportConfig:
    exp = cfg.get("export") or {}
    return ExportConfig(
        config_fields=list(exp.get("config_fields") or []),
        summary_fields=list(exp.get("summary_fields") or []),
        history_fields=list(exp.get("history_fields") or []),
        transforms=dict(
            OmegaConf.to_container(exp.get("transforms") or {}, resolve=True)
        ),
    )


def _aggregate_config_from_cfg(cfg: DictConfig) -> AggregateConfig:
    agg = cfg.get("aggregate") or {}
    raw_aggs = OmegaConf.to_container(agg.get("aggregations") or {}, resolve=True)
    aggregations = {
        col: ([v] if isinstance(v, str) else list(v)) for col, v in raw_aggs.items()
    }
    return AggregateConfig(
        group_by=list(agg.get("group_by") or []),
        aggregations=aggregations,
        nan_policy=agg.get("nan_policy", "warn"),
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

    if not cfg.get("output"):
        logger.error("'output' path is required in the config.")
        sys.exit(1)

    store = HDF5Store(cfg.output)

    source = _source_from_cfg(cfg)
    export_config = _export_config_from_cfg(cfg)
    export(export_config, source, store)

    if cfg.get("aggregate"):
        aggregate_config = _aggregate_config_from_cfg(cfg)
        aggregate(aggregate_config, store)

    logger.info("Done -> %s", cfg.output)


def main() -> None:
    """Console script entry point."""
    _run()
