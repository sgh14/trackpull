"""Export step — fetch runs from a source and write ``points/`` to a store."""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from tqdm import tqdm

from trackpull.source import RunRecord, WandbSource
from trackpull.store import POINTS_GROUP, HDF5Store
from trackpull.transforms import _resolve_transform

logger = logging.getLogger(__name__)


@dataclass
class ExportConfig:
    """Configuration for the export step.

    Args:
        config_fields:  Run config paths to extract (dot-separated for nested
                        keys, e.g. ``"model.width"``).
        summary_fields: Run summary field names to extract.
        history_fields: Per-step history field names to extract.  When empty
                        (the default), ``fetch_history`` is never called.
        transforms:     ``{field: transform_name}`` mapping applied before
                        writing.  See :mod:`trackpull.transforms`.
    """

    config_fields: list[str]
    summary_fields: list[str]
    history_fields: list[str] = field(default_factory=list)
    transforms: dict[str, str] = field(default_factory=dict)


def _get_field_value(data: dict[str, Any], key: str, default: Any = None) -> Any:
    """Retrieve a value from a nested dict using a dot-separated key path."""
    value: Any = data
    for part in key.split("."):
        if isinstance(value, dict) and part in value:
            value = value[part]
        else:
            return default
    return value


def _build_array(values: list[Any]) -> np.ndarray:
    """Build a numpy array from a list of per-run values.

    - Scalar per run → shape ``(N,)``.
    - Sequence per run → shape ``(N, max_length)`` with NaN / empty-string padding.
    - ``None`` entries are treated as missing (NaN or empty string).
    """
    arrays = []
    for value in values:
        arrays.append(np.atleast_1d(np.asarray(value)) if value is not None else None)

    non_none = [a for a in arrays if a is not None]
    if not non_none:
        return np.full(len(values), np.nan)

    is_string = non_none[0].dtype.kind in ("U", "S", "O")
    if is_string:
        empty_value = ""
        dtype = object
    else:
        empty_value = np.nan
        dtype = float
        arrays = [a.astype(float) if a is not None else None for a in arrays]

    max_length = max(a.shape[0] for a in non_none)
    final_array = np.full((len(values), max_length), empty_value, dtype=dtype)
    for i, array in enumerate(arrays):
        if array is not None:
            final_array[i, : array.shape[0]] = array

    return final_array.squeeze(axis=1) if max_length == 1 else final_array


def export_run_ids(runs: list[RunRecord], store: HDF5Store) -> None:
    """Export run IDs as a separate field."""
    runs_id = np.array([run.id for run in runs], dtype=object)
    store.write_field(POINTS_GROUP, "run_id", runs_id)


def export_config_fields(
    runs: list[RunRecord], store: HDF5Store, config: ExportConfig
) -> None:
    """Export config fields."""
    for field_name in config.config_fields:
        values = [_get_field_value(run.config, field_name) for run in runs]
        if field_name in config.transforms:
            transform = _resolve_transform(config.transforms[field_name])
            values = [transform(value) for value in values]

        store.write_field(POINTS_GROUP, field_name, _build_array(values))


def export_summary_fields(
    runs: list[RunRecord], store: HDF5Store, config: ExportConfig
) -> None:
    """Export summary fields."""
    for field_name in config.summary_fields:
        values = [run.summary.get(field_name) for run in runs]
        if field_name in config.transforms:
            transform = _resolve_transform(config.transforms[field_name])
            values = [transform(value) for value in values]

        store.write_field(POINTS_GROUP, field_name, _build_array(values))


def export_history_fields(
    runs: list[RunRecord], store: HDF5Store, config: ExportConfig
) -> None:
    """Export history fields."""
    for field_name in tqdm(config.history_fields, desc="history fields", unit="field"):
        values = [
            [step.get(field_name) for step in run.fetch_history(field_name)]
            for run in runs
        ]
        store.write_field(POINTS_GROUP, field_name, _build_array(values))


def export(config: ExportConfig, source: WandbSource, store: HDF5Store) -> None:
    """Fetch runs from *source* and write ``points/`` to *store* one field at a time."""
    runs = list(source.fetch())
    if not runs:
        logger.error("No runs found. Check source/filter settings.")
        sys.exit(1)

    N = len(runs)
    logger.info("Found %d runs", N)

    store.clear_group(POINTS_GROUP)
    export_run_ids(runs, store)
    export_config_fields(runs, store, config)
    export_summary_fields(runs, store, config)
    export_history_fields(runs, store, config)

    logger.info(
        "Export complete — %d runs, %d fields",
        N,
        1
        + len(config.config_fields)
        + len(config.summary_fields)
        + len(config.history_fields),
    )
