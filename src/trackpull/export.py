"""Export step — fetch runs from a source and write raw data to a store.

Writes the ``points/`` group.  The aggregate step reads this group.

Memory strategy
---------------
Scalar data (config, summary) for all runs is accumulated in a list of dicts
and written once at the end.  History data is streamed row-by-row via the
store's ``open_writer`` context manager so peak memory is *O(T_i)* per run
rather than *O(N × T_max × n_fields)*.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from trackpull.source import RunSource
from trackpull.store import AnalysisStore
from trackpull.transforms import apply_transforms, warn_untransformed_lists

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class ExportConfig:
    """Configuration for the export step.

    Args:
        config_fields:   Run config field paths to extract, supporting
                         dot-separated nested keys (e.g. ``"model.width"``).
        summary_fields:  Run summary field names to extract.
        history_fields:  Per-step history field names to extract.  When empty
                         (the default), ``fetch_history`` is never called and
                         no 2-D arrays are written.
        transforms:      ``{field: transform_name}`` mapping applied before
                         writing.  See :mod:`trackpull.transforms` for the
                         full catalogue.
    """

    config_fields: list[str]
    summary_fields: list[str]
    history_fields: list[str] = field(default_factory=list)
    transforms: dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_nested_value(data: dict[str, Any], key: str, default: Any = None) -> Any:
    """Retrieve a value from a nested dict using a dot-separated key path.

    Examples::

        _get_nested_value({"model": {"width": 64}}, "model.width")  # → 64
        _get_nested_value({"lr": 0.01}, "lr")                        # → 0.01
        _get_nested_value({}, "missing")                             # → None
    """
    parts = key.split(".")
    value: Any = data
    for part in parts:
        if isinstance(value, dict) and part in value:
            value = value[part]
        else:
            return default
    return value


def _extract_fields(
    run,  # RunRecord
    config_fields: list[str],
    summary_fields: list[str],
) -> dict[str, Any]:
    """Extract named fields from a :class:`~trackpull.source.RunRecord`."""
    data: dict[str, Any] = {"run_id": run.id}
    for f in config_fields:
        data[f] = _get_nested_value(run.config, f)
    for f in summary_fields:
        data[f] = _get_nested_value(run.summary, f)
    return data


def _rows_to_scalar_arrays(
    rows: list[dict[str, Any]],
    all_fields: list[str],
) -> dict[str, np.ndarray]:
    """Convert a list of per-run dicts to a dict of per-field numpy arrays.

    - ``None`` values → ``np.nan`` for numeric fields.
    - String values (including ``run_id``) → ``object`` dtype array.
    - List values are skipped with a warning (untransformed lists cannot be
      stored as 1-D arrays).
    """
    result: dict[str, np.ndarray] = {}
    for f in all_fields:
        values = [row.get(f) for row in rows]
        # Skip fields where any value is still a list (user forgot transform)
        if any(isinstance(v, (list, tuple)) for v in values):
            logger.warning(
                "Field '%s' still contains list values after transforms — "
                "skipping.  Add a transform to your config.",
                f,
            )
            continue
        # Detect string fields
        non_none = [v for v in values if v is not None]
        if non_none and isinstance(non_none[0], str):
            result[f] = np.array(
                [v if v is not None else "" for v in values], dtype=object
            )
        else:
            result[f] = np.array(
                [v if v is not None else np.nan for v in values], dtype=float
            )
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def export(
    config: ExportConfig,
    source: RunSource,
    store: AnalysisStore,
) -> None:
    """Fetch runs from *source* and write raw data to *store*.

    Writes the ``points/`` group.  If called on a store that already contains
    a ``points/`` group, it is overwritten.

    Args:
        config: Export configuration.
        source: Run data source (e.g. :class:`~trackpull.source.WandbSource`).
        store:  Storage backend (e.g. :class:`~trackpull.store.HDF5Store`).

    Raises:
        SystemExit: If no runs are returned by the source.
    """
    run_list = list(source.fetch())
    N = len(run_list)
    logger.info("Found %d runs", N)

    if not run_list:
        logger.error("No runs found. Check source/filter settings.")
        sys.exit(1)

    all_scalar_fields = ["run_id"] + config.config_fields + config.summary_fields
    rows: list[dict[str, Any]] = []
    n_history_missing = 0

    with store.open_writer(N, config.history_fields) as writer:
        for i, run in enumerate(run_list):
            data = _extract_fields(run, config.config_fields, config.summary_fields)
            rows.append(data)

            if config.history_fields:
                history_rows = list(run.fetch_history())

                if not history_rows:
                    logger.warning(
                        "Run %s: no history data — history columns will be NaN",
                        run.id,
                    )
                    n_history_missing += 1
                else:
                    run_arrays = {
                        f: np.array(
                            [r.get(f, np.nan) for r in history_rows], dtype=float
                        )
                        for f in config.history_fields
                    }
                    del history_rows

                    if all(np.all(np.isnan(v)) for v in run_arrays.values()):
                        logger.warning(
                            "Run %s: all-NaN history — history columns will be NaN",
                            run.id,
                        )
                        n_history_missing += 1
                    else:
                        T_i = len(run_arrays[config.history_fields[0]])
                        writer.ensure_history_capacity(T_i)
                        for f in config.history_fields:
                            writer.write_history_row(f, i, run_arrays[f])

                    del run_arrays

        if not rows:
            logger.error("No valid run data extracted. Exiting.")
            sys.exit(1)

        logger.info("Extracted data from %d runs", len(rows))

        apply_transforms(rows, config.transforms)
        warn_untransformed_lists(rows, config.transforms)

        scalar_arrays = _rows_to_scalar_arrays(rows, all_scalar_fields)
        writer.write_scalars(scalar_arrays)

    if n_history_missing:
        logger.warning(
            "%d/%d run(s) had missing/all-NaN history; those rows are NaN-filled.",
            n_history_missing,
            N,
        )

    logger.info("Export complete → points/ group written")
