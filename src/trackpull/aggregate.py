"""Aggregate step — group and aggregate exported data.

Reads the ``points/`` group written by the export step and writes the
``statistics/`` group to the same store.

Aggregation functions use ``axis=0`` so they work uniformly on both
scalar columns ``(N,)`` and history columns ``(N, T)``:

- ``(N,)`` → scalar per group
- ``(N, T)`` → curve ``(T,)`` per group

Output key naming
-----------------
- ``mean_<column>``, ``std_<column>``, ``min_<column>``, etc.
- Exception: aggregation ``"first"`` writes ``<column>`` directly, because
  it represents the group-identifying value rather than a derived statistic.

NaN policy
----------
Applies to *input* NaNs present in the ``points/`` data.  Output NaNs are
always warned about regardless of policy, because they silently break plots.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from trackpull.store import POINTS_GROUP, STATISTICS_GROUP, AnalysisStore

logger = logging.getLogger(__name__)

NAN_POLICIES = ("ignore", "warn", "raise")


# ---------------------------------------------------------------------------
# Aggregation functions
# ---------------------------------------------------------------------------


#: NaN-ignoring aggregation functions (default).
#: Using ``axis=0`` makes them polymorphic: ``(N,)→scalar``, ``(N,T)→(T,)``.
AGG_FUNCTIONS: dict[str, Any] = {
    "mean": lambda x: np.nanmean(x, axis=0),
    "std": lambda x: np.nanstd(x, axis=0),
    "min": lambda x: np.nanmin(x, axis=0),
    "max": lambda x: np.nanmax(x, axis=0),
    "median": lambda x: np.nanmedian(x, axis=0),
    "first": lambda x: x[0] if len(x) > 0 else np.nan,
}

#: Strict variants — propagate NaNs (used when ``nan_policy="raise"``).
_STRICT_AGG_FUNCTIONS: dict[str, Any] = {
    "mean": lambda x: np.mean(x, axis=0),
    "std": lambda x: np.std(x, axis=0),
    "min": lambda x: np.min(x, axis=0),
    "max": lambda x: np.max(x, axis=0),
    "median": lambda x: np.median(x, axis=0),
    "first": lambda x: x[0] if len(x) > 0 else np.nan,
}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class AggregateConfig:
    """Configuration for the aggregate step.

    Args:
        group_by:     Fields to group runs by.  Must be present in
                      :attr:`~trackpull.export.ExportConfig.config_fields`.
                      An empty list puts all runs in one group.
        aggregations: ``{column: [agg_func, ...]}`` mapping, e.g.
                      ``{"energy": ["mean", "std"]}``.  Supported functions:
                      ``mean``, ``std``, ``min``, ``max``, ``median``,
                      ``first``.
        nan_policy:   How to handle NaN values in the input data:

                      ``"ignore"``
                          Silently use NaN-ignoring functions.
                      ``"warn"``  *(default)*
                          Use NaN-ignoring functions but emit a warning.
                      ``"raise"``
                          Use strict (NaN-propagating) functions; raises
                          ``ValueError`` if NaN inputs are detected.
    """

    group_by: list[str] = field(default_factory=list)
    aggregations: dict[str, list[str]] = field(default_factory=dict)
    nan_policy: Literal["ignore", "warn", "raise"] = "warn"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def aggregate(config: AggregateConfig, store: AnalysisStore) -> None:
    """Group and aggregate data from the ``points/`` group.

    Reads ``points/`` from *store* and writes ``statistics/`` to the same
    store.

    Args:
        config: Aggregate configuration.
        store:  Storage backend that already contains a ``points/`` group
                (written by :func:`~trackpull.export.export`).

    Raises:
        ValueError: If a ``group_by`` field is not found in ``points/``, if
                    an aggregation function name is unknown, or if
                    ``nan_policy="raise"`` and NaN inputs are detected.
    """
    if config.nan_policy not in NAN_POLICIES:
        raise ValueError(
            f"Invalid nan_policy '{config.nan_policy}'. "
            f"Must be one of {NAN_POLICIES}."
        )

    available_fields = store.list_fields(POINTS_GROUP)
    logger.info("points/ fields: %s", available_fields)

    # Validate group_by fields
    for f in config.group_by:
        if f not in available_fields:
            raise ValueError(
                f"group_by field '{f}' not found in points/. "
                f"Available: {available_fields}"
            )

    # Validate aggregation columns and function names
    for column, funcs in config.aggregations.items():
        if column not in available_fields:
            logger.warning(
                "Aggregation column '%s' not in points/ — skipping", column
            )
        for func_name in funcs:
            if func_name not in AGG_FUNCTIONS:
                raise ValueError(
                    f"Unknown aggregation function '{func_name}'. "
                    f"Available: {sorted(AGG_FUNCTIONS)}"
                )

    # Build composite group index (only loads group_by columns)
    if config.group_by:
        n_runs = len(store.read_column(POINTS_GROUP, config.group_by[0]))
        group_arrays = [
            store.read_column(POINTS_GROUP, f) for f in config.group_by
        ]
        composite_keys: dict[tuple, list[int]] = {}
        for i in range(n_runs):
            key = tuple(arr[i] for arr in group_arrays)
            composite_keys.setdefault(key, []).append(i)
    else:
        n_runs = len(store.read_column(POINTS_GROUP, available_fields[0]))
        composite_keys = {(): list(range(n_runs))}

    sorted_keys = sorted(composite_keys.keys())
    G = len(sorted_keys)
    logger.info("Found %d unique group(s) for group_by=%s", G, config.group_by)

    # Choose aggregation functions according to nan_policy
    agg_fns = (
        _STRICT_AGG_FUNCTIONS
        if config.nan_policy == "raise"
        else AGG_FUNCTIONS
    )

    results: dict[str, Any] = {}

    # Store group-by columns in statistics
    for col_idx, f in enumerate(config.group_by):
        results[f] = np.array([k[col_idx] for k in sorted_keys])

    # NaN tracking
    nan_input_report: dict[str, list[tuple]] = {}
    nan_output_report: dict[str, list[tuple]] = {}

    for column, funcs in config.aggregations.items():
        if column not in available_fields:
            continue

        # Load one column at a time; release memory after aggregation
        col_data = store.read_column(POINTS_GROUP, column).astype(float)

        for func_name in funcs:
            if func_name not in agg_fns:
                continue

            # Output key: "first" uses the column name directly
            output_key = column if func_name == "first" else f"{func_name}_{column}"
            agg_fn = agg_fns[func_name]
            values = []

            for composite_key in sorted_keys:
                indices = composite_keys[composite_key]
                group_slice = col_data[indices]  # (n_group,) or (n_group, T)

                # Track input NaNs
                n_nan = int(np.isnan(group_slice).sum())
                if n_nan > 0:
                    nan_input_report.setdefault(column, []).append(
                        (composite_key, n_nan)
                    )

                value = agg_fn(group_slice)
                values.append(value)

                # Track output NaNs
                if np.any(np.isnan(value)):
                    nan_output_report.setdefault(output_key, []).append(
                        composite_key
                    )

            results[output_key] = np.array(values)

        del col_data

    # Handle input NaN report according to policy
    if nan_input_report:
        lines = []
        for col, entries in sorted(nan_input_report.items()):
            total = sum(cnt for _, cnt in entries)
            lines.append(
                f"  '{col}': {len(entries)} group(s), {total} NaN value(s)"
            )
        msg = "NaN inputs detected during aggregation:\n" + "\n".join(lines)
        if config.nan_policy == "ignore":
            pass
        elif config.nan_policy == "warn":
            logger.warning(msg)
        elif config.nan_policy == "raise":
            raise ValueError(msg)

    # Always warn about NaN outputs — these silently break downstream plots
    if nan_output_report:
        lines = [
            f"  '{k}': {len(groups)} group(s)"
            for k, groups in sorted(nan_output_report.items())
        ]
        logger.warning(
            "NaN values in aggregated statistics (will affect plots):\n%s",
            "\n".join(lines),
        )

    store.write(
        STATISTICS_GROUP,
        results,
        metadata={
            "group_by": str(config.group_by),
            "n_groups": G,
        },
    )
    logger.info("Aggregate complete → statistics/ group written (%d groups)", G)
