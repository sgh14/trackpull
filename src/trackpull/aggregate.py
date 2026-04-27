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

from trackpull.store import POINTS_GROUP, STATISTICS_GROUP, HDF5Store

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


def validate_nan_policy(policy: str) -> None:
    if policy not in NAN_POLICIES:
        raise ValueError(
            f"Invalid nan_policy '{policy}'. Must be one of {NAN_POLICIES}."
        )


def validate_group_by_fields(group_by: list[str], available_fields: list[str]) -> None:
    for f in group_by:
        if f not in available_fields:
            raise ValueError(
                f"group_by field '{f}' not found in points/. "
                f"Available: {available_fields}"
            )


def validate_aggregations(
    aggregations: dict[str, list[str]], available_fields: list[str]
) -> None:
    for column, funcs in aggregations.items():
        if column not in available_fields:
            raise ValueError(
                f"Aggregation column '{column}' not found in points/. "
                f"Available: {available_fields}"
            )
        for func_name in funcs:
            if func_name not in AGG_FUNCTIONS:
                raise ValueError(
                    f"Unknown aggregation function '{func_name}'. "
                    f"Available: {sorted(AGG_FUNCTIONS)}"
                )


def get_group_indices(
    store: HDF5Store, config: AggregateConfig, available_fields: list[str]
) -> tuple[list[tuple], dict[tuple, list[int]]]:
    if config.group_by:
        n_runs = len(store.read_field(POINTS_GROUP, config.group_by[0]))
        group_arrays = [
            store.read_field(POINTS_GROUP, group_field)
            for group_field in config.group_by
        ]
        groups_indices: dict[tuple, list[int]] = {}
        for i in range(n_runs):
            key = tuple(arr[i] for arr in group_arrays)
            groups_indices.setdefault(key, []).append(i)
    else:
        n_runs = len(store.read_field(POINTS_GROUP, available_fields[0]))
        groups_indices = {(): list(range(n_runs))}

    sorted_keys = sorted(groups_indices.keys())
    logger.info(
        "Found %d unique group(s) for group_by=%s", len(sorted_keys), config.group_by
    )

    return sorted_keys, groups_indices


def aggregate_field(
    field_array: np.ndarray,
    sorted_keys: list[tuple],
    groups_indices: dict[tuple, list[int]],
    agg_func: Any,
) -> tuple[list[Any], list[tuple], list[tuple]]:
    values = []
    input_nans = []
    output_nans = []

    for group in sorted_keys:
        indices = groups_indices[group]
        field_slice = field_array[indices]  # (n_group,) or (n_group, T)
        value = agg_func(field_slice)
        values.append(value)

        # Track input NaNs
        n_nan = int(np.isnan(field_slice).sum())
        if n_nan > 0:
            input_nans.append((group, n_nan))

        # Track output NaNs
        if np.any(np.isnan(value)):
            output_nans.append(group)

    return values, input_nans, output_nans


def handle_nan_inputs(
    nan_policy: str, nan_input_report: dict[str, list[tuple]]
) -> None:
    if nan_input_report:
        lines = []
        for col, entries in sorted(nan_input_report.items()):
            total = sum(cnt for _, cnt in entries)
            lines.append(f"  '{col}': {len(entries)} group(s), {total} NaN value(s)")
        msg = "NaN inputs detected during aggregation:\n" + "\n".join(lines)
        if nan_policy == "ignore":
            pass
        elif nan_policy == "warn":
            logger.warning(msg)
        elif nan_policy == "raise":
            raise ValueError(msg)


def handle_nan_outputs(nan_output_report: dict[str, list[tuple]]) -> None:
    if nan_output_report:
        lines = [
            f"  '{k}': {len(groups)} group(s)"
            for k, groups in sorted(nan_output_report.items())
        ]
        logger.warning(
            "NaN values in aggregated statistics (will affect plots):\n%s",
            "\n".join(lines),
        )


def aggregate(config: AggregateConfig, store: HDF5Store) -> None:
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
    available_fields = store.list_fields(POINTS_GROUP)
    logger.info("points/ fields: %s", available_fields)

    validate_nan_policy(config.nan_policy)
    validate_group_by_fields(config.group_by, available_fields)
    validate_aggregations(config.aggregations, available_fields)

    # Choose aggregation functions according to nan_policy
    if config.nan_policy == "raise":
        agg_funcs = _STRICT_AGG_FUNCTIONS
    else:
        agg_funcs = AGG_FUNCTIONS

    nan_input_report: dict[str, list[tuple]] = {}
    nan_output_report: dict[str, list[tuple]] = {}

    # Build composite group index (only loads group_by columns)
    sorted_keys, groups_indices = get_group_indices(store, config, available_fields)

    store.clear_group(STATISTICS_GROUP)

    # Write group-by index columns
    for i, group_field in enumerate(config.group_by):
        store.write_field(
            STATISTICS_GROUP,
            group_field,
            np.array([key[i] for key in sorted_keys]),
        )

    for column_name, func_names in config.aggregations.items():
        field_array = store.read_field(POINTS_GROUP, column_name).astype(float)
        for func_name in func_names:
            output_key = f"{func_name}_{column_name}"
            agg_func = agg_funcs[func_name]
            values, input_nans, output_nans = aggregate_field(
                field_array,
                sorted_keys,
                groups_indices,
                agg_func,
            )
            store.write_field(STATISTICS_GROUP, output_key, np.array(values))
            nan_input_report.setdefault(column_name, []).extend(input_nans)
            nan_output_report.setdefault(output_key, []).extend(output_nans)

    handle_nan_inputs(config.nan_policy, nan_input_report)
    handle_nan_outputs(nan_output_report)
    logger.info(
        "Aggregate complete → statistics/ group written (%d groups)", len(sorted_keys)
    )
