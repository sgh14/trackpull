"""Field transforms — convert non-scalar config values to scalars.

Applied during the export step to ensure all values entering the numpy/HDF5/
aggregation pipeline are scalar-typed.

Transform catalogue
-------------------

.. list-table::
   :header-rows: 1

   * - Name
     - Behaviour
   * - ``first``
     - First element of a list; value unchanged if scalar
   * - ``last``
     - Last element of a list
   * - ``max``
     - Maximum of a list
   * - ``min``
     - Minimum of a list
   * - ``sum``
     - Sum of a list
   * - ``len``
     - Length of a list; ``1`` for scalars
   * - ``mean``
     - Arithmetic mean of a list
   * - ``str``
     - String representation
   * - ``index:<n>``
     - Pick element *n* from a list (may be negative)

Example YAML::

    transforms:
      model.hidden_dims: first    # [64, 32] → 64
      model.layers: index:2       # [a, b, c, d] → c

No external dependencies.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_sequence(value: Any) -> bool:
    """Return ``True`` if *value* is a list or tuple (strings excluded)."""
    return isinstance(value, (list, tuple))


# ---------------------------------------------------------------------------
# Individual transform functions
# ---------------------------------------------------------------------------


def _transform_first(value: Any) -> Any:
    if value is None:
        return None
    if _is_sequence(value):
        return value[0] if value else None
    return value


def _transform_last(value: Any) -> Any:
    if value is None:
        return None
    if _is_sequence(value):
        return value[-1] if value else None
    return value


def _transform_max(value: Any) -> Any:
    if value is None:
        return None
    return max(value) if _is_sequence(value) else value


def _transform_min(value: Any) -> Any:
    if value is None:
        return None
    return min(value) if _is_sequence(value) else value


def _transform_sum(value: Any) -> Any:
    if value is None:
        return None
    return sum(value) if _is_sequence(value) else value


def _transform_len(value: Any) -> Any:
    if value is None:
        return None
    return len(value) if _is_sequence(value) else 1


def _transform_mean(value: Any) -> Any:
    if value is None:
        return None
    if _is_sequence(value):
        return sum(value) / len(value) if value else None
    return float(value)


def _transform_str(value: Any) -> Any:
    if value is None:
        return None
    return str(value)


def _make_index_transform(index: int) -> Any:
    """Return a transform that picks element *index* from a sequence."""

    def _transform_index(value: Any) -> Any:
        if value is None:
            return None
        if _is_sequence(value):
            return value[index] if value else None
        # For scalars, only index 0 or -1 is meaningful
        if index in (0, -1):
            return value
        raise IndexError(f"Cannot apply index {index} to a scalar value: {value!r}")

    return _transform_index


# ---------------------------------------------------------------------------
# Transform registry
# ---------------------------------------------------------------------------


TRANSFORM_FUNCTIONS: dict[str, Any] = {
    "first": _transform_first,
    "last": _transform_last,
    "max": _transform_max,
    "min": _transform_min,
    "sum": _transform_sum,
    "len": _transform_len,
    "mean": _transform_mean,
    "str": _transform_str,
}

PARAMETERISED_PREFIXES: dict[str, Any] = {
    "index": lambda arg: _make_index_transform(int(arg)),
}


def _resolve_transform(func_name: str) -> Any:
    """Look up a transform by name, handling parameterised forms.

    Args:
        func_name: Transform name, e.g. ``"first"`` or ``"index:2"``.

    Returns:
        The callable transform function.

    Raises:
        ValueError: If the name is not recognised.
    """
    if func_name in TRANSFORM_FUNCTIONS:
        return TRANSFORM_FUNCTIONS[func_name]

    if ":" in func_name:
        prefix, arg = func_name.split(":", 1)
        if prefix in PARAMETERISED_PREFIXES:
            return PARAMETERISED_PREFIXES[prefix](arg)

    available = ", ".join(
        sorted(TRANSFORM_FUNCTIONS) + [f"{p}:<n>" for p in PARAMETERISED_PREFIXES]
    )
    raise ValueError(
        f"Unknown transform '{func_name}'. Available transforms: {available}"
    )
