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
   * - ``unwrap``
     - Unwrap a single-element list; raises on multi-element lists
   * - ``index:<n>``
     - Pick element *n* from a list (may be negative)

Example YAML::

    transforms:
      model.hidden_dims: first    # [64, 32] → 64
      model.seed: unwrap          # [42] → 42
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


def _transform_unwrap(value: Any) -> Any:
    if value is None:
        return None
    if _is_sequence(value):
        if len(value) == 1:
            return value[0]
        raise ValueError(
            f"Cannot unwrap sequence of length {len(value)}: {value!r}. "
            "Use 'first', 'last', 'index:<n>', etc. instead."
        )
    return value


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
        raise IndexError(
            f"Cannot apply index {index} to a scalar value: {value!r}"
        )

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
    "unwrap": _transform_unwrap,
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
        sorted(TRANSFORM_FUNCTIONS)
        + [f"{p}:<n>" for p in PARAMETERISED_PREFIXES]
    )
    raise ValueError(
        f"Unknown transform '{func_name}'. Available transforms: {available}"
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def apply_transforms(
    rows: list[dict[str, Any]],
    transforms: dict[str, str],
) -> None:
    """Apply named transforms to extracted run data, **in place**.

    Args:
        rows:       List of dicts returned by the export step — modified in
                    place.
        transforms: ``{field_name: transform_name}`` mapping, e.g.
                    ``{"model.hidden_dims": "first"}``.

    Raises:
        ValueError: If a transform name is not recognised, or if ``unwrap``
                    is applied to a multi-element sequence.
    """
    if not transforms:
        return

    for field_name, func_name in transforms.items():
        try:
            transform_fn = _resolve_transform(func_name)
        except ValueError as exc:
            raise ValueError(f"Field '{field_name}': {exc}") from exc

        applied = 0
        for row in rows:
            if field_name in row:
                row[field_name] = transform_fn(row[field_name])
                applied += 1

        logger.debug(
            "Applied transform '%s' to field '%s' (%d values)",
            func_name,
            field_name,
            applied,
        )


def warn_untransformed_lists(
    rows: list[dict[str, Any]],
    transforms: dict[str, str] | None = None,
) -> None:
    """Warn about list-valued fields that have no transform configured.

    Inspects the first row only; emits one warning per affected field.
    Should be called *after* :func:`apply_transforms` so already-handled
    fields are scalar.

    Args:
        rows:       Post-transform run data dicts.
        transforms: The configured transforms dict (to skip already-handled
                    fields).
    """
    if not rows:
        return
    transforms = transforms or {}
    for field_name, value in rows[0].items():
        if field_name in transforms:
            continue
        if _is_sequence(value):
            logger.warning(
                "Field '%s' contains list values but has no transform configured. "
                "This may cause errors in downstream aggregation or plotting. "
                "Add a transform in your config:\n  transforms:\n    %s: first",
                field_name,
                field_name,
            )
