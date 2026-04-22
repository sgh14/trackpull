"""Run data model and source protocol.

``WandbSource`` requires the ``[wandb]`` optional extra::

    pip install trackpull[wandb]
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class RunRecord:
    """Normalised representation of a single experiment run.

    ``config`` and ``summary`` are flat or nested dicts.  Nested keys are
    accessed via dot-separated strings in :class:`~trackpull.export.ExportConfig`
    (e.g. ``"model.width"``).

    ``fetch_history`` is a zero-argument callable that returns an iterator of
    per-step dicts, e.g. ``[{"loss": 0.5, "_step": 0}, ...]``.  It is called
    at most once per run and only when
    :attr:`~trackpull.export.ExportConfig.history_fields` is non-empty.
    """

    id: str
    config: dict[str, Any]
    summary: dict[str, Any]
    fetch_history: Callable[[], Iterator[dict[str, Any]]] = field(
        default_factory=lambda: (lambda: iter([]))
    )


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class RunSource(Protocol):
    """Protocol for experiment run data sources.

    Any class with a ``fetch()`` method that yields :class:`RunRecord` objects
    satisfies this protocol — no inheritance required.
    """

    def fetch(self) -> Iterator[RunRecord]:
        """Yield :class:`RunRecord` objects one at a time."""
        ...


# ---------------------------------------------------------------------------
# W&B implementation
# ---------------------------------------------------------------------------


class WandbSource:
    """Fetch runs from a Weights & Biases project.

    Satisfies :class:`RunSource`.

    Args:
        project: W&B project name (e.g. ``"my-project"``).
        entity:  W&B entity (username or team).  ``None`` uses the default
                 entity from the active wandb login.
        filters: W&B API filter dict.  Supported keys:

                 ``tags``
                     List of tag strings.  W&B applies AND semantics.

                 ``None`` means no filtering (all runs are returned).
                 Keys whose value is ``None`` are silently dropped so that
                 YAML ``null`` values do not accidentally filter runs.

    Raises:
        ImportError: If ``wandb`` is not installed.
    """

    def __init__(
        self,
        project: str,
        entity: str | None = None,
        filters: dict[str, Any] | None = None,
    ) -> None:
        try:
            import wandb  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "WandbSource requires wandb.  Install it with:\n"
                "  pip install trackpull[wandb]"
            ) from exc
        self.project = project
        self.entity = entity
        self.filters = filters

    def fetch(self) -> Iterator[RunRecord]:
        """Yield one :class:`RunRecord` per W&B run matching the filters."""
        import wandb

        api = wandb.Api()
        project_path = (
            f"{self.entity}/{self.project}" if self.entity else self.project
        )

        # Strip None values — YAML nulls must not be forwarded to the W&B API
        filters = self.filters or {}
        filters = {k: v for k, v in filters.items() if v is not None} or None

        logger.info("Querying W&B project: %s", project_path)
        if filters:
            logger.info("Filters: %s", filters)

        runs = api.runs(project_path, filters=filters)
        for run in runs:
            yield self._to_record(run)

    @staticmethod
    def _to_record(run: Any) -> RunRecord:
        return RunRecord(
            id=run.id,
            config=dict(run.config),
            summary=dict(run.summary),
            # Bind run in default arg to capture by value, not by reference
            fetch_history=lambda r=run: iter(r.scan_history()),
        )
