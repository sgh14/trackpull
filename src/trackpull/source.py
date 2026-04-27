"""Run data model and W&B source.

``WandbSource`` requires the ``[wandb]`` optional extra::

    pip install trackpull[wandb]
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator

import wandb

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
    fetch_history: Callable[[str], Iterator[dict[str, Any]]] = field(
        default_factory=lambda: lambda key: iter([])
    )


# ---------------------------------------------------------------------------
# W&B implementation
# ---------------------------------------------------------------------------


class WandbSource:
    """Fetch runs from a Weights & Biases project.

    Args:
        project: W&B project name (e.g. ``"my-project"``).
        entity:  W&B entity (username or team).  ``None`` uses the default
                 entity from the active wandb login.
        filters: W&B API filter dict passed directly to ``wandb.Api().runs()``.
                 ``None`` means no filtering (all runs are returned).

    Raises:
        ImportError: If ``wandb`` is not installed.
    """

    def __init__(
        self,
        project: str,
        entity: str | None = None,
        filters: dict[str, Any] | None = None,
    ) -> None:
        self.project = project
        self.entity = entity
        self.filters = filters

    def fetch(self) -> Iterator[RunRecord]:
        """Yield one :class:`RunRecord` per W&B run matching the filters."""
        api = wandb.Api()
        project_path = f"{self.entity}/{self.project}" if self.entity else self.project

        logger.info("Querying W&B project: %s", project_path)
        if self.filters:
            logger.info("Filters: %s", self.filters)

        runs = api.runs(project_path, filters=self.filters)
        for run in runs:
            yield self._to_record(run)

    @staticmethod
    def _to_record(run: Any) -> RunRecord:
        return RunRecord(
            id=run.id,
            config=dict(run.config),
            summary=dict(run.summary),
            # Bind run in default arg to capture by value, not by reference
            fetch_history=lambda key, r=run: iter(r.scan_history(keys=[key])),
        )
