"""trackpull — Export and aggregate experiment tracking runs to HDF5.

Public API::

    from trackpull import (
        RunRecord, RunSource, WandbSource,
        AnalysisStore, HDF5Store, InMemoryStore,
        ExportConfig, export,
        AggregateConfig, aggregate,
        apply_transforms, warn_untransformed_lists,
    )
"""

from trackpull._version import __version__
from trackpull.aggregate import AggregateConfig, aggregate
from trackpull.export import ExportConfig, export
from trackpull.source import RunRecord, RunSource, WandbSource
from trackpull.store import AnalysisStore, HDF5Store, InMemoryStore
from trackpull.transforms import apply_transforms, warn_untransformed_lists

__all__ = [
    "__version__",
    # source
    "RunRecord",
    "RunSource",
    "WandbSource",
    # store
    "AnalysisStore",
    "HDF5Store",
    "InMemoryStore",
    # pipeline
    "ExportConfig",
    "export",
    "AggregateConfig",
    "aggregate",
    # transforms
    "apply_transforms",
    "warn_untransformed_lists",
]
