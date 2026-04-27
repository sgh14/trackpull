"""trackpull — Export and aggregate experiment tracking runs to HDF5.

Public API::

    from trackpull import (
        RunRecord, WandbSource,
        HDF5Store,
        ExportConfig, export,
        AggregateConfig, aggregate,
    )
"""

from trackpull._version import __version__
from trackpull.aggregate import AggregateConfig, aggregate
from trackpull.export import ExportConfig, export
from trackpull.source import RunRecord, WandbSource
from trackpull.store import HDF5Store

__all__ = [
    "__version__",
    # source
    "RunRecord",
    "WandbSource",
    # store
    "HDF5Store",
    # pipeline
    "ExportConfig",
    "export",
    "AggregateConfig",
    "aggregate",
]
