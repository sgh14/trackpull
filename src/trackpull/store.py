"""HDF5 storage for trackpull.

HDF5 schema::

    <name>.h5
    ├── @created              (root attribute: ISO timestamp)
    ├── points/               (written by export step)
    │   ├── run_id            (string array, shape (N,))
    │   ├── <config_field>    (float or string array, shape (N,))
    │   ├── <summary_field>   (float array, shape (N,))
    │   └── <history_field>   (float array, shape (N, T), NaN-padded)
    └── statistics/           (written by aggregate step)
        ├── <group_by_field>  (unique group values, shape (G,))
        ├── mean_<column>     (shape (G,) or (G, T))
        ├── std_<column>      (shape (G,) or (G, T))
        └── ...
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import h5py
import numpy as np

logger = logging.getLogger(__name__)

POINTS_GROUP = "points"
STATISTICS_GROUP = "statistics"


class HDF5Store:
    """Wraps a single ``.h5`` file.

    Args:
        path: Path to the ``.h5`` file.  Parent directories are created
              automatically on the first write.
    """

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)

    def clear_group(self, group: str) -> None:
        """Delete *group* if it exists, so subsequent writes start fresh."""
        if self.path.exists():
            with h5py.File(self.path, "a") as f:
                if group in f:
                    del f[group]

    def write_field(self, group: str, name: str, data: Any) -> None:
        """Write a single dataset *name* into *group*, creating the group if needed."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(self.path, "a") as f:
            if "created" not in f.attrs:
                f.attrs["created"] = datetime.now().isoformat()

            grp = f.require_group(group)
            if name in grp:
                del grp[name]

            _write_value(grp, name, data)

    def list_fields(self, group: str) -> list[str]:
        """Return field names in *group* without loading data."""
        with h5py.File(self.path, "r") as f:
            return list(f[group].keys())

    def read_field(self, group: str, field: str) -> np.ndarray:
        """Return a single *field* from *group* as a numpy array."""
        with h5py.File(self.path, "r") as f:
            arr = np.asarray(f[group][field])
            if arr.dtype.kind in ("S", "O"):
                arr = arr.astype(str)

            return arr


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _write_value(group: h5py.Group, key: str, value: Any) -> None:
    """Write a single key/value pair into an HDF5 group."""
    if isinstance(value, (np.ndarray, list, tuple)):
        arr = np.asarray(value)
        if arr.dtype.kind in ("U", "S", "O"):
            group.create_dataset(
                key, data=[str(v) for v in arr.flat], dtype=h5py.string_dtype()
            )
        else:
            group.create_dataset(key, data=arr)
    elif isinstance(value, (int, float, np.integer, np.floating)):
        group.create_dataset(key, data=value)
    elif isinstance(value, str):
        group.create_dataset(key, data=value, dtype=h5py.string_dtype())
    else:
        group.create_dataset(key, data=str(value), dtype=h5py.string_dtype())
