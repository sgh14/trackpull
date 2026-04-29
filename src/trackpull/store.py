"""HDF5 storage for trackpull.

HDF5 schema::

    <name>.h5
    ├── @created              (root attribute: ISO timestamp)
    ├── runs/                 (transient — written by export, cleared after points/)
    │   └── <run_id>/
    │       ├── config        (JSON string dataset)
    │       ├── summary       (JSON string dataset)
    │       └── history/
    │           └── <field>   (JSON string dataset, list of step values)
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

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import h5py
import numpy as np

logger = logging.getLogger(__name__)

POINTS_GROUP = "points"
STATISTICS_GROUP = "statistics"
RUNS_GROUP = "runs"


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

    def write_run_cache(
        self,
        run_id: str,
        config: dict[str, Any],
        summary: dict[str, Any],
        history: dict[str, list[Any]],
    ) -> None:
        """Write one run to ``runs/<run_id>`` as JSON payloads."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(self.path, "a") as f:
            runs_grp = f.require_group(RUNS_GROUP)
            if run_id in runs_grp:
                del runs_grp[run_id]

            run_grp = runs_grp.require_group(run_id)
            run_grp.create_dataset(
                "config",
                data=json.dumps(config, default=_json_default),
                dtype=h5py.string_dtype(),
            )
            run_grp.create_dataset(
                "summary",
                data=json.dumps(summary, default=_json_default),
                dtype=h5py.string_dtype(),
            )

            hist_grp = run_grp.require_group("history")
            for field_name, values in history.items():
                hist_grp.create_dataset(
                    field_name,
                    data=json.dumps(values, default=_json_default),
                    dtype=h5py.string_dtype(),
                )

    def read_run_cache_config(self, run_id: str) -> dict[str, Any]:
        """Read cached config from ``runs/<run_id>/config``."""
        with h5py.File(self.path, "r") as f:
            return _read_json_dataset(f[RUNS_GROUP][run_id]["config"])

    def read_run_cache_configs(self, run_ids: list[str]) -> list[dict[str, Any]]:
        """Read cached configs for *run_ids* in order."""
        with h5py.File(self.path, "r") as f:
            return [
                _read_json_dataset(f[RUNS_GROUP][run_id]["config"])
                for run_id in run_ids
            ]

    def read_run_cache_summary(self, run_id: str) -> dict[str, Any]:
        """Read cached summary from ``runs/<run_id>/summary``."""
        with h5py.File(self.path, "r") as f:
            return _read_json_dataset(f[RUNS_GROUP][run_id]["summary"])

    def read_run_cache_summaries(self, run_ids: list[str]) -> list[dict[str, Any]]:
        """Read cached summaries for *run_ids* in order."""
        with h5py.File(self.path, "r") as f:
            return [
                _read_json_dataset(f[RUNS_GROUP][run_id]["summary"])
                for run_id in run_ids
            ]

    def read_run_cache_history_field(self, run_id: str, field_name: str) -> list[Any]:
        """Read a single cached history field; missing fields return an empty list."""
        with h5py.File(self.path, "r") as f:
            run_grp = f[RUNS_GROUP][run_id]
            if "history" not in run_grp or field_name not in run_grp["history"]:
                return []
            return _read_json_dataset(run_grp["history"][field_name])

    def read_run_cache_history_fields(
        self, run_ids: list[str], field_name: str
    ) -> list[list[Any]]:
        """Read one cached history field for *run_ids* in order."""
        with h5py.File(self.path, "r") as f:
            values: list[list[Any]] = []
            for run_id in run_ids:
                run_grp = f[RUNS_GROUP][run_id]
                if "history" not in run_grp or field_name not in run_grp["history"]:
                    values.append([])
                else:
                    values.append(_read_json_dataset(run_grp["history"][field_name]))

            return values


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


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def _read_json_dataset(dataset: h5py.Dataset) -> Any:
    data = dataset[()]
    if isinstance(data, bytes):
        data = data.decode("utf-8")
    return json.loads(data)
