"""Storage backends for trackpull.

:class:`HDF5Store`
    Production backend.  Requires ``h5py`` (included in core dependencies).

:class:`InMemoryStore`
    In-memory backend for tests and notebooks.  No disk I/O.

Both satisfy the :class:`AnalysisStore` protocol.

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
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Generator, Protocol, runtime_checkable

import h5py
import numpy as np

logger = logging.getLogger(__name__)

#: HDF5 group written by the export step.
POINTS_GROUP = "points"
#: HDF5 group written by the aggregate step.
STATISTICS_GROUP = "statistics"


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class AnalysisStore(Protocol):
    """Protocol for analysis result storage backends."""

    def write(
        self,
        group: str,
        data: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Write *data* to a named group, replacing any existing content."""
        ...

    def read(self, group: str) -> dict[str, Any]:
        """Return all data in *group* as a dict of numpy arrays."""
        ...

    def list_fields(self, group: str) -> list[str]:
        """Return top-level field names in *group* without loading data."""
        ...

    def read_column(self, group: str, field: str) -> np.ndarray:
        """Return a single *field* from *group* as a numpy array."""
        ...


# ---------------------------------------------------------------------------
# HDF5Store
# ---------------------------------------------------------------------------


class HDF5Store:
    """HDF5-backed storage.

    Wraps a single ``.h5`` file.  Parent directories are created automatically
    on the first write.

    Args:
        path: Path to the ``.h5`` file.
    """

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)

    # -- AnalysisStore protocol ------------------------------------------------

    def write(
        self,
        group: str,
        data: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(self.path, "a") as f:
            if "created" not in f.attrs:
                f.attrs["created"] = datetime.now().isoformat()
            if group in f:
                del f[group]
            grp = f.create_group(group)
            _write_dict(grp, data)
            if metadata:
                for key, value in metadata.items():
                    try:
                        grp.attrs[key] = value
                    except TypeError:
                        grp.attrs[key] = str(value)
        logger.info("Saved group '%s' → %s", group, self.path)

    def read(self, group: str) -> dict[str, Any]:
        result: dict[str, Any] = {}
        try:
            f_ctx = h5py.File(self.path, "r")
        except FileNotFoundError:
            raise KeyError(f"Group '{group}' not found (file does not exist): {self.path}")
        with f_ctx as f:
            if group not in f:
                raise KeyError(
                    f"Group '{group}' not found in {self.path}. "
                    f"Available groups: {list(f.keys())}"
                )
            _read_group(f[group], result)
        return result

    def list_fields(self, group: str) -> list[str]:
        try:
            f_ctx = h5py.File(self.path, "r")
        except FileNotFoundError:
            raise KeyError(f"Group '{group}' not found (file does not exist): {self.path}")
        with f_ctx as f:
            if group not in f:
                raise KeyError(
                    f"Group '{group}' not found in {self.path}. "
                    f"Available groups: {list(f.keys())}"
                )
            return list(f[group].keys())

    def read_column(self, group: str, field: str) -> np.ndarray:
        try:
            f_ctx = h5py.File(self.path, "r")
        except FileNotFoundError:
            raise KeyError(f"Group '{group}' not found (file does not exist): {self.path}")
        with f_ctx as f:
            if group not in f:
                raise KeyError(
                    f"Group '{group}' not found in {self.path}."
                )
            if field not in f[group]:
                raise KeyError(
                    f"Field '{field}' not found in group '{group}' of {self.path}."
                )
            return np.asarray(f[group][field])

    # -- Streaming writer ------------------------------------------------------

    @contextmanager
    def open_writer(
        self,
        N: int,
        history_fields: list[str],
        chunk_cols: int = 1000,
    ) -> Generator[_PointsWriter, None, None]:
        """Context manager for streaming writes to the ``points/`` group.

        Opens the file fresh (any existing content is discarded), creates
        resizable history datasets pre-filled with NaN, and yields a
        :class:`_PointsWriter`.  Datasets are expanded on demand as runs of
        varying history length arrive.

        Args:
            N:              Total number of runs (row dimension).
            history_fields: Names of history fields to pre-allocate.
            chunk_cols:     HDF5 chunk size along the time axis.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(self.path, "w") as hf:
            hf.attrs["created"] = datetime.now().isoformat()
            grp = hf.create_group(POINTS_GROUP)
            history_datasets: dict[str, h5py.Dataset] = {}
            for field in history_fields:
                history_datasets[field] = grp.create_dataset(
                    field,
                    shape=(N, 0),
                    maxshape=(N, None),
                    dtype=float,
                    fillvalue=np.nan,
                    chunks=(1, chunk_cols),
                )
            yield _PointsWriter(grp, history_datasets)


class _PointsWriter:
    """Stateful writer for the open ``points/`` group.

    Obtained via :meth:`HDF5Store.open_writer`.  Not intended for direct use.
    """

    def __init__(
        self,
        grp: h5py.Group,
        history_datasets: dict[str, h5py.Dataset],
    ) -> None:
        self._grp = grp
        self._datasets = history_datasets
        self._max_T = 0

    def ensure_history_capacity(self, T: int) -> None:
        """Expand all history datasets to at least *T* columns."""
        if T > self._max_T:
            N = self._datasets[next(iter(self._datasets))].shape[0]
            for ds in self._datasets.values():
                ds.resize((N, T))
            self._max_T = T

    def write_history_row(self, field: str, i: int, arr: np.ndarray) -> None:
        """Write *arr* into row *i* of the *field* history dataset."""
        self._datasets[field][i, : len(arr)] = arr

    def write_scalars(
        self,
        data: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Write scalar/array datasets and optional group-level attributes."""
        _write_dict(self._grp, data)
        if metadata:
            for key, value in metadata.items():
                try:
                    self._grp.attrs[key] = value
                except TypeError:
                    self._grp.attrs[key] = str(value)


# ---------------------------------------------------------------------------
# InMemoryStore
# ---------------------------------------------------------------------------


class InMemoryStore:
    """In-memory storage backend.

    Groups are stored as nested dicts.  No disk I/O.  Useful in tests and
    notebooks.  Provides :meth:`open_writer` for API compatibility with
    :class:`HDF5Store`.
    """

    def __init__(self) -> None:
        self._groups: dict[str, dict[str, Any]] = {}
        self._metadata: dict[str, dict[str, Any]] = {}

    def write(
        self,
        group: str,
        data: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._groups[group] = {
            k: np.asarray(v) if not isinstance(v, str) else v
            for k, v in data.items()
        }
        self._metadata[group] = metadata or {}

    def read(self, group: str) -> dict[str, Any]:
        if group not in self._groups:
            raise KeyError(
                f"Group '{group}' not found. Available: {list(self._groups)}"
            )
        return dict(self._groups[group])

    def list_fields(self, group: str) -> list[str]:
        if group not in self._groups:
            raise KeyError(
                f"Group '{group}' not found. Available: {list(self._groups)}"
            )
        return list(self._groups[group].keys())

    def read_column(self, group: str, field: str) -> np.ndarray:
        if group not in self._groups:
            raise KeyError(
                f"Group '{group}' not found. Available: {list(self._groups)}"
            )
        if field not in self._groups[group]:
            raise KeyError(
                f"Field '{field}' not found in group '{group}'."
            )
        return np.asarray(self._groups[group][field])

    @contextmanager
    def open_writer(
        self,
        N: int,
        history_fields: list[str],
        chunk_cols: int = 1000,
    ) -> Generator[_InMemoryWriter, None, None]:
        """Context manager matching :meth:`HDF5Store.open_writer`."""
        writer = _InMemoryWriter(N, history_fields)
        yield writer
        self._groups[POINTS_GROUP] = writer.finalise()


class _InMemoryWriter:
    """Writer counterpart to :class:`_PointsWriter` for :class:`InMemoryStore`."""

    def __init__(self, N: int, history_fields: list[str]) -> None:
        self._N = N
        self._history_fields = history_fields
        self._scalars: dict[str, Any] = {}
        self._history: dict[str, list[np.ndarray | None]] = {
            f: [None] * N for f in history_fields
        }
        self._max_T = 0

    def ensure_history_capacity(self, T: int) -> None:
        if T > self._max_T:
            self._max_T = T

    def write_history_row(self, field: str, i: int, arr: np.ndarray) -> None:
        self._history[field][i] = np.asarray(arr, dtype=float)

    def write_scalars(
        self,
        data: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._scalars.update(data)

    def finalise(self) -> dict[str, Any]:
        """Return a dict merging scalars and NaN-padded history arrays."""
        result = dict(self._scalars)
        T = self._max_T
        for field, rows in self._history.items():
            if T > 0:
                mat = np.full((self._N, T), np.nan, dtype=float)
                for i, row in enumerate(rows):
                    if row is not None:
                        mat[i, : len(row)] = row
            else:
                mat = np.full((self._N,), np.nan, dtype=float)
            result[field] = mat
        return result


# ---------------------------------------------------------------------------
# Internal HDF5 helpers
# ---------------------------------------------------------------------------


def _write_dict(group: h5py.Group, data: dict[str, Any]) -> None:
    """Recursively write a nested dict into an HDF5 group."""
    for key, value in data.items():
        if isinstance(value, dict):
            sub = group.create_group(key)
            _write_dict(sub, value)
        elif isinstance(value, np.ndarray):
            if value.dtype.kind in ("U", "S", "O"):
                dt = h5py.string_dtype()
                group.create_dataset(key, data=[str(v) for v in value.flat], dtype=dt)
            else:
                group.create_dataset(key, data=value)
        elif isinstance(value, (list, tuple)):
            arr = np.asarray(value)
            if arr.dtype.kind in ("U", "S", "O"):
                dt = h5py.string_dtype()
                group.create_dataset(key, data=[str(v) for v in arr.flat], dtype=dt)
            else:
                group.create_dataset(key, data=arr)
        elif isinstance(value, (int, float, np.integer, np.floating)):
            group.create_dataset(key, data=value)
        elif isinstance(value, str):
            group.create_dataset(key, data=value, dtype=h5py.string_dtype())
        else:
            group.create_dataset(key, data=str(value), dtype=h5py.string_dtype())


def _read_group(group: h5py.Group, target: dict[str, Any]) -> None:
    """Recursively read an HDF5 group into a dict."""
    for key in group.keys():
        item = group[key]
        if isinstance(item, h5py.Group):
            target[key] = {}
            _read_group(item, target[key])
        else:
            arr = np.asarray(item)
            if arr.dtype.kind in ("S", "O"):
                arr = arr.astype(str)
            target[key] = arr
