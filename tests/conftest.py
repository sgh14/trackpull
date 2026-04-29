"""Shared test fixtures."""

from __future__ import annotations

import pytest

from trackpull.source import RunRecord
from trackpull.store import HDF5Store

# ---------------------------------------------------------------------------
# Mock run data
# ---------------------------------------------------------------------------


def _make_runs() -> list[RunRecord]:
    return [
        RunRecord(
            id="r1",
            config={"model": {"width": 64}, "seed": 0},
            summary={"energy": -1.2, "variance": 0.05},
            fetch_history=lambda keys=None: iter(
                [
                    {"energy_step": -0.5, "_step": 0},
                    {"energy_step": -1.0, "_step": 1},
                    {"energy_step": -1.2, "_step": 2},
                ]
            ),
        ),
        RunRecord(
            id="r2",
            config={"model": {"width": 64}, "seed": 1},
            summary={"energy": -1.5, "variance": 0.04},
            fetch_history=lambda keys=None: iter(
                [
                    {"energy_step": -0.6, "_step": 0},
                    {"energy_step": -1.1, "_step": 1},
                    {"energy_step": -1.5, "_step": 2},
                ]
            ),
        ),
        RunRecord(
            id="r3",
            config={"model": {"width": 128}, "seed": 0},
            summary={"energy": -2.0, "variance": 0.02},
            fetch_history=lambda keys=None: iter(
                [
                    {"energy_step": -0.8, "_step": 0},
                    {"energy_step": -1.5, "_step": 1},
                    {"energy_step": -2.0, "_step": 2},
                ]
            ),
        ),
    ]


class _MockSource:
    def __init__(self, runs: list[RunRecord]) -> None:
        self._runs = runs

    def fetch(self):
        return iter(self._runs)


@pytest.fixture
def mock_runs() -> list[RunRecord]:
    return _make_runs()


@pytest.fixture
def mock_source(mock_runs):
    return _MockSource(mock_runs)


@pytest.fixture
def hdf5_store(tmp_path) -> HDF5Store:
    return HDF5Store(tmp_path / "results.h5")
