# Contributing to trackpull

## Development Environment

### Option A: Use the Devcontainer (recommended)

```bash
cp .devcontainer/.env.example .devcontainer/.env
# Edit .env with your credentials
# Then open in VS Code and click "Reopen in Container"
```

### Option B: Standalone Setup

```bash
git clone git@github.com:sgh14/trackpull.git
cd trackpull
pip install uv
uv venv .venv && source .venv/bin/activate
uv pip install -e ".[dev]"
pre-commit install
```

---

## Workflow

1. **Create a feature branch:**
   ```bash
   git checkout -b feature/my-change
   ```

2. **Develop and test:**
   ```bash
   pytest tests/ -v
   ruff check . --fix && ruff format .
   ```

3. **Commit and push** — pre-commit hooks run automatically.

4. **Open a PR** targeting `main`.

---

## Code Quality

| Layer | Tool | When |
|-------|------|------|
| Editor | Ruff (VS Code extension) | Format on save |
| Pre-commit | Ruff + standard hooks | Blocks bad commits locally |
| CI | GitHub Actions | Blocks bad PRs |

**Ruff config:** line length 88, rules `E`, `F`, `I`, `W`.

If CI fails:
```bash
ruff check . --fix
ruff format .
git add -u && git commit --amend
```

---

## Tests

```bash
pytest tests/ -v                  # All tests
pytest tests/unit/ -v             # Unit tests only
pytest tests/ -k "store" -v       # Filter by name
pytest tests/ --cov=trackpull     # With coverage
```

### Test structure

```
tests/
├── conftest.py                   # Shared fixtures (MockSource, InMemoryStore, HDF5Store)
├── unit/
│   ├── test_transforms.py
│   ├── test_aggregate_fns.py
│   └── test_store.py
└── integration/
    └── test_pipeline.py          # Full export → aggregate pipeline
```

---

## Type Annotations

Always annotate function parameters and return types. Skip local variables.

```python
# ✅ Good
def export(config: ExportConfig, source: RunSource, store: AnalysisStore) -> None:
    rows = _extract_fields(run, ...)  # no annotation needed for locals

# ❌ Bad
def export(config, source, store):
    ...
```

---

## Adding a New Source Backend

Implement the `RunSource` protocol — no inheritance needed:

```python
from typing import Iterator
from trackpull import RunRecord

class MySource:
    def fetch(self) -> Iterator[RunRecord]:
        for run in my_backend.list_runs():
            yield RunRecord(
                id=run.id,
                config=run.config,
                summary=run.metrics,
            )
```

## Adding a New Store Backend

Implement the `AnalysisStore` protocol:

```python
from trackpull.store import AnalysisStore  # for reference only — no inheritance

class MyStore:
    def write(self, group, data, metadata=None): ...
    def read(self, group): ...
    def list_fields(self, group): ...
    def read_column(self, group, field): ...
    def open_writer(self, n_runs, history_fields): ...
```
