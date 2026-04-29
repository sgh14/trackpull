# trackpull

Export and aggregate experiment tracking runs to HDF5.

**trackpull** is a minimal, dependency-light Python library that pulls run
data from experiment tracking backends (currently W&B) into a local HDF5
file and optionally aggregates the results by any grouping key.

## Install

```bash
# Core only (no W&B, no Hydra CLI)
pip install trackpull

# With W&B support
pip install trackpull[wandb]

# With Hydra CLI
pip install trackpull[hydra,wandb]

# Everything
pip install trackpull[all]
```

## Quick start (Python API)

```python
from trackpull import (
    WandbSource, HDF5Store, ExportConfig, AggregateConfig,
    export, aggregate,
)

source = WandbSource(project="me/my-project", filters={"tags": ["v2"]})
store  = HDF5Store("results/energy.h5")

export_cfg = ExportConfig(
    config_fields=["model.width", "model.depth"],
    summary_fields=["energy", "variance"],
    history_fields=["energy_step"],
    transforms={"model.hidden_dims": "first"},
)
export(export_cfg, source, store)

agg_cfg = AggregateConfig(
    group_by=["model.width"],
    aggregations={"energy": ["mean", "std"], "energy_step": ["mean"]},
)
aggregate(agg_cfg, store)
```

## Hydra CLI

Create a YAML file:

```yaml
# conf/analysis/my_experiment.yaml
# @package _global_

source:
  project: me/my-project
  entity: null
  filters:
    tags: [v2]

export:
  config_fields: [model.width, model.depth]
  summary_fields: [energy, variance]
  history_fields: [energy_step]
  transforms:
    model.hidden_dims: first

aggregate:
  group_by: [model.width]
  aggregations:
    energy: [mean, std]
  nan_policy: warn

output: results/my_experiment.h5
```

Run:

```bash
# Full pipeline (export + aggregate)
trackpull --config-dir=conf/analysis -cn my_experiment

# Hydra multirun sweep
trackpull --config-dir=conf/analysis -cn my_experiment \
    "source.filters.tags=[v1]","source.filters.tags=[v2]" -m
```

Note: the CLI runs export first, then aggregate when an `aggregate:` section is
present in the config.

## HDF5 schema

```
results.h5
├── runs/            ← transient run cache used during export
│   └── <run_id>/
│       ├── config   (JSON string)
│       ├── summary  (JSON string)
│       └── history/...
├── points/          ← one value per run
│   ├── run_id       (N,) str
│   ├── model.width  (N,) float
│   ├── energy       (N,) float
│   └── energy_step  (N, T) float   # NaN-padded to max T
└── statistics/      ← aggregated
    ├── model.width  (G,) float
    ├── mean_energy  (G,) float
    ├── std_energy   (G,) float
    └── mean_energy_step  (G, T) float
```

## Extending trackpull

Custom backends are supported via structural subtyping — no inheritance
required. A source backend only needs a `fetch()` method that yields
`RunRecord` objects.

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
                fetch_history=lambda keys=None, r=run: iter(r.history(keys=keys)),
            )
```

## License

MIT
