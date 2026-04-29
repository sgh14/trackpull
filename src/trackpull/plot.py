"""Core plot logic for trackplot.

Organised in four layers — each function has exactly one responsibility:

  1. Config dataclasses  — typed containers, no logic
  2. Data helpers        — pure functions, no matplotlib
  3. Drawing helpers     — each touches ``ax`` for exactly one purpose
  4. Orchestrators       — call helpers only, no inline logic

Plot types
----------
timeseries
    History curves from ``statistics/``.  X axis is a history field (or the
    implicit step index when omitted).  One curve per ``color_by`` group.

trend
    Scalar x vs mean ± std from ``statistics/``.  One sorted line per
    ``color_by`` group.

Both types read exclusively from the ``statistics/`` HDF5 group (pre-computed
aggregates).  Field naming convention inherited from the aggregate step:

- Group keys are stored as bare names, e.g. ``model.hidden_dims``
- Metrics are stored as ``mean_<col>`` and ``std_<col>``
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np

from trackpull.store import STATISTICS_GROUP, HDF5Store

# ---------------------------------------------------------------------------
# 1. Config dataclasses
# ---------------------------------------------------------------------------


@dataclass
class InputConfig:
    path: str
    label: str | None = None
    label_template: str | None = None


@dataclass
class SelectConfig:
    by: str
    criterion: Literal["min", "max"] = "min"


@dataclass
class AxisConfig:
    field: str | None = None
    label: str = ""
    scale: str = "linear"
    log_base: int = 10
    ticks: str = "auto"
    tick_rotation: int = 0
    clip_to_shortest: bool = False


@dataclass
class LineConfig:
    linewidth: float = 1.5
    marker: str | None = None


@dataclass
class BandConfig:
    enabled: bool = True
    alpha: float = 0.2


@dataclass
class ReferenceLineConfig:
    field: str
    label: str = ""


@dataclass
class TimeseriesPlotConfig:
    output_name: str
    color_by: str | list[str] = "label"
    x_axis: AxisConfig | None = None
    y_axis: AxisConfig = field(default_factory=AxisConfig)
    line: LineConfig = field(default_factory=LineConfig)
    band: BandConfig = field(default_factory=BandConfig)


@dataclass
class TrendPlotConfig:
    output_name: str
    color_by: str | list[str] = "label"
    x_axis: AxisConfig = field(default_factory=AxisConfig)
    y_axis: AxisConfig = field(default_factory=AxisConfig)
    line: LineConfig = field(default_factory=LineConfig)
    band: BandConfig = field(default_factory=BandConfig)
    reference_lines: list[ReferenceLineConfig] = field(default_factory=list)


@dataclass
class MasterPlotConfig:
    inputs: list[InputConfig]
    filter: dict = field(default_factory=dict)
    select: SelectConfig | None = None
    figure_style: str | None = None
    output_dir: str = "."
    output_formats: list[str] = field(default_factory=lambda: ["pdf"])


# ---------------------------------------------------------------------------
# 2. Data helpers — pure functions, no matplotlib
# ---------------------------------------------------------------------------


def read_fields(store: HDF5Store, field_names: list[str]) -> dict[str, np.ndarray]:
    """Batch-read named fields from the statistics/ group."""
    return {name: store.read_field(STATISTICS_GROUP, name) for name in field_names}


def filter_mask(fields: dict[str, np.ndarray], filter_dict: dict) -> np.ndarray:
    """Boolean mask: AND of per-key equality (scalar) or containment (list)."""
    n = len(next(iter(fields.values())))
    mask = np.ones(n, dtype=bool)
    for key, value in filter_dict.items():
        col = fields[key]
        if isinstance(value, (list, tuple)):
            mask &= np.isin(col, value)
        else:
            mask &= col == value
    return mask


def select_mask(fields: dict[str, np.ndarray], select_cfg: SelectConfig) -> np.ndarray:
    """Boolean mask with a single True at the argmin or argmax of select_cfg.by."""
    values = fields[select_cfg.by]
    if len(values) == 0:
        return np.zeros(0, dtype=bool)
    fn = np.argmin if select_cfg.criterion == "min" else np.argmax
    idx = int(fn(values))
    mask = np.zeros(len(values), dtype=bool)
    mask[idx] = True
    return mask


def combined_mask(
    fields: dict[str, np.ndarray],
    filter_dict: dict,
    select_cfg: SelectConfig | None,
) -> np.ndarray:
    """AND of filter_mask with an optional select_mask applied to filtered rows."""
    mask = filter_mask(fields, filter_dict)
    if select_cfg is None:
        return mask
    filtered = {k: v[mask] for k, v in fields.items()}
    inner = select_mask(filtered, select_cfg)
    indices = np.where(mask)[0][inner]
    full = np.zeros(len(mask), dtype=bool)
    full[indices] = True
    return full


def group_indices(values: np.ndarray) -> dict[Any, np.ndarray]:
    """Map each unique value to its row indices."""
    groups: dict[Any, list[int]] = {}
    for i, v in enumerate(values):
        key = v.item() if hasattr(v, "item") else v
        groups.setdefault(key, []).append(i)
    return {k: np.array(v) for k, v in groups.items()}


def resolve_label(
    input_cfg: InputConfig, fields: dict[str, np.ndarray], row: int
) -> str:
    """Return the display label for one row of one input file."""
    if input_cfg.label_template:
        return _expand_template(input_cfg.label_template, fields, row)
    return input_cfg.label or Path(input_cfg.path).stem


def trim_to_valid(arr: np.ndarray) -> np.ndarray:
    """Strip trailing NaNs from a 1-D curve."""
    valid = ~np.isnan(arr)
    last = int(np.flatnonzero(valid)[-1]) + 1 if valid.any() else 0
    return arr[:last]


def clip_curves(curves: list[np.ndarray]) -> list[np.ndarray]:
    """Truncate all curves to the length of the shortest one."""
    if not curves:
        return curves
    min_len = min(len(c) for c in curves)
    return [c[:min_len] for c in curves]


def _flatten_dict(d: dict, prefix: str = "") -> dict:
    """Flatten a nested dict to a dotted-key flat dict.

    Handles Hydra's conversion of dotted YAML keys to nested dicts:
    ``{"model": {"width": 64}}`` → ``{"model.width": 64}``.
    """
    result = {}
    for k, v in d.items():
        full_key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            result.update(_flatten_dict(v, full_key))
        else:
            result[full_key] = v
    return result


def _template_fields(template: str) -> list[str]:
    """Extract field names referenced as ``$field.name`` in a label template."""
    return re.findall(r"\$([\w.]+)", template)


def _expand_template(template: str, fields: dict[str, np.ndarray], row: int) -> str:
    """Replace ``$field.name`` occurrences with the field value at row."""

    def _sub(match: re.Match) -> str:
        key = match.group(1)
        return str(fields[key][row]) if key in fields else match.group(0)

    return re.sub(r"\$([\w.]+)", _sub, template)


# ---------------------------------------------------------------------------
# 3. Drawing helpers — each operates on ax for exactly one purpose
# ---------------------------------------------------------------------------


def apply_style(style_path: str | None) -> None:
    """Load a matplotlib style file, or do nothing if not provided."""
    if style_path:
        plt.style.use(style_path)


def apply_scale(
    ax: plt.Axes, axis: str, axis_cfg: AxisConfig, data: np.ndarray
) -> None:
    """Set axis scale: linear, log, or auto-detected from data range."""
    scale = axis_cfg.scale if axis_cfg.scale != "auto" else _detect_scale(data)
    setter = ax.set_xscale if axis == "x" else ax.set_yscale
    if scale == "log":
        setter("log", base=axis_cfg.log_base, nonpositive="clip")
    else:
        setter("linear")


def apply_ticks(ax: plt.Axes, axis: str, axis_cfg: AxisConfig) -> None:
    """Apply tick label rotation when configured."""
    if axis_cfg.tick_rotation == 0:
        return
    ax.tick_params(axis=axis, labelrotation=axis_cfg.tick_rotation)


def save_figure(
    fig: plt.Figure, output_dir: str, output_name: str, formats: list[str]
) -> None:
    """Save the figure to each requested format, then close it."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(out / f"{output_name}.{fmt}", bbox_inches="tight")
    plt.close(fig)


def draw_line(
    ax: plt.Axes, x: np.ndarray, y: np.ndarray, label: str, line_cfg: LineConfig
) -> None:
    """Plot one line onto ax."""
    ax.plot(x, y, label=label, linewidth=line_cfg.linewidth, marker=line_cfg.marker)


def draw_band(
    ax: plt.Axes,
    x: np.ndarray,
    y_mean: np.ndarray,
    y_std: np.ndarray,
    band_cfg: BandConfig,
) -> None:
    """Draw a ± std confidence band onto ax."""
    if not band_cfg.enabled:
        return
    ax.fill_between(x, y_mean - y_std, y_mean + y_std, alpha=band_cfg.alpha)


def draw_hlines(
    ax: plt.Axes,
    fields: dict[str, np.ndarray],
    ref_cfgs: list[ReferenceLineConfig],
) -> None:
    """Add a horizontal reference line for each ReferenceLineConfig."""
    for ref in ref_cfgs:
        value = float(fields[ref.field][0])
        ax.axhline(value, linestyle="--", label=ref.label or None)


def _finalize_axes(
    ax: plt.Axes,
    x_cfg: AxisConfig | None,
    y_cfg: AxisConfig,
    x_data: np.ndarray,
    y_data: np.ndarray,
) -> None:
    """Set axis labels, scales, and tick rotation."""
    if x_cfg is not None:
        ax.set_xlabel(x_cfg.label)
        apply_scale(ax, "x", x_cfg, x_data)
        apply_ticks(ax, "x", x_cfg)
    ax.set_ylabel(y_cfg.label)
    apply_scale(ax, "y", y_cfg, y_data)
    apply_ticks(ax, "y", y_cfg)
    # fill_between may introduce negative axis limits before log scale is set;
    # force ylim bottom to a positive floor so the log locator can function.
    y_scale = y_cfg.scale if y_cfg.scale != "auto" else _detect_scale(y_data)
    if y_scale == "log":
        ylim = ax.get_ylim()
        if ylim[0] <= 0:
            pos = y_data[y_data > 0] if y_data.size else np.array([])
            floor = pos.min() * 0.5 if pos.size else 1e-10
            ax.set_ylim(bottom=floor)


def _detect_scale(data: np.ndarray) -> str:
    """Return 'log' if data is all-positive and spans more than 2 decades."""
    finite = data[np.isfinite(data)]
    if finite.size == 0 or finite.min() <= 0:
        return "linear"
    return "log" if finite.max() / finite.min() > 100 else "linear"


# ---------------------------------------------------------------------------
# 4. Internal iterators — shared data-loading logic for orchestrators
# ---------------------------------------------------------------------------


def _color_values(
    inp: InputConfig,
    fields: dict[str, np.ndarray],
    color_by: str | list[str],
    n: int,
) -> np.ndarray:
    """Return the per-row color group key: input label or a field array."""
    color_fields = _color_by_fields(color_by)
    if color_fields:
        if len(color_fields) == 1:
            return fields[color_fields[0]]

        values: list[tuple[Any, ...]] = []
        for i in range(n):
            values.append(
                tuple(
                    fields[name][i].item()
                    if hasattr(fields[name][i], "item")
                    else fields[name][i]
                    for name in color_fields
                )
            )
        keys = np.empty(n, dtype=object)
        keys[:] = values
        return keys

    label = (
        resolve_label(inp, fields, row=0)
        if n > 0
        else (inp.label or Path(inp.path).stem)
    )
    return np.array([label] * n)


def _color_by_fields(color_by: str | list[str]) -> list[str]:
    """Return normalized color-by field names; empty means use input label."""
    if isinstance(color_by, str):
        return [] if color_by == "label" else [color_by]

    return [name for name in color_by if name != "label"]


_INTERP_GRID_N = 1000  # resolution of the common time grid used when averaging runs


def _interp_group_mean(
    y_2d: np.ndarray,
    std_2d: np.ndarray,
    x_2d: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Average N runs by interpolating onto a shared x grid.

    Runs may have different step counts (and therefore non-aligned x values).
    The common grid spans the intersection of all runs' x ranges so every run
    contributes at every grid point and no extrapolation is needed.

    Returns (grid_x, mean_y, mean_std) of length ``_INTERP_GRID_N``.
    """
    rows: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for i in range(y_2d.shape[0]):
        valid = ~np.isnan(x_2d[i]) & ~np.isnan(y_2d[i])
        if valid.sum() < 2:
            continue
        rows.append((x_2d[i][valid], y_2d[i][valid], std_2d[i][valid]))

    if not rows:
        empty: np.ndarray = np.array([])
        return empty, empty, empty

    # Intersection: from the latest start to the earliest end
    grid_start = max(r[0][0] for r in rows)
    grid_end = min(r[0][-1] for r in rows)
    if grid_start >= grid_end:  # no overlap — fall back to full union range
        grid_start = min(r[0][0] for r in rows)
        grid_end = max(r[0][-1] for r in rows)

    grid = np.linspace(grid_start, grid_end, _INTERP_GRID_N)
    y_interp = np.array([np.interp(grid, rx, ry) for rx, ry, _ in rows])
    s_interp = np.array([np.interp(grid, rx, rs) for rx, _, rs in rows])
    return grid, np.mean(y_interp, axis=0), np.mean(s_interp, axis=0)


def _iter_timeseries_groups(
    master_cfg: MasterPlotConfig,
    y_field: str,
    x_field: str | None,
    color_by: str,
):
    """Yield (label, x, y_mean, y_std) per color group across all input files.

    When multiple runs share a color-group key and an x_field is given, all
    runs are interpolated onto a common x grid so that the average is computed
    at the same x value for every run.  This avoids artefacts from runs with
    different step counts (different x spacing but the same total x range).
    """
    for inp in master_cfg.inputs:
        store = HDF5Store(inp.path)
        needed = _timeseries_needed(y_field, x_field, color_by, master_cfg, inp)
        fields = read_fields(store, needed)
        mask = combined_mask(fields, master_cfg.filter, master_cfg.select)
        f = {k: v[mask] for k, v in fields.items()}
        n = int(mask.sum())
        for key, idxs in group_indices(_color_values(inp, f, color_by, n)).items():
            y_raw = f[f"mean_{y_field}"][idxs]  # (N, T) or (N,)
            s_raw = f[f"std_{y_field}"][idxs]

            if y_raw.ndim == 2 and x_field is not None and len(idxs) > 1:
                # Multiple runs with a real x axis — interpolate onto common grid
                x_arr, y_mean, y_std = _interp_group_mean(
                    y_raw, s_raw, f[f"mean_{x_field}"][idxs]
                )
            elif y_raw.ndim == 2:
                # Single run, or no x_field — nanmean collapses to the run itself
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore", "Mean of empty slice", RuntimeWarning
                    )
                    y_mean = np.nanmean(y_raw, axis=0)
                    y_std = np.nanmean(s_raw, axis=0)
                x_arr = (
                    np.nanmean(f[f"mean_{x_field}"][idxs], axis=0)
                    if x_field is not None
                    else np.arange(y_mean.shape[0])
                )
            else:
                y_mean = y_raw.flatten()
                y_std = s_raw.flatten()
                x_arr = (
                    f[f"mean_{x_field}"][idxs].flatten()
                    if x_field is not None
                    else np.arange(len(y_mean))
                )

            yield str(key), x_arr, y_mean, y_std


def _final_value(arr: np.ndarray) -> np.ndarray:
    """Return per-row last non-NaN value.  Works for both 1-D and 2-D arrays.

    If *arr* is 1-D (scalar field) it is returned unchanged.
    If *arr* is 2-D (timeseries field, shape ``(N, T)``), the last non-NaN
    element in each row is extracted and returned as shape ``(N,)``.
    """
    if arr.ndim == 1:
        return arr
    result = np.full(arr.shape[0], np.nan)
    for i, row in enumerate(arr):
        valid = row[~np.isnan(row)]
        if valid.size:
            result[i] = valid[-1]
    return result


def _iter_trend_groups(
    master_cfg: MasterPlotConfig,
    y_field: str,
    x_field: str,
    color_by: str,
    ref_fields: list[str],
):
    """Yield (label, x, y_mean, y_std, fields) per color group across all inputs.

    x, y_mean, y_std are 1-D scalar arrays — one entry per row in the group.
    If the y field is a 2-D timeseries, the last valid timestep is used.
    """
    for inp in master_cfg.inputs:
        store = HDF5Store(inp.path)
        needed = _trend_needed(y_field, x_field, color_by, master_cfg, inp, ref_fields)
        fields = read_fields(store, needed)
        mask = combined_mask(fields, master_cfg.filter, master_cfg.select)
        f = {k: v[mask] for k, v in fields.items()}
        n = int(mask.sum())
        for key, idxs in group_indices(_color_values(inp, f, color_by, n)).items():
            x_arr = f[x_field][idxs]
            y_mean = _final_value(f[f"mean_{y_field}"][idxs])
            y_std = _final_value(f[f"std_{y_field}"][idxs])
            yield str(key), x_arr, y_mean, y_std, f


def _timeseries_needed(
    y_field: str,
    x_field: str | None,
    color_by: str | list[str],
    master_cfg: MasterPlotConfig,
    inp: InputConfig,
) -> list[str]:
    """Collect statistics/ field names needed for a timeseries group iteration."""
    names: list[str] = [f"mean_{y_field}", f"std_{y_field}"]
    if x_field:
        names.append(f"mean_{x_field}")
    names.extend(_color_by_fields(color_by))
    names.extend(master_cfg.filter.keys())
    if master_cfg.select:
        names.append(master_cfg.select.by)
    if inp.label_template:
        names.extend(_template_fields(inp.label_template))
    return list(dict.fromkeys(names))


def _trend_needed(
    y_field: str,
    x_field: str,
    color_by: str | list[str],
    master_cfg: MasterPlotConfig,
    inp: InputConfig,
    ref_fields: list[str],
) -> list[str]:
    """Collect statistics/ field names needed for a trend group iteration."""
    names: list[str] = [x_field, f"mean_{y_field}", f"std_{y_field}"]
    names.extend(_color_by_fields(color_by))
    names.extend(master_cfg.filter.keys())
    if master_cfg.select:
        names.append(master_cfg.select.by)
    if inp.label_template:
        names.extend(_template_fields(inp.label_template))
    names.extend(ref_fields)
    return list(dict.fromkeys(names))


# ---------------------------------------------------------------------------
# 5. Orchestrators — call helpers only, no inline logic
# ---------------------------------------------------------------------------


def plot_timeseries(
    master_cfg: MasterPlotConfig, plot_cfg: TimeseriesPlotConfig
) -> None:
    """Plot time-series history curves from statistics/."""
    apply_style(master_cfg.figure_style)
    fig, ax = plt.subplots()

    x_field = plot_cfg.x_axis.field if plot_cfg.x_axis else None
    clip = plot_cfg.x_axis.clip_to_shortest if plot_cfg.x_axis else False

    curves: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []
    for label, x, y_mean, y_std in _iter_timeseries_groups(
        master_cfg, plot_cfg.y_axis.field, x_field, plot_cfg.color_by
    ):
        x, y_mean, y_std = _trim_series(x, y_mean, y_std)
        curves.append((label, x, y_mean, y_std))

    if clip and curves:
        min_len = min(len(x) for _, x, _, _ in curves)
        curves = [
            (label, x[:min_len], ym[:min_len], ys[:min_len])
            for label, x, ym, ys in curves
        ]

    for label, x, y_mean, y_std in curves:
        draw_line(ax, x, y_mean, label, plot_cfg.line)
        draw_band(ax, x, y_mean, y_std, plot_cfg.band)

    x_data = np.concatenate([x for _, x, _, _ in curves]) if curves else np.array([])
    y_data = np.concatenate([ym for _, _, ym, _ in curves]) if curves else np.array([])
    _finalize_axes(ax, plot_cfg.x_axis, plot_cfg.y_axis, x_data, y_data)
    ax.legend()
    save_figure(
        fig, master_cfg.output_dir, plot_cfg.output_name, master_cfg.output_formats
    )


def plot_trend(master_cfg: MasterPlotConfig, plot_cfg: TrendPlotConfig) -> None:
    """Plot trend lines (scalar x vs mean ± std) from statistics/."""
    apply_style(master_cfg.figure_style)
    fig, ax = plt.subplots()

    ref_fields = [r.field for r in plot_cfg.reference_lines]
    all_x: list[np.ndarray] = []
    all_y: list[np.ndarray] = []
    last_fields: dict[str, np.ndarray] = {}

    for label, x, y_mean, y_std, fields in _iter_trend_groups(
        master_cfg,
        plot_cfg.y_axis.field,
        plot_cfg.x_axis.field,
        plot_cfg.color_by,
        ref_fields,
    ):
        sort_idx = np.argsort(x)
        x, y_mean, y_std = x[sort_idx], y_mean[sort_idx], y_std[sort_idx]
        draw_line(ax, x, y_mean, label, plot_cfg.line)
        draw_band(ax, x, y_mean, y_std, plot_cfg.band)
        all_x.append(x)
        all_y.append(y_mean)
        last_fields = fields

    if plot_cfg.reference_lines and last_fields:
        draw_hlines(ax, last_fields, plot_cfg.reference_lines)

    x_data = np.concatenate(all_x) if all_x else np.array([])
    y_data = np.concatenate(all_y) if all_y else np.array([])
    _finalize_axes(ax, plot_cfg.x_axis, plot_cfg.y_axis, x_data, y_data)
    ax.legend()
    save_figure(
        fig, master_cfg.output_dir, plot_cfg.output_name, master_cfg.output_formats
    )


def _trim_series(
    x: np.ndarray, y_mean: np.ndarray, y_std: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Trim x, y_mean, y_std to the valid (non-NaN) length of y_mean."""
    valid_len = len(trim_to_valid(y_mean))
    return x[:valid_len], y_mean[:valid_len], y_std[:valid_len]
