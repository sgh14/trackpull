"""Unit tests for trackpull.plot helper functions.

Each test targets exactly one helper — no HDF5, no matplotlib rendering.
"""

from __future__ import annotations

import numpy as np

from trackpull.plot import (
    InputConfig,
    SelectConfig,
    _expand_template,
    _flatten_dict,
    _template_fields,
    clip_curves,
    combined_mask,
    filter_mask,
    group_indices,
    resolve_label,
    select_mask,
    trim_to_valid,
)

# ---------------------------------------------------------------------------
# _flatten_dict
# ---------------------------------------------------------------------------


def test_flatten_dict_already_flat():
    assert _flatten_dict({"a": 1, "b": 2}) == {"a": 1, "b": 2}


def test_flatten_dict_one_level_nested():
    assert _flatten_dict({"model": {"width": 64}}) == {"model.width": 64}


def test_flatten_dict_two_levels_nested():
    result = _flatten_dict({"a": {"b": {"c": 1}}})
    assert result == {"a.b.c": 1}


def test_flatten_dict_mixed():
    result = _flatten_dict({"graph": {"length": 12}, "seed": 0})
    assert result == {"graph.length": 12, "seed": 0}


def test_flatten_dict_empty():
    assert _flatten_dict({}) == {}


# ---------------------------------------------------------------------------
# filter_mask
# ---------------------------------------------------------------------------


def _fields(n=4):
    return {
        "graph.length": np.array([10.0, 12.0, 12.0, 16.0]),
        "vstate.n_samples": np.array([512, 1024, 2048, 1024]),
    }


def test_filter_mask_scalar_equality():
    mask = filter_mask(_fields(), {"graph.length": 12.0})
    assert list(mask) == [False, True, True, False]


def test_filter_mask_list_containment():
    mask = filter_mask(_fields(), {"graph.length": [10.0, 16.0]})
    assert list(mask) == [True, False, False, True]


def test_filter_mask_and_of_two_keys():
    mask = filter_mask(_fields(), {"graph.length": 12.0, "vstate.n_samples": 1024})
    assert list(mask) == [False, True, False, False]


def test_filter_mask_empty_filter():
    mask = filter_mask(_fields(), {})
    assert mask.all()


# ---------------------------------------------------------------------------
# select_mask
# ---------------------------------------------------------------------------


def test_select_mask_min():
    fields = {"error": np.array([0.5, 0.1, 0.3])}
    mask = select_mask(fields, SelectConfig(by="error", criterion="min"))
    assert list(mask) == [False, True, False]


def test_select_mask_max():
    fields = {"score": np.array([0.2, 0.8, 0.5])}
    mask = select_mask(fields, SelectConfig(by="score", criterion="max"))
    assert list(mask) == [False, True, False]


def test_select_mask_empty():
    fields = {"error": np.array([])}
    mask = select_mask(fields, SelectConfig(by="error", criterion="min"))
    assert len(mask) == 0


# ---------------------------------------------------------------------------
# combined_mask
# ---------------------------------------------------------------------------


def test_combined_mask_filter_only():
    fields = {"graph.length": np.array([10.0, 12.0, 16.0])}
    mask = combined_mask(fields, {"graph.length": 12.0}, select_cfg=None)
    assert list(mask) == [False, True, False]


def test_combined_mask_filter_then_select():
    fields = {
        "graph.length": np.array([10.0, 12.0, 12.0, 16.0]),
        "error": np.array([0.1, 0.4, 0.2, 0.9]),
    }
    mask = combined_mask(
        fields,
        {"graph.length": 12.0},
        SelectConfig(by="error", criterion="min"),
    )
    # filter keeps indices 1,2; select picks index 2 (error=0.2)
    assert list(mask) == [False, False, True, False]


def test_combined_mask_no_select():
    fields = {"x": np.array([1.0, 2.0, 3.0])}
    mask = combined_mask(fields, {}, select_cfg=None)
    assert mask.all()


# ---------------------------------------------------------------------------
# group_indices
# ---------------------------------------------------------------------------


def test_group_indices_single_group():
    result = group_indices(np.array(["a", "a", "a"]))
    assert set(result.keys()) == {"a"}
    assert list(result["a"]) == [0, 1, 2]


def test_group_indices_multiple_groups():
    result = group_indices(np.array([1.0, 2.0, 1.0, 2.0]))
    assert set(result.keys()) == {1.0, 2.0}
    assert list(result[1.0]) == [0, 2]
    assert list(result[2.0]) == [1, 3]


def test_group_indices_preserves_order():
    result = group_indices(np.array(["b", "a", "b"]))
    keys = list(result.keys())
    assert keys == ["b", "a"]


# ---------------------------------------------------------------------------
# resolve_label
# ---------------------------------------------------------------------------


def test_resolve_label_uses_label():
    inp = InputConfig(path="some/file.h5", label="SR")
    label = resolve_label(inp, {}, row=0)
    assert label == "SR"


def test_resolve_label_falls_back_to_stem():
    inp = InputConfig(path="results/plain.h5")
    label = resolve_label(inp, {}, row=0)
    assert label == "plain"


def test_resolve_label_expands_template():
    inp = InputConfig(path="file.h5", label_template="QPS (n=$n_steps)")
    fields = {"n_steps": np.array([32, 64])}
    assert resolve_label(inp, fields, row=0) == "QPS (n=32)"
    assert resolve_label(inp, fields, row=1) == "QPS (n=64)"


def test_resolve_label_template_missing_field_kept():
    inp = InputConfig(path="file.h5", label_template="val=$missing")
    assert resolve_label(inp, {}, row=0) == "val=$missing"


# ---------------------------------------------------------------------------
# _template_fields
# ---------------------------------------------------------------------------


def test_template_fields_single():
    assert _template_fields("QPS (n=$n_steps)") == ["n_steps"]


def test_template_fields_dotted():
    assert _template_fields("$driver.n_ppo_steps steps") == ["driver.n_ppo_steps"]


def test_template_fields_multiple():
    assert _template_fields("$a and $b.c") == ["a", "b.c"]


def test_template_fields_no_fields():
    assert _template_fields("no substitution here") == []


# ---------------------------------------------------------------------------
# _expand_template
# ---------------------------------------------------------------------------


def test_expand_template_latex_dollar_preserved():
    # LaTeX $\varepsilon$ should not be treated as a field reference
    # because \varepsilon contains backslash which is not in [\w.]
    fields = {"eps": np.array([0.1])}
    result = _expand_template(r"$\varepsilon$ = $eps", fields, row=0)
    assert result == r"$\varepsilon$ = 0.1"


# ---------------------------------------------------------------------------
# trim_to_valid
# ---------------------------------------------------------------------------


def test_trim_to_valid_no_nans():
    arr = np.array([1.0, 2.0, 3.0])
    result = trim_to_valid(arr)
    np.testing.assert_array_equal(result, arr)


def test_trim_to_valid_trailing_nans():
    arr = np.array([1.0, 2.0, np.nan, np.nan])
    result = trim_to_valid(arr)
    np.testing.assert_array_equal(result, [1.0, 2.0])


def test_trim_to_valid_all_nan():
    arr = np.array([np.nan, np.nan])
    result = trim_to_valid(arr)
    assert len(result) == 0


def test_trim_to_valid_interior_nan_kept():
    arr = np.array([1.0, np.nan, 3.0, np.nan])
    result = trim_to_valid(arr)
    np.testing.assert_array_equal(result, [1.0, np.nan, 3.0])


# ---------------------------------------------------------------------------
# clip_curves
# ---------------------------------------------------------------------------


def test_clip_curves_equal_length():
    curves = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]
    result = clip_curves(curves)
    assert all(len(c) == 2 for c in result)


def test_clip_curves_unequal_length():
    curves = [np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0])]
    result = clip_curves(curves)
    assert all(len(c) == 2 for c in result)
    np.testing.assert_array_equal(result[0], [1.0, 2.0])


def test_clip_curves_empty_list():
    assert clip_curves([]) == []


def test_clip_curves_single_curve():
    curves = [np.array([1.0, 2.0, 3.0])]
    result = clip_curves(curves)
    np.testing.assert_array_equal(result[0], [1.0, 2.0, 3.0])
