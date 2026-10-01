"""共享分箱引擎、固定定义和监督参考集合同的行为回归。"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from mars.analysis import cross_scores, get_score_bin_definitions
from mars.feature.binning import MarsNativeBinner
from mars.reporting import load_report


def _reference() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "old": np.linspace(0.01, 0.99, 40),
            "new": np.linspace(0.99, 0.01, 40),
            "bad": [0] * 20 + [1] * 20,
            "late": [0, 1] * 20,
            "sample": ["development/任意"] * 40,
            "date": ["2026-01-01"] * 20 + ["2026-02-01"] * 20,
        }
    )


def _cross(data: pl.DataFrame | None = None, **kwargs: Any) -> Any:
    return cross_scores(
        _reference() if data is None else data,
        score_x="old",
        score_y="new",
        score_directions={"old": "higher_risk", "new": "lower_risk"},
        **kwargs,
    )


@pytest.mark.parametrize("method", ["quantile", "uniform", "cart"])
def test_existing_native_engine_supplies_actual_cuts_and_right_closed_assignment(
    method: str,
) -> None:
    reference = _reference()
    options: dict[str, Any] = {"method": method, "n_bins": 4, "n_jobs": 1}
    if method == "cart":
        options["cart_params"] = {"max_depth": 2, "random_state": 17}
    expected = MarsNativeBinner(**options).fit(
        reference.select("old", "new"), reference["bad"] if method == "cart" else None
    )
    report = _cross(
        targets=["bad", "late"],
        method=method,
        n_bins=4,
        n_jobs=1,
        binner_params={"cart_params": options["cart_params"]} if method == "cart" else None,
    )
    definitions = get_score_bin_definitions(report)
    for axis, score in [("x", "old"), ("y", "new")]:
        definition = definitions[axis]
        assert definition["cutpoints"] == expected.bin_cuts_[score][1:-1]
        assert definition["actual_n_bins"] == len(expected.bin_cuts_[score]) - 1
        assert definition["fit"]["binner_params"]["method"] == method
        assert definition["fit"]["target"] == ("bad" if method == "cart" else None)
        assert definition["closed"] == "right"
    cuts = definitions["x"]["cutpoints"]
    values = [v for cut in cuts for v in [np.nextafter(cut, -np.inf), cut, np.nextafter(cut, np.inf)]]
    boundary = _cross(
        pl.DataFrame({"old": values, "new": [0.0] * len(values)}), bin_definitions=definitions
    )
    counts = {
        row["x_bin"]: row["sample_count"]
        for row in boundary.get_table("row_summary").to_dicts()
    }
    expected_counts = {
        f"b{i}": sum(sum(value > cut for cut in cuts) == i for value in values)
        for i in range(len(cuts) + 1)
    }
    assert {key: counts[key] for key in expected_counts} == expected_counts


@pytest.mark.parametrize("binning_type", ["native", "optimal", "lite_opt"])
def test_supervised_reference_is_explicit_and_definitions_fit_once_then_survive_load(
    binning_type: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from mars.analysis import _score_cross_binning as adapter

    calls: list[tuple[list[str], str, int]] = []
    original = adapter.build_binner

    def counted(**kwargs: Any) -> Any:
        calls.append((kwargs["features"], kwargs["target"], kwargs["fit_df"].height))
        return original(**kwargs)

    monkeypatch.setattr(adapter, "build_binner", counted)
    reference = _reference().with_columns(pl.lit(1).alias("bad"))
    data = _reference().with_columns(
        pl.when(pl.col("late") == 0).then(pl.lit("holdout A")).otherwise(pl.lit("OOT/外部"))
        .alias("sample"),
        pl.lit(0).alias("bad"),
    )
    params: dict[str, Any] = {}
    if binning_type == "optimal":
        params = {"min_bin_n_event": 1, "min_n_bins": 1, "n_prebins": 6, "time_limit": 1}
    elif binning_type == "lite_opt":
        params = {"n_prebins": 6}
    report = _cross(
        data,
        targets=["bad", "late"],
        binning_reference=reference,
        binning_target="late",
        binning_type=binning_type,
        method="cart" if binning_type == "native" else "quantile",
        n_bins={"old": 3, "new": 2},
        binner_params=params,
        n_jobs=1,
        group_col="sample",
        time_col="date",
        time_grain="month",
    )
    assert calls == [(["old"], "late", 40), (["new"], "late", 40)]
    definitions = get_score_bin_definitions(report)
    for axis in ["x", "y"]:
        fit = definitions[axis]["fit"]
        assert fit["target"] == "late" and fit["source"] == "reference"
        assert fit["observed_class_count"] == 2 and fit["fit_row_count"] == 40
        assert fit["binning_type"] == binning_type
        assert fit["actual_n_bins"] == definitions[axis]["actual_n_bins"]
        assert fit["actual_n_bins"] <= fit["requested_n_bins"]
    assert report.describe()["parameters"]["binning_target"] == "late"
    report.save(tmp_path / "supervised.marsreport")
    loaded = load_report(tmp_path / "supervised.marsreport")
    assert get_score_bin_definitions(loaded) == definitions
    replay = _cross(
        data,
        targets=["bad", "late"],
        bin_definitions=get_score_bin_definitions(loaded),
        group_col="sample",
        time_col="date",
        time_grain="month",
    )
    assert replay.describe()["parameters"]["binning_target"] == "late"
    assert replay.describe()["parameters"]["fit_performed"] is False
    assert calls == [(["old"], "late", 40), (["new"], "late", 40)]
    for table in ["cells", "row_summary", "column_summary", "overall"]:
        assert_frame_equal(report.get_table(table), replay.get_table(table))


def test_shared_native_advanced_parameters_and_actual_different_axis_counts() -> None:
    reference = pl.DataFrame({"old": [0.0] * 18 + [0.8, 1.0], "new": [0.2] * 20})
    report = _cross(
        reference,
        method="uniform",
        n_bins={"old": 7, "new": 3},
        binner_params={"merge_small_bins": True, "remove_empty_bins": True},
        min_bin_size=0.2,
        n_jobs=1,
    )
    definitions = get_score_bin_definitions(report)
    assert definitions["x"]["actual_n_bins"] < 7
    assert definitions["y"]["actual_n_bins"] == 1
    fit = definitions["x"]["fit"]["binner_params"]
    assert fit["merge_small_bins"] is True and fit["remove_empty_bins"] is True
    assert fit["min_bin_size"] == 0.2
    bins = report.get_table("bins").filter(pl.col("kind") == "normal")
    for row in bins.to_dicts():
        assert row["display_label"] == row["axis"].upper() + str(row["risk_rank"])
        definition = definitions[row["axis"]]
        assert row["risk_rank"] == (
            row["raw_order"] + 1 if row["axis"] == "x" else definition["actual_n_bins"] - row["raw_order"]
        )
    assert _cross(n_bins=1).describe()["parameters"]["actual_n_bins"] == {"old": 1, "new": 1}


@pytest.mark.parametrize("method", ["quantile", "uniform", "cart"])
def test_one_native_bin_uses_shared_engine_without_failed_tree_or_extra_median(
    method: str,
) -> None:
    reference = _reference()
    binner = MarsNativeBinner(method=method, n_bins=1, n_jobs=1).fit(
        reference.select("old", "new"), reference["bad"]
    )
    assert binner.bin_cuts_ == {
        "old": [float("-inf"), float("inf")],
        "new": [float("-inf"), float("inf")],
    }
    assert binner.fit_failures_ == {}
    assert binner.get_fit_report()["n_bins"].to_list() == [1, 1]
    assert binner.get_fit_report()["usable"].to_list() == [True, True]
    transformed = binner.transform(reference.select("old", "new"), return_type="index")
    assert transformed["old_bin"].unique().to_list() == [0]
    assert transformed["new_bin"].unique().to_list() == [0]
    woes = binner.transform(reference.select("old", "new"), return_type="woe")
    assert woes["old_woe"].is_finite().all() and woes["new_woe"].is_finite().all()
    assert set(binner.bin_woes_) == {"old", "new"}
    report = _cross(targets=["bad"], method=method, n_bins=1, n_jobs=1)
    definitions = get_score_bin_definitions(report)
    for definition in definitions.values():
        assert definition["cutpoints"] == []
        assert definition["fit"]["diagnostic"] is None
        assert definition["actual_n_bins"] == 1


def test_probability_domain_and_missing_special_exclusions_preserve_reference_size() -> None:
    data = pl.DataFrame(
        {
            "old": [0.0, 0.1, 0.5, 0.9, 1.0, -99.0, -7.0, 2.0, float("nan"), None],
            "new": [0.5] * 10,
        }
    )
    report = _cross(
        data,
        method="uniform",
        n_bins=2,
        probability_scores=["old"],
        special_values={"old": [-99.0]},
        missing_values=[-7.0],
        n_jobs=1,
    )
    definition = get_score_bin_definitions(report)["x"]
    assert definition["cutpoints"] == [0.5]
    assert definition["fit"]["reference_row_count"] == 10
    assert definition["fit"]["fit_row_count"] == 10
    assert definition["fit"]["valid_score_row_count"] == 5
    assert definition["fit"]["usable_fit_row_count"] == 5
    counts = {r["x_bin"]: r["sample_count"] for r in report.get_table("row_summary").to_dicts()}
    assert counts == {"b0": 3, "b1": 2, "missing": 3, "invalid": 1, "s0": 1}
    assert report.get_table("overall")["sample_count"][0] == 10


def test_custom_missing_probabilities_specials_and_legacy_saved_definitions() -> None:
    data = pl.DataFrame(
        {
            "old": ["0", "0.5", "1", "N/A", "-99", "-7", "abc", "inf", "nan", None, "2"],
            "new": [0.0] * 11,
        }
    )
    report = _cross(
        data,
        cutpoints={"old": [0.5, 0.5], "new": []},
        probability_scores=["old"],
        special_values={"old": [-99.0]},
        missing_values={"old": ["N/A", "-7"]},
    )
    counts = {r["x_bin"]: r["sample_count"] for r in report.get_table("row_summary").to_dicts()}
    assert counts == {"b0": 2, "b1": 1, "missing": 4, "invalid": 3, "s0": 1}
    definitions = get_score_bin_definitions(report)
    assert get_score_bin_definitions(_cross(data, bin_definitions=definitions)) == definitions
    legacy = {axis: {k: v for k, v in d.items() if k != "missing_values"} for axis, d in definitions.items()}
    assert get_score_bin_definitions(_cross(data, bin_definitions=legacy)) == legacy
    for changes in [
        {"missing_values": ["different"]},
        {"probability_scores": []},
        {"special_values": {}},
    ]:
        with pytest.raises(ValueError, match="differ from saved"):
            _cross(data, bin_definitions=definitions, **changes)


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"method": "custom"}, "method must"),
        ({"binning_type": "opt"}, "binning_type"),
        ({"method": "cart"}, "requires binning_target"),
        ({"binning_type": "lite_opt"}, "requires binning_target"),
        ({"binning_target": "bad"}, "supervised"),
        ({"n_bins": {"old": 3}}, "both score IDs"),
        ({"n_bins": True}, "integers"),
        ({"min_bin_size": -0.1}, "min_bin_size"),
        ({"n_jobs": 0}, "n_jobs"),
        ({"binner_params": {"n_bins": 2}}, "unsupported keys"),
        ({"binner_params": {"method": "cart"}}, "unsupported keys"),
        ({"binner_params": {"prebinning_method": "cart"}}, "unsupported keys"),
        ({"binner_params": {"unknown": True}}, "unsupported keys"),
        ({"method": "cart", "targets": ["bad"], "binner_params": {"cart_params": {"max_leaf_nodes": 2}}}, "cart_params"),
        ({"cutpoints": {"old": [], "new": []}, "method": "quantile"}, "Fitting options"),
        ({"cutpoints": {"old": [], "new": []}, "binning_reference": _reference()}, "mutually exclusive"),
    ],
)
def test_illegal_configuration_is_explicit(kwargs: dict[str, Any], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        _cross(**kwargs)


@pytest.mark.parametrize("labels", [[0] * 40, [None] * 40, [2] * 40])
def test_supervised_labels_and_usable_classes_are_validated(labels: list[int | None]) -> None:
    reference = _reference().with_columns(pl.Series("bad", labels))
    with pytest.raises(ValueError, match="usable observed|invalid values"):
        _cross(targets=["bad"], method="cart", binning_reference=reference, n_jobs=1)
    if labels[0] == 2:
        return
    sparse = _reference().with_columns(
        pl.when(pl.col("bad") == 1).then(None).otherwise(pl.col("old")).alias("old")
    )
    with pytest.raises(ValueError, match="usable observed"):
        _cross(targets=["bad"], method="cart", binning_reference=sparse, n_jobs=1)


def test_shared_warnings_default_primary_target_and_explicit_no_fit_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.warns(UserWarning, match="monotonic_trend.*ignored"):
        _cross(monotonic_trend="ascending", n_jobs=1)
    with pytest.warns(UserWarning, match="binner_params ignored"):
        _cross(binner_params={"n_prebins": 8}, n_jobs=1)
    report = _cross(targets=["bad", "late"], method="cart", n_jobs=1)
    assert report.describe()["parameters"]["binning_target"] == "bad"
    definitions = get_score_bin_definitions(report)
    saved = deepcopy(definitions)
    saved["x"]["fit"]["actual_n_bins"] += 1
    with pytest.raises(ValueError, match="provenance"):
        _cross(bin_definitions=saved)
    from mars.analysis import _score_cross_binning as adapter

    def forbidden(**kwargs: Any) -> Any:
        raise AssertionError("Explicit paths must not fit.")

    monkeypatch.setattr(adapter, "build_binner", forbidden)
    _cross(cutpoints={"old": [], "new": []})
    _cross(bin_definitions=definitions)
    with pytest.raises(ValueError, match="Fitting options"):
        _cross(bin_definitions=definitions, binning_target="bad")
