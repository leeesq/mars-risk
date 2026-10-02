from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from mars.analysis import MarsBinEvaluator, profile_risk
from mars.core.exceptions import NotFittedError
from mars.feature import MarsLiteOptBinner, MarsNativeBinner, MarsOptimalBinner
from mars.feature.binning.base import MarsBinnerBase


def _binner(kind: str, **params: Any) -> MarsBinnerBase:
    config: dict[str, Any] = {"n_bins": 2, "n_jobs": 1, **params}
    if kind == "native":
        return MarsNativeBinner(method="quantile", **config)
    if kind == "lite_opt":
        return MarsLiteOptBinner(n_prebins=3, min_bin_size=0.05, **config)
    return MarsOptimalBinner(min_bin_n_event=30, n_prebins=3, **config)


def _frame(data: dict[str, Any], backend: str) -> pl.DataFrame | pd.DataFrame:
    return pl.DataFrame(data) if backend == "polars" else pd.DataFrame(data)


def _polars(frame: pl.DataFrame | pd.DataFrame) -> pl.DataFrame:
    return frame if isinstance(frame, pl.DataFrame) else pl.from_pandas(frame)


@pytest.mark.parametrize("backend", ["polars", "pandas"])
@pytest.mark.parametrize("kind", ["native", "lite_opt", "optimal"])
def test_refit_rebuilds_woe_rules_and_type_specific_state(kind: str, backend: str) -> None:
    X = _frame({"x": [0, 1, 2, 3, 4, 5]}, backend)
    y = pl.Series("target", [0, 0, 0, 1, 1, 1])
    binner = _binner(kind).fit(X, y)
    original = _polars(binner.transform(X, return_type="woe"))
    binner.fit(X, y)
    fresh = _binner(kind).fit(X, y)
    assert_frame_equal(_polars(binner.transform(X, return_type="woe")), original)
    assert binner.bin_cuts_ == fresh.bin_cuts_
    binner.fit(X, 1 - y)
    fresh.fit(X, 1 - y)
    assert_frame_equal(
        _polars(binner.transform(X, return_type="woe")),
        _polars(fresh.transform(X, return_type="woe")),
    )
    assert binner.bin_woes_ == fresh.bin_woes_
    assert not np.array_equal(_polars(binner.transform(X, return_type="woe"))["x_woe"], original["x_woe"])

    Z = _frame({"z": [0, 1, 2, 3, 4, 5]}, backend)
    binner.fit(Z, y)
    assert binner.feature_names_in_ == ["z"]
    assert set(binner.bin_cuts_) == {"z"}
    assert set(binner.bin_mappings_) == {"z"}
    assert binner.bin_woes_ == {}
    assert "z_bin" in binner.transform(Z).columns
    assert binner.get_fit_report()["feature"].to_list() == ["z"]

    categorical = _frame({"z": ["a", "a", "a", "b", "b", "b"]}, backend)
    binner.fit(categorical, y, cat_features=["z"])
    assert not binner.bin_cuts_
    assert set(binner.cat_cuts_) == {"z"}
    assert binner.cat_features == ["z"]
    binner.fit(Z, y)
    assert not binner.cat_cuts_
    assert set(binner.bin_cuts_) == {"z"}


@pytest.mark.parametrize("backend", ["polars", "pandas"])
@pytest.mark.parametrize("kind", ["native", "lite_opt", "optimal"])
@pytest.mark.parametrize("failure", ["missing_feature", "missing_target", "unsupported_dtype"])
def test_failed_refit_rejects_old_results_and_can_recover(
    kind: str, backend: str, failure: str, tmp_path: Path,
) -> None:
    binner = _binner(kind)
    if failure == "missing_target" and kind == "native":
        binner = MarsNativeBinner(method="cart", n_bins=2, n_jobs=1)
    X = _frame({"x": [0, 1, 2, 3, 4, 5]}, backend)
    y = pl.Series("target", [0, 0, 0, 1, 1, 1])
    binner.fit(X, y)
    binner.transform(X, return_type="woe")
    with pytest.raises(ValueError):
        if failure == "missing_feature":
            binner.fit(X, y, features=["absent"])
        elif failure == "missing_target":
            binner.fit(X)
        else:
            binner.fit(_frame({"unsupported": [[1], [2], [3], [4], [5], [6]]}, backend), y)
    assert not binner.__sklearn_is_fitted__()
    assert not binner.bin_cuts_ and not binner.cat_cuts_ and not binner.bin_woes_
    assert not binner.bin_mappings_
    for read in (
        lambda: binner.transform(X),
        binner.to_dict,
        lambda: binner.save_json(tmp_path / "stale.json"),
        lambda: binner.get_bin_mapping("x"),
        lambda: binner.generate_sql(return_type="index"),
        lambda: binner.profile_bin_performance(X, y),
    ):
        with pytest.raises(NotFittedError):
            read()
    assert not tmp_path.joinpath("stale.json").exists()
    assert "x" not in binner.get_fit_report()["feature"].to_list()
    binner.fit(X, y)
    assert binner.__sklearn_is_fitted__()
    assert "x_bin" in binner.transform(X).columns


@pytest.mark.parametrize("backend", ["polars", "pandas"])
@pytest.mark.parametrize("join_threshold", [0, 100])
@pytest.mark.parametrize("nullable", [False, True])
def test_boolean_binning_uses_both_encoding_paths_and_json_roundtrip(
    backend: str, join_threshold: int, nullable: bool, tmp_path: Path,
) -> None:
    values = [False, False, True, True] + ([None] if nullable else [])
    data: dict[str, Any] = {"flag": values, "_flag_utf8_tmp": ["user"] * len(values), "_idx_flag": [42] * len(values)}
    X = _frame(data, backend)
    if backend == "pandas":
        X["flag"] = pd.array(values, dtype="boolean" if nullable else "bool")
    y = pl.Series("target", [0, 0, 1, 1] + ([None] if nullable else []))
    binner = MarsNativeBinner(n_bins=2, join_threshold=join_threshold, n_jobs=1)
    binner.fit(X, y, features=["flag"], cat_features=["flag"])
    bins = _polars(binner.transform(X))
    assert set(bins.columns) == set(X.columns) | {"flag_bin"}
    assert bins["_flag_utf8_tmp"].to_list() == ["user"] * len(values)
    assert bins["_idx_flag"].to_list() == [42] * len(values)
    assert set(bins["flag_bin"].head(4).to_list()) == {0, 1}
    assert bins["flag_bin"][0] == bins["flag_bin"][1]
    assert bins["flag_bin"][2] == bins["flag_bin"][3]
    assert bins["flag_bin"][0] != bins["flag_bin"][2]
    if nullable:
        assert bins["flag_bin"][-1] == -1
    stats = _polars(binner.profile_bin_performance(X, y, include_bin_index=True))
    assert sorted(stats["count"].to_list()) == [2, 2]
    assert sorted(stats["bad"].to_list()) == [0, 2]
    assert stats["IV"][0] > 0 and stats["KS"][0] > 0
    payload = json.loads(json.dumps(binner.to_dict(), allow_nan=False))
    restored = MarsBinnerBase.from_dict(payload)
    assert_frame_equal(_polars(restored.transform(X)), bins)
    path = tmp_path / "boolean.json"
    binner.save_json(path)
    assert_frame_equal(_polars(MarsBinnerBase.load_json(path).transform(X)), bins)
    profile = profile_risk(X, target=None, features=["flag"], n_bins=2, method="quantile")
    assert profile.binner.get_fit_report()["usable"].to_list() == [True]
    evaluation_frame = _polars(X).with_columns(y.alias("target"))
    integer_frame = evaluation_frame.with_columns(pl.col("flag").cast(pl.Int8))
    comparison_binner = MarsNativeBinner(n_bins=2).fit(
        integer_frame.select("flag"), y, cat_features=["flag"],
    )
    if backend == "pandas":
        evaluation_frame = evaluation_frame.to_pandas(use_pyarrow_extension_array=True)
        integer_frame = integer_frame.to_pandas(use_pyarrow_extension_array=True)
    risk = profile_risk(evaluation_frame, target="target", features=["flag"], n_bins=2, method="quantile")
    comparison = MarsBinEvaluator().evaluate(
        integer_frame, target="target", features=["flag"], binner=comparison_binner,
    )
    risk_summary = _polars(risk.report.summary_table).select("iv", "ks")
    assert_frame_equal(risk_summary, _polars(comparison.report.summary_table).select("iv", "ks"))
    assert risk_summary["iv"][0] > 0 and risk_summary["ks"][0] > 0
    detail_columns = ["count", "bad", "bad_rate", "iv_bin"]
    assert_frame_equal(
        _polars(risk.report.detail_table).select(detail_columns).sort(detail_columns),
        _polars(comparison.report.detail_table).select(detail_columns).sort(detail_columns),
    )


@pytest.mark.parametrize("backend", ["polars", "pandas"])
@pytest.mark.parametrize("join_threshold", [0, 100])
def test_string_categories_keep_case_ids_and_missing_values(backend: str, join_threshold: int) -> None:
    values = ["True", "False", "true", "false", "1", "0", "001", "nan", None]
    X = _frame({"category": values}, backend)
    binner = MarsNativeBinner(n_bins=10, join_threshold=join_threshold).fit(X)
    bins = _polars(binner.transform(X))["category_bin"].to_list()
    assert len(set(bins[:-1])) == 8
    assert all(index >= 0 for index in bins[:-1])
    assert bins[-1] == -1
    unseen = _polars(binner.transform(_frame({"category": ["new", None]}, backend)))
    assert unseen["category_bin"].to_list() == [-2, -1]


@pytest.mark.parametrize("backend", ["polars", "pandas"])
@pytest.mark.parametrize("invalid", [2, -1, 0.5])
def test_public_bin_profile_validates_labels_before_updating_woe(backend: str, invalid: float) -> None:
    X = _frame({"x": [0, 1, 2, 3, 4, 5]}, backend)
    binner = MarsNativeBinner(n_bins=2).fit(X, pl.Series("target", [0, 0, 0, 1, 1, 1]))
    binner.transform(X, return_type="woe")
    before = copy.deepcopy(binner.bin_woes_)
    labels = pl.Series("target", [0.0, 0.0, 0.0, invalid, invalid, invalid])
    y = labels if backend == "polars" else labels.to_pandas()
    with pytest.raises(ValueError, match="Target.*invalid"):
        binner.profile_bin_performance(X, y)
    assert binner.bin_woes_ == before
    assert binner.__sklearn_is_fitted__()


@pytest.mark.parametrize("backend", ["polars", "pandas"])
@pytest.mark.parametrize("labels", [[False, False, True, True, None, None], [0.0, 0.0, 1.0, 1.0, None, np.nan], [0, 0, 0, 0, None, None]])
def test_public_bin_profile_excludes_unobserved_and_keeps_single_class(
    backend: str, labels: list[Any],
) -> None:
    X = _frame({"x": [0, 1, 2, 3, 4, 5]}, backend)
    binner = MarsNativeBinner(n_bins=2).fit(X)
    y = pl.Series("target", labels)
    stats = _polars(binner.profile_bin_performance(X, y))
    assert stats["count"].sum() == 4
    assert stats["bad"].sum() == sum(value for value in labels[:4])
    assert stats["bad_rate"].is_between(0, 1).all()
    assert (stats["bad"] <= stats["count"]).all()
    woes = copy.deepcopy(binner.bin_woes_)
    with pytest.raises(ValueError, match="observed"):
        binner.profile_bin_performance(X, pl.Series("target", [None] * 6))
    assert binner.bin_woes_ == woes


@pytest.mark.parametrize("backend", ["polars", "pandas"])
@pytest.mark.parametrize("kind", ["native", "lite_opt", "optimal"])
def test_refit_preserves_configuration_and_explicit_rules(kind: str, backend: str) -> None:
    X = _frame({"x": [0, 1, 2, 3, 4, 5], "category": ["a", "a", "a", "b", "b", "b"]}, backend)
    y = pl.Series("target", [0, 0, 0, 1, 1, 1])
    binner = _binner(kind, special_values=[-999, "unknown"], missing_values=[-777], join_threshold=1)
    config = copy.deepcopy(binner._serialization_params())
    binner.fit(X, y, cat_features=["category"])
    binner.update_bins({"x": [1.5], "category": [["a", "b"]]})
    numeric_rule = copy.deepcopy(binner.bin_cuts_["x"])
    binner.fit(X, 1 - y, cat_features=["category"])
    assert binner._serialization_params() == config
    assert binner.bin_cuts_["x"] == numeric_rule
    assert binner.cat_cuts_["category"] == [["a", "b"]]
    assert binner.cat_features == ["category"]
    assert set(binner.bin_woes_) == set()
    restored = MarsBinnerBase.from_dict(json.loads(json.dumps(binner.to_dict(), allow_nan=False)))
    restored.fit(X, y, cat_features=["category"])
    assert restored.bin_cuts_["x"] == numeric_rule
    assert restored.cat_cuts_["category"] == [["a", "b"]]
    restored.prune(["x"])
    assert "category" not in restored._user_bin_rules
    restored.fit(X, y, cat_features=["category"])
    fresh = _binner(kind, special_values=[-999, "unknown"], missing_values=[-777], join_threshold=1)
    fresh.fit(X, y, cat_features=["category"])
    assert restored.cat_cuts_["category"] == fresh.cat_cuts_["category"]
    numeric_category = _frame({"x": ["a", "a", "a", "b", "b", "b"]}, backend)
    restored.fit(numeric_category, y)
    assert set(restored.cat_cuts_) == {"x"} and not restored.bin_cuts_
    fresh.fit(numeric_category, y)
    assert_frame_equal(_polars(restored.transform(numeric_category)), _polars(fresh.transform(numeric_category)))


def test_binner_old_state_restoration_does_not_require_new_explicit_rule_field() -> None:
    X = pl.DataFrame({"x": [0, 1, 2, 3]})
    original = MarsNativeBinner(n_bins=2).fit(X)
    state = original.__getstate__()
    del state["_user_bin_rules"]
    restored = MarsNativeBinner.__new__(MarsNativeBinner)
    restored.__setstate__(state)
    assert_frame_equal(restored.fit(X).transform(X), original.transform(X))
    artifact = original.to_dict()
    artifact["state"].pop("user_bin_rules")
    assert_frame_equal(MarsBinnerBase.from_dict(artifact).transform(X), original.transform(X))
    artifact["state"]["user_bin_rules"] = []
    with pytest.raises(ValueError, match="user_bin_rules"):
        MarsBinnerBase.from_dict(artifact)


@pytest.mark.parametrize("backend", ["polars", "pandas"])
@pytest.mark.parametrize("join_threshold", [0, 100])
def test_boolean_unseen_typed_categories_all_missing_and_empty_input(
    backend: str, join_threshold: int,
) -> None:
    X = _frame({"flag": [False, False]}, backend)
    binner = MarsNativeBinner(n_bins=2, join_threshold=join_threshold).fit(X)
    inference = _frame({"flag": [False, True, None]}, backend)
    if backend == "pandas":
        inference["flag"] = pd.array([False, True, None], dtype="boolean")
    assert _polars(binner.transform(inference))["flag_bin"].to_list() == [0, -2, -1]
    assert _polars(binner.transform(_frame({"flag": [0, 1]}, backend)))["flag_bin"].to_list() == [-2, -2]
    assert _polars(binner.transform(_frame({"flag": ["false", "False"]}, backend)))["flag_bin"].to_list() == [-2, -2]
    empty = pl.DataFrame(schema={"flag": pl.Boolean})
    assert binner.transform(empty).columns == ["flag", "flag_bin"]
    assert binner.transform(empty).height == 0
    null_frame = pl.DataFrame({"flag": pl.Series([None, None], dtype=pl.Boolean)})
    null_input = null_frame if backend == "polars" else null_frame.to_pandas(use_pyarrow_extension_array=True)
    binner.fit(null_input)
    assert binner.get_fit_report()["status"].to_list() == ["all_missing"]
    assert _polars(binner.transform(null_input))["flag_bin"].to_list() == [-1, -1]
    assert _polars(binner.transform(inference))["flag_bin"].to_list() == [-2, -2, -1]


@pytest.mark.parametrize("backend", ["polars", "pandas"])
@pytest.mark.parametrize("kind", ["native", "lite_opt", "optimal"])
def test_boolean_categories_preserve_native_types_for_all_binners(kind: str, backend: str) -> None:
    X = _frame({"flag": [False, False, False, False, True, True, True, True]}, backend)
    y = pl.Series("target", [0, 0, 1, 0, 1, 1, 0, 1])
    binner = _binner(kind).fit(X, y, cat_features=["flag"])
    bins = _polars(binner.transform(X))["flag_bin"]
    assert set(bins.to_list()) == {0, 1}
    assert {type(value) for group in binner.cat_cuts_["flag"] for value in group} == {bool}
    assert _polars(binner.profile_bin_performance(X, y))["KS"][0] > 0


@pytest.mark.parametrize("backend", ["polars", "pandas"])
def test_real_optimal_solver_refit_matches_fresh_instance(backend: str, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("optbinning")
    X = _frame({"x": np.repeat(np.arange(12), 20)}, backend)
    y = pl.Series("target", [int(index % 20 < (index // 20 + 2)) for index in range(240)])
    params: dict[str, Any] = {"n_bins": 3, "min_bin_n_event": 1, "min_bin_size": 0.05, "n_prebins": 6, "n_jobs": 1}
    binner = MarsOptimalBinner(**params)
    solver = binner.OptimalBinning
    statuses: list[str] = []

    class RecordingSolver(solver):
        def fit(self, *args: Any, **kwargs: Any) -> Any:
            result = super().fit(*args, **kwargs)
            statuses.append(self.status)
            return result

    monkeypatch.setattr(binner, "OptimalBinning", RecordingSolver)
    binner.fit(X, y)
    assert statuses and set(statuses) <= {"OPTIMAL", "FEASIBLE"}
    assert not binner.fit_failures_
    original = _polars(binner.transform(X, return_type="woe"))
    binner.fit(X, 1 - y)
    fresh = MarsOptimalBinner(**params).fit(X, 1 - y)
    assert not binner.fit_failures_ and not fresh.fit_failures_
    assert binner.bin_cuts_ == fresh.bin_cuts_
    assert_frame_equal(_polars(binner.transform(X, return_type="woe")), _polars(fresh.transform(X, return_type="woe")))
    assert not np.array_equal(original["x_woe"], _polars(binner.transform(X, return_type="woe"))["x_woe"])


@pytest.mark.parametrize("backend", ["polars", "pandas"])
def test_legal_string_labels_and_invalid_update_leave_rules_untouched(backend: str) -> None:
    X = _frame({"x": [0, 1, 2, 3, 4, 5]}, backend)
    binner = MarsNativeBinner(n_bins=2).fit(X)
    y = pl.Series("target", ["false", "False", "true", "True", "", None])
    stats = _polars(binner.profile_bin_performance(X, y))
    assert stats["count"].sum() == 4 and stats["bad"].sum() == 2
    rules = copy.deepcopy(binner.bin_cuts_)
    woes = copy.deepcopy(binner.bin_woes_)
    with pytest.raises(ValueError, match="invalid"):
        binner.update_bins({"x": [0.5]}, X=X, y=pl.Series("target", [0, 0, 0, 2, 2, 2]))
    assert binner.bin_cuts_ == rules and binner.bin_woes_ == woes


def test_boolean_rules_load_and_transform_in_independent_process(tmp_path: Path) -> None:
    X = pl.DataFrame({"flag": [False, True, None]})
    path = tmp_path / "rules.json"
    MarsNativeBinner(n_bins=2).fit(X).save_json(path)
    script = "\n".join([
        "import json,sys,polars as pl",
        "from mars.feature import MarsBinnerBase",
        "b=MarsBinnerBase.load_json(sys.argv[1])",
        "out=b.transform(pl.DataFrame({'flag':[False,True,None]}))",
        "print(json.dumps(out.to_dict(as_series=False),allow_nan=False))",
    ])
    completed = subprocess.run([sys.executable, "-c", script, str(path)], capture_output=True, text=True, check=True)
    payload = json.loads(completed.stdout)
    assert payload["flag"] == [False, True, None]
    assert set(payload["flag_bin"][:2]) == {0, 1} and payload["flag_bin"][-1] == -1


@pytest.mark.parametrize("backend", ["polars", "pandas"])
@pytest.mark.parametrize("join_threshold", [0, 100])
def test_nonboolean_categorical_numbers_keep_existing_string_matching(
    backend: str, join_threshold: int,
) -> None:
    X = _frame({"code": [0, 0, 1, 1]}, backend)
    binner = MarsNativeBinner(n_bins=2, join_threshold=join_threshold).fit(X, cat_features=["code"])
    inferred = _polars(binner.transform(_frame({"code": ["0", "1", "001"]}, backend)))
    fitted = _polars(binner.transform(X))["code_bin"].to_list()
    assert inferred["code_bin"].to_list() == [fitted[0], fitted[2], -2]
    float_inferred = _polars(binner.transform(_frame({"code": [0.0, 1.0]}, backend)))
    assert float_inferred["code_bin"].to_list() == [-2, -2]
    literal_strings = _frame({"code": ["b:true", "true", "True", "1", "001"]}, backend)
    string_binner = MarsNativeBinner(n_bins=5, join_threshold=join_threshold).fit(literal_strings)
    assert _polars(string_binner.transform(literal_strings))["code_bin"].n_unique() == 5


@pytest.mark.parametrize("kind", ["native", "lite_opt"])
def test_all_missing_boolean_keeps_missing_and_other_protocol(kind: str) -> None:
    X = pl.DataFrame({"flag": pl.Series([None] * 4, dtype=pl.Boolean)})
    binner = _binner(kind).fit(X, pl.Series("target", [0, 0, 1, 1]))
    assert binner.cat_cuts_ == {"flag": []} and binner.bin_cuts_ == {}
    assert binner.transform(pl.DataFrame({"flag": [False, True, None]}))["flag_bin"].to_list() == [-2, -2, -1]


def test_optimal_all_missing_boolean_preserves_failure_contract() -> None:
    X = pl.DataFrame({"flag": pl.Series([None] * 4, dtype=pl.Boolean)})
    binner = _binner("optimal")
    with pytest.raises(ValueError, match="no usable"):
        binner.fit(X, pl.Series("target", [0, 0, 1, 1]))
    assert not binner.__sklearn_is_fitted__()
    assert binner.get_fit_report()["usable"].to_list() == [False]


def test_legacy_boolean_solver_json_and_pickle_rules_preserve_transform() -> None:
    X = pl.DataFrame({"flag": [False] * 100 + [True] * 100})
    y = pl.Series("target", [0] * 90 + [1] * 10 + [0] * 10 + [1] * 90)
    binner = MarsOptimalBinner(n_bins=2, max_cats_to_solver=None, min_cat_fraction=0.01, n_jobs=1).fit(X, y)
    expected = binner.transform(X)
    artifact = binner.to_dict()
    artifact["state"].pop("user_bin_rules")
    artifact["state"]["cat_cuts_"]["flag"] = [["false"], ["true"]]
    restored = MarsBinnerBase.from_dict(artifact)
    assert_frame_equal(restored.transform(X), expected)
    state = binner.__getstate__()
    state.pop("_user_bin_rules")
    state["cat_cuts_"] = {"flag": [["false"], ["true"]]}
    legacy_pickle = MarsOptimalBinner.__new__(MarsOptimalBinner)
    legacy_pickle.__setstate__(state)
    assert_frame_equal(legacy_pickle.transform(X), expected)
    strings = pl.DataFrame({"flag": ["false", "true"]})
    string_binner = MarsNativeBinner(n_bins=2).fit(strings)
    assert_frame_equal(MarsBinnerBase.from_dict(string_binner.to_dict()).transform(strings), string_binner.transform(strings))
