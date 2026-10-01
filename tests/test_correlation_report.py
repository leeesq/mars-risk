"""相关性真实口径、生命周期、事件与保存后的操作回归。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import pytest
from pandas.testing import assert_frame_equal
from polars.testing import assert_frame_equal as assert_pl_equal

from mars.feature import MarsLinearSelector, MarsStatsSelector
from mars.reporting import (
    get_correlation_matrix,
    get_related_features,
    load_report,
    show_correlation_matrix,
)


def test_linear_reuses_signed_matrix_and_preserves_complete_pool(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    data = pd.DataFrame(
        {
            "a": [-1.0, -1.0, 1.0, 1.0, 9.0],
            "negative": [1.0, 1.0, -1.0, -1.0, 9.0],
            "orthogonal": [-1.0, 1.0, -1.0, 1.0, None],
            "constant": [2.0] * 5,
            "text": ["none"] * 5,
        }
    )
    y = pd.Series([0, 0, 1, 0, 1], name="bad")
    metadata = {
        "a": {"display_name": "重复名", "data_source": "A"},
        "negative": {"display_name": "重复名", "data_source": "B"},
        "orthogonal": {"data_source": "B"},
    }
    original = pd.DataFrame.corr
    calls: list[list[str]] = []

    def counted(frame: pd.DataFrame, *args: Any, **kwargs: Any) -> pd.DataFrame:
        calls.append(list(frame.columns))
        return original(frame, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "corr", counted)
    selector = MarsLinearSelector(corr_thr=1).fit(data, y, feature_metadata=metadata)
    assert calls == [["a", "negative", "orthogonal"]]
    assert selector.selected_features_ == ["a", "orthogonal"]
    report = selector.get_correlation_report()
    assert report.describe()["parameters"]["prepared_row_count"] == 4
    assert report.describe()["parameters"]["cleaning_field_pool"] == [
        "a",
        "negative",
        "orthogonal",
        "constant",
    ]
    matrix = report.get_matrix()
    assert matrix.loc["a", "negative"] == -1
    assert matrix.loc["a", "orthogonal"] == 0
    drop = report.get_table("correlation_decisions", filters={"action": "drop"}).to_dicts()[0]
    assert (
        drop["feature"],
        drop["trigger_feature"],
        drop["operator"],
        drop["correlation"],
        drop["event_order"],
    ) == ("negative", "a", ">=", -1, 0)
    assert report.get_table("features", features="negative")["selected"][0] is False
    assert report.get_table("features", features="constant")["diagonal_status"][0] == "uncomputed"
    # 两条件可由不同端点匹配；peer 的来源过滤只作用在另一端。
    assert len(report.get_table("pairs", features="a", sources="B")) == 2
    assert get_related_features(report, "negative", sources="A")["peer_feature"].to_list() == ["a"]
    assert report.get_feature("negative")["tables"]["pairs"].height == 2
    context = json.loads(
        report.to_ai_context(tables=["pairs", "correlation_decisions"], max_chars=16000)
    )
    assert context["description"]["tables"]["pairs"]["feature_roles"] == {
        "feature_a": "left endpoint",
        "feature_b": "right endpoint",
    }
    assert set(context["description"]["feature_metadata"]) >= {"a", "negative"}
    assert context["evidence"][0]["report_id"] == report.report_id
    path = tmp_path / "corr.marsreport"
    report.save(path)
    restored = load_report(path)
    assert_frame_equal(get_correlation_matrix(restored), matrix)
    assert_pl_equal(get_related_features(restored, "a"), report.get_related("a"))
    assert_pl_equal(
        restored.get_table("correlation_decisions"), report.get_table("correlation_decisions")
    )
    assert "omitted features: 1" in show_correlation_matrix(restored, max_features=2).to_html()
    assert len(calls) == 1
    assert selector._corr_matrix is None


def test_linear_latest_fit_failure_or_skip_has_no_old_result() -> None:
    selector = MarsLinearSelector()
    with pytest.raises(ValueError, match="not fitted"):
        selector.get_correlation_report()
    data = pd.DataFrame({"a": [1, 2, 3, 4], "b": [4, 3, 2, 1]})
    selector.fit(data, [0, 0, 1, 1])
    with pytest.raises(ValueError, match="both classes"):
        selector.fit(data, [0, 0, 0, 0])
    with pytest.raises(ValueError, match="not fitted"):
        selector.get_correlation_report()
    selector.enable_corr_filter = False
    selector.fit(data, [0, 0, 1, 1])
    assert (
        selector.get_correlation_report().describe()["parameters"]["status"] == "skipped_disabled"
    )
    assert selector.get_correlation_report().get_table("pairs").height == 0
    selector.enable_corr_filter = True
    selector.fit(data[["a"]], [0, 0, 1, 1])
    report = selector.get_correlation_report()
    assert report.describe()["parameters"]["status"] == "skipped_insufficient_candidates"
    assert pd.isna(report.get_matrix(["a"]).iloc[0, 0])
    with pytest.raises(ValueError, match="candidate"):
        report.get_matrix(["bogus"])


def test_stats_fit_captures_real_woe_rows_and_threshold(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    rng = np.random.default_rng(46)
    x = rng.normal(size=150)
    target: list[int | None] = (rng.random(150) < 1 / (1 + np.exp(-x))).astype(int).tolist()
    target[-12:] = [None] * 12
    data = pl.DataFrame({"a": x, "b": x, "c": rng.normal(size=150), "bad": target})
    captures: list[pl.DataFrame] = []
    original = pl.DataFrame.corr

    def capture(frame: pl.DataFrame, **kwargs: Any) -> pl.DataFrame:
        if all(c.endswith("_woe") for c in frame.columns):
            captures.append(frame.clone())
        return original(frame, **kwargs)

    monkeypatch.setattr(pl.DataFrame, "corr", capture)
    selector = MarsStatsSelector(
        skip_fine_scan=True,
        psi_thr=None,
        rc_thr=None,
        rough_iv_thr=-1,
        rough_lift_thr=0,
        corr_thr=0.9,
        rough_binning_params={"method": "quantile", "n_bins": 3},
        n_jobs=1,
    )
    selector.fit(data, target="bad", features=["a", "b", "c"], white_list=["c"])
    report = selector.get_correlation_report()
    assert len(captures) == 1 and captures[0].height == 138
    expected = np.corrcoef(captures[0].to_numpy(), rowvar=False)
    np.testing.assert_allclose(report.get_matrix().to_numpy(), expected, equal_nan=True)
    parameters = report.describe()["parameters"]
    assert parameters["representation"] == "woe" and parameters["operator"] == ">"
    assert parameters["correlation_row_count"] == 138
    assert report.get_table("pairs").height == 3
    assert report.get_table("correlation_decisions", filters={"action": "protect"})[
        "feature"
    ].to_list() == ["c"]
    assert selector.selected_features_ == ["a", "c"]
    report.save(tmp_path / "woe.marsreport")
    assert_frame_equal(
        get_correlation_matrix(load_report(tmp_path / "woe.marsreport")), report.get_matrix()
    )
    with pytest.raises(ValueError):
        selector.fit(data.with_columns(pl.lit(0).alias("bad")), target="bad")
    with pytest.raises(ValueError, match="not fitted"):
        selector.get_correlation_report()


def test_actual_diagonal_and_nonfinite_values_are_not_fabricated() -> None:
    # 引擎无法区分原因时只保留 unavailable；此测试独立覆盖保存层，不引入筛选逻辑。
    selector = MarsLinearSelector()
    selector._reset_correlation()
    selector._corr_input_features = ["a", "constant", "invalid"]
    selector._corr_candidates = list(selector._corr_input_features)
    selector._corr_matrix = np.array(
        [[1.0, 0.0, -0.5], [0.0, np.nan, np.inf], [-0.5, np.inf, np.nan]]
    )
    selector._corr_parameters = {"representation": "raw", "method": "pearson"}
    selector._finish_correlation()
    selector._is_fitted = True
    report = selector.get_correlation_report()
    pairs = report.get_table("pairs")
    assert pairs["correlation"].to_list() == [0.0, -0.5, None]
    assert pairs["status"].to_list() == ["valid", "valid", "unavailable"]
    assert pd.isna(report.get_matrix().loc["constant", "constant"])


def test_large_pair_context_does_not_embed_entire_enum_catalog() -> None:
    selector = MarsLinearSelector()
    selector._reset_correlation()
    selector._corr_input_features = [f"f{i:04d}" for i in range(500)]
    selector._corr_candidates = list(selector._corr_input_features)
    selector._corr_matrix = np.eye(500)
    selector._corr_parameters = {"representation": "raw", "method": "pearson"}
    selector._finish_correlation()
    selector._is_fitted = True
    report = selector.get_correlation_report()
    context = report.to_ai_context(tables=["pairs"], features="f0000", limit=3, max_chars=8000)
    payload = json.loads(context)
    assert len(context) <= 8000 and payload["evidence"][0]["returned_rows"] == 3
    assert payload["description"]["tables"]["pairs"]["fields"]["feature_a"]["dtype"] == "Enum"
    assert "f0499" not in json.dumps(payload["description"]["tables"])
