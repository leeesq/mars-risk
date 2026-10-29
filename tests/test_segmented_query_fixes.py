"""客群趋势状态与直接身份过滤的数值、分页和保存恢复回归。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from mars.analysis import profile_stats
from mars.feature._scopes import resolve_scopes
from mars.reporting import MarsProfileReport, load_report
from mars.reporting._query import ReportFrame
from mars.reporting._scope_view import scope_views


def _records(frame: ReportFrame) -> list[dict[str, Any]]:
    """统一后端的有限结果，保留公共字段身份。"""
    return frame.to_dicts() if isinstance(frame, pl.DataFrame) else frame.to_dict("records")


@pytest.fixture(params=["polars", "pandas"])
def scoped_profile(request: pytest.FixtureRequest) -> MarsProfileReport:
    """用真实画像同时生成有效客群和缺基准客群。"""
    frame = pl.DataFrame({
        "s": ["seg1"] * 10 + ["seg2"] * 10 + ["All"] * 10,
        "g": ["Train", "Val"] * 15,
        "x": [float(i % 5) for i in range(30)],
        "z": [float(i % 7) for i in range(30)],
    })
    benchmark = frame.filter(pl.col("s") != "seg2")
    return profile_stats(
        frame.to_pandas() if request.param == "pandas" else frame,
        features=["x", "z"], segment_col="s", group_col="g",
        benchmark_df=benchmark.to_pandas() if request.param == "pandas" else benchmark,
        metrics=["mean", "psi"],
    )


@pytest.mark.parametrize("segments,include_all,failed", [
    (["seg1"], False, False), (["seg2"], False, True),
    (None, True, False), (["seg1", "seg2"], False, True),
])
@pytest.mark.parametrize("columns", [None, ["Train", "Val", "total"]])
def test_trend_status_uses_displayed_scopes_and_named_fields(
    scoped_profile: MarsProfileReport,
    segments: list[str] | None,
    include_all: bool,
    failed: bool,
    columns: list[str] | None,
) -> None:
    view = scoped_profile.show_trend(
        "psi", features="x", segments=segments, include_all=include_all, columns=columns,
    )
    assert ("missing_segment_benchmark" in view.caption) is failed
    assert "<caption>" in view.to_html()
    if segments == ["seg2"]:
        assert view.data[["Train", "Val", "total"]].isna().all().all()
    elif segments == ["seg1"]:
        assert view.data[["Train", "Val", "total"]].notna().all().all()


def test_trend_empty_selection_and_limited_rows_have_no_foreign_status(
    scoped_profile: MarsProfileReport,
) -> None:
    report = scoped_profile
    for options in ({"features": []}, {"limit": 0}, {"segments": [], "include_all": False}):
        view = report.show_trend("psi", **options)
        assert view.data.empty
        assert "missing_segment_benchmark" not in view.caption
    # 相同特征出现在两个客群时，页长截断后只解释实际展示的那一行。
    view = report.show_trend(
        "psi", features="x", segments=["seg1", "seg2"], include_all=False,
        limit=1, sort_by="scope_id", sort_ascending=True,
    )
    identity = view.data["scope_id"].iloc[0]
    failed_identity = resolve_scopes(report._scopes, ["seg2"], False)[0]
    assert ("missing_segment_benchmark" in view.caption) is (identity == failed_identity)
    assert "missing_segment_benchmark" not in report.show_trend("mean").caption


@pytest.mark.parametrize("segments,include_all,condition,expected", [
    (None, None, "absent_filter", ["all", "seg1", "seg2", "All"]),
    (None, None, "seg2", ["seg2"]),
    (None, True, "absent_filter", ["all"]),
    (None, True, "seg2", ["seg2"]),
    (None, False, "absent_filter", []),
    (None, False, "seg2", ["seg2"]),
    (None, False, "all", []),
    (None, False, ["all", "seg2"], ["seg2"]),
    (["seg1"], False, "seg2", []),
    (["seg1"], False, "all", []),
    (["seg1"], True, ["all", "seg1", "seg2"], ["all", "seg1"]),
    (["seg1"], None, ["all", "seg1", "seg2"], ["all", "seg1"]),
    ([], False, "all", []),
    ([], False, "seg2", []),
    ([], False, ["all", "seg2"], []),
    ([], True, ["all", "seg1", "seg2"], ["all"]),
    ([], None, ["all", "seg1", "seg2"], ["all"]),
    ([], True, "seg2", []),
    (None, False, None, []),
    (None, True, None, []),
    (None, None, None, []),
    (None, True, "unknown", []),
    (None, None, [], []),
    (None, False, ["all", "All"], ["All"]),
])
def test_scope_filter_intersects_explicit_exclusions_before_page_and_replay(
    scoped_profile: MarsProfileReport,
    tmp_path: Path,
    segments: list[str] | None,
    include_all: bool | None,
    condition: str | list[str] | None,
    expected: list[str],
) -> None:
    report = scoped_profile
    identities = {name: resolve_scopes(report._scopes, [name], False)[0]
                  for name in ("seg1", "seg2", "All")}
    identities["all"] = "all"
    if condition == "absent_filter":
        filters: dict[str, Any] = {}
    elif isinstance(condition, list):
        filters = {"scope_id": {"op": "in", "value": [identities[item] for item in condition]}}
    else:
        filters = {"scope_id": identities.get(condition, condition)}
    options: dict[str, Any] = {
        "features": "x", "segments": segments, "include_all": include_all,
        "filters": filters, "columns": ["feature", "mean"], "sort_by": "scope_id",
    }
    wanted = {identities[name] for name in expected}
    report.save(tmp_path / "query.marsreport")
    for consumer in (report, load_report(tmp_path / "query.marsreport")):
        selected = _records(consumer.get_table("overview", **options))
        assert {row["scope_id"] for row in selected} == wanted
        page = consumer.query_page("overview", **options, limit=1)
        assert page["total_rows"] == len(wanted)
        assert page["returned_rows"] == min(len(wanted), 1)
        assert page["next_offset"] == (1 if len(wanted) > 1 else None)
        assert page["omitted_rows"] == max(len(wanted) - 1, 0)
        assert page["omission_reason"] == ("pagination" if len(wanted) > 1 else None)
        reference = page["reference"]
        assert _records(consumer.get_table(reference["table"], **reference["query"])) == _records(page["data"])
        context = json.loads(consumer.to_ai_context(
            tables=["overview"], queries={"overview": {**options, "limit": 1}}, max_chars=30000,
        ))
        evidence = context["evidence"][0]
        assert evidence["total_rows"] == len(wanted)
        assert evidence["returned_rows"] == min(len(wanted), 1)
        assert evidence["rows"] == _records(page["data"])
        assert _records(consumer.get_table(evidence["reference"], **evidence["query"])) == evidence["rows"]
        if not wanted:
            assert evidence["next_offset"] is None
            assert context["omitted"] == []


def test_single_scope_default_and_explicit_identity_remain_independent(
    scoped_profile: MarsProfileReport,
) -> None:
    report = scoped_profile
    identity = resolve_scopes(report._scopes, ["seg2"], False)[0]
    report.report_meta["default_scope_id"] = identity
    assert set(row["scope_id"] for row in _records(report.get_table("overview", include_all=True))) == {identity}
    assert set(row["scope_id"] for row in _records(report.get_table(
        "overview", include_all=True, filters={"scope_id": "all"},
    ))) == {"all"}
    assert not len(report.get_table("overview", include_all=False, filters={"scope_id": "all"}))


def test_saved_profile_native_scope_view_keeps_only_own_status(
    scoped_profile: MarsProfileReport, tmp_path: Path,
) -> None:
    path = tmp_path / "states.marsreport"
    scoped_profile.save(path)
    restored = load_report(path)
    # 快照恢复已保存的统计表，原生 adapter 由 scope_views 构造，不访问原数据。
    for name, failed in (("seg1", False), ("seg2", True)):
        view = scope_views(restored, [name], False)[0][1]
        caption = view.show_trend("psi", features="x").caption
        assert ("missing_segment_benchmark" in caption) is failed
        state = restored.get_table("calculation_status", segments=[name], include_all=False)
        assert bool(len(state)) is failed
