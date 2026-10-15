"""原生查询、展示前缩小范围、AI 序列化与预算契约。"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd
import polars as pl
import pytest

from mars.reporting import MarsBinningReport, MarsProfileReport, Report, ReportSnapshot, load_report
from mars.reporting._query import ReportFrame


@pytest.fixture(params=["polars", "pandas"])
def report(request: pytest.FixtureRequest) -> MarsBinningReport:
    summary = pl.DataFrame(
        {
            "feature": ["a", "b", "c"],
            "iv": [0.1, 0.3, None],
            "ks": [10.0, 20.0, None],
            "data_source": ["base", "other", "base"],
        }
    )
    detail = pl.DataFrame(
        {
            "feature": ["a", "a", "b"],
            "bin_index": [-1, 0, 0],
            "count": [0.0, 2.0, 3.0],
            "day": [date(2026, 1, 1)] * 3,
        }
    )
    trend = pl.DataFrame({"feature": ["a", "b", "c"], "Total": [0.1, 0.3, None]})
    if request.param == "pandas":
        summary, detail, trend = summary.to_pandas(), detail.to_pandas(), trend.to_pandas()
    return MarsBinningReport(
        summary,
        {"iv": trend},
        detail,
        feature_data_source={"a": "base", "b": "other", "c": "base"},
        report_meta={"weights_col": None, "has_target": True},
    )


def records(frame: pl.DataFrame | pd.DataFrame) -> list[dict]:
    return frame.to_dicts() if isinstance(frame, pl.DataFrame) else frame.to_dict("records")


def _relation_snapshot(backend: str, same_column: bool) -> ReportSnapshot:
    """创建含重复统计身份的桥接快照，验证成员关系不会展开指标行。"""
    identity = "feature" if same_column else "member_id"
    ids = ["x", "x", "y", "z"] if same_column else ["k-x", "k-x", "k-y", "k-z"]
    overview: ReportFrame = pl.DataFrame(
        {identity: ids, "mean": [2.0, 3.0, 4.0, 5.0], "detail": ["证据" * 800] * 4}
    )
    members: ReportFrame = pl.DataFrame({"feature": ["x", "x", "y", "z", "unused"]})
    if not same_column:
        members = members.with_columns(
            pl.Series("member_id", ["k-x", "k-x", "k-y", "k-z", "k-unused"])
        )
    if backend == "pandas":
        overview, members = overview.to_pandas(), members.to_pandas()
    profile = MarsProfileReport(
        overview,
        {},
        {"members": members},
        feature_metadata={feature: {"data_source": "bank"} for feature in ["x", "y", "z"]},
    )
    description = profile.describe()
    description["tables"]["overview"]["feature_relation"] = {
        "table": "stats.members",
        "key": "feature" if same_column else "member_id",
        "feature": "feature",
        "roles": {identity: "member"},
    }
    return ReportSnapshot({"overview": overview, "stats.members": members}, description)


@pytest.mark.parametrize("backend", ["polars", "pandas"])
@pytest.mark.parametrize("same_column", [True, False], ids=["same_column", "different_columns"])
def test_snapshot_relation_preserves_rows_page_members_and_replay(
    tmp_path: Path, backend: str, same_column: bool
) -> None:
    original = _relation_snapshot(backend, same_column)
    path = tmp_path / "relation.marsreport"
    original.save(path)
    restored = load_report(path)
    assert restored.describe() == original.describe()
    for report in (original, restored):
        selected = report.get_table("overview", features="x", sources="bank", columns=["mean"])
        assert records(selected) == [{"mean": 2.0}, {"mean": 3.0}]
        assert type(selected) is type(original.get_table("overview"))
        page = report.query_page("overview", features="x", columns=["mean"], limit=1)
        assert page["total_rows"] == 2 and page["returned_rows"] == 1
        assert page["next_offset"] == 1 and page["omitted_rows"] == 1
        assert records(report.get_table(page["reference"]["table"], **page["reference"]["query"])) == [
            {"mean": 2.0}
        ]
        # 投影省略关系身份时仍只关联当前有限页的成员，不带入后续 y/z 或 unused。
        context = report.to_ai_context(tables=["overview"], columns=["mean"], limit=1)
        payload = json.loads(context)
        evidence = payload["evidence"][0]
        assert evidence["rows"] == [{"mean": 2.0}]
        assert evidence["total_rows"] == 4 and evidence["returned_rows"] == 1
        assert evidence["report_id"] == report.report_id
        assert evidence["next_offset"] == 1 and evidence["truncated"] is True
        assert set(payload["description"]["feature_metadata"]) == {"x"}
        assert records(report.get_table(evidence["reference"], **evidence["query"])) == evidence["rows"]
        next_query = {**evidence["query"], "offset": evidence["next_offset"]}
        assert records(report.get_table(evidence["reference"], **next_query)) == [{"mean": 3.0}]
        assert json.loads(report.to_ai_context(tables=["overview"], features="x"))["evidence"][0][
            "returned_rows"
        ] == 2


@pytest.mark.parametrize("backend", ["polars", "pandas"])
@pytest.mark.parametrize("same_column", [True, False], ids=["same_column", "different_columns"])
def test_snapshot_relation_budget_keeps_only_retained_evidence_members(
    tmp_path: Path, backend: str, same_column: bool
) -> None:
    original = _relation_snapshot(backend, same_column)
    path = tmp_path / "relation-budget.marsreport"
    original.save(path)
    for report in (original, load_report(path)):
        context = report.to_ai_context(
            tables=["overview"], columns=["mean", "detail"], limit=4, max_chars=5500
        )
        assert len(context) <= 5500
        payload = json.loads(context)
        evidence = payload["evidence"][0]
        assert 0 < evidence["returned_rows"] < 4
        assert evidence["next_offset"] == evidence["returned_rows"]
        assert evidence["truncated"] is True
        expected = {2.0: "x", 3.0: "x", 4.0: "y", 5.0: "z"}
        assert set(payload["description"]["feature_metadata"]) == {
            expected[row["mean"]] for row in evidence["rows"]
        }
        replay = report.get_table(
            evidence["reference"], **{**evidence["query"], "limit": evidence["returned_rows"]}
        )
        assert records(replay) == evidence["rows"]
        omitted = [item for item in payload["omitted"] if item.get("reason") == "output_budget"]
        assert sum(item.get("rows", 0) for item in omitted) == 4 - evidence["returned_rows"]
        assert omitted[0]["continue_at"]["offset"] == evidence["next_offset"]


def test_query_preserves_type_filters_before_projection_and_does_not_mutate(
    report: MarsBinningReport,
) -> None:
    result = report.get_table(
        "summary",
        features=["a", "b"],
        filters={"iv": {"op": "gt", "value": 0.05}},
        sort_by="iv",
        descending=True,
        columns=["feature"],
        offset=1,
        limit=1,
    )
    assert type(result) is type(report.summary_table)
    assert records(result) == [{"feature": "a"}]
    assert len(report.summary_table) == 3
    if isinstance(result, pd.DataFrame):
        result.iloc[0, 0] = "mutated"
    else:
        result.replace_column(0, pl.Series("feature", ["mutated"]))
    assert records(report.get_table("summary", features="a", columns=["feature"])) == [
        {"feature": "a"}
    ]
    assert records(report.get_table("summary", features="a", filters={"feature": "b"})) == []
    assert records(report.get_table("summary", sources="other", columns=["feature"])) == [
        {"feature": "b"}
    ]


def test_query_pages_advance_through_filtered_results_and_finish(
    report: MarsBinningReport, tmp_path: Path
) -> None:
    frame: ReportFrame = pl.DataFrame(
        {
            "feature": ["a", "b", "a", "c", "a", "b", "a"],
            "mean": [1.0, 10.0, 3.0, 9.0, -1.0, 8.0, 2.0],
        }
    )
    if isinstance(report.summary_table, pd.DataFrame):
        frame = frame.to_pandas()
    original = MarsProfileReport(
        frame,
        {},
        {},
        feature_metadata={
            "a": {"data_source": "base"},
            "b": {"data_source": "other"},
            "c": {"data_source": "base"},
        },
    )
    path = tmp_path / "pages.marsreport"
    original.save(path)
    for saved in (original, load_report(path)):
        options = {
            "features": ["a", "b"],
            "sources": "base",
            "filters": {"mean": {"op": "ge", "value": 0}},
            "sort_by": "mean",
            "descending": True,
            "columns": ["mean", "feature"],
        }
        offset: int | None = 0
        rows: list[dict] = []
        pages: list[dict] = []
        while offset is not None:
            page = saved.query_page("overview", **options, offset=offset, limit=1)
            assert page["total_rows"] == 3
            assert page["returned_rows"] == 1
            assert page["reference"]["report_id"] == original.report_id
            assert list(page["data"].columns) == ["mean", "feature"]
            assert records(saved.get_table("overview", **page["reference"]["query"])) == records(
                page["data"]
            )
            rows.extend(records(page["data"]))
            pages.append(page)
            assert len(pages) <= 3, "pagination must terminate after the finite matching rows"
            next_offset = page["next_offset"]
            assert next_offset is None or next_offset > offset
            offset = next_offset
        assert rows == [{"mean": value, "feature": "a"} for value in (3.0, 2.0, 1.0)]
        assert pages[0]["next_offset"] == 1 and pages[0]["omitted_rows"] == 2
        assert pages[-1]["next_offset"] is None and pages[-1]["truncated"] is False
        assert pages[-1]["omitted_rows"] == 0
        default = saved.query_page("overview", **options)
        assert default["reference"]["query"]["limit"] == 10
        assert default["returned_rows"] == 3 and default["next_offset"] is None
        all_remaining = saved.query_page("overview", **options, offset=1, limit=None)
        assert records(all_remaining["data"]) == rows[1:]
        assert all_remaining["next_offset"] is None and all_remaining["truncated"] is False
        for start in (3, 20):
            ended = saved.query_page("overview", **options, offset=start, limit=1)
            assert ended["returned_rows"] == 0 and ended["total_rows"] == 3
            assert ended["next_offset"] is None and ended["truncated"] is False
        empty = saved.query_page("overview", filters={"feature": "unknown"}, limit=1)
        assert empty["returned_rows"] == empty["total_rows"] == empty["omitted_rows"] == 0
        assert empty["next_offset"] is None and empty["truncated"] is False


@pytest.mark.parametrize("invalid", [0, -1, True, False, 1.0, 0.5, "1"])
def test_query_page_rejects_nonpositive_or_noninteger_limits(
    report: MarsBinningReport, invalid: object, tmp_path: Path
) -> None:
    path = tmp_path / "invalid-pages.marsreport"
    report.save(path)
    for saved in (report, load_report(path)):
        with pytest.raises(ValueError, match="query_page limit.*positive integer or None"):
            saved.query_page("summary", limit=invalid)


def _assert_zero_row_context_replays_and_continues(
    report: Report, context: str, table: str, expected_remaining: int
) -> None:
    """零行证据与继续取证分别回放，防止定位重新形成零页长循环。"""
    payload = json.loads(context)
    evidence = next(item for item in payload["evidence"] if item["reference"] == table)
    query = evidence["query"]
    assert query["limit"] == 0
    assert evidence["returned_rows"] == 0 and evidence["rows"] == []
    assert evidence["next_offset"] is None and evidence["truncated"] is True
    assert evidence["total_rows"] == query["offset"] + expected_remaining
    assert records(report.get_table(table, **query)) == evidence["rows"]
    assert list(report.get_table(table, **query).columns) == list(
        payload["description"]["tables"][table]["fields"]
    )
    with pytest.raises(ValueError, match="positive integer"):
        report.query_page(table, **query)
    omission = next(item for item in payload["omitted"] if item["reference"] == table)
    assert omission["rows"] == expected_remaining and omission["reason"] == "row_limit"
    assert omission["next_offset"] is None
    continuation = omission["continue_at"]
    assert continuation == {**query, "limit": 10}
    page = report.query_page(table, **continuation)
    assert page["returned_rows"] == expected_remaining
    assert page["next_offset"] is None
    assert evidence["report_id"] == omission["report_id"] == report.report_id
    assert len(context) <= payload["budget"]["max_chars"]


def test_zero_row_table_feature_and_ai_queries_keep_evidence_and_positive_continuation(
    report: MarsBinningReport, tmp_path: Path
) -> None:
    path = tmp_path / "zero-row-evidence.marsreport"
    report.save(path)
    for saved in (report, load_report(path)):
        before = {name: saved.get_table(name) for name in saved.describe()["tables"]}
        selected = saved.get_table("summary", columns=["ks", "feature"], limit=0)
        assert len(selected) == 0 and list(selected.columns) == ["ks", "feature"]
        feature = saved.get_feature("a", limit=0)
        assert feature["evidence_status"] == "available"
        assert feature["omitted_rows"] == {"summary": 1, "detail": 2, "trend.iv": 1}
        for name, frame in feature["tables"].items():
            assert len(frame) == 0
            assert list(frame.columns) == list(saved.get_table(name).columns)
            assert type(frame) is type(saved.get_table(name))
        _assert_zero_row_context_replays_and_continues(
            saved,
            saved.to_ai_context(
                tables=["summary"], columns=["ks", "feature"], limit=0, max_chars=16000
            ),
            "summary",
            3,
        )
        options = {
            "detail": {
                "features": "a",
                "sources": "base",
                "filters": {"count": {"op": "ge", "value": 0}},
                "sort_by": "count",
                "descending": True,
                "columns": ["count", "feature"],
                "offset": 1,
                "limit": 0,
            },
            "summary": {"columns": ["feature"], "limit": 1},
        }
        # 顶层和每表覆盖均保留零行请求；另一表的正页长覆盖仍可返回证据。
        for top_limit in (0, 2):
            context = saved.to_ai_context(queries=options, limit=top_limit, max_chars=16000)
            _assert_zero_row_context_replays_and_continues(saved, context, "detail", 1)
            summary = json.loads(context)["evidence"][1]
            assert summary["returned_rows"] == 1 and summary["query"]["limit"] == 1
        for name, frame in before.items():
            assert saved.get_table(name).equals(frame)


def test_empty_table_has_no_continuation_in_pages_or_zero_row_context(
    report: MarsBinningReport, tmp_path: Path
) -> None:
    empty = MarsProfileReport(report.get_table("summary", limit=0), {}, {})
    path = tmp_path / "empty-page.marsreport"
    empty.save(path)
    for saved in (empty, load_report(path)):
        for limit in (1, None):
            page = saved.query_page("overview", limit=limit)
            assert page["returned_rows"] == page["total_rows"] == page["omitted_rows"] == 0
            assert page["next_offset"] is None and page["truncated"] is False
        payload = json.loads(saved.to_ai_context(limit=0))
        evidence = payload["evidence"][0]
        assert evidence["query"]["limit"] == 0 and evidence["rows"] == []
        assert evidence["total_rows"] == 0 and evidence["next_offset"] is None
        assert evidence["truncated"] is False and payload["omitted"] == []


@pytest.mark.parametrize(
    "options",
    [
        {"name": "bad"},
        {"columns": ["bad"]},
        {"sort_by": "bad"},
        {"offset": -1},
        {"limit": True},
        {"descending": 1},
        {"filters": {"iv": {"op": "eval", "value": "x"}}},
        {"filters": {"iv": {"op": "in", "value": "bad"}}},
        {"sources": "bad"},
    ],
)
def test_invalid_queries_have_explicit_errors(report: MarsBinningReport, options: dict) -> None:
    with pytest.raises(ValueError):
        report.get_table(**{"name": "summary", **options})


def test_describe_definitions_and_feature_navigation(report: MarsBinningReport) -> None:
    description = report.describe()
    assert description["tables"]["summary"]["fields"]["ks"]["unit"] == "points_0_100"
    assert description["tables"]["summary"]["fields"]["iv"]["unit"] == "dimensionless"
    assert description["business_context"]["currency"] == "unknown"
    description["parameters"]["has_target"] = False
    assert report.report_meta["has_target"] is True
    selected = report.get_feature("a", limit=1)
    assert len(selected["tables"]["detail"]) == 1
    assert selected["omitted_rows"]["detail"] == 1
    assert "overview" in selected["unavailable"]
    with pytest.raises(ValueError, match="Unknown feature"):
        report.get_feature("unknown")


def test_ai_serializes_date_null_zero_and_nonfinite_with_evidence(
    report: MarsBinningReport,
) -> None:
    payload = json.loads(
        report.to_ai_context(tables=["detail"], features="a", columns=["feature", "count", "day"])
    )
    evidence = payload["evidence"][0]
    assert evidence["reference"] == "detail"
    assert evidence["query"]["features"] == "a"
    assert evidence["rows"][0]["count"] == 0.0
    assert evidence["rows"][0]["day"] in {"2026-01-01", "2026-01-01T00:00:00"}
    special = MarsProfileReport(
        pl.DataFrame(
            {"feature": ["x", "y", "z", "w"], "mean": [None, float("nan"), float("inf"), 0.0]}
        ),
        {},
        {},
    )
    rows = json.loads(special.to_ai_context())["evidence"][0]["rows"]
    assert [row["mean"] for row in rows] == [
        None,
        {"$mars": "float", "value": "nan"},
        {"$mars": "float", "value": "inf"},
        0.0,
    ]


def test_ai_budget_omits_complete_rows_and_parameter_values() -> None:
    report = MarsProfileReport(
        pl.DataFrame({"feature": [f"x{i}" for i in range(100)], "mode_value": ["a" * 400] * 100}),
        {},
        {},
        report_meta={"features": ["b" * 50] * 100},
    )
    context = report.to_ai_context(limit=100, max_chars=3000)
    assert len(context) <= 3000
    payload = json.loads(context)
    assert payload["omitted"]
    evidence = payload["evidence"][0]
    assert len(evidence["rows"]) == evidence["returned_rows"] < 100
    assert (
        sum(item.get("rows", 0) for item in payload["omitted"]) == 100 - evidence["returned_rows"]
    )
    with pytest.raises(ValueError, match="budget|description|max_chars"):
        report.to_ai_context(max_chars=512)


def test_profile_navigation_recognizes_quality_trends() -> None:
    report = MarsProfileReport(
        pl.DataFrame({"feature": ["x"], "missing_rate": [0.0]}),
        {"missing": pl.DataFrame({"feature": ["x"], "total": [0.0]})},
        {},
    )
    assert "trend" not in report.get_feature("x")["unavailable"]


def test_context_keeps_last_evidence_with_scope_or_rejects_budget() -> None:
    report = MarsProfileReport(
        pl.DataFrame({"feature": ["x"], "mean": [1.5]}),
        {},
        {},
        feature_metadata={"x": {"data_source": "bank", "description": "业务定义" * 100}},
        report_meta={"weights_col": "weight", "group_col": "cohort"},
        business_context={"labels": {"bad": {"definition": "逾期", "performance_window": "90天"}}},
    )
    for budget in (2500, 3000, 5000):
        try:
            context = report.to_ai_context(max_chars=budget)
        except ValueError as exc:
            assert "max_chars" in str(exc)
        else:
            payload = json.loads(context)
            assert payload["evidence"][0]["rows"] == [{
                "feature": "x", "display_name": None, "dtype": None, "data_source": "bank", "mean": 1.5,
            }]
            assert payload["description"]["parameters"]["weights_col"] == "weight"
            assert (
                payload["description"]["business_context"]["labels"]["bad"]["definition"] == "逾期"
            )
            assert len(context) <= budget


def test_display_uses_small_native_query_and_projection(
    report: MarsBinningReport, monkeypatch: pytest.MonkeyPatch
) -> None:
    import mars.reporting.binning_report as module

    original = module.to_pandas_frame
    sizes: list[tuple[int, list[str]]] = []

    def capture(frame: object) -> pd.DataFrame:
        sizes.append((len(frame), list(frame.columns)))
        return original(frame)

    monkeypatch.setattr(module, "to_pandas_frame", capture)
    styler = report.show_summary(sort_by="iv", columns=["feature", "iv"], limit=1)
    assert styler.data["feature"].tolist() == ["b"]
    assert sizes == [(1, ["feature", "iv"])]
    assert report.show_trend("iv", limit=1, columns=["Total"]).data.shape == (1, 1)
