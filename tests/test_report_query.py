"""原生查询、展示前缩小范围、AI 序列化与预算契约。"""

from __future__ import annotations

import json
from datetime import date

import pandas as pd
import polars as pl
import pytest

from mars.reporting import MarsBinningReport, MarsProfileReport


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
        None, {"$mars": "float", "value": "nan"}, {"$mars": "float", "value": "inf"}, 0.0,
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
