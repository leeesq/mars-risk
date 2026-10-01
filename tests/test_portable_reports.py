"""公共业务语义、可携带报告及外部消费者回归。"""

from __future__ import annotations

import json
import math
import sys
import zipfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import polars as pl
import pytest
from pandas.testing import assert_frame_equal
from polars.testing import assert_frame_equal as assert_polars_equal

from mars.analysis import MarsBinEvaluator, MarsDataProfiler, profile_risk, profile_stats
from mars.feature import MarsStatsSelector
from mars.reporting import (
    MarsBinningReport,
    MarsProfileReport,
    Report,
    load_report,
    snapshot_report,
)

METADATA = {
    "income": {
        "display_name": "月收入",
        "description": "申请时申报的税后月收入",
        "data_source": "application",
        "unit": "CNY/month",
    },
    "salary": {
        "display_name": "月收入",
        "description": "授权流水入账金额",
        "data_source": "bank",
        "unit": "CNY/month",
    },
    "unused": {"display_name": "未分析字段"},
}
CONTEXT = {
    "labels": {
        "y": {
            "definition": "放款后首次还款逾期",
            "positive_class": "逾期",
            "negative_class": "履约",
            "performance_window": "放款后30天",
        },
        "later": {"definition": "第二次还款逾期", "performance_window": "放款后60天"},
    },
    "sample": {
        "scope": "模拟获批申请",
        "filter": "示例全样本",
        "time_range": ["2026-01-01", "2026-02-01"],
    },
    "splits": {"train": "首月开发", "oot": "次月观察"},
    "currency": "CNY",
    "score_direction": {"income": "unknown"},
}


def sample() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "income": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            "salary": [8.0, 6.0, 4.0, 2.0, 7.0, 5.0, 3.0, 1.0],
            "y": [0, 1, 0, 1, None, None, None, None],
            "later": [None] * 8,
            "month": ["train"] * 4 + ["oot"] * 4,
        }
    )


@pytest.mark.parametrize("form", ["dict", "pandas", "polars"])
@pytest.mark.parametrize("kind", ["profile", "risk"])
def test_metadata_and_business_context_through_analysis_query_and_exports(
    form: str,
    kind: str,
    tmp_path: Path,
) -> None:
    rows = [{"feature": key, **value} for key, value in METADATA.items()]
    dictionary = (
        METADATA
        if form == "dict"
        else pd.DataFrame(rows)
        if form == "pandas"
        else pl.DataFrame(rows)
    )
    data = sample().to_pandas() if form == "pandas" else sample()
    if kind == "profile":
        report = profile_stats(
            data,
            features=["income", "salary"],
            metrics=["missing", "mean"],
            feature_metadata=dictionary,
            business_context=CONTEXT,
            group_col="month",
        )
        displayed = report.show_overview(limit=1).data
    else:
        report = profile_risk(
            data,
            features=["income", "salary"],
            target="y",
            feature_metadata=dictionary,
            business_context=CONTEXT,
            group_col="month",
            n_bins=2,
        ).report
        displayed = report.show_summary(limit=1).data
    assert isinstance(report, Report)
    assert "display_name" in displayed.columns and "feature" in displayed.columns
    assert set(report.feature_metadata) == {"income", "salary"}
    assert report.get_feature("income")["metadata"] == METADATA["income"]
    assert {r["feature"] for r in report.search_features("月收入")} == {"income", "salary"}
    assert report.search_features("月收入", sources="bank")[0]["feature"] == "salary"
    assert report.search_features("税后")[0]["feature"] == "income"
    assert len(report.search_features("", limit=1)) == 1
    assert (
        report.describe()["business_context"]["labels"]["y"]["definition"]
        == CONTEXT["labels"]["y"]["definition"]
    )
    path = tmp_path / "analysis.marsreport"
    report.save(path)
    restored = load_report(path)
    assert restored.describe() == report.describe()
    assert restored.get_feature("income")["metadata"] == METADATA["income"]
    html = tmp_path / "report.html"
    if kind == "risk":
        report.write_html(str(html), include_charts=False)
    else:
        report.write_html(str(html))
    assert "月收入" in html.read_text(encoding="utf-8")
    assert "申请时申报" in html.read_text(encoding="utf-8")
    excel = tmp_path / "report.xlsx"
    report.write_excel(str(excel))
    metadata_sheet = pd.read_excel(excel, sheet_name="FeatureMetadata")
    assert metadata_sheet.loc[metadata_sheet.feature == "income", "display_name"].item() == "月收入"
    assert "CNY" in pd.read_excel(excel, sheet_name="BusinessContext").to_string()
    restored.write_html(str(tmp_path / "restored.html"))
    restored.write_excel(str(tmp_path / "restored.xlsx"))


@pytest.mark.parametrize(
    "dictionary",
    [
        {"income": "月收入"},
        {"income": {"display_name": 3}},
        {"income": {"unknown": "x"}},
        pl.DataFrame({"feature": ["income", "income"], "display_name": ["x", "x"]}),
        pd.DataFrame({"display_name": ["x"]}),
        1,
    ],
)
def test_invalid_feature_dictionaries_are_explicit(dictionary: object) -> None:
    with pytest.raises(ValueError, match="feature_metadata"):
        profile_stats(sample(), features=["income"], metrics=["mean"], feature_metadata=dictionary)


@pytest.mark.parametrize("sources", [{"a": ["income"], "b": ["income"]}, {"bank": ["income"]}])
def test_source_conflicts_are_rejected(sources: dict[str, list[str]]) -> None:
    with pytest.raises(ValueError, match="Conflicting data_source"):
        profile_risk(
            sample(),
            target="y",
            features=["income"],
            feature_data_source=sources,
            feature_metadata={"income": METADATA["income"]},
        )


def test_legacy_sources_merge_missing_semantics_and_dictionary_clipping() -> None:
    report = (
        MarsBinEvaluator()
        .evaluate(
            sample(),
            features=["income"],
            target="y",
            feature_metadata={"income": {"display_name": "收入"}, "unused": {}},
            feature_data_source={"application": ["income"]},
        )
        .report
    )
    assert report.feature_metadata == {
        "income": {"display_name": "收入", "data_source": "application"}
    }
    unknown = MarsDataProfiler().generate_profile(sample(), features=["income"], metrics=["mean"])
    assert unknown.get_feature("income")["metadata"] == {}
    assert unknown.describe()["business_context"]["currency"] == "unknown"
    assert "display_name" not in unknown.overview_table.columns
    with pytest.raises(ValueError, match="Unknown feature"):
        unknown.get_feature("月收入")


@pytest.mark.parametrize(
    "context", [{"labels": {"y": "bad"}}, {"sample": object()}, {"sample": (1, 2)}, {1: "x"}]
)
def test_context_rejects_unsupported_objects(context: dict) -> None:
    with pytest.raises(ValueError):
        profile_stats(sample(), features=["income"], metrics=["mean"], business_context=context)


@pytest.mark.parametrize("backend", ["pandas", "polars"])
def test_full_artifact_preserves_schema_order_values_and_replay(
    backend: str, tmp_path: Path
) -> None:
    frame = pl.DataFrame(
        {
            "feature": ["income"] * 6,
            "value": [None, float("nan"), float("inf"), float("-inf"), 0.0, 2.0],
            "text": ["NaN", "Infinity", "-Infinity", "0", "正常", None],
            "day": [date(2026, 1, 1)] * 6,
            "at": [datetime(2026, 1, 1, tzinfo=timezone.utc)] * 6,
            "category": pl.Series(["a", "b", "a", "b", "a", None], dtype=pl.Categorical),
        }
    )
    if backend == "pandas":
        frame = frame.to_pandas()
        frame.index = pd.MultiIndex.from_arrays(
            [["a"] * 6, [8, 6, 4, 2, 0, -2]], names=["batch", "position"]
        )
    report = MarsProfileReport(
        frame,
        {},
        {},
        report_meta={"nan": float("nan"), "inf": float("inf"), "zero": 0, "date": date(2026, 1, 1)},
        feature_metadata=METADATA,
    )
    query = dict(
        features="income",
        filters={"value": {"op": "ge", "value": 0.0}},
        sort_by="value",
        descending=True,
        columns=["feature", "value", "text"],
        offset=1,
        limit=1,
    )
    page = report.query_page("overview", **query)
    path = tmp_path / "test.marsreport"
    report.save(path)
    del frame
    restored = load_report(path)
    assert restored.report_id == report.report_id
    if backend == "pandas":
        assert_frame_equal(restored.get_table("overview"), report.get_table("overview"))
        assert_frame_equal(restored.get_table("overview", **query), page["data"])
    else:
        assert_polars_equal(restored.get_table("overview"), report.get_table("overview"))
        assert_polars_equal(restored.get_table("overview", **query), page["data"])
    assert restored.query_page("overview", **query)["reference"] == page["reference"]
    payload = json.loads(restored.to_ai_context())
    rows = payload["evidence"][0]["rows"]
    assert rows[1]["value"] == {"$mars": "float", "value": "nan"}
    assert rows[2]["value"] == {"$mars": "float", "value": "inf"}
    assert rows[3]["value"] == {"$mars": "float", "value": "-inf"}
    assert rows[4]["value"] == 0
    assert rows[0]["text"] == "NaN" and rows[0]["day"] in {"2026-01-01", "2026-01-01T00:00:00"}
    if backend == "polars":
        assert rows[0]["value"] is None
    assert restored.describe()["parameters"]["nan"] == {"$mars": "float", "value": "nan"}
    with zipfile.ZipFile(path) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        assert manifest["format_version"] == 1
        assert (
            manifest["tables"]["overview"]["columns"]
            == report.get_table("overview").columns.tolist()
            if backend == "pandas"
            else manifest["tables"]["overview"]["columns"] == report.get_table("overview").columns
        )
        assert not any(name.endswith(".pkl") for name in archive.namelist())


def test_atomic_write_overwrite_corruption_and_unsupported_versions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = profile_stats(sample(), features=["income"], metrics=["mean"])
    path = tmp_path / "test.marsreport"
    report.save(path)
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        report.save(path)
    report.save(path, overwrite=True)
    assert load_report(path).report_id == report.report_id
    original_get = report.get_table

    def failed(name: str, **options: object) -> pl.DataFrame:
        raise ValueError("simulated writer failure")

    monkeypatch.setattr(report, "get_table", failed)
    with pytest.raises(ValueError, match="simulated"):
        report.save(path, overwrite=True)
    assert path.read_bytes() == original
    assert not list(tmp_path.glob(".marsreport-*"))
    monkeypatch.setattr(report, "get_table", original_get)
    for version, tables in [(2, True), (1, False)]:
        with zipfile.ZipFile(path) as source:
            manifest = json.loads(source.read("manifest.json"))
            manifest["format_version"] = version
            data = {
                name: source.read(name)
                for name in source.namelist()
                if tables and name != "manifest.json"
            }
        broken = tmp_path / f"broken-{version}.marsreport"
        with zipfile.ZipFile(broken, "w") as archive:
            archive.writestr("manifest.json", json.dumps(manifest))
            for name, value in data.items():
                archive.writestr(name, value)
        with pytest.raises(ValueError, match="format|version|incomplete"):
            load_report(broken)
    malformed = tmp_path / "invalid-json.marsreport"
    with zipfile.ZipFile(malformed, "w") as archive:
        archive.writestr("manifest.json", '{"format":"marsreport","format_version":NaN}')
    with pytest.raises(ValueError, match="Invalid marsreport JSON"):
        load_report(malformed)


@pytest.mark.parametrize(
    "values,expected",
    [([1.0] * 8, "insufficient_normal_bins"), (list(range(8)), "constant_or_missing_bad_rate")],
)
def test_monotonicity_undefined_is_not_perfect(values: list[float], expected: str) -> None:
    data = pl.DataFrame({"x": values, "y": [0, 1] * 4})
    report = profile_risk(data, features=["x"], target="y", n_bins=2, method="uniform").report
    assert report.get_table("summary")["mono"].item() is None
    state = report.get_table("calculation_status", filters={"metric": "mono"}).row(0, named=True)
    assert state["status"] == "undefined" and state["reason"] == expected


def test_multilabel_unobserved_context_status_and_no_label_mode(tmp_path: Path) -> None:
    report = profile_risk(
        sample(),
        features=["income"],
        target=["y", "later"],
        group_col="month",
        n_bins=2,
        feature_metadata=METADATA,
        business_context=CONTEXT,
    ).report
    state = report.get_table(
        "calculation_status", filters={"target": "later", "metric": "risk_metrics"}
    )
    assert state["status"].unique().to_list() == ["unobserved"]
    assert (
        report.describe()["business_context"]["labels"]["y"]
        != report.describe()["business_context"]["labels"]["later"]
    )
    report.save(tmp_path / "labels.marsreport")
    assert_polars_equal(
        load_report(tmp_path / "labels.marsreport").get_table("calculation_status"),
        report.get_table("calculation_status"),
    )
    unlabelled = profile_risk(sample(), features=["income"], target=None).report
    assert unlabelled.get_table("calculation_status")["status"].unique().to_list() == [
        "not_computed"
    ]
    assert unlabelled.get_table("summary")["mono"].item() is None


def test_wide_trend_budget_heterogeneous_queries_and_persistent_evidence(tmp_path: Path) -> None:
    dates = [(date(2026, 1, 1) + timedelta(days=i)).isoformat() for i in range(365)]
    trend = pl.DataFrame({"feature": ["income"], **{day: [i / 365] for i, day in enumerate(dates)}})
    report = MarsBinningReport(
        pl.DataFrame({"feature": ["income"], "iv": [0.2]}),
        {"ks": trend},
        pl.DataFrame({"feature": ["income"] * 2, "bin_index": [0, 1], "count": [10.0, 20.0]}),
        feature_metadata=METADATA,
    )
    queries = {
        "trend.ks": {"features": "income"},
        "detail": {
            "columns": ["feature", "bin_index", "count"],
            "sort_by": "count",
            "descending": True,
            "limit": 1,
        },
    }
    context = report.to_ai_context(queries=queries)
    payload = json.loads(context)
    assert len(context) <= 16000
    assert payload["evidence"][0]["returned_rows"] == 1
    assert len(payload["evidence"][0]["rows"][0]) == 366
    assert "value_definition" in payload["description"]["tables"]["trend.ks"]
    short = json.loads(report.to_ai_context(queries=queries, max_chars=5000))
    assert short["evidence"][0]["rows"] and short["omitted"]
    assert len(report.to_ai_context(queries=queries, max_chars=5000)) <= 5000
    report.save(tmp_path / "wide.marsreport")
    restored = load_report(tmp_path / "wide.marsreport")
    for evidence in payload["evidence"]:
        replay = restored.query_page(evidence["reference"], **evidence["query"])
        assert replay["reference"]["report_id"] == evidence["report_id"]
        assert replay["data"].to_dicts() == evidence["rows"]
    assert "unused" not in short["description"]["feature_metadata"]
    with pytest.raises(ValueError, match="budget|description|max_chars"):
        report.to_ai_context(queries=queries, max_chars=512)


def test_selector_carries_semantics_into_final_report() -> None:
    data = pl.DataFrame({"income": list(range(40)), "y": [0, 1] * 20})
    selector = MarsStatsSelector(
        skip_rough_scan=True,
        iv_thr=0,
        psi_thr=None,
        rc_thr=None,
        corr_thr=None,
        binning_params={"n_bins": 2},
    )
    selector.fit(
        data,
        target="y",
        features=["income"],
        feature_metadata=METADATA,
        business_context=CONTEXT,
        white_list=["income"],
    )
    report = selector.get_binning_report(data)
    assert report.get_feature("income")["metadata"] == METADATA["income"]
    assert report.describe()["business_context"]["currency"] == "CNY"


@pytest.mark.skipif(sys.version_info < (3, 10), reason="mars.agent requires Python >= 3.10")
def test_snapshot_registration_and_agent_share_public_queries(tmp_path: Path) -> None:
    from mars.agent import MarsAgentSession, MarsRiskAgent

    report = profile_stats(
        sample(),
        features=["income", "salary"],
        metrics=["mean"],
        feature_metadata=METADATA,
        business_context=CONTEXT,
    )
    report.save(tmp_path / "saved.marsreport")
    persistent_id = report.report_id
    del report
    restored = load_report(tmp_path / "saved.marsreport")
    session = MarsAgentSession()
    handle = session.register_report(restored)
    assert handle != persistent_id
    restored.feature_metadata["income"]["display_name"] = "changed"
    restored.get_table("overview")
    agent = MarsRiskAgent()
    found = agent.execute_tool(
        "search_report_features", {"report_id": handle, "query": "月收入"}, session=session
    )
    assert found.success and len(found.data["candidates"]) == 2
    query = agent.execute_tool(
        "get_report_table",
        {
            "report_id": handle,
            "table": "overview",
            "sources": ["bank"],
            "features": ["salary"],
            "columns": ["feature", "mean"],
        },
        session=session,
    )
    assert query.success and query.data["rows"][0]["feature"] == "salary"
    assert query.data["evidence_reference"]["report_id"] == persistent_id
    context = agent.execute_tool(
        "get_report_context", {"report_id": handle, "features": ["income"]}, session=session
    )
    assert context.success
    assert context.data["description"]["feature_metadata"]["income"]["display_name"] == "月收入"
    assert session.get_report(handle).dataset_id is None
    assert snapshot_report(restored).report_id == persistent_id


@pytest.mark.skipif(sys.version_info < (3, 10), reason="mars.agent requires Python >= 3.10")
def test_serialization_rejects_objects_and_agent_preserves_nonfinite(tmp_path: Path) -> None:
    from mars.agent import MarsAgentSession, MarsRiskAgent

    frame = pl.DataFrame(
        {"feature": ["income"] * 5, "value": [None, float("nan"), float("inf"), float("-inf"), 0.0]}
    )
    report = MarsProfileReport(frame, {}, {})
    session = MarsAgentSession()
    handle = session.register_report(report)
    result = MarsRiskAgent().execute_tool(
        "get_report_table", {"report_id": handle, "table": "overview"}, session=session
    )
    assert result.success
    assert result.data["rows"] == json.loads(report.to_ai_context())["evidence"][0]["rows"]
    report.report_meta["invalid"] = object()
    with pytest.raises(ValueError, match="Unsupported report value"):
        report.save(tmp_path / "invalid.marsreport")
    assert not (tmp_path / "invalid.marsreport").exists()
    assert math.isnan(frame["value"][1])


@pytest.mark.parametrize("backend", ["pandas", "polars"])
def test_iso_date_query_replays_json_evidence_after_restore(backend: str, tmp_path: Path) -> None:
    frame = pl.DataFrame(
        {
            "feature": ["income"] * 3,
            "day": [date(2026, 1, i) for i in (1, 2, 3)],
            "at": [datetime(2026, 1, i, tzinfo=timezone.utc) for i in (1, 2, 3)],
        }
    )
    if backend == "pandas":
        frame = frame.to_pandas()
    report = MarsProfileReport(frame, {}, {})
    conditions = {
        "day": {"op": "ge", "value": date(2026, 1, 2)},
        "at": {"op": "in", "value": [datetime(2026, 1, 3, tzinfo=timezone.utc)]},
    }
    evidence = json.loads(report.to_ai_context(filters=conditions))["evidence"][0]
    report.save(tmp_path / "dated.marsreport")
    restored = load_report(tmp_path / "dated.marsreport")
    page = restored.query_page(evidence["reference"], **evidence["query"])
    assert len(page["data"]) == 1 and page["total_rows"] == 1
    assert restored.get_feature("income", limit=0)["omitted_rows"]["overview"] == 3


@pytest.mark.skipif(sys.version_info < (3, 10), reason="mars.agent requires Python >= 3.10")
def test_external_contract_snapshot_isolates_data_and_keeps_real_source(tmp_path: Path) -> None:
    from mars.agent import MarsAgentSession, MarsRiskAgent

    report = MarsProfileReport(
        pd.DataFrame({"feature": ["income"], "mean": [2.0]}), {}, {}, feature_metadata=METADATA
    )
    report.source = {"kind": "external", "producer": "other-agent", "uri": "analysis:42"}

    class External:
        def __init__(self) -> None:
            # 3.12 的 runtime Protocol 按静态属性检查；公开成员显式提供。
            for name in (
                "report_id",
                "report_type",
                "format_version",
                "get_table",
                "get_feature",
                "search_features",
                "query_page",
                "to_ai_context",
                "save",
            ):
                setattr(self, name, getattr(report, name))

        def describe(self) -> dict[str, Any]:
            description = report.describe()
            description.pop("serialization")
            return description

        def __getattr__(self, name: str) -> Any:
            if name.startswith("_"):
                raise AssertionError("consumer attempted a private interface")
            return getattr(report, name)

    external = External()
    assert isinstance(external, Report)
    session = MarsAgentSession()
    handle = session.register_report(external)
    isolated = snapshot_report(external)
    assert json.loads(isolated.to_ai_context())["evidence"][0]["rows"][0]["mean"] == 2
    report.overview_table.loc[0, "mean"] = 999
    report.feature_metadata["income"]["display_name"] = "changed"
    assert isolated.get_table("overview")["mean"].item() == 2
    result = MarsRiskAgent().execute_tool(
        "get_report_table", {"report_id": handle, "table": "overview"}, session=session
    )
    assert result.success and result.data["rows"][0]["mean"] == 2
    assert session.get_report(handle).dataset_id is None
    assert isolated.describe()["source"]["producer"] == "other-agent"
    isolated.save(tmp_path / "external.marsreport")
    assert load_report(tmp_path / "external.marsreport").describe() == isolated.describe()


def test_nullable_float_and_enum_roundtrip(tmp_path: Path) -> None:
    pandas_frame = pd.DataFrame(
        {
            "feature": ["income"] * 4,
            "value": pd.array([None, 0.0, float("inf"), float("-inf")], dtype="Float64"),
        }
    )
    pandas_report = MarsProfileReport(pandas_frame, {}, {})
    pandas_report.save(tmp_path / "nullable.marsreport")
    assert_frame_equal(
        pandas_frame, load_report(tmp_path / "nullable.marsreport").get_table("overview")
    )
    polars_frame = pl.DataFrame(
        {"feature": ["income"] * 2, "kind": pl.Series(["a", None], dtype=pl.Enum(["a", "b"]))}
    )
    report = MarsProfileReport(polars_frame, {}, {})
    report.save(tmp_path / "enum.marsreport")
    assert_polars_equal(
        polars_frame, load_report(tmp_path / "enum.marsreport").get_table("overview")
    )


def test_profile_states_are_exceptional_and_survive_export_and_restore(tmp_path: Path) -> None:
    report = profile_stats(
        pl.DataFrame({"income": [1.0], "category": ["a"]}),
        features=["income", "category"],
        metrics=["mean", "std"],
    )
    state = report.get_table("calculation_status")
    assert state.filter(pl.col("feature") == "category")["status"].unique().to_list() == ["skipped"]
    assert state.filter(pl.col("feature") == "income")["status"].to_list() == [
        "insufficient_samples"
    ]
    report.save(tmp_path / "states.marsreport")
    restored = load_report(tmp_path / "states.marsreport")
    assert_polars_equal(restored.get_table("calculation_status"), state)
    restored.write_excel(str(tmp_path / "states.xlsx"))
    assert "calculation_status" in " ".join(pd.ExcelFile(tmp_path / "states.xlsx").sheet_names)


def test_large_business_descriptions_leave_useful_evidence_under_budget() -> None:
    report = profile_stats(
        sample(),
        features=["income"],
        metrics=["mean"],
        feature_metadata={"income": {"description": "长定义" * 10000}},
        business_context={"sample": "长口径" * 10000},
    )
    context = report.to_ai_context(max_chars=5000)
    payload = json.loads(context)
    assert len(context) <= 5000 and payload["evidence"][0]["rows"]
    assert len(payload["omitted"]) >= 2


@pytest.mark.skipif(sys.version_info < (3, 10), reason="mars.agent requires Python >= 3.10")
@pytest.mark.parametrize("tool", ["profile_data", "evaluate_risk"])
def test_agent_analysis_propagates_registered_business_semantics(tool: str) -> None:
    from mars.agent import MarsAgentSession, MarsRiskAgent

    session = MarsAgentSession()
    session.register_dataset(
        "applications",
        sample(),
        features=["income"],
        target="y",
        group_columns=["month"],
        feature_metadata=METADATA,
        business_context=CONTEXT,
    )
    agent = MarsRiskAgent()
    options = {"dataset_id": "applications", "group_col": "month"}
    if tool == "profile_data":
        options["metrics"] = ["mean"]
    result = agent.execute_tool(tool, options, session=session)
    assert result.success
    handle = result.data["report_id"]
    description = agent.execute_tool("describe_report", {"report_id": handle}, session=session)
    assert description.success
    assert description.data["description"]["feature_metadata"]["income"] == METADATA["income"]
    assert (
        description.data["description"]["business_context"]["labels"]["y"]["definition"]
        == CONTEXT["labels"]["y"]["definition"]
    )
    assert session.get_report(handle).dataset_id == "applications"
