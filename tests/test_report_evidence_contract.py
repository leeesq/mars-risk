"""报告状态、受限浮点、有效预算查询及当前页身份的回归。"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import polars as pl
import pytest

import mars
from mars.analysis import cross_scores, profile_risk, profile_stats
from mars.reporting import MarsProfileReport, ReportSnapshot, load_report, snapshot_report
from mars.reporting._serialization import encode, table_rows


def _frame(data: dict[str, Any], backend: str) -> pl.DataFrame | pd.DataFrame:
    """让同一反例在两种原生报告后端执行。"""
    frame = pl.DataFrame(data)
    return frame.to_pandas() if backend == "pandas" else frame


def _payload(report: Any, **query: Any) -> dict[str, Any]:
    """严格读取完整 JSON，拒绝标准以外的浮点字面量。"""

    def reject(value: str) -> None:
        raise AssertionError(f"Non-standard JSON constant: {value}")

    return json.loads(report.to_ai_context(**query), parse_constant=reject)


def _assert_replay(report: Any, payload: dict[str, Any]) -> None:
    """直接使用有效证据参数，不能在测试中替消费者补 limit 或 columns。"""
    for evidence in payload["evidence"]:
        replay = report.get_table(evidence["reference"], **evidence["query"])
        assert json.loads(encode(table_rows(replay))) == evidence["rows"]


@pytest.mark.parametrize("backend", ["polars", "pandas"])
@pytest.mark.parametrize("sign", [1, -1])
@pytest.mark.parametrize("form", ["scalar", "eq", "in"])
def test_tagged_nonfinite_filters_replay_across_snapshot_and_disk(
    backend: str, sign: int, form: str, tmp_path: Path
) -> None:
    value = sign * float("inf")
    condition: Any = value
    if form != "scalar":
        condition = {"op": form, "value": [value, 0.0] if form == "in" else value}
    report = MarsProfileReport(
        _frame(
            {
                "feature": ["positive", "negative", "zero"],
                "mean": [float("inf"), -float("inf"), 0.0],
            },
            backend,
        ),
        {},
        {},
    )
    path = tmp_path / "nonfinite.marsreport"
    report.save(path)
    for saved in (report, snapshot_report(report), load_report(path)):
        payload = _payload(saved, filters={"mean": condition})
        _assert_replay(saved, payload)
        assert len(payload["evidence"][0]["rows"]) == (2 if form == "in" else 1)


@pytest.mark.parametrize(
    "tag",
    [
        {"$mars": "unknown", "value": "inf"},
        {"$mars": "float"},
        {"$mars": "float", "value": "Infinity"},
        {"$mars": "float", "value": 1},
        {"$mars": "float", "value": "inf", "extra": True},
    ],
)
@pytest.mark.parametrize("form", ["scalar", "eq", "in"])
def test_malformed_float_filters_fail_at_the_protocol_boundary(
    tag: dict[str, Any], form: str
) -> None:
    report = MarsProfileReport(pl.DataFrame({"feature": ["x"], "mean": [1.0]}), {}, {})
    condition: Any = (
        tag if form == "scalar" else {"op": form, "value": [tag] if form == "in" else tag}
    )
    with pytest.raises(ValueError, match="tagged float"):
        report.get_table("overview", filters={"mean": condition})


@pytest.mark.parametrize("backend", ["polars", "pandas"])
def test_strings_null_and_existing_nan_predicates_keep_their_types(backend: str) -> None:
    report = MarsProfileReport(
        _frame(
            {
                "feature": ["a", "b", "c", "d"],
                "text": ["inf", "-inf", "NaN", '{"$mars":"float","value":"inf"}'],
            },
            backend,
        ),
        {},
        {},
    )
    for value in report.get_table("overview")["text"]:
        payload = _payload(report, filters={"text": {"op": "eq", "value": value}})
        _assert_replay(report, payload)
        assert payload["evidence"][0]["rows"][0]["text"] == value
    numeric = MarsProfileReport(
        _frame({"feature": ["null", "nan", "zero"], "mean": [None, float("nan"), 0.0]}, backend),
        {},
        {},
    )
    for condition in (
        None,
        {"op": "is_null"},
        {"op": "is_not_null"},
        {"op": "ne", "value": 0.0},
        {"op": "eq", "value": float("nan")},
        {"op": "in", "value": [float("nan"), 0.0]},
    ):
        _assert_replay(numeric, _payload(numeric, filters={"mean": condition}))


@pytest.mark.parametrize("backend", ["polars", "pandas"])
def test_numeric_projection_keeps_only_the_page_identity_and_metadata(
    backend: str, tmp_path: Path
) -> None:
    report = MarsProfileReport(
        _frame({"feature": ["a", "b", "c"], "mean": [1.0, 3.0, 2.0]}, backend),
        {},
        {},
        feature_metadata={
            name: {"data_source": "bank", "display_name": name.upper(), "unit": "CNY"}
            for name in ["a", "b", "c"]
        },
    )
    path = tmp_path / "identities.marsreport"
    report.save(path)
    for saved in (report, snapshot_report(report), load_report(path)):
        payload = _payload(
            saved,
            features=["a", "b", "c"],
            columns=["mean"],
            sort_by="mean",
            descending=True,
            offset=1,
            limit=1,
        )
        evidence = payload["evidence"][0]
        assert evidence["rows"] == [{"mean": 2.0}]
        assert evidence["identities"] == [{"feature": "c"}]
        assert payload["description"]["feature_metadata"] == {"c": report.feature_metadata["c"]}
        _assert_replay(saved, payload)
        empty = _payload(saved, features="unknown", columns=["mean"])
        assert empty["description"]["feature_metadata"] == {}
        assert empty["evidence"][0].get("identities", []) == []


@pytest.mark.parametrize("backend", ["polars", "pandas"])
def test_relation_projection_keeps_both_actual_endpoints(backend: str, tmp_path: Path) -> None:
    frame = _frame(
        {
            "feature_a": ["a", "a", "b"],
            "feature_b": ["b", "c", "c"],
            "correlation": [0.9, 0.3, 0.4],
        },
        backend,
    )
    profile = MarsProfileReport(
        pl.DataFrame({"feature": ["a", "b", "c"]}),
        {},
        {},
        feature_metadata={name: {"display_name": name.upper()} for name in ["a", "b", "c"]},
    )
    description = profile.describe()
    description["report_type"] = "correlation"
    description["tables"] = {
        "pairs": {
            "rows": 3,
            "grain": "feature pair",
            "fields": {
                column: {"dtype": str(dtype), "unit": "unknown"}
                for column, dtype in (
                    frame.schema.items()
                    if isinstance(frame, pl.DataFrame)
                    else frame.dtypes.items()
                )
            },
            "feature_roles": {"feature_a": "left endpoint", "feature_b": "right endpoint"},
        }
    }
    report = ReportSnapshot({"pairs": frame}, description)
    path = tmp_path / "pairs.marsreport"
    report.save(path)
    for saved in (report, load_report(path)):
        payload = _payload(
            saved,
            tables=["pairs"],
            features="a",
            columns=["correlation"],
            sort_by="correlation",
            descending=True,
            limit=1,
        )
        assert payload["evidence"][0]["identities"] == [{"feature_a": "a", "feature_b": "b"}]
        assert set(payload["description"]["feature_metadata"]) == {"a", "b"}
        _assert_replay(saved, payload)


@pytest.mark.parametrize("backend", ["polars", "pandas"])
@pytest.mark.parametrize("kind", ["narrow", "wide", "combined", "empty", "uncropped"])
def test_budget_is_complete_stable_and_exactly_replayable(
    backend: str, kind: str, tmp_path: Path
) -> None:
    names = [f"feature_{index}" for index in range(12)]
    overview = _frame(
        {"feature": names, "mean": list(range(12)), "detail": ["evidence" * 60] * 12}, backend
    )
    trend = _frame(
        {"feature": names, **{f"day_{index:03d}": [float(index)] * 12 for index in range(365)}},
        backend,
    )
    report = MarsProfileReport(
        overview,
        {},
        {"mean": trend},
        feature_metadata={name: {"display_name": name.upper()} for name in names},
    )
    path = tmp_path / "budget.marsreport"
    report.save(path)
    options: dict[str, Any] = {"limit": 12, "max_chars": 7000}
    if kind in {"wide", "combined"}:
        options["tables"] = ["stats.mean"]
    if kind == "wide":
        options["limit"] = 1
    if kind == "empty":
        options["features"] = "unknown"
    if kind == "uncropped":
        options.update(limit=1, columns=["mean"], max_chars=16000)
    for saved in (report, load_report(path)):
        context = saved.to_ai_context(**options)
        payload = json.loads(context)
        assert len(context) <= options["max_chars"]
        assert saved.to_ai_context(**options) == context
        _assert_replay(saved, payload)
        evidence = payload["evidence"][0]
        if kind == "narrow":
            assert 0 < evidence["returned_rows"] < 12
        if kind in {"wide", "combined"}:
            assert len(evidence["rows"][0]) < 366
        identities = evidence.get("identities", [])
        assert not identities or len(identities) == evidence["returned_rows"]
        expected = {row["feature"] for row in evidence["rows"] if "feature" in row}
        expected.update(row["feature"] for row in identities if "feature" in row)
        assert set(payload["description"]["feature_metadata"]) == expected
    with pytest.raises(ValueError, match="max_chars|description"):
        report.to_ai_context(max_chars=512)


def test_status_definitions_use_the_actual_report_table_and_survive_disk(tmp_path: Path) -> None:
    no_target = profile_risk(
        pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]}), features=["x"], n_bins=2
    ).report
    observed = profile_risk(
        pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "y": [0, 1, 0, 1]}),
        features=["x"],
        target="y",
        n_bins=2,
    ).report
    unobserved = profile_risk(
        pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "y": pl.Series([None] * 4, dtype=pl.Int64)}),
        features=["x"],
        target="y",
        n_bins=2,
    ).report
    profile = profile_stats(
        pl.DataFrame({"x": [1.0], "text": ["a"]}), features=["x", "text"], metrics=["mean", "std"]
    )
    cross = cross_scores(
        pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "z": [4.0, 3.0, 2.0, 1.0], "y": [0, 1, 0, 1]}),
        score_x="x",
        score_y="z",
        targets=["y"],
        score_directions={"x": "higher_risk", "z": "higher_risk"},
        cutpoints={"x": [2.5], "z": [2.5]},
    )
    for index, report in enumerate((no_target, observed, unobserved, profile, cross)):
        path = tmp_path / f"status-{index}.marsreport"
        report.save(path)
        restored = load_report(path)
        assert restored.describe() == report.describe()
        table = "cells" if report is cross else "calculation_status"
        meaning = restored.describe()["tables"][table]["fields"]["status"]["meaning"]
        values = set(restored.get_table(table)["status"])
        assert all(value in meaning for value in values)
        assert ("low_sample" in meaning) == (report is cross)
        context = _payload(restored, tables=[table], columns=["status"], limit=1)
        assert context["description"]["tables"][table]["fields"]["status"]["meaning"] == meaning
    assert no_target.get_table("calculation_status")["reason"].unique().to_list() == ["no_target"]


def test_real_e990813_version_one_report_keeps_data_and_corrects_legacy_semantics() -> None:
    report = load_report(Path(__file__).parent / "fixtures" / "legacy_e990813_no_target.marsreport")
    assert report.format_version == 1
    state = report.get_table("calculation_status")
    assert state["status"].unique().to_list() == ["not_computed"]
    assert state["reason"].unique().to_list() == ["no_target"]
    meaning = report.describe()["tables"]["calculation_status"]["fields"]["status"]["meaning"]
    assert "not_computed" in meaning and "low_sample" not in meaning


@pytest.mark.parametrize("table", ["calculation_status", "comparison.schema", "comparison.unseen"])
def test_custom_reports_keep_their_own_same_named_state_definitions(table: str) -> None:
    profile = MarsProfileReport(pl.DataFrame({"feature": ["x"]}), {}, {})
    description = profile.describe()
    description["report_type"] = "custom_domain"
    description["tables"] = {
        table: {
            "rows": 1,
            "grain": "custom entity",
            "fields": {
                "status": {"dtype": "String", "unit": "business_custom_state", "meaning": "custom"},
                "reason": {
                    "dtype": "String",
                    "unit": "business_reason",
                    "meaning": "custom_reason",
                },
            },
        }
    }
    report = ReportSnapshot(
        {table: pl.DataFrame({"status": ["custom"], "reason": ["custom_reason"]})}, description
    )
    assert report.describe()["tables"] == description["tables"]


def test_new_process_uses_only_saved_reports_for_agent_context_and_evidence_replay(
    tmp_path: Path,
) -> None:
    report = MarsProfileReport(
        pl.DataFrame(
            {
                "feature": ["a", "b", "c"],
                "mean": [float("inf"), -float("inf"), 0.0],
                "detail": ["evidence" * 80] * 3,
            }
        ),
        {},
        {
            "mean": pl.DataFrame(
                {
                    "feature": ["a", "b", "c"],
                    **{f"day_{index:03d}": [float(index)] * 3 for index in range(365)},
                }
            )
        },
        feature_metadata={
            name: {"display_name": name.upper(), "data_source": "bank"} for name in ["a", "b", "c"]
        },
    )
    path = tmp_path / "consumer.marsreport"
    report.save(path)
    del report
    code = """
import json, sys
from mars.reporting import load_report
from mars.reporting._serialization import encode, table_rows
r = load_report(sys.argv[1])
p = json.loads(r.to_ai_context(columns=["mean"], filters={"mean": {"op": "in", "value": [{"$mars": "float", "value": "inf"}, {"$mars": "float", "value": "-inf"}]}}, max_chars=5000))
e = p["evidence"][0]
assert json.loads(encode(table_rows(r.get_table(e["reference"], **e["query"])))) == e["rows"]
assert e["identities"] == [{"feature": "a"}, {"feature": "b"}]
assert set(p["description"]["feature_metadata"]) == {"a", "b"}
wide_context = r.to_ai_context(tables=["stats.mean"], limit=3, max_chars=5000)
assert len(wide_context) <= 5000
wide = json.loads(wide_context)["evidence"][0]
assert len(wide["rows"][0]) < 366 and wide["query"]["columns"] is not None
assert json.loads(encode(table_rows(r.get_table(wide["reference"], **wide["query"])))) == wide["rows"]
legacy = load_report(sys.argv[2])
assert "not_computed" in legacy.describe()["tables"]["calculation_status"]["fields"]["status"]["meaning"]
if sys.version_info >= (3, 10):
 from mars.agent import MarsAgentSession, MarsRiskAgent
 s = MarsAgentSession(); identifier = s.register_report(r)
 a = MarsRiskAgent(max_result_chars=5000)
 response = a.execute_tool("get_report_context", {"report_id": identifier, "queries": {"overview": e["query"]}}, session=s)
 assert response.success, response.error_message
 agent_evidence = json.loads(encode(response.data))["evidence"][0]
 assert json.loads(encode(table_rows(r.get_table(agent_evidence["reference"], **agent_evidence["query"])))) == agent_evidence["rows"]
 assert a.execute_tool("describe_report", {"report_id": identifier, "table": "overview"}, session=s).success
print("saved-only replay passed")
"""
    legacy = Path(__file__).parent / "fixtures" / "legacy_e990813_no_target.marsreport"
    result = subprocess.run(
        [sys.executable, "-c", code, str(path), str(legacy)],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(Path(mars.__file__).resolve().parent.parent)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "saved-only replay passed" in result.stdout
