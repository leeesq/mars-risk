"""旧 Monitoring 消费路径的真实来源筛选与证据回归。"""

from __future__ import annotations

from typing import Any

import polars as pl
import pytest

from mars.agent import MarsAgentSession, MarsRiskAgent


@pytest.mark.parametrize("backend", ["polars", "pandas"])
def test_monitor_source_filters_intersect_features_and_match_the_reference(backend: str) -> None:
    frame = pl.DataFrame(
        {
            "score": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "income": [6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
            "y": [0, 0, 0, 1, 1, 1],
        }
    )
    data = frame.to_pandas() if backend == "pandas" else frame
    session = MarsAgentSession()
    for identifier in ("current", "benchmark"):
        session.register_dataset(
            identifier,
            data,
            features=["score", "income"],
            target="y",
            feature_metadata={
                "score": {"data_source": "bureau"},
                "income": {"data_source": "application"},
            },
        )
    agent = MarsRiskAgent()
    computed = agent.execute_tool(
        "monitor_data",
        {"dataset_id": "current", "benchmark_id": "benchmark", "n_bins": 2},
        session=session,
    )
    assert computed.success, computed.error_message
    identifier = computed.data["report_id"]
    for source, selected in (("bureau", "score"), ("application", "income")):
        result = agent.execute_tool(
            "get_report_table",
            {"report_id": identifier, "table": "summary", "sources": [source]},
            session=session,
        )
        assert result.success, result.error_message
        assert [row["feature"] for row in result.data["rows"]] == [selected]
        assert result.data["rows"][0]["data_source"] == source
        reference = result.data["evidence_reference"]
        replay = agent.execute_tool(
            "get_report_table",
            {"report_id": identifier, "table": reference["table"], **reference["query"]},
            session=session,
        )
        assert replay.success and replay.data["rows"] == result.data["rows"]
        empty = agent.execute_tool(
            "get_report_table",
            {
                "report_id": identifier,
                "table": "summary",
                "sources": [source],
                "features": ["income" if selected == "score" else "score"],
            },
            session=session,
        )
        assert empty.success and empty.data["rows"] == []
    unknown = agent.execute_tool(
        "get_report_table",
        {"report_id": identifier, "table": "summary", "sources": ["nonexistent"]},
        session=session,
    )
    assert unknown.error_code == "INVALID_ARGUMENTS" and "source" in unknown.error_message


@pytest.mark.parametrize("metadata", [{}, {"feature_metadata": {}}])
def test_legacy_monitor_without_source_information_rejects_source_filters(
    metadata: dict[str, Any],
) -> None:
    session = MarsAgentSession()
    stored = session._save_report(
        "monitor_data",
        None,
        None,
        {"summary": pl.DataFrame({"feature": ["x"], "mean": [1.0]})},
        metadata,
    )
    agent = MarsRiskAgent()
    result = agent.execute_tool(
        "get_report_table",
        {"report_id": stored.id, "table": "summary", "sources": ["bank"]},
        session=session,
    )
    assert result.error_code == "INVALID_ARGUMENTS"
    assert "source" in result.error_message and "unsupported" in result.error_message.lower()
    plain = agent.execute_tool(
        "get_report_table", {"report_id": stored.id, "table": "summary"}, session=session
    )
    assert plain.success and plain.data["rows"] == [{"feature": "x", "mean": 1.0}]


@pytest.mark.parametrize("source", [None, "", "UNMAPPED"])
def test_fresh_monitor_without_a_registered_source_rejects_unmapped_queries(
    source: str | None,
) -> None:
    frame = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "y": [0, 0, 1, 1]})
    session = MarsAgentSession()
    for identifier in ("current", "benchmark"):
        session.register_dataset(
            identifier,
            frame,
            features=["x"],
            target="y",
            feature_metadata={"x": {"data_source": source}},
        )
    agent = MarsRiskAgent()
    computed = agent.execute_tool(
        "monitor_data",
        {"dataset_id": "current", "benchmark_id": "benchmark", "n_bins": 2},
        session=session,
    )
    assert computed.success, computed.error_message
    result = agent.execute_tool(
        "get_report_table",
        {"report_id": computed.data["report_id"], "table": "summary", "sources": ["UNMAPPED"]},
        session=session,
    )
    assert (
        result.error_code == "INVALID_ARGUMENTS" and "unsupported" in result.error_message.lower()
    )


def test_legacy_monitor_consumes_reliable_sources_already_present_in_tables() -> None:
    session = MarsAgentSession()
    stored = session._save_report(
        "monitor_data",
        None,
        None,
        {
            "summary": pl.DataFrame(
                {"feature": ["x", "y"], "data_source": ["bank", "bureau"], "mean": [1.0, 2.0]}
            )
        },
        {},
    )
    result = MarsRiskAgent().execute_tool(
        "get_report_table",
        {"report_id": stored.id, "table": "summary", "sources": ["bank"]},
        session=session,
    )
    assert result.success and [row["feature"] for row in result.data["rows"]] == ["x"]


@pytest.mark.parametrize("declared", ["bank", "conflicting"])
def test_monitor_combines_partial_source_metadata_and_tables_without_overwriting_conflicts(
    declared: str,
) -> None:
    session = MarsAgentSession()
    stored = session._save_report(
        "monitor_data",
        None,
        None,
        {"summary": pl.DataFrame({"feature": ["x", "y"], "data_source": ["bank", "bureau"]})},
        {"feature_metadata": {"x": {"data_source": declared}}},
    )
    result = MarsRiskAgent().execute_tool(
        "get_report_table",
        {"report_id": stored.id, "table": "summary", "sources": ["bureau"]},
        session=session,
    )
    if declared == "bank":
        assert result.success and [row["feature"] for row in result.data["rows"]] == ["y"]
    else:
        assert result.error_code == "INVALID_ARGUMENTS" and "Conflicting" in result.error_message
