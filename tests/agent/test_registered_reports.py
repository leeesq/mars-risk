"""外部已有报告登记、来源、快照及会话隔离。"""

from __future__ import annotations

import polars as pl
import pytest

from mars.agent import MarsAgentSession, MarsRiskAgent
from mars.analysis import profile_risk, profile_stats


@pytest.mark.parametrize("pandas", [False, True])
@pytest.mark.parametrize("kind", ["profile", "risk"])
def test_external_reports_are_queryable_snapshots_without_datasets(pandas: bool, kind: str) -> None:
    frame = pl.DataFrame({"x": [1.0, 2.0, None, 4.0], "y": [0, 0, 1, 1]})
    data = frame.to_pandas() if pandas else frame
    result = (
        profile_stats(data, features=["x"], metrics=["missing", "mean"])
        if kind == "profile"
        else profile_risk(data, features=["x"], target="y", n_bins=2)
    )
    session = MarsAgentSession()
    identifier = session.register_report(result, report_id="report_2")
    report = session.get_report(identifier)
    assert report.dataset_id is None
    assert report.metadata["source"]["kind"] == "external_report"
    assert report.metadata["report_description"]["report_type"] in {
        "MarsProfileReport",
        "MarsBinningReport",
    }
    original = result if kind == "profile" else result.report
    original.report_meta["mutated"] = True
    table_name = "overview" if kind == "profile" else "summary"
    table = original.get_table(table_name)
    if pandas:
        original._query_tables()[table_name].iloc[0, 0] = "changed"
    else:
        original._query_tables()[table_name].replace_column(0, pl.Series("feature", ["changed"]))
    assert "mutated" not in session.get_report(identifier).metadata
    agent = MarsRiskAgent()
    query = agent.execute_tool(
        "get_report_table",
        {"report_id": identifier, "table": table_name, "columns": ["feature"]},
        session=session,
    )
    assert query.success and query.data["rows"] == [{"feature": "x"}]
    assert agent.execute_tool("list_reports", {}, session=session).success
    described = agent.execute_tool(
        "describe_report",
        {
            "report_id": identifier,
            "table": table_name,
            "columns": ["feature"],
            "parameter_keys": ["row_count"],
        },
        session=session,
    )
    assert described.success
    assert described.data["description"]["parameters"]["row_count"] == 4
    assert not agent.execute_tool(
        "get_report_table",
        {"report_id": identifier, "table": table_name},
        session=MarsAgentSession(),
    ).success
    with pytest.raises(ValueError, match="already registered"):
        session.register_report(original, report_id=identifier)
    assert session.register_report(original) != identifier
    copied = session.get_report(identifier)
    copied.tables[table_name].replace_column(0, pl.Series("feature", ["local_change"]))
    assert session.get_report(identifier).tables[table_name]["feature"].to_list() == ["x"]
    assert len(table) == 1


def test_registration_rejects_invalid_type_identifier_and_busy_session() -> None:
    session = MarsAgentSession()
    report = profile_stats(pl.DataFrame({"x": [1, 2]}), metrics=["mean"])
    with pytest.raises(ValueError):
        session.register_report(object())
    with pytest.raises(ValueError):
        session.register_report(report, report_id="bad/path")
    session._lock.acquire()
    try:
        with pytest.raises(RuntimeError, match="busy"):
            session.register_report(report)
    finally:
        session._lock.release()


def test_report_catalog_pages_under_budget_and_description_rejects_unknown_keys() -> None:
    session = MarsAgentSession()
    report = profile_stats(pl.DataFrame({"x": [1, 2]}), metrics=["mean"])
    for i in range(12):
        session.register_report(report, report_id=f"external_{i}")
    agent = MarsRiskAgent(max_result_chars=1024)
    offset, seen = 0, []
    while True:
        page = agent.execute_tool("list_reports", {"offset": offset, "limit": 50}, session=session)
        assert page.success and page.data["reports"]
        seen.extend(item["report_id"] for item in page.data["reports"])
        if page.data["next_offset"] is None:
            break
        assert page.data["next_offset"] > offset
        offset = page.data["next_offset"]
    assert seen == list(session.report_ids)
    for options in [
        {"table": "unknown"},
        {"table": "overview", "columns": ["bad"]},
        {"parameter_keys": ["unknown"]},
        {"columns": ["feature"]},
    ]:
        result = agent.execute_tool(
            "describe_report", {"report_id": seen[0], **options}, session=session
        )
        assert result.error_code == "INVALID_ARGUMENTS"
