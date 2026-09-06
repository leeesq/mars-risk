from __future__ import annotations

import json

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from mars.agent import MarsAgentSession, MarsRiskAgent
from mars.analysis import profile_risk, profile_stats
from mars.monitoring import MarsMonitor


def make_session() -> tuple[MarsAgentSession, pl.DataFrame, pl.DataFrame]:
    baseline = pl.DataFrame(
        {
            "score": [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9],
            "income": [8.0, 7, 6, 5, 4, 3, 2, 1],
            "target": [0, 0, 0, 0, 1, 1, 1, 1],
        }
    )
    current = pl.DataFrame(
        {
            "score": [0.15, 0.3, 0.65, 0.85, 0.2, 0.4, 0.7, 0.95],
            "income": [8.0, 6, None, 1, 7, -999, 2, 1],
            "target": [0, 0, 1, 1, None, None, None, None],
            "month": ["2026-01"] * 4 + ["2026-02"] * 4,
            "channel": ["A", "B"] * 4,
            "customer_secret": ["never-send-this-value"] * 8,
        }
    )
    session = MarsAgentSession()
    session.register_dataset(
        "baseline",
        baseline,
        features=["score", "income"],
        target="target",
        missing_values=[-999],
    )
    session.register_dataset(
        "current",
        current,
        features=["score", "income"],
        target="target",
        group_columns=["month", "channel"],
        missing_values=[-999],
    )
    return session, baseline, current


def test_profile_tool_matches_public_api_for_pandas_and_polars() -> None:
    session, _, current = make_session()
    session.register_dataset(
        "pandas",
        current.to_pandas(),
        features=["income"],
        group_columns=["month"],
        missing_values=[-999],
    )
    agent = MarsRiskAgent()
    expected = profile_stats(
        current,
        metrics=["missing", "mean"],
        features=["income"],
        group_col="month",
        missing_values=[-999],
    )
    for dataset_id in ("current", "pandas"):
        result = agent.execute_tool(
            "profile_data",
            {
                "dataset_id": dataset_id,
                "features": ["income"],
                "metrics": ["missing", "mean"],
                "group_col": "month",
            },
            session=session,
        )
        assert result.success, result.error_message
        report = session.get_report(result.data["report_id"])
        assert_frame_equal(report.tables["dq.missing"], expected.dq_tables["missing"])
        assert_frame_equal(report.tables["stats.mean"], expected.stats_tables["mean"])


def test_monitor_matches_mars_and_preserves_unobserved_targets() -> None:
    session, baseline, current = make_session()
    result = MarsRiskAgent().execute_tool(
        "monitor_data",
        {
            "dataset_id": "current",
            "benchmark_id": "baseline",
            "group_col": "month",
            "n_bins": 2,
        },
        session=session,
    )
    assert result.success, result.error_message
    report = session.get_report(result.data["report_id"])
    expected = MarsMonitor(
        binner_params={"method": "quantile", "n_bins": 2, "missing_values": [-999]}
    ).monitor(
        current,
        features=["score", "income"],
        target="target",
        benchmark_df=baseline,
        group_col="month",
    )
    assert_frame_equal(report.tables["summary"], expected.summary_table)
    assert_frame_equal(
        report.tables["target_observation"], expected.target_observation_table
    )
    last = report.tables["target_observation"].filter(
        pl.col(report.metadata["group_col"]) == "2026-02"
    )
    assert last["target_observed_rate"].item() == 0
    assert last["bad_rate_observed"].item() is None
    assert report.metadata["agent_parameters"]["n_bins"] == 2
    assert report.metadata["agent_parameters"]["missing_values"] == [-999]


def test_risk_tool_matches_public_api() -> None:
    session, baseline, current = make_session()
    result = MarsRiskAgent().execute_tool(
        "evaluate_risk",
        {
            "dataset_id": "current",
            "benchmark_id": "baseline",
            "features": ["score"],
            "n_bins": 2,
        },
        session=session,
    )
    assert result.success, result.error_message
    expected = profile_risk(
        current,
        target="target",
        features=["score"],
        benchmark_df=baseline,
        method="quantile",
        n_bins=2,
        missing_values=[-999],
    )
    assert_frame_equal(
        session.get_report(result.data["report_id"]).tables["summary"],
        expected.report.summary_table,
    )


def test_grouped_query_is_filterable_sortable_and_paginated() -> None:
    session, _, _ = make_session()
    agent = MarsRiskAgent()
    result = agent.execute_tool(
        "monitor_data",
        {
            "dataset_id": "current",
            "benchmark_id": "baseline",
            "group_col": "channel",
            "n_bins": 2,
        },
        session=session,
    )
    assert result.success
    report_id = result.data["report_id"]
    detail = session.get_report(report_id).tables["detail"]
    expected = detail.filter(pl.col("feature") == "score").sort(
        "count", descending=True, nulls_last=True
    )
    rows = []
    offset = 0
    while True:
        page = agent.execute_tool(
            "get_report_table",
            {
                "report_id": report_id,
                "table": "detail",
                "filters": {"feature": "score"},
                "sort_by": "count",
                "descending": True,
                "offset": offset,
                "limit": 2,
            },
            session=session,
        )
        assert page.success
        assert page.data["reference"] == f"{report_id}/detail"
        rows.extend(page.data["rows"])
        offset = page.data["next_offset"]
        if offset is None:
            break
    assert len(rows) == expected.height
    assert [row["count"] for row in rows] == expected["count"].to_list()


@pytest.mark.parametrize(
    ("name", "arguments"),
    [
        ("profile_data", {"dataset_id": "current", "metrics": ["psi"]}),
        ("monitor_data", {"dataset_id": "current"}),
        ("monitor_data", {"dataset_id": "current", "benchmark_id": "current"}),
        (
            "profile_data",
            {
                "dataset_id": "current",
                "metrics": ["missing"],
                "features": ["customer_secret"],
            },
        ),
        ("evaluate_risk", {"dataset_id": "current", "target": "customer_secret"}),
        ("evaluate_risk", {"dataset_id": "current", "group_col": "customer_secret"}),
        ("evaluate_risk", {"dataset_id": "current", "n_bins": True}),
        ("evaluate_risk", {"dataset_id": "current", "n_bins": 1}),
        ("describe_dataset", {"dataset_id": "/etc/passwd"}),
        ("get_report_table", {"report_id": "report_1", "table": "summary"}),
    ],
)
def test_tool_rejects_unknown_roles_and_invalid_arguments(
    name: str, arguments: dict
) -> None:
    session, _, _ = make_session()
    result = MarsRiskAgent().execute_tool(name, arguments, session=session)
    assert not result.success
    assert result.error_code == "INVALID_ARGUMENTS"
    assert session.report_ids == ()


def test_unknown_tool_cannot_execute_code_or_shell() -> None:
    session, _, _ = make_session()
    result = MarsRiskAgent().execute_tool(
        "bash", {"command": "echo unexpected"}, session=session
    )
    assert result.error_code == "UNKNOWN_TOOL"


def test_raw_rows_and_unregistered_columns_are_not_sent() -> None:
    session, _, _ = make_session()
    result = MarsRiskAgent().execute_tool(
        "describe_dataset", {"dataset_id": "current"}, session=session
    )
    serialized = json.dumps(result.data)
    assert "customer_secret" not in serialized
    assert "never-send-this-value" not in serialized
    assert "rows" not in result.data
    assert result.data["row_count"] == 8


def test_invalid_labels_return_sanitized_error_and_session_remains_usable() -> None:
    session, _, current = make_session()
    bad = current.with_columns(pl.lit("private-invalid-target").alias("target"))
    session.register_dataset(
        "bad", bad, features=["score"], target="target", missing_values=[-999]
    )
    agent = MarsRiskAgent()
    result = agent.execute_tool(
        "monitor_data",
        {"dataset_id": "bad", "benchmark_id": "baseline"},
        session=session,
    )
    assert result.error_code == "COMPUTATION_FAILED"
    assert "private-invalid-target" not in result.error_message
    assert agent.execute_tool("list_datasets", {}, session=session).success


def test_missing_target_uses_unlabeled_monitoring_with_baseline() -> None:
    session, _, current = make_session()
    session.register_dataset(
        "unlabeled",
        current.drop("target"),
        features=["score"],
        target="target",
        missing_values=[-999],
    )
    result = MarsRiskAgent().execute_tool(
        "monitor_data",
        {"dataset_id": "unlabeled", "benchmark_id": "baseline"},
        session=session,
    )
    assert result.success, result.error_message
    assert (
        "target_observation" not in session.get_report(result.data["report_id"]).tables
    )


def test_report_snapshot_is_independent_from_caller_mutation() -> None:
    session, _, _ = make_session()
    result = MarsRiskAgent().execute_tool(
        "profile_data",
        {"dataset_id": "current", "metrics": ["missing"]},
        session=session,
    )
    report = session.get_report(result.data["report_id"])
    report.metadata["agent_parameters"]["features"].clear()
    report.tables.clear()
    intact = session.get_report(report.id)
    assert intact.tables
    assert intact.metadata["agent_parameters"]["features"] == ["score", "income"]


def test_dataset_registration_does_not_overwrite_and_rejects_role_overlap() -> None:
    session, baseline, _ = make_session()
    with pytest.raises(ValueError, match="already registered"):
        session.register_dataset("baseline", baseline, features=["score"])
    with pytest.raises(ValueError, match="overlap"):
        session.register_dataset(
            "overlap", baseline, features=["target"], target="target"
        )


def test_report_output_reduces_page_size_without_dropping_rows() -> None:
    session = MarsAgentSession()
    session._save_report(
        "test",
        "data",
        None,
        {
            "summary": pl.DataFrame(
                {"feature": ["x" * 150] * 20, "value": list(range(20))}
            ),
        },
        {},
    )
    agent = MarsRiskAgent(max_result_chars=1024)
    result = agent.execute_tool(
        "get_report_table",
        {"report_id": "report_1", "table": "summary"},
        session=session,
    )
    assert result.success
    assert 0 < result.data["returned_rows"] < 20
    assert result.data["next_offset"] == result.data["returned_rows"]
    assert len(json.dumps(result.data, ensure_ascii=False)) <= 1024


def test_nonfinite_report_values_are_json_null() -> None:
    session = MarsAgentSession()
    session._save_report(
        "test",
        "data",
        None,
        {"summary": pl.DataFrame({"x": [float("nan"), float("inf")]})},
        {},
    )
    result = MarsRiskAgent().execute_tool(
        "get_report_table",
        {"report_id": "report_1", "table": "summary"},
        session=session,
    )
    assert result.data["rows"] == [{"x": None}, {"x": None}]
    json.dumps(result.data, allow_nan=False)


def test_projection_can_query_a_wide_report_with_small_budget() -> None:
    session = MarsAgentSession()
    session._save_report(
        "test",
        "data",
        None,
        {
            "summary": pl.DataFrame(
                {"feature": ["x"], "value": [1], "large": ["a" * 5000]}
            ),
        },
        {},
    )
    agent = MarsRiskAgent(max_result_chars=1024)
    result = agent.execute_tool(
        "get_report_table",
        {
            "report_id": "report_1",
            "table": "summary",
            "columns": ["feature", "value"],
        },
        session=session,
    )
    assert result.success
    assert result.data["rows"] == [{"feature": "x", "value": 1}]
    invalid = agent.execute_tool(
        "get_report_table",
        {
            "report_id": "report_1",
            "table": "summary",
            "columns": ["unknown"],
        },
        session=session,
    )
    assert invalid.error_code == "INVALID_ARGUMENTS"


def test_benchmark_role_mismatch_is_rejected() -> None:
    session, baseline, _ = make_session()
    session.register_dataset(
        "other", baseline, features=["score"], target=None, missing_values=[-999]
    )
    result = MarsRiskAgent().execute_tool(
        "monitor_data",
        {
            "dataset_id": "current",
            "benchmark_id": "other",
            "features": ["score"],
        },
        session=session,
    )
    assert result.error_code == "INVALID_ARGUMENTS"


def test_dataset_snapshot_survives_caller_frame_mutation() -> None:
    original = pl.DataFrame({"x": [1, None]})
    session = MarsAgentSession()
    session.register_dataset("snapshot", original, features=["x"])
    original.replace_column(0, pl.Series("x", [1, 2]))
    result = MarsRiskAgent().execute_tool(
        "profile_data",
        {
            "dataset_id": "snapshot",
            "metrics": ["missing"],
        },
        session=session,
    )
    expected = profile_stats(
        pl.DataFrame({"x": [1, None]}), features=["x"], metrics=["missing"]
    )
    assert_frame_equal(
        session.get_report(result.data["report_id"]).tables["dq.missing"],
        expected.dq_tables["missing"],
    )
