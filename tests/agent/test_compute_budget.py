from __future__ import annotations

from dataclasses import replace
from typing import Any

import polars as pl
import pytest

from mars.agent import MarsAgentComputeBudget, MarsAgentSession, MarsRiskAgent


def _session(features: int = 1) -> MarsAgentSession:
    df = pl.DataFrame({f"x{i}": [1, 2, 3, 4] for i in range(features)}).with_columns(
        pl.Series("target", [0, 1, None, 1]),
        pl.Series("group", ["a", "b", "a", "b"]),
        pl.Series("dt", ["2026-01-01", "2026-01-02", "2026-02-01", "2026-02-02"]),
    )
    session = MarsAgentSession()
    for name in ("current", "baseline"):
        session.register_dataset(
            name,
            df,
            features=[f"x{i}" for i in range(features)],
            target="target",
            group_columns=["group"],
            time_col="dt",
        )
    return session


@pytest.mark.parametrize("explicit", [True, False])
@pytest.mark.parametrize("tool", ["profile_data", "evaluate_risk", "monitor_data"])
def test_201_features_rejected_before_domain(
    monkeypatch: pytest.MonkeyPatch,
    explicit: bool,
    tool: str,
) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> None:
        pytest.fail("domain calculation must not run")

    for name in ("profile_stats", "profile_risk", "MarsMonitor"):
        monkeypatch.setattr(f"mars.agent._tools.{name}", forbidden)
    session = _session(201)
    args: dict[str, Any] = {"dataset_id": "current"}
    if tool == "profile_data":
        args["metrics"] = ["mean"]
    if tool == "monitor_data":
        args["benchmark_id"] = "baseline"
    if explicit:
        args["features"] = [f"x{i}" for i in range(201)]
    result = MarsRiskAgent().execute_tool(tool, args, session=session)
    assert result.error_code == "COMPUTE_BUDGET_EXCEEDED"
    assert result.data["dimension"] == "features"
    assert result.data["actual"] == 201 and result.data["limit"] == 200
    assert session.report_ids == ()


@pytest.mark.parametrize(
    ("limit_name", "limit", "dimension", "tool", "extra"),
    [
        ("max_current_rows", 3, "current_rows", "evaluate_risk", {}),
        ("max_benchmark_rows", 3, "benchmark_rows", "evaluate_risk", {}),
        ("max_input_cells", 7, "input_cells", "monitor_data", {}),
        ("max_groups", 3, "groups", "evaluate_risk", {"group_col": "group"}),
        ("max_time_windows", 7, "time_windows", "evaluate_risk", {}),
        ("max_bins", 4, "bins", "evaluate_risk", {}),
        ("max_bins", 9, "bins", "profile_data", {"metrics": ["psi"]}),
        ("max_estimated_rows", 1, "estimated_rows", "evaluate_risk", {}),
        ("max_estimated_cells", 1, "estimated_cells", "evaluate_risk", {}),
    ],
)
def test_scale_checks_precede_domain(
    monkeypatch: pytest.MonkeyPatch,
    limit_name: str,
    limit: int,
    dimension: str,
    tool: str,
    extra: dict[str, Any],
) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> None:
        pytest.fail("domain calculation must not run")

    for name in ("profile_stats", "profile_risk", "MarsMonitor"):
        monkeypatch.setattr(f"mars.agent._tools.{name}", forbidden)
    budget = replace(MarsAgentComputeBudget(), **{limit_name: limit})
    result = MarsRiskAgent(compute_budget=budget).execute_tool(
        tool,
        {"dataset_id": "current", "benchmark_id": "baseline", **extra},
        session=_session(),
    )
    assert result.error_code == "COMPUTE_BUDGET_EXCEEDED"
    assert result.data["dimension"] == dimension
    assert result.data["suggestion"]


def test_wide_registration_small_request_and_large_report_query() -> None:
    session = _session(201)
    agent = MarsRiskAgent()
    result = agent.execute_tool(
        "profile_data",
        {"dataset_id": "current", "features": ["x0"], "metrics": ["mean"]},
        session=session,
    )
    assert result.success, result.error_message
    report = session.get_report(result.data["report_id"])
    assert report.metadata["agent_compute_budget"]["scale"]["features"] == 1
    assert report.metadata["agent_output_budget"] == {"max_result_chars": 16000}
    # 已有报告不受当前计算预算限制。
    restricted = MarsRiskAgent(
        compute_budget=MarsAgentComputeBudget(max_features=1, max_current_rows=1)
    )
    query = restricted.execute_tool(
        "get_report_table",
        {"report_id": report.id, "table": "overview", "limit": 1},
        session=session,
    )
    assert query.success
    assert restricted.execute_tool(
        "get_report_context",
        {"report_id": report.id, "tables": ["overview"]},
        session=session,
    ).success


def test_configurable_schema_matches_runtime_and_n_bins_changes_estimate() -> None:
    agent = MarsRiskAgent(compute_budget=MarsAgentComputeBudget(max_features=1, max_bins=6))
    tool = next(t for t in agent.tools if t.name == "evaluate_risk")
    assert tool.parameters["properties"]["features"]["maxItems"] == 1
    assert tool.parameters["properties"]["n_bins"]["maximum"] == 6
    estimates = []
    for bins in (2, 6):
        session = _session()
        result = agent.execute_tool(
            "evaluate_risk", {"dataset_id": "current", "n_bins": bins}, session=session
        )
        assert result.success, result.error_message
        estimates.append(
            session.get_report(result.data["report_id"]).metadata["agent_compute_budget"]["scale"]
        )
    assert estimates[1]["estimated_rows"] > estimates[0]["estimated_rows"]


def test_categorical_bins_are_budgeted_before_domain(monkeypatch: pytest.MonkeyPatch) -> None:
    session = MarsAgentSession()
    session.register_dataset(
        "data", pl.DataFrame({"x": ["a", "b", "c"], "y": [0, 1, 0]}), features=["x"], target="y"
    )
    monkeypatch.setattr(
        "mars.agent._tools.profile_risk", lambda *a, **kw: pytest.fail("must not compute")
    )
    result = MarsRiskAgent(compute_budget=MarsAgentComputeBudget(max_bins=2)).execute_tool(
        "evaluate_risk",
        {"dataset_id": "data", "n_bins": 2},
        session=session,
    )
    assert result.data["dimension"] == "bins"
    assert result.data["actual"] == 3


def test_date_grain_group_uses_parsed_windows_before_domain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = MarsAgentSession()
    session.register_dataset(
        "data",
        pl.DataFrame({"x": [1, 2], "month": ["same", "same"], "dt": ["2026-01-01", "2026-02-01"]}),
        features=["x"],
        group_columns=["month"],
        time_col="dt",
    )
    monkeypatch.setattr(
        "mars.agent._tools.profile_stats", lambda *a, **kw: pytest.fail("must not compute")
    )
    result = MarsRiskAgent(compute_budget=MarsAgentComputeBudget(max_groups=1)).execute_tool(
        "profile_data",
        {"dataset_id": "data", "metrics": ["mean"], "group_col": "month"},
        session=session,
    )
    assert result.data["dimension"] == "groups"
    assert result.data["actual"] == 2


def test_existing_report_over_feature_budget_remains_queryable() -> None:
    from mars.analysis import profile_stats

    data = pl.DataFrame({f"x{i}": [1, 2] for i in range(201)})
    session = MarsAgentSession()
    handle = session.register_report(profile_stats(data, features=data.columns, metrics=["mean"]))
    agent = MarsRiskAgent(compute_budget=MarsAgentComputeBudget(max_features=1, max_current_rows=1))
    page = agent.execute_tool(
        "get_report_table",
        {"report_id": handle, "table": "overview", "offset": 100, "limit": 2},
        session=session,
    )
    assert page.success and page.data["total_rows"] == 201
    assert page.data["returned_rows"] == 2
    assert agent.execute_tool(
        "get_report_context",
        {"report_id": handle, "tables": ["overview"], "features": ["x0"]},
        session=session,
    ).success


@pytest.mark.parametrize("value", [0, -1, True, 2.5])
def test_invalid_budget(value: Any) -> None:
    with pytest.raises(ValueError, match="max_features"):
        MarsAgentComputeBudget(max_features=value)
