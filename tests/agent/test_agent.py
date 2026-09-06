from __future__ import annotations

from copy import deepcopy

import polars as pl
import pytest

from mars.agent import (
    MarsAgentMessage,
    MarsAgentResponse,
    MarsAgentSession,
    MarsAgentTool,
    MarsAgentToolCall,
    MarsRiskAgent,
)


class ScriptedProvider:
    def __init__(self, responses: list[MarsAgentResponse | Exception]) -> None:
        self.responses = iter(responses)
        self.requests: list[list[MarsAgentMessage]] = []

    def chat(
        self,
        messages: list[MarsAgentMessage],
        *,
        system: str,
        tools: tuple[MarsAgentTool, ...],
    ) -> MarsAgentResponse:
        self.requests.append(deepcopy(messages))
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response


def tool_response(*calls: MarsAgentToolCall) -> MarsAgentResponse:
    return MarsAgentResponse(
        tool_calls=calls, finish_reason="tool_calls", input_tokens=10, output_tokens=5
    )


def session_with_data() -> MarsAgentSession:
    session = MarsAgentSession()
    session.register_dataset("data", pl.DataFrame({"x": [1, 2, None]}), features=["x"])
    return session


def test_agent_creates_reads_report_and_preserves_history_for_followup() -> None:
    provider = ScriptedProvider(
        [
            tool_response(
                MarsAgentToolCall(
                    "c1", "profile_data", {"dataset_id": "data", "metrics": ["missing"]}
                )
            ),
            tool_response(
                MarsAgentToolCall(
                    "c2",
                    "get_report_table",
                    {"report_id": "report_1", "table": "dq.missing"},
                )
            ),
            MarsAgentResponse(
                "证据见 report_1/dq.missing。", input_tokens=5, output_tokens=3
            ),
            MarsAgentResponse("继续引用 report_1/dq.missing。"),
        ]
    )
    session = session_with_data()
    agent = MarsRiskAgent(provider)
    result = agent.run("分析缺失", session=session)
    assert result.status == "completed"
    assert result.iterations == 3
    assert result.report_ids == ("report_1",)
    assert result.input_tokens == 25
    assert result.output_tokens == 13
    assert [message.role for message in session.messages] == [
        "user",
        "assistant",
        "tool",
        "assistant",
        "tool",
        "assistant",
    ]
    assert provider.requests[2][-1].tool_call_id == "c2"
    agent.run("继续说明", session=session)
    assert provider.requests[-1][0].content == "分析缺失"
    assert provider.requests[-1][-2].content == result.text
    assert provider.requests[-1][-1].content == "继续说明"


def test_tool_failure_is_returned_to_model_for_recovery() -> None:
    provider = ScriptedProvider(
        [
            tool_response(
                MarsAgentToolCall("c1", "describe_dataset", {"dataset_id": "missing"})
            ),
            tool_response(MarsAgentToolCall("c2", "list_datasets", {})),
            MarsAgentResponse("请指定登记的数据。"),
        ]
    )
    result = MarsRiskAgent(provider).run("分析", session=session_with_data())
    assert result.status == "completed"
    assert not result.tool_results[0].success
    assert result.tool_results[1].success
    assert "INVALID_ARGUMENTS" in provider.requests[1][-1].content


def test_same_response_tools_execute_in_dependency_order() -> None:
    provider = ScriptedProvider(
        [
            tool_response(
                MarsAgentToolCall(
                    "c1", "profile_data", {"dataset_id": "data", "metrics": ["missing"]}
                ),
                MarsAgentToolCall(
                    "c2",
                    "get_report_table",
                    {"report_id": "report_1", "table": "dq.missing"},
                ),
            ),
            MarsAgentResponse("完成。"),
        ]
    )
    result = MarsRiskAgent(provider).run("分析", session=session_with_data())
    assert all(item.success for item in result.tool_results)


def test_iteration_limit_keeps_tool_call_result_pairs() -> None:
    provider = ScriptedProvider(
        [tool_response(MarsAgentToolCall("c1", "list_datasets", {}))]
    )
    session = session_with_data()
    result = MarsRiskAgent(provider, max_iterations=1).run("分析", session=session)
    assert result.status == "max_iterations"
    assert [message.role for message in session.messages] == [
        "user",
        "assistant",
        "tool",
        "assistant",
    ]


def test_tool_budget_does_not_execute_partial_batch() -> None:
    provider = ScriptedProvider(
        [
            tool_response(
                MarsAgentToolCall("c1", "list_datasets", {}),
                MarsAgentToolCall("c2", "list_datasets", {}),
            )
        ]
    )
    session = session_with_data()
    result = MarsRiskAgent(provider, max_tool_calls=1).run("分析", session=session)
    assert result.status == "max_tool_calls"
    assert result.tool_results == ()
    assert [message.role for message in session.messages] == ["user", "assistant"]


def test_context_budget_prevents_network_call_and_does_not_commit_large_prompt() -> (
    None
):
    provider = ScriptedProvider([])
    session = session_with_data()
    result = MarsRiskAgent(provider, max_context_chars=4096).run(
        "x" * 5000, session=session
    )
    assert result.status == "context_limit"
    assert provider.requests == []
    assert session.messages == ()


def test_length_stop_is_incomplete() -> None:
    result = MarsRiskAgent(
        ScriptedProvider([MarsAgentResponse("部分回答", finish_reason="length")])
    ).run(
        "分析",
        session=session_with_data(),
    )
    assert result.status == "incomplete"


def test_failed_provider_does_not_leave_unmatched_tool_messages_or_lock() -> None:
    session = session_with_data()
    provider = ScriptedProvider(
        [
            tool_response(
                MarsAgentToolCall(
                    "c1", "profile_data", {"dataset_id": "data", "metrics": ["missing"]}
                )
            ),
            RuntimeError("provider unavailable"),
            MarsAgentResponse("已恢复。"),
        ]
    )
    agent = MarsRiskAgent(provider)
    with pytest.raises(RuntimeError, match="provider unavailable"):
        agent.run("分析", session=session)
    assert session.messages == ()
    assert session.report_ids == ("report_1",)
    assert agent.run("读取已有报告", session=session).status == "completed"


def test_duplicate_ids_are_rejected_before_execution() -> None:
    session = session_with_data()
    provider = ScriptedProvider(
        [
            tool_response(
                MarsAgentToolCall("same", "list_datasets", {}),
                MarsAgentToolCall("same", "list_datasets", {}),
            )
        ]
    )
    with pytest.raises(ValueError, match="invalid tool-call"):
        MarsRiskAgent(provider).run("分析", session=session)
    assert session.messages == ()


def test_session_isolation_and_history_reset() -> None:
    first, second = session_with_data(), session_with_data()
    agent = MarsRiskAgent(ScriptedProvider([MarsAgentResponse("回答。")]))
    agent.run("分析", session=first)
    result = agent.execute_tool(
        "profile_data", {"dataset_id": "data", "metrics": ["missing"]}, session=first
    )
    assert not agent.execute_tool(
        "get_report_table",
        {"report_id": result.data["report_id"], "table": "overview"},
        session=second,
    ).success
    first.clear_history()
    assert first.messages == ()
    assert first.report_ids == ("report_1",)


def test_concurrent_session_use_is_rejected() -> None:
    session = session_with_data()
    session._lock.acquire()
    try:
        with pytest.raises(RuntimeError, match="busy"):
            MarsRiskAgent().execute_tool("list_datasets", {}, session=session)
        with pytest.raises(RuntimeError, match="busy"):
            session.clear_history()
    finally:
        session._lock.release()
