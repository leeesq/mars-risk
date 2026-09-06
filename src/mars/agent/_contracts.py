"""Agent 的消息、工具和结果契约；JSON 仅用于模型及工具边界。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

import polars as pl


@dataclass(frozen=True)
class MarsAgentToolCall:
    """
    模型发起的一次结构化工具请求。

    Parameters
    ----------
    id : str
        当前响应内唯一的调用标识。
    name : str
        已登记的工具名称。
    arguments : dict[str, Any]
        JSON 对象参数，执行前仍须校验。
    """

    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MarsAgentMessage:
    """
    与模型厂商无关的会话消息。

    Parameters
    ----------
    role : Literal['user', 'assistant', 'tool']
        消息角色，系统提示词由运行器单独提供。
    content : str
        文本或 JSON 工具结果。
    tool_calls : tuple[MarsAgentToolCall, ...]
        助手发起的工具调用。
    tool_call_id : str | None
        工具结果对应的调用标识。
    """

    role: Literal["user", "assistant", "tool"]
    content: str = ""
    tool_calls: tuple[MarsAgentToolCall, ...] = ()
    tool_call_id: str | None = None


@dataclass(frozen=True)
class MarsAgentTool:
    """
    模型可见的工具说明及参数约束。

    Parameters
    ----------
    name : str
        工具名称。
    description : str
        用途和业务边界。
    parameters : dict[str, Any]
        工具输入的 JSON Schema。
    """

    name: str
    description: str
    parameters: dict[str, Any]


@dataclass(frozen=True)
class MarsAgentResponse:
    """
    一次模型调用的完整响应。

    Parameters
    ----------
    text : str
        模型输出文本。
    tool_calls : tuple[MarsAgentToolCall, ...]
        待执行的工具请求。
    finish_reason : str
        ``stop`` 表示完成，``tool_calls`` 表示继续，其他原因视为未完成。
    input_tokens : int
        模型服务返回的输入 token 数。
    output_tokens : int
        模型服务返回的输出 token 数。
    """

    text: str = ""
    tool_calls: tuple[MarsAgentToolCall, ...] = ()
    finish_reason: str = "stop"
    input_tokens: int = 0
    output_tokens: int = 0


class MarsAgentProvider(Protocol):
    """运行器要求的同步模型接口，可由自定义后端或测试替身实现。"""

    def chat(
        self,
        messages: list[MarsAgentMessage],
        *,
        system: str,
        tools: tuple[MarsAgentTool, ...],
    ) -> MarsAgentResponse:
        """
        获取一次完整模型响应。

        Parameters
        ----------
        messages : list[MarsAgentMessage]
            当前会话消息副本。
        system : str
            系统提示词和业务边界。
        tools : tuple[MarsAgentTool, ...]
            当前可使用的工具。

        Returns
        -------
        MarsAgentResponse
            完整响应，工具参数仍由执行器验证。
        """
        ...


@dataclass(frozen=True)
class MarsAgentToolResult:
    """
    工具执行结果及可定位的错误状态。

    Parameters
    ----------
    call_id : str
        对应调用标识。
    name : str
        工具名称。
    success : bool
        工具是否成功。
    data : dict[str, Any]
        经过大小限制的 JSON 结果。
    error_code : str | None
        失败类别；成功时为空。
    error_message : str | None
        不含底层数据值的错误提示。
    """

    call_id: str
    name: str
    success: bool
    data: dict[str, Any] = field(default_factory=dict)
    error_code: str | None = None
    error_message: str | None = None


@dataclass(frozen=True)
class MarsAgentReport:
    """
    会话内可追溯的 MARS 计算结果快照。

    Parameters
    ----------
    id : str
        会话内唯一报告标识。
    kind : str
        来源工具名称。
    dataset_id : str
        当前数据标识。
    benchmark_id : str | None
        基准数据标识。
    tables : dict[str, pl.DataFrame]
        规范命名的完整聚合结果表。
    metadata : dict[str, Any]
        MARS 元数据及实际生效参数，仅保存在本地。
    """

    id: str
    kind: str
    dataset_id: str
    benchmark_id: str | None
    tables: dict[str, pl.DataFrame]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class MarsAgentResult:
    """
    一轮用户请求的回答和执行记录。

    Parameters
    ----------
    text : str
        回答文本；模型结论仍需结合证据复核。
    status : str
        ``completed``、``incomplete`` 或达到资源上限的停止原因。
    iterations : int
        本轮模型调用次数。
    tool_results : tuple[MarsAgentToolResult, ...]
        本轮工具调用记录。
    report_ids : tuple[str, ...]
        本轮创建或读取的报告，不代表文本中的结论已经自动验证。
    input_tokens : int
        本轮输入 token 累计量。
    output_tokens : int
        本轮输出 token 累计量。
    """

    text: str
    status: str
    iterations: int
    tool_results: tuple[MarsAgentToolResult, ...] = ()
    report_ids: tuple[str, ...] = ()
    input_tokens: int = 0
    output_tokens: int = 0
