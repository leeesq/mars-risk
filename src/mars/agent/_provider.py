"""延迟加载 OpenAI SDK 的兼容接口适配器，不在导入 MARS 时读取凭据或联网。"""

from __future__ import annotations

import json
import math
from typing import Any

from mars.utils.imports import require_optional_module

from ._contracts import (
    MarsAgentMessage,
    MarsAgentResponse,
    MarsAgentTool,
    MarsAgentToolCall,
)
from ._tools import encode_json


class MarsOpenAIProvider:
    """
    通过 Chat Completions 接口调用支持工具调用的模型。

    Parameters
    ----------
    model : str
        服务端模型名称，由调用方显式指定。
    api_key : str
        对应服务的凭据，不写入消息或日志。
    base_url : str | None
        兼容服务地址；为空时使用 SDK 默认地址。
    timeout : float
        单次请求超时秒数。
    max_retries : int
        SDK 的请求重试次数，不重复执行本地工具。
    max_tokens : int
        单次模型响应 token 上限。

    Raises
    ------
    ValueError
        模型、凭据或资源参数无效时抛出。

    Notes
    -----
    SDK 仅在创建实例时导入。使用自定义 MarsAgentProvider 无需安装 SDK。
    """

    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        base_url: str | None = None,
        timeout: float = 60.0,
        max_retries: int = 2,
        max_tokens: int = 4096,
    ) -> None:
        if not model.strip() or not api_key.strip():
            raise ValueError("model and api_key must be non-empty")
        if (
            not math.isfinite(timeout)
            or timeout <= 0
            or type(max_retries) is not int
            or max_retries < 0
        ):
            raise ValueError(
                "timeout must be positive and max_retries must be a non-negative integer"
            )
        if type(max_tokens) is not int or max_tokens < 1:
            raise ValueError("max_tokens must be a positive integer")
        sdk = require_optional_module(
            "openai",
            feature_name="MARS Agent OpenAI-compatible provider",
            extra_hint='pip install "mars-risk[agent]" (Python >=3.10)',
        )
        self._client = sdk.OpenAI(
            api_key=api_key, base_url=base_url, timeout=timeout, max_retries=max_retries
        )
        self._model = model
        self._max_tokens = max_tokens

    def chat(
        self,
        messages: list[MarsAgentMessage],
        *,
        system: str,
        tools: tuple[MarsAgentTool, ...],
    ) -> MarsAgentResponse:
        """
        转换消息和工具协议并解析完整响应。

        Parameters
        ----------
        messages : list[MarsAgentMessage]
            用户、助手和配对的工具结果。
        system : str
            系统提示词。
        tools : tuple[MarsAgentTool, ...]
            允许模型调用的工具定义。

        Returns
        -------
        MarsAgentResponse
            统一响应及真实停止原因。

        Raises
        ------
        ValueError
            服务未返回候选，或工具参数不是有效 JSON 对象时抛出。
        """
        formatted: list[dict[str, Any]] = [{"role": "system", "content": system}]
        for message in messages:
            item: dict[str, Any] = {"role": message.role, "content": message.content}
            if message.tool_calls:
                item["tool_calls"] = [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.name,
                            "arguments": encode_json(call.arguments),
                        },
                    }
                    for call in message.tool_calls
                ]
            if message.role == "tool":
                item["tool_call_id"] = message.tool_call_id
            formatted.append(item)
        response = self._client.chat.completions.create(
            model=self._model,
            messages=formatted,
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters,
                    },
                }
                for tool in tools
            ],
            max_tokens=self._max_tokens,
        )
        if not response.choices:
            raise ValueError("provider returned no choices")
        choice = response.choices[0]
        calls = []
        for call in choice.message.tool_calls or []:
            try:
                arguments = json.loads(call.function.arguments)
            except (TypeError, json.JSONDecodeError) as exc:
                raise ValueError("provider tool arguments must be valid JSON") from exc
            if not isinstance(arguments, dict):
                raise ValueError("provider tool arguments must be a JSON object")
            calls.append(MarsAgentToolCall(call.id, call.function.name, arguments))
        usage = response.usage
        return MarsAgentResponse(
            text=choice.message.content or "",
            tool_calls=tuple(calls),
            finish_reason=choice.finish_reason or "unknown",
            input_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
            output_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
        )

    def close(self) -> None:
        """关闭 SDK 客户端持有的 HTTP 连接。"""
        self._client.close()
