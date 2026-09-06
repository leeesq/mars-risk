from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from mars.agent import (
    MarsAgentMessage,
    MarsAgentToolCall,
    MarsOpenAIProvider,
    MarsRiskAgent,
    _provider,
)


def make_provider(
    monkeypatch: pytest.MonkeyPatch, arguments: str = '{"dataset_id":"data"}'
) -> tuple[MarsOpenAIProvider, Mock]:
    client = Mock()
    client.chat.completions.create.return_value = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=None,
                    tool_calls=[
                        SimpleNamespace(
                            id="call_2",
                            function=SimpleNamespace(
                                name="describe_dataset", arguments=arguments
                            ),
                        )
                    ],
                ),
                finish_reason="tool_calls",
            )
        ],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=4),
    )
    sdk = Mock()
    sdk.OpenAI.return_value = client
    monkeypatch.setattr(
        _provider, "require_optional_module", lambda *args, **kwargs: sdk
    )
    return (
        MarsOpenAIProvider(
            model="test-model",
            api_key="test-key",
            base_url="https://example.invalid/v1",
        ),
        client,
    )


def test_provider_formats_tool_messages_and_preserves_protocol(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider, client = make_provider(monkeypatch)
    response = provider.chat(
        [
            MarsAgentMessage("user", "分析"),
            MarsAgentMessage(
                "assistant",
                tool_calls=(MarsAgentToolCall("call_1", "list_datasets", {}),),
            ),
            MarsAgentMessage("tool", '{"datasets":[]}', tool_call_id="call_1"),
        ],
        system="system rules",
        tools=MarsRiskAgent().tools,
    )
    request = client.chat.completions.create.call_args.kwargs
    assert request["messages"][0] == {"role": "system", "content": "system rules"}
    assert request["messages"][2]["tool_calls"][0]["function"]["arguments"] == "{}"
    assert request["messages"][3]["tool_call_id"] == "call_1"
    assert len(request["tools"]) == 6
    assert response.tool_calls[0].arguments == {"dataset_id": "data"}
    assert response.finish_reason == "tool_calls"
    assert response.input_tokens == 10
    provider.close()
    client.close.assert_called_once()


@pytest.mark.parametrize("arguments", ["{bad json}", "[]", "null", '"text"'])
def test_provider_rejects_nonobject_tool_arguments(
    monkeypatch: pytest.MonkeyPatch, arguments: str
) -> None:
    provider, _ = make_provider(monkeypatch, arguments)
    with pytest.raises(ValueError, match="JSON"):
        provider.chat([], system="rules", tools=MarsRiskAgent().tools)


def test_missing_sdk_has_optional_install_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    import mars.utils.imports as imports

    original = imports.importlib.import_module

    def missing(name: str):
        if name == "openai":
            raise ImportError("missing")
        return original(name)

    monkeypatch.setattr(imports.importlib, "import_module", missing)
    with pytest.raises(ImportError, match=r"mars-risk\[agent\]"):
        MarsOpenAIProvider(model="test", api_key="test")
