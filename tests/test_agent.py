from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from agent import MAX_TOOL_ITERATIONS, OrderStatusAgent


@dataclass
class FakeFunctionCall:
    name: str
    arguments: str


@dataclass
class FakeToolCall:
    id: str
    function: FakeFunctionCall
    type: str = "function"


@dataclass
class FakeMessage:
    content: str | None
    tool_calls: list[FakeToolCall] | None = None


@dataclass
class FakeChoice:
    message: FakeMessage


@dataclass
class FakeResponse:
    choices: list[FakeChoice] = field(default_factory=list)


def _text_response(text: str) -> FakeResponse:
    return FakeResponse(choices=[FakeChoice(message=FakeMessage(content=text, tool_calls=None))])


def _tool_call_response(tool_call_id: str, order_number: str) -> FakeResponse:
    tool_call = FakeToolCall(
        id=tool_call_id,
        function=FakeFunctionCall(name="get_order_status", arguments=f'{{"order_number": "{order_number}"}}'),
    )
    return FakeResponse(choices=[FakeChoice(message=FakeMessage(content=None, tool_calls=[tool_call]))])


def _client_with_responses(responses: list[FakeResponse]) -> MagicMock:
    client = MagicMock()
    client.chat.completions.create = MagicMock(side_effect=responses)
    return client


@pytest.fixture(autouse=True)
def api_key_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-123")


def test_missing_api_key_raises() -> None:
    import os

    env = dict(os.environ)
    env.pop("OPENAI_API_KEY", None)
    with patch.dict(os.environ, env, clear=True):
        with pytest.raises(ValueError):
            OrderStatusAgent()


def test_default_base_url_is_not_overridden() -> None:
    with patch("agent.OpenAI") as mock_openai_cls:
        OrderStatusAgent(api_key="test-key")

    # base_url=None lets the OpenAI SDK use its own default endpoint.
    assert mock_openai_cls.call_args.kwargs["base_url"] is None


def test_explicit_base_url_is_passed_to_client() -> None:
    with patch("agent.OpenAI") as mock_openai_cls:
        OrderStatusAgent(api_key="test-key", base_url="https://llm-gateway.example.com/v1")

    assert mock_openai_cls.call_args.kwargs["base_url"] == "https://llm-gateway.example.com/v1"


def test_run_executes_tool_and_returns_final_text() -> None:
    tool_use_response = _tool_call_response("call_1", "CS-2026-100401")
    final_response = _text_response("Your order CS-2026-100401 was delivered on 2026-09-05.")

    with patch("agent.OpenAI") as mock_openai_cls:
        mock_openai_cls.return_value = _client_with_responses([tool_use_response, final_response])

        agent = OrderStatusAgent(api_key="test-key")
        result = agent.run("What's the status of CS-2026-100401?")

    assert result["reply"] == "Your order CS-2026-100401 was delivered on 2026-09-05."
    assert len(result["tools_called"]) == 1
    assert result["tools_called"][0]["name"] == "get_order_status"
    assert result["tools_called"][0]["result"]["found"] is True
    assert result["tools_called"][0]["result"]["order"]["status"] == "DELIVERED"

    assert result["order"] is not None
    assert result["order"]["order_number"] == "CS-2026-100401"
    assert result["order"]["status"] == "DELIVERED"
    assert result["order"]["distribution_center"] == "Robesonia, PA DC"

    # history should contain: user msg, assistant tool_call, tool result, assistant final text
    assert len(result["conversation_history"]) == 4
    assert result["conversation_history"][0]["role"] == "user"
    assert result["conversation_history"][-1]["role"] == "assistant"


def test_run_with_no_tool_call_returns_text_immediately() -> None:
    final_response = _text_response("Sure, what's your order number?")

    with patch("agent.OpenAI") as mock_openai_cls:
        mock_openai_cls.return_value = _client_with_responses([final_response])

        agent = OrderStatusAgent(api_key="test-key")
        result = agent.run("Can you check my order?")

    assert result["reply"] == "Sure, what's your order number?"
    assert result["tools_called"] == []
    assert result["order"] is None
    assert len(result["conversation_history"]) == 2


def test_run_is_stateless_and_accepts_prior_history() -> None:
    final_response = _text_response("Got it, thanks!")

    with patch("agent.OpenAI") as mock_openai_cls:
        mock_openai_cls.return_value = _client_with_responses([final_response])

        agent = OrderStatusAgent(api_key="test-key")
        prior_history = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello, how can I help?"},
        ]
        result = agent.run("thanks", conversation_history=prior_history)

    # original list passed in must not be mutated
    assert len(prior_history) == 2
    assert len(result["conversation_history"]) == 4


def test_run_stops_after_max_iterations() -> None:
    responses = [_tool_call_response("call_x", "CS-2026-100401") for _ in range(MAX_TOOL_ITERATIONS)]

    with patch("agent.OpenAI") as mock_openai_cls:
        client = _client_with_responses(responses)
        mock_openai_cls.return_value = client

        agent = OrderStatusAgent(api_key="test-key")
        result = agent.run("status please")

    assert client.chat.completions.create.call_count == MAX_TOOL_ITERATIONS
    assert "trouble" in result["reply"].lower()
    assert len(result["tools_called"]) == MAX_TOOL_ITERATIONS
    assert result["order"]["order_number"] == "CS-2026-100401"


def test_run_order_is_none_when_lookup_not_found() -> None:
    tool_use_response = _tool_call_response("call_1", "CS-2026-999999")
    final_response = _text_response("I couldn't find that order.")

    with patch("agent.OpenAI") as mock_openai_cls:
        mock_openai_cls.return_value = _client_with_responses([tool_use_response, final_response])

        agent = OrderStatusAgent(api_key="test-key")
        result = agent.run("What's the status of CS-2026-999999?")

    assert result["order"] is None
    assert result["tools_called"][0]["result"]["found"] is False
