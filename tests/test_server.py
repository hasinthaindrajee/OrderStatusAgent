from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from cs_order_agent import server


@pytest.fixture(autouse=True)
def _clear_sessions() -> None:
    server._sessions.clear()
    yield
    server._sessions.clear()


def _make_result(reply: str, history: list[dict]) -> dict:
    return {"reply": reply, "conversation_history": history, "tools_called": [], "order": None}


def test_health_does_not_touch_agent() -> None:
    agent_mock = MagicMock()

    with patch("cs_order_agent.server.OrderStatusAgent", return_value=agent_mock):
        with TestClient(server.app) as client:
            response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    agent_mock.run.assert_not_called()


def test_chat_missing_message_returns_400() -> None:
    agent_mock = MagicMock()

    with patch("cs_order_agent.server.OrderStatusAgent", return_value=agent_mock):
        with TestClient(server.app) as client:
            response = client.post("/chat", json={"session_id": "s1"})

    assert response.status_code == 400
    agent_mock.run.assert_not_called()


def test_chat_returns_response_field_matching_platform_contract() -> None:
    agent_mock = MagicMock()
    agent_mock.run.return_value = _make_result(
        "Order CS-2026-100401 was delivered on 2026-09-05.",
        [{"role": "user", "content": "hi"}],
    )

    with patch("cs_order_agent.server.OrderStatusAgent", return_value=agent_mock):
        with TestClient(server.app) as client:
            response = client.post(
                "/chat",
                json={
                    "message": "status of CS-2026-100401?",
                    "session_id": "s1",
                    "context": {"some": "metadata"},
                },
            )

    assert response.status_code == 200
    assert response.json() == {"response": "Order CS-2026-100401 was delivered on 2026-09-05."}


def test_chat_without_session_id_still_works() -> None:
    agent_mock = MagicMock()
    agent_mock.run.return_value = _make_result("Sure, what's your order number?", [])

    with patch("cs_order_agent.server.OrderStatusAgent", return_value=agent_mock):
        with TestClient(server.app) as client:
            response = client.post("/chat", json={"message": "hi"})

    assert response.status_code == 200
    assert response.json() == {"response": "Sure, what's your order number?"}


def test_chat_session_continuity_passes_prior_history() -> None:
    agent_mock = MagicMock()
    agent_mock.run.side_effect = [
        _make_result("first reply", [{"role": "user", "content": "a"}]),
        _make_result(
            "second reply",
            [{"role": "user", "content": "a"}, {"role": "user", "content": "b"}],
        ),
    ]

    with patch("cs_order_agent.server.OrderStatusAgent", return_value=agent_mock):
        with TestClient(server.app) as client:
            client.post("/chat", json={"message": "a", "session_id": "s1"})
            client.post("/chat", json={"message": "b", "session_id": "s1"})

    first_call_kwargs = agent_mock.run.call_args_list[0].kwargs
    second_call_kwargs = agent_mock.run.call_args_list[1].kwargs

    assert first_call_kwargs["conversation_history"] is None
    assert second_call_kwargs["conversation_history"] == [{"role": "user", "content": "a"}]


def test_chat_different_sessions_do_not_share_history() -> None:
    agent_mock = MagicMock()
    agent_mock.run.side_effect = [
        _make_result("reply for s1", [{"role": "user", "content": "a"}]),
        _make_result("reply for s2", [{"role": "user", "content": "z"}]),
    ]

    with patch("cs_order_agent.server.OrderStatusAgent", return_value=agent_mock):
        with TestClient(server.app) as client:
            client.post("/chat", json={"message": "a", "session_id": "s1"})
            client.post("/chat", json={"message": "z", "session_id": "s2"})

    second_call_kwargs = agent_mock.run.call_args_list[1].kwargs
    assert second_call_kwargs["conversation_history"] is None


def test_chat_agent_failure_returns_500_without_leaking_traceback() -> None:
    agent_mock = MagicMock()
    agent_mock.run.side_effect = RuntimeError("boom - secret internal detail")

    with patch("cs_order_agent.server.OrderStatusAgent", return_value=agent_mock):
        with TestClient(server.app) as client:
            response = client.post("/chat", json={"message": "hi", "session_id": "s1"})

    assert response.status_code == 500
    assert "secret internal detail" not in response.text
