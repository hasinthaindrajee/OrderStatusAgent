"""FastAPI app exposing OrderStatusAgent over HTTP.

Implements the hosting platform's standard Chat Agent contract:

    POST /chat   {message: str, session_id: str, context: JSON} -> {response: str}
    GET  /health -> {status: "ok"}

OrderStatusAgent.run() itself is stateless — callers pass conversation
history in and get it back out. Since the platform's contract only
round-trips `message` and `session_id` (not history), this module keeps
conversation history server-side in memory, keyed by session_id. This
trades away horizontal-scaling-friendly statelessness for platform
compatibility: history is lost on restart and isn't shared across
replicas. If this needs to scale out, swap `_sessions` for a shared store
(e.g. Redis) keyed the same way.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from cs_order_agent.agent import OrderStatusAgent

logger = logging.getLogger("cs_order_agent")

app = FastAPI(title="C&S Wholesale Order Status Agent")

_agent: OrderStatusAgent | None = None
_sessions: dict[str, list[dict[str, Any]]] = {}

_DEFAULT_SESSION_ID = "default"


class ChatRequest(BaseModel):
    message: str | None = None
    session_id: str | None = None
    context: Any | None = None


@app.on_event("startup")
def _startup() -> None:
    global _agent
    _agent = OrderStatusAgent()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat")
def chat(request: ChatRequest) -> dict[str, str]:
    if not request.message or not request.message.strip():
        raise HTTPException(status_code=400, detail="'message' is required and cannot be empty.")

    assert _agent is not None, "agent was not initialized at startup"

    session_id = request.session_id or _DEFAULT_SESSION_ID
    history = _sessions.get(session_id)

    try:
        result = _agent.run(user_message=request.message, conversation_history=history)
    except Exception:
        logger.exception("OrderStatusAgent failed while handling /chat request")
        raise HTTPException(
            status_code=500,
            detail="Something went wrong while processing your request. Please try again.",
        ) from None

    _sessions[session_id] = result["conversation_history"]
    return {"response": result["reply"]}


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
