"""FastAPI app exposing OrderStatusAgent over HTTP."""

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


class ChatRequest(BaseModel):
    message: str | None = None
    conversation_history: list[dict[str, Any]] | None = None


@app.on_event("startup")
def _startup() -> None:
    global _agent
    _agent = OrderStatusAgent()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat")
def chat(request: ChatRequest) -> dict[str, Any]:
    if not request.message or not request.message.strip():
        raise HTTPException(status_code=400, detail="'message' is required and cannot be empty.")

    assert _agent is not None, "agent was not initialized at startup"

    try:
        return _agent.run(
            user_message=request.message,
            conversation_history=request.conversation_history,
        )
    except Exception:
        logger.exception("OrderStatusAgent failed while handling /chat request")
        raise HTTPException(
            status_code=500,
            detail="Something went wrong while processing your request. Please try again.",
        ) from None


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=port)
