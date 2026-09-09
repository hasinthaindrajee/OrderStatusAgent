"""C&S Wholesale order-status agent — FastAPI service exposing POST /chat.

Uses the OpenAI Chat Completions tool-calling API directly (no agent
framework) to answer order-status questions, backed by hardcoded data in
data.py.

POST /chat only round-trips {message, session_id, context} -> {response},
so conversation history lives here in memory, keyed by session_id, rather
than being passed by the caller the way OrderStatusAgent.run() otherwise
expects. This is lost on restart and isn't shared across replicas — swap
SESSIONS for a shared store (e.g. Redis) if this needs to scale out.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI, Omit
from pydantic import BaseModel

from tools import ORDER_STATUS_TOOL_SCHEMA, get_order_status

LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("order_status_agent")

OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_URL = os.environ.get("OPENAI_URL")


def _mask_key(key: str | None) -> str:
    """Partially reveal a secret for logging: enough to tell keys apart
    without ever writing the full value to logs."""
    if not key:
        return "<missing>"
    if len(key) <= 10:
        return f"...{key[-4:]}"
    return f"{key[:6]}...{key[-4:]}"


def _resolve_api_key(explicit: str | None) -> tuple[str | None, str]:
    """Resolve the OpenAI/gateway API key and where it came from.

    AGENT_OPENAI_API_KEY takes priority over OPENAI_API_KEY. This exists as
    an override escape hatch for platforms that auto-manage or auto-inject
    OPENAI_API_KEY themselves (e.g. from their own secret store) in a way
    you can't directly control — set AGENT_OPENAI_API_KEY instead and it
    always wins, regardless of what OPENAI_API_KEY ends up being.
    """
    if explicit:
        return explicit, "explicit api_key argument"
    override = os.environ.get("AGENT_OPENAI_API_KEY")
    if override:
        return override, "AGENT_OPENAI_API_KEY"
    return os.environ.get("OPENAI_API_KEY"), "OPENAI_API_KEY"


def _key_fingerprint(key: str | None) -> str:
    """A short SHA-256 fingerprint of a secret, safe to log.

    Two keys produce the same fingerprint only if they're byte-for-byte
    identical, so this lets you confirm a deployed key matches a known-good
    one (compute the same fingerprint locally and compare) without ever
    exposing the actual value: `echo -n "$KEY" | shasum -a 256 | cut -c1-12`.
    """
    if not key:
        return "<missing>"
    return hashlib.sha256(key.encode()).hexdigest()[:12]

SYSTEM_PROMPT = """You are a customer service assistant for C&S Wholesale Grocers, \
helping customers check the status of their orders.

Rules you must follow:
- If the customer hasn't given you an order number, ask for one before doing \
anything else. Order numbers look like CS-2026-100482.
- Always use the get_order_status tool to look up an order. Never guess, \
assume, or make up a status.
- When you report on an order, don't just give the status — summarize the \
full order in plain, friendly language: status, the promised (and actual, \
if delivered) delivery date, distribution center, carrier and tracking \
number when available, the purchase order number, and the line item/case \
counts.
- If an order is in EXCEPTION status, clearly explain the exception reason.
- If the tool reports the order was not found or the order number was \
malformed, tell the customer clearly and ask them to double-check the \
order number.
- You only handle order status questions. If asked about anything else \
(pricing, placing new orders, account changes, general chit-chat, etc.), \
politely explain that you can only help with order status and redirect \
the customer to the right channel.
"""

MAX_TOOL_ITERATIONS = 5


class OrderStatusAgent:
    """Stateless wrapper around the OpenAI Chat Completions tool-call loop."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = OPENAI_MODEL,
        base_url: str | None = OPENAI_URL,
    ) -> None:
        raw_key, key_source = _resolve_api_key(api_key)
        if not raw_key:
            raise ValueError(
                "No OpenAI API key found. Pass api_key explicitly, or set "
                "AGENT_OPENAI_API_KEY (takes priority — a safe override if "
                "the platform manages OPENAI_API_KEY itself) or OPENAI_API_KEY."
            )

        # Some platforms mount secrets as files and expose them as env vars
        # verbatim, trailing newline/whitespace included — which silently
        # changes the value's identity (and its fingerprint) without
        # changing how it looks when printed. Strip defensively, and log
        # when this actually happened so a mismatch like that is visible
        # rather than a mysterious "wrong key" that isn't actually wrong.
        resolved_key = raw_key.strip()
        if resolved_key != raw_key:
            log.warning(
                "%s had leading/trailing whitespace stripped (raw_len=%s, "
                "stripped_len=%s, raw_fingerprint=%s, stripped_fingerprint=%s) "
                "— the platform may be injecting it with extra characters "
                "(e.g. a trailing newline from a mounted secret file). "
                "Compare stripped_fingerprint against your known-good key's "
                "fingerprint.",
                key_source,
                len(raw_key),
                len(resolved_key),
                _key_fingerprint(raw_key),
                _key_fingerprint(resolved_key),
            )
        if base_url:
            base_url = base_url.strip()

        # base_url=None lets the OpenAI SDK use its own default endpoint
        # (https://api.openai.com/v1).
        self._client = OpenAI(api_key=resolved_key, base_url=base_url)
        self._model = model

        # For logging only — never the raw key. Lets a failed request's log
        # line show exactly which endpoint/key combination was in effect,
        # without scrolling back to find the startup log. api_key_fingerprint
        # lets you confirm byte-for-byte whether a deployed key matches a
        # known-good one, without ever exposing the actual value — compute
        # the same fingerprint locally: echo -n "$KEY" | shasum -a 256 | cut -c1-12
        self.masked_api_key = _mask_key(resolved_key)
        self.api_key_fingerprint = _key_fingerprint(resolved_key)
        self.api_key_length = len(resolved_key)
        self.api_key_source = key_source
        self.base_url_label = base_url or "(OpenAI default)"

        if base_url:
            # Routing through a custom endpoint (e.g. an LLM gateway/proxy)
            # instead of OpenAI directly. These typically expect their own
            # API key header rather than OpenAI's "Authorization: Bearer"
            # scheme, so swap it in per-request: Omit() drops the SDK's
            # default Authorization header. This has to be passed as
            # per-request extra_headers (not client-level default_headers)
            # — openai>=2.0 only recognizes an omitted Authorization header
            # when it's set at the request level; setting it at client
            # construction time raises "Could not resolve authentication
            # method" even though the header ends up correct either way.
            self._extra_headers: dict[str, Any] | None = {
                "Authorization": Omit(),
                "X-API-Key": resolved_key,
            }
            # Same headers, but with the key masked — safe to put in logs.
            self._loggable_extra_headers = {
                "Authorization": "<omitted>",
                "X-API-Key": self.masked_api_key,
            }
        else:
            self._extra_headers = None
            self._loggable_extra_headers = {"Authorization": f"Bearer {self.masked_api_key}"}

    def run(
        self,
        user_message: str,
        conversation_history: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Run one turn of the tool-call loop and return the final reply.

        Stateless: `conversation_history` is read and a new, extended list is
        returned. Nothing is kept on `self`, so callers can hand the same
        agent instance concurrent requests for different conversations. The
        system prompt is not part of the stored/returned history — it's
        prepended fresh on every call.

        The returned dict includes `order`: the full structured order record
        (every field from data.py, not just status) from the most recent
        successful lookup this turn, or None if no order was found/looked up.
        """
        history: list[dict[str, Any]] = list(conversation_history) if conversation_history else []
        history.append({"role": "user", "content": user_message})

        tools_called: list[dict[str, Any]] = []

        for _ in range(MAX_TOOL_ITERATIONS):
            log.debug(
                "Calling %s/chat/completions with headers=%s",
                str(self._client.base_url).rstrip("/"),
                self._loggable_extra_headers,
            )
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "system", "content": SYSTEM_PROMPT}] + history,
                tools=[ORDER_STATUS_TOOL_SCHEMA],
                tool_choice="auto",
                extra_headers=self._extra_headers,
            )

            assistant_message = _serialize_assistant_message(response.choices[0].message)
            history.append(assistant_message)

            tool_calls = assistant_message.get("tool_calls")
            if not tool_calls:
                return {
                    "reply": assistant_message.get("content") or "",
                    "conversation_history": history,
                    "tools_called": tools_called,
                    "order": _last_found_order(tools_called),
                }

            for tool_call in tool_calls:
                function = tool_call["function"]

                if function["name"] == "get_order_status":
                    arguments = _parse_arguments(function["arguments"])
                    order_number = arguments.get("order_number", "")
                    result = get_order_status(order_number)
                    tools_called.append(
                        {"name": function["name"], "input": arguments, "result": result}
                    )
                    history.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call["id"],
                            "content": json.dumps(result),
                        }
                    )
                else:
                    history.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call["id"],
                            "content": json.dumps({"error": f"unknown tool: {function['name']}"}),
                        }
                    )

        return {
            "reply": (
                "I'm having trouble completing that request right now. "
                "Please try again in a moment or contact support."
            ),
            "conversation_history": history,
            "tools_called": tools_called,
            "order": _last_found_order(tools_called),
        }


def _parse_arguments(raw_arguments: str) -> dict[str, Any]:
    if not raw_arguments:
        return {}
    try:
        parsed = json.loads(raw_arguments)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _last_found_order(tools_called: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Return the full order record from the most recent successful lookup."""
    for call in reversed(tools_called):
        result = call["result"]
        if result.get("found"):
            return result["order"]
    return None


def _serialize_assistant_message(message: Any) -> dict[str, Any]:
    """Convert an SDK ChatCompletionMessage into a plain JSON-serializable dict."""
    serialized: dict[str, Any] = {"role": "assistant", "content": message.content}
    if message.tool_calls:
        serialized["tool_calls"] = [
            {
                "id": tool_call.id,
                "type": "function",
                "function": {
                    "name": tool_call.function.name,
                    "arguments": tool_call.function.arguments,
                },
            }
            for tool_call in message.tool_calls
        ]
    return serialized


# --- FastAPI service --------------------------------------------------------

_agent: OrderStatusAgent | None = None
SESSIONS: dict[str, list[dict[str, Any]]] = {}
_DEFAULT_SESSION_ID = "default"

app = FastAPI(title="C&S Wholesale Order Status Agent")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str | None = None
    session_id: str | None = None
    context: Any | None = None


class ChatResponse(BaseModel):
    response: str


@app.on_event("startup")
def _startup() -> None:
    global _agent
    log.info("Starting order-status agent: log_level=%s", LOG_LEVEL)
    try:
        _agent = OrderStatusAgent()
    except Exception:
        log.exception("Failed to initialize OrderStatusAgent at startup")
        raise
    # Logged from the agent's own (post-stripping) attributes, so this
    # reflects the value actually in effect — not just what the raw env
    # var looked like before any whitespace was stripped from it.
    log.info(
        "Order-status agent ready: model=%s base_url=%s openai_api_key=%s "
        "api_key_length=%s api_key_fingerprint=%s api_key_source=%s",
        OPENAI_MODEL,
        _agent.base_url_label,
        _agent.masked_api_key,
        _agent.api_key_length,
        _agent.api_key_fingerprint,
        _agent.api_key_source,
    )


@app.get("/")
def root() -> dict[str, Any]:
    return {
        "service": "C&S Wholesale Order Status Agent",
        "tip": "POST /chat with {message, session_id, context}. GET /health for status.",
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    if not request.message or not request.message.strip():
        raise HTTPException(status_code=400, detail="'message' is required and cannot be empty.")

    assert _agent is not None, "agent was not initialized at startup"

    session_id = request.session_id or _DEFAULT_SESSION_ID
    history = SESSIONS.get(session_id)

    log.debug(
        "chat request: session=%s history_turns=%s message=%r",
        session_id,
        len(history) if history else 0,
        request.message,
    )

    try:
        result = _agent.run(user_message=request.message, conversation_history=history)
    except Exception as exc:
        # A compact single-line summary first, in case a log viewer truncates
        # or reorders the full traceback that follows. Includes which
        # endpoint/key combination was actually in effect — key masked, plus
        # a fingerprint you can compare against a known-good key computed
        # locally (see _key_fingerprint) — so an auth failure is
        # diagnosable without cross-referencing the startup log or ever
        # exposing the actual key value.
        log.error(
            "OrderStatusAgent failed while handling /chat request "
            "(session=%s, base_url=%s, api_key=%s, api_key_length=%s, "
            "api_key_fingerprint=%s, api_key_source=%s): %s: %s",
            session_id,
            _agent.base_url_label,
            _agent.masked_api_key,
            _agent.api_key_length,
            _agent.api_key_fingerprint,
            _agent.api_key_source,
            type(exc).__name__,
            exc,
        )
        log.exception("Full traceback for the error above:")
        raise HTTPException(
            status_code=500,
            detail="Something went wrong while processing your request. Please try again.",
        ) from None

    SESSIONS[session_id] = result["conversation_history"]
    log.debug(
        "chat response: session=%s tools_called=%s order_found=%s",
        session_id,
        [call["name"] for call in result["tools_called"]],
        result["order"] is not None,
    )
    return ChatResponse(response=result["reply"])
