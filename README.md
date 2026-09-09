# Wholesale Order Status Agent

A small agent that answers order-status questions in natural language for
C&S Wholesale Grocers customers. It uses the OpenAI Chat Completions API
tool-call loop with a single tool, `get_order_status`, backed today by
hardcoded data.

## Project layout

```
main.py       # `python main.py` entry point
agent.py      # OrderStatusAgent (the tool-call loop) + FastAPI app: POST /chat, GET /health
data.py       # hardcoded order records (swap this for a real API later)
tools.py      # get_order_status() + its OpenAI function-calling schema
cli.py        # terminal REPL for local testing
tests/
  test_tools.py
  test_agent.py
  test_server.py
```

Flat, top-level modules rather than a package — no Dockerfile either.
This matches what the hosting platform expects: it detects Python from
`requirements.txt` and runs the process directly with the start command
you configure (`python main.py`), so there's no build step to keep in
sync with a separate container image.

## Setup

Requires Python 3.11 or 3.12 (avoid 3.13/3.14).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` (or export directly) and set:

- `OPENAI_API_KEY` — required, your OpenAI API key.
- `OPENAI_MODEL` — optional, defaults to `gpt-4o-mini`.
- `OPENAI_URL` — optional, defaults to OpenAI's own API (`https://api.openai.com/v1`).
  Set this to point at a compatible endpoint instead — e.g. an LLM gateway
  or proxy in front of OpenAI. When set, `OPENAI_API_KEY` is sent as an
  `X-API-Key` header instead of OpenAI's own `Authorization: Bearer`
  scheme, since gateways/proxies typically expect their own key header
  rather than OpenAI's. Left unset, requests go straight to OpenAI with
  its normal `Authorization: Bearer` header.
- `PORT` — optional, defaults to `8000`. Used by the server.
- `LOG_LEVEL` — optional, defaults to `INFO`. Set to `DEBUG` to also log
  each `/chat` request's message and session, the response's tool calls
  and whether an order was found, and the OpenAI SDK's own HTTP-level
  debug logs (request/response headers, with the API key masked) —
  useful when debugging against a gateway/proxy deployment where you
  can't attach a local debugger. This is verbose; leave it at `INFO`
  normally.

On startup the server logs the effective model, `OPENAI_URL` (or "OpenAI
default"), a masked `OPENAI_API_KEY`, and an `api_key_fingerprint` — check
this first when a deployed instance behaves unexpectedly, to confirm it
picked up the environment variables you think it did. A failed `/chat`
request logs a compact one-line summary — exception type + message, plus
the same `base_url`/masked-key/fingerprint that was actually in effect
for that request — before the full traceback, so the cause is
diagnosable from that one line alone even if a log viewer truncates or
reorders multi-line output. At `LOG_LEVEL=DEBUG`, each outbound request
also logs the exact headers being sent (e.g. `Authorization: <omitted>,
X-API-Key: sk-proj...ab3f` when routed through a custom `OPENAI_URL`) —
useful for confirming exactly what's on the wire when a gateway rejects
a request.

The key itself is never logged in full — only masked (first 6 + last 4
characters) or as a short SHA-256 `api_key_fingerprint`. This app may run
on a shared platform where log access is broader than just you, and logs
often get retained or shipped elsewhere, so a full key is never worth the
risk. The fingerprint still lets you confirm with certainty whether a
deployed key is identical to one you already have, without ever exposing
either value — compute the same fingerprint locally for the key you
expect and compare:

```bash
echo -n "$OPENAI_API_KEY" | shasum -a 256 | cut -c1-12
```

If it matches the fingerprint in the logs, the deployed key is that
exact value; if not, the deployment has a different key than you think —
fix that by setting the correct value on the platform and redeploying,
not by trying to read the wrong one back out.

## Running the CLI

For quick local multi-turn testing in the terminal:

```bash
export OPENAI_API_KEY=sk-...
python cli.py
```

Type an order number or question, and `quit`/`exit` to leave. Conversation
history is kept in memory for the life of the REPL session.

## Running the server

```bash
export OPENAI_API_KEY=sk-...
python main.py
# → listening on http://localhost:8000
```

or directly with uvicorn:

```bash
export OPENAI_API_KEY=sk-...
uvicorn agent:app --host 0.0.0.0 --port 8000
```

### Sample request

The server implements a standard "Chat Agent" HTTP contract:
`POST /chat` takes `{message, session_id, context}` and returns
`{response}`.

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is the status of order CS-2026-100563?", "session_id": "demo-session-1"}'
```

Response:

```json
{
  "response": "Order CS-2026-100563 is currently in transit..."
}
```

`context` is accepted (any JSON value) but not currently used — it's part
of the standard contract for compatibility with the hosting platform.

Conversation history is kept server-side in memory, keyed by `session_id`
(see `SESSIONS` in `agent.py`). Send the same `session_id` on later
requests to continue that conversation. This means:

- History does **not** survive a server restart.
- History is **not** shared across replicas if you scale the server out
  horizontally — each replica has its own in-memory session store. For
  multi-replica deployments, swap `SESSIONS` for a shared store (e.g.
  Redis) keyed the same way.

`OrderStatusAgent.run()` itself remains stateless — it's only the FastAPI
layer in `agent.py` that adds the session store, to match the platform's
contract.

`GET /health` returns `{"status": "ok"}` and does not call the model, so
it's safe to use as a liveness/readiness probe. `GET /` returns a short
service description.

## Tests

```bash
python -m pytest
```

`test_tools.py` covers every order status, unknown orders, malformed input,
and order-number normalization variants. `test_agent.py` mocks the OpenAI
client, so the tool-call loop is tested without any live API calls.
`test_server.py` mocks `OrderStatusAgent` and covers the `/chat` contract,
session continuity across requests, and error handling.

## Replacing the hardcoded data with a real backend

All order data currently lives in `data.py` as an in-memory dict. The rest
of the codebase — `agent.py`, `cli.py` — only ever calls
`get_order_status()` from `tools.py` and never touches `data.py` directly,
so swapping the backend means changing one function:

1. In `tools.py`, replace the `ORDERS.get(normalized)` lookup inside
   `get_order_status()` with a call to the real order-management API/DB
   (e.g. an HTTP client call or a repository/DAO call), keeping the input
   normalization and the `{"found": ..., "order": ...}` /
   `{"found": False, "error": ..., "message": ...}` response shape the same.
2. Handle the new failure modes a real backend introduces (timeouts,
   5xx responses, auth errors) by mapping them to the existing
   `{"found": False, "error": ..., "message": ...}` shape rather than
   raising — `get_order_status()` must still never raise.
3. Delete or repurpose `data.py` once nothing references `ORDERS` directly.

No changes are needed in `agent.py` or `cli.py` — they only depend on the
`get_order_status` function signature and `ORDER_STATUS_TOOL_SCHEMA`, both
of which stay the same.
# OrderStatusAgent
