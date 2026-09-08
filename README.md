# C&S Wholesale Order Status Agent

A small agent that answers order-status questions in natural language for
C&S Wholesale Grocers customers. It uses the OpenAI Chat Completions API
tool-call loop with a single tool, `get_order_status`, backed today by
hardcoded data.

## Project layout

```
cs_order_agent/
  data.py     # hardcoded order records (swap this for a real API later)
  tools.py    # get_order_status() + its OpenAI function-calling schema
  agent.py    # OrderStatusAgent — the tool-use loop
  server.py   # FastAPI app: POST /chat, GET /health
  cli.py      # terminal REPL for local testing
tests/
  test_tools.py
  test_agent.py
```

## Setup

Requires Python 3.11+.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` (or export directly) and set:

- `OPENAI_API_KEY` — required, your OpenAI API key.
- `PORT` — optional, defaults to `8080`. Used by the server.

## Running the CLI

For quick local multi-turn testing in the terminal:

```bash
export OPENAI_API_KEY=sk-...
python -m cs_order_agent.cli
```

Type an order number or question, and `quit`/`exit` to leave. Conversation
history is kept in memory for the life of the REPL session.

## Running the server

```bash
export OPENAI_API_KEY=sk-...
uvicorn cs_order_agent.server:app --host 0.0.0.0 --port 8080
```

or with Docker:

```bash
docker build -t cs-order-agent .
docker run -p 8080:8080 -e OPENAI_API_KEY=sk-... cs-order-agent
```

### Sample request

```bash
curl -X POST http://localhost:8080/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is the status of order CS-2026-100563?"}'
```

Response:

```json
{
  "reply": "Order CS-2026-100563 is currently in transit...",
  "conversation_history": [ ... ],
  "tools_called": [ ... ],
  "order": {
    "order_number": "CS-2026-100563",
    "customer_name": "Tops Friendly Markets #029",
    "status": "IN_TRANSIT",
    "carrier": "Estes Express Lines",
    "tracking_number": "EXL9928374615",
    "...": "..."
  }
}
```

`order` is the full structured order record (every field from `data.py`) for
the most recent successful lookup in that turn, or `null` if no order was
found or looked up — use it to render order details in a UI without parsing
the natural-language `reply`.

Pass the returned `conversation_history` back in on the next call (as
`conversation_history` in the request body) to continue the conversation —
the server and the agent are both stateless between requests.

`GET /health` returns `{"status": "ok"}` and does not call the model, so it's
safe to use as a liveness/readiness probe.

## Tests

```bash
python -m pytest
```

`test_tools.py` covers every order status, unknown orders, malformed input,
and order-number normalization variants. `test_agent.py` mocks the OpenAI
client, so the tool-call loop is tested without any live API calls.

## Replacing the hardcoded data with a real backend

All order data currently lives in `cs_order_agent/data.py` as an in-memory
dict. The rest of the codebase — `agent.py`, `server.py`, `cli.py` — only
ever calls `get_order_status()` from `tools.py` and never touches `data.py`
directly, so swapping the backend means changing one function:

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

No changes are needed in `agent.py`, `server.py`, or `cli.py` — they only
depend on the `get_order_status` function signature and
`ORDER_STATUS_TOOL_SCHEMA`, both of which stay the same.
# OrderStatusAgent
