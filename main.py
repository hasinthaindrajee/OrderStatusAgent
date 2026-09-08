"""Entry point for platforms that require a `python main.py` start command.

Equivalent to `uvicorn cs_order_agent.server:app --host 0.0.0.0 --port $PORT`.
"""

from __future__ import annotations

import os

import uvicorn

from cs_order_agent.server import app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
