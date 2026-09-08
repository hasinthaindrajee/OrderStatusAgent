"""The order-status tool exposed to the agent.

get_order_status() is the only function the agent ever calls. It is a thin
lookup wrapper around cs_order_agent.data — swapping the hardcoded dict for
a real order-management API means changing data.py (or replacing the lookup
inside this function) without touching agent.py at all.
"""

from __future__ import annotations

import re
from typing import Any

from cs_order_agent.data import ORDERS

_ORDER_CORE_RE = re.compile(r"^\d{4}\d{6}$")


def _normalize_order_number(raw: str) -> str | None:
    """Normalize an order number to the canonical CS-YYYY-XXXXXX form.

    Accepts leading/trailing whitespace, an optional "CS" prefix, and
    dashes in any/no position. Returns None if the input can't be
    normalized to a valid-shaped order number.
    """
    if not isinstance(raw, str):
        return None

    candidate = raw.strip().upper()
    if not candidate:
        return None

    candidate = candidate.replace("-", "").replace(" ", "")
    if candidate.startswith("CS"):
        candidate = candidate[2:]

    if not _ORDER_CORE_RE.match(candidate):
        return None

    year, sequence = candidate[:4], candidate[4:]
    return f"CS-{year}-{sequence}"


def get_order_status(order_number: str) -> dict[str, Any]:
    """Look up the current status of a C&S Wholesale order.

    Always returns a dict and never raises, regardless of input.
    """
    normalized = _normalize_order_number(order_number)

    if normalized is None:
        return {
            "found": False,
            "error": "invalid_format",
            "message": (
                f"'{order_number}' doesn't look like a valid C&S order number. "
                "Order numbers look like CS-2026-100482."
            ),
        }

    order = ORDERS.get(normalized)
    if order is None:
        return {
            "found": False,
            "error": "not_found",
            "message": f"No order found with number {normalized}.",
        }

    return {"found": True, "order": order}


ORDER_STATUS_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "get_order_status",
        "description": (
            "Look up the current status of a C&S Wholesale Grocers order by its "
            "order number. Returns delivery status, dates, distribution center, "
            "carrier, and tracking information when available. Use this any time "
            "a customer asks about the status of a specific order — never guess "
            "or make up an order status."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "order_number": {
                    "type": "string",
                    "description": (
                        "The order number to look up, e.g. 'CS-2026-100482'. "
                        "The 'CS-' prefix and dashes are optional — pass "
                        "whatever the customer provided."
                    ),
                },
            },
            "required": ["order_number"],
        },
    },
}
