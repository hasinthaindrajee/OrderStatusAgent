from __future__ import annotations

import pytest

from data import ORDERS
from tools import get_order_status

ALL_STATUSES = {
    "RECEIVED",
    "PICKING",
    "STAGED",
    "IN_TRANSIT",
    "DELIVERED",
    "EXCEPTION",
    "CANCELLED",
}


def test_data_covers_every_status() -> None:
    statuses = {order["status"] for order in ORDERS.values()}
    assert ALL_STATUSES.issubset(statuses)


@pytest.mark.parametrize("status", sorted(ALL_STATUSES))
def test_get_order_status_for_each_status(status: str) -> None:
    order_number = next(
        num for num, order in ORDERS.items() if order["status"] == status
    )

    result = get_order_status(order_number)

    assert result["found"] is True
    assert result["order"]["status"] == status
    assert result["order"]["order_number"] == order_number


def test_unknown_order_returns_not_found() -> None:
    result = get_order_status("CS-2026-999999")

    assert result == {
        "found": False,
        "error": "not_found",
        "message": "No order found with number CS-2026-999999.",
    }


@pytest.mark.parametrize(
    "malformed",
    [
        "",
        "   ",
        "not-an-order",
        "CS-2026-1",
        "CS-2026-1000004829",
        "CS-ABCD-100482",
        "12345",
        "CS--2026-100482x",
    ],
)
def test_malformed_input_returns_invalid_format(malformed: str) -> None:
    result = get_order_status(malformed)

    assert result["found"] is False
    assert result["error"] == "invalid_format"
    assert "message" in result


def test_get_order_status_never_raises_on_non_string() -> None:
    result = get_order_status(None)  # type: ignore[arg-type]

    assert result["found"] is False
    assert result["error"] == "invalid_format"


@pytest.mark.parametrize(
    "variant",
    [
        "CS-2026-100482",
        "cs-2026-100482",
        "CS2026100482",
        "2026100482",
        "  CS-2026-100482  ",
        "cs 2026 100482",
        "CS-2026100482",
        "2026-100482",
    ],
)
def test_input_normalization_variants_resolve_to_same_order(variant: str) -> None:
    result = get_order_status(variant)

    assert result["found"] is True
    assert result["order"]["order_number"] == "CS-2026-100482"
