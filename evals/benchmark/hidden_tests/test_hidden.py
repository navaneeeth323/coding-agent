"""Scoring-only tests, one per planted bug. Never copied into the agent's workspace."""

import pytest

from stockroom.events import EventBus, counters
from stockroom.inventory import InsufficientStock, Inventory
from stockroom.models import Item, Order
from stockroom.money import apply_discount, parse_price
from stockroom.parser import normalize_sku
from stockroom.pricing import PriceBook, order_total
from stockroom.reports import paginate, summary
from stockroom.storage import load_csv
from stockroom.utils import clamp


def _inv(*items):
    inv = Inventory()
    for it in items:
        inv.add(it)
    return inv


# ---- easy ---------------------------------------------------------------

def test_e1_orders_do_not_share_lines():
    first = Order("a")
    first.add("X", 1)
    assert Order("b").lines == []


def test_e2_bad_quantity_raises(tmp_path):
    path = tmp_path / "s.csv"
    path.write_text("sku,name,qty,price_cents\nA,Apple,lots,5\n")
    with pytest.raises(ValueError):
        load_csv(path)


def test_e3_summary():
    assert summary(_inv(Item("A", "a", qty=2, price_cents=100))) == {"skus": 1, "units": 2, "value": 200}


def test_e4_counters_per_kind():
    bus = EventBus()
    counts = counters(bus, ["added", "removed"])
    bus.emit("added", "A")
    bus.emit("added", "B")
    assert counts == {"added": 2, "removed": 0}


def test_e5_normalize_sku():
    assert normalize_sku("  wid-1 ") == "WID-1"


# ---- medium -------------------------------------------------------------

def test_m1_paginate_exact_multiple():
    assert paginate([1, 2, 3, 4], 2) == [[1, 2], [3, 4]]
    assert paginate([], 3) == []


def test_m2_item_at_end():
    assert _inv(Item("A", "a"), Item("B", "b")).item_at(2) is None


def test_m3_unknown_sku_price():
    assert PriceBook(_inv(Item("A", "a", price_cents=5))).unit_price("NOPE") is None


def test_m4_low_stock_inclusive():
    assert _inv(Item("A", "a", qty=3), Item("B", "b", qty=4)).low_stock(3) == ["A"]


def test_m5_discount():
    assert apply_discount(1000, 25) == 750
    assert apply_discount(999, 0) == 999


def test_m6_clamp():
    assert clamp(5, 0, 10) == 5 and clamp(-1, 0, 10) == 0 and clamp(11, 0, 10) == 10


def test_m7_parse_price_rounding():
    assert parse_price("19.99") == 1999 and parse_price("0.29") == 29


# ---- hard ---------------------------------------------------------------

def test_h1_shipping_uses_kilograms():
    inv = _inv(Item("BOX", "Box", qty=5, price_cents=1000, weight_g=1500))
    order = Order("c", [])
    order.add("BOX", 1)
    assert order_total(order, PriceBook(inv)) == 1000 + 83 + 899  # 1.5 kg parcel


def test_h2_price_change_not_stale():
    inv = _inv(Item("A", "a", price_cents=100))
    book = PriceBook(inv)
    assert book.unit_price("A") == 100
    inv.set_price("A", 150)
    assert book.unit_price("A") == 150


def test_h3_failed_remove_changes_nothing():
    inv = _inv(Item("A", "a", qty=2))
    with pytest.raises(InsufficientStock):
        inv.remove("A", 5)
    assert inv.get("A").qty == 2 and "removed" not in inv.bus.kinds()
