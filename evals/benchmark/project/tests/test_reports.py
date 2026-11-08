from stockroom.inventory import Inventory
from stockroom.models import Item
from stockroom.reports import paginate, render_pages, stock_lines, top_by_value


def test_paginate():
    assert paginate(list(range(5)), 2) == [[0, 1], [2, 3], [4]]


def test_render_pages():
    assert list(render_pages(["a", "b", "c"], 2)) == ["page 1/2", "  a", "  b", "page 2/2", "  c"]
    assert list(render_pages([], 2)) == ["page 1/1"]


def test_top_and_lines():
    inv = Inventory()
    inv.add(Item("A", "Apple", qty=1, price_cents=50))
    inv.add(Item("B", "Banana", qty=10, price_cents=20))
    assert [i.sku for i in top_by_value(inv, 1)] == ["B"]
    assert stock_lines(inv)[0].split() == ["A", "1", "$0.50", "Apple"]
