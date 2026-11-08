import pytest

from stockroom.inventory import Inventory
from stockroom.models import Item


@pytest.fixture
def inv():
    inv = Inventory()
    inv.add(Item("A", "Apple", qty=1, price_cents=50))
    inv.add(Item("B", "Banana", qty=10, price_cents=20))
    return inv


def test_add_and_get(inv):
    assert inv.get("A").name == "Apple" and inv.get("Z") is None
    assert len(inv) == 2 and [i.sku for i in inv] == ["A", "B"]
    with pytest.raises(ValueError):
        inv.add(Item("A", "Again"))


def test_item_at(inv):
    assert inv.item_at(0).sku == "A" and inv.item_at(1).sku == "B"
    assert inv.item_at(-1) is None and inv.item_at(7) is None


def test_restock_and_remove(inv):
    inv.restock("A", 4)
    inv.remove("B", 3)
    assert inv.get("A").qty == 5 and inv.get("B").qty == 7
    assert inv.bus.kinds() == ["added", "added", "restocked", "removed"]


def test_low_stock(inv):
    assert inv.low_stock(5) == ["A"]
    assert inv.low_stock(0) == []
