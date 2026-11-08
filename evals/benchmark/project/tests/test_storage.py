import pytest

from stockroom.models import Item
from stockroom.storage import load_csv, save_csv


def test_roundtrip(tmp_path):
    path = tmp_path / "stock.csv"
    items = [Item("A", "Apple", 3, 50, 120), Item("B", "Banana, ripe", 0, 20)]
    save_csv(path, items)
    assert load_csv(path) == items


def test_missing_sku(tmp_path):
    path = tmp_path / "stock.csv"
    path.write_text("sku,name,qty,price_cents\n,Nameless,1,5\n")
    with pytest.raises(ValueError, match="line 2"):
        load_csv(path)
