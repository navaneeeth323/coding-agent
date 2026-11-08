import pytest

from stockroom.parser import Command, parse_command, parse_qty


def test_add():
    assert parse_command('add WID-1 "Big widget" 5 @2.50') == Command(
        "add", sku="WID-1", qty=5, label="Big widget", price_cents=250)


def test_remove_defaults_to_one():
    assert parse_command("remove WID-1").qty == 1
    assert parse_command("restock WID-1 4").qty == 4
    assert parse_command("report") == Command("report")
    assert parse_command("   ") is None


def test_errors():
    with pytest.raises(ValueError):
        parse_command("sell WID-1")
    with pytest.raises(ValueError):
        parse_qty("0")
    assert parse_qty(None, default=3) == 3
