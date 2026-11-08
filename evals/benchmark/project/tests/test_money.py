import pytest

from stockroom.money import apply_discount, format_cents, parse_price


def test_parse_price():
    assert parse_price("3") == 300
    assert parse_price("$1,250.50") == 125050
    with pytest.raises(ValueError):
        parse_price("  ")


def test_format_cents():
    assert format_cents(123456) == "$1,234.56"
    assert format_cents(-5) == "-$0.05"


def test_apply_discount():
    assert apply_discount(1000, 50) == 500
    with pytest.raises(ValueError):
        apply_discount(1000, 150)
