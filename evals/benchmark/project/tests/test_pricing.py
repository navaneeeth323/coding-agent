from stockroom.inventory import Inventory
from stockroom.models import Item, Order
from stockroom.pricing import PriceBook, bulk_discount_percent, order_total, tax_cents
from stockroom.shipping import shipping_cost


def test_tiers_and_tax():
    assert bulk_discount_percent(9) == 0 and bulk_discount_percent(10) == 5 and bulk_discount_percent(250) == 15
    assert tax_cents(10000) == 825 and tax_cents(0) == 0


def test_shipping():
    assert shipping_cost(0) == 0
    assert shipping_cost(0.4) == 499 and shipping_cost(2.0) == 899
    assert shipping_cost(12) == 1599 + 2 * 150


def test_price_book_and_total():
    inv = Inventory()
    inv.add(Item("PEN", "Pen", qty=100, price_cents=250))
    book = PriceBook(inv)
    assert book.unit_price("PEN") == 250
    order = Order("bob", [])
    order.add("PEN", 2)
    assert order.total_units() == 2
    assert order_total(order, book) == 500 + 41  # digital goods: no weight, no shipping
