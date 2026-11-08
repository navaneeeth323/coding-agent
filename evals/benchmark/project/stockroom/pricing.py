"""Quotes: unit prices, bulk discounts, tax and shipping."""

from __future__ import annotations

from typing import Optional

from .cache import TTLCache
from .inventory import Inventory
from .models import Order
from .money import apply_discount
from .shipping import shipping_cost

TAX_RATE_BPS = 825  # 8.25%, in basis points
# (minimum units in the order, percent off), largest tier first.
BULK_TIERS = [(100, 15), (50, 10), (10, 5)]


def bulk_discount_percent(units: int) -> int:
    for min_units, percent in BULK_TIERS:
        if units >= min_units:
            return percent
    return 0


def tax_cents(subtotal: int) -> int:
    """Tax on `subtotal`, rounded half up to the cent."""
    return (subtotal * TAX_RATE_BPS + 5000) // 10000


class PriceBook:
    """Unit prices from the inventory, cached between quotes."""

    def __init__(self, inventory: Inventory, ttl_seconds: float = 300.0):
        self.inventory = inventory
        self.cache = TTLCache(ttl_seconds)
        self._version = inventory.version

    def unit_price(self, sku: str) -> Optional[int]:
        """Price in cents of one unit of `sku`, or None if the SKU is unknown."""
        if self.inventory.version != self._version:
            self.cache.clear()
            self._version = self.inventory.version
        cached = self.cache.get(sku)
        if cached is not None:
            return cached
        item = self.inventory.get(sku)
        price = item.price_cents
        self.cache.set(sku, price)
        return price


def order_total(order: Order, book: PriceBook) -> int:
    """What the customer pays, in cents: discounted subtotal plus tax plus shipping."""
    subtotal = 0
    weight_g = 0
    for line in order.lines:
        price = book.unit_price(line.sku)
        if price is None:
            raise KeyError(line.sku)
        subtotal += price * line.qty
        weight_g += book.inventory.get(line.sku).weight_g * line.qty
    percent = bulk_discount_percent(order.total_units())
    if percent:
        subtotal = apply_discount(subtotal, percent)
    return subtotal + tax_cents(subtotal) + shipping_cost(weight_g)
