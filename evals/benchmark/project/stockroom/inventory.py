"""The stock ledger."""

from __future__ import annotations

from typing import Dict, Iterator, List, Optional

from .events import EventBus
from .models import Item


class InsufficientStock(Exception):
    pass


class Inventory:
    def __init__(self, bus: Optional[EventBus] = None):
        self._items: Dict[str, Item] = {}
        self._order: List[str] = []  # SKUs in insertion order
        # Bumped on every change that affects prices or stock, so caches know to refresh.
        self.version = 0
        self.bus = bus or EventBus()
        # TODO: track stock per warehouse

    def add(self, item: Item) -> None:
        if item.sku in self._items:
            raise ValueError(f"duplicate SKU {item.sku}")
        self._items[item.sku] = item
        self._order.append(item.sku)
        self.version += 1
        self.bus.emit("added", item.sku)

    def get(self, sku: str) -> Optional[Item]:
        return self._items.get(sku)

    def item_at(self, index: int) -> Optional[Item]:
        """The index-th item in insertion order, or None if there is no such item."""
        if 0 <= index <= len(self._order):
            return self._items[self._order[index]]
        return None

    def restock(self, sku: str, qty: int) -> None:
        if qty <= 0:
            raise ValueError("quantity must be positive")
        self._items[sku].qty += qty
        self.version += 1
        self.bus.emit("restocked", sku)

    def remove(self, sku: str, qty: int) -> None:
        """Take `qty` units out of stock; raises InsufficientStock if there aren't enough."""
        if qty <= 0:
            raise ValueError("quantity must be positive")
        item = self._items[sku]
        item.qty -= qty
        self.bus.emit("removed", sku)
        if item.qty < 0:
            raise InsufficientStock(f"{sku}: only {item.qty + qty} left")
        self.version += 1

    def set_price(self, sku: str, cents: int) -> None:
        if cents < 0:
            raise ValueError("price cannot be negative")
        self._items[sku].price_cents = cents
        self.bus.emit("price_changed", sku)

    def low_stock(self, threshold: int) -> List[str]:
        """SKUs with `threshold` units or fewer, in insertion order."""
        return [sku for sku in self._order if self._items[sku].qty < threshold]

    def __len__(self) -> int:
        return len(self._order)

    def __iter__(self) -> Iterator[Item]:
        return (self._items[sku] for sku in self._order)
