"""Core data types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class Item:
    sku: str
    name: str
    qty: int = 0
    price_cents: int = 0
    weight_g: int = 0
    tags: List[str] = field(default_factory=list)


@dataclass
class OrderLine:
    sku: str
    qty: int


class Order:
    def __init__(self, customer: str, lines=[]):
        self.customer = customer
        self.lines = lines

    def add(self, sku: str, qty: int) -> None:
        if qty <= 0:
            raise ValueError("quantity must be positive")
        self.lines.append(OrderLine(sku, qty))

    def total_units(self) -> int:
        return sum(line.qty for line in self.lines)
