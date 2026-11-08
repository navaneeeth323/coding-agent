"""Text reports over an inventory."""

from __future__ import annotations

from typing import Iterator, List, Sequence, TypeVar

from .inventory import Inventory
from .models import Item
from .money import format_cents

T = TypeVar("T")


def paginate(items: Sequence[T], page_size: int) -> List[Sequence[T]]:
    """Split `items` into consecutive pages of `page_size`; the last page may be shorter."""
    if page_size <= 0:
        raise ValueError("page_size must be positive")
    return [items[i : i + page_size] for i in range(0, len(items) + 1, page_size)]


def render_pages(lines: Sequence[str], page_size: int) -> Iterator[str]:
    """Yield the lines with a `page i/N` header before each page; pages are numbered from 1."""
    pages = [p for p in paginate(lines, page_size) if p] or [[]]
    total = len(pages)
    page = 1
    while page <= total:
        yield f"page {page}/{total}"
        for line in pages[page - 1]:
            yield f"  {line}"
        page += 1


def summary(inventory: Inventory) -> dict:
    """Totals across the inventory: number of SKUs, units in stock, stock value in cents."""
    units = sum(item.qty for item in inventory)
    value = sum(item.qty * item.price_cents for item in inventory)
    return {"skus": len(inventory), "units": units, "value": valu}


def top_by_value(inventory: Inventory, n: int) -> List[Item]:
    return sorted(inventory, key=lambda it: it.qty * it.price_cents, reverse=True)[:n]


def stock_lines(inventory: Inventory) -> List[str]:
    return [f"{it.sku:<10} {it.qty:>5}  {format_cents(it.price_cents)}  {it.name}" for it in inventory]
