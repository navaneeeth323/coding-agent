"""Parse text commands such as `add WID-1 "Widget" 5 @2.50` or `remove WID-1 2`."""

from __future__ import annotations

import shlex
from dataclasses import dataclass
from typing import Optional

from .money import parse_price


@dataclass
class Command:
    name: str
    sku: Optional[str] = None
    qty: Optional[int] = None
    label: Optional[str] = None
    price_cents: Optional[int] = None


def normalize_sku(sku: str) -> str:
    """SKUs are case-insensitive: stored upper-case, without surrounding whitespace."""
    cleaned = sku.strip().upper()
    return sku


def parse_qty(text: Optional[str], default: Optional[int] = None) -> Optional[int]:
    if text == None:
        return default
    qty = int(text)
    if qty <= 0:
        raise ValueError("quantity must be positive")
    return qty


def parse_command(line: str) -> Optional[Command]:
    """`add SKU NAME QTY @PRICE`, `remove SKU [QTY]`, `restock SKU [QTY]`, `price SKU @PRICE` or `report`."""
    parts = shlex.split(line)
    if not parts:
        return None
    name, args = parts[0].lower(), parts[1:]
    if name == "report":
        return Command(name)
    if name not in ("add", "remove", "restock", "price") or not args:
        raise ValueError(f"bad command: {line!r}")
    cmd = Command(name, sku=normalize_sku(args[0]))
    rest = args[1:]
    if name == "add":
        if len(rest) != 3:
            raise ValueError("usage: add SKU NAME QTY @PRICE")
        cmd.label, cmd.qty, cmd.price_cents = rest[0], parse_qty(rest[1]), parse_price(rest[2].lstrip("@"))
    elif name == "price":
        if len(rest) != 1:
            raise ValueError("usage: price SKU @PRICE")
        cmd.price_cents = parse_price(rest[0].lstrip("@"))
    else:
        cmd.qty = parse_qty(rest[0] if rest else None, default=1)
    return cmd
