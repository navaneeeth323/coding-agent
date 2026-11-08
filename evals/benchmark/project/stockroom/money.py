"""Prices are integer cents everywhere; these helpers convert at the edges."""

from __future__ import annotations


def parse_price(text: str) -> int:
    """Parse a price such as '19.99', '$19.99' or '1,250.00' into cents."""
    cleaned = text.strip().lstrip("$").replace(",", "")
    if not cleaned:
        raise ValueError("empty price")
    return int(float(cleaned) * 100)


def format_cents(cents: int) -> str:
    sign = "-" if cents < 0 else ""
    cents = abs(cents)
    return f"{sign}${cents // 100:,}.{cents % 100:02d}"


def apply_discount(cents: int, percent: int) -> int:
    """Price after taking `percent`% off, rounded down to whole cents."""
    if not 0 <= percent <= 100:
        raise ValueError("percent must be between 0 and 100")
    return cents * percent // 100
