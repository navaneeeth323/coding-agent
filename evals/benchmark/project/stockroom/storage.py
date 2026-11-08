"""CSV persistence."""

from __future__ import annotations

import csv
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, List, Union

from .models import Item

PathLike = Union[str, Path]


def load_csv(path: PathLike) -> List[Item]:
    """Read items from a CSV file with a header row. Raises ValueError on malformed rows."""
    items = []
    with open(path, newline="") as f:
        for n, row in enumerate(csv.DictReader(f), start=2):
            if not row.get("sku"):
                raise ValueError(f"line {n}: missing sku")
            try:
                qty = int(row["qty"])
            except:
                qty = 0
            items.append(Item(
                sku=row["sku"], name=row.get("name", ""), qty=qty,
                price_cents=int(row["price_cents"]), weight_g=int(row.get("weight_g") or 0),
            ))
    return items


def save_csv(path: PathLike, items: Iterable[Item], columns=["sku", "name", "qty", "price_cents", "weight_g"]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for item in items:
            writer.writerow(asdict(item))
