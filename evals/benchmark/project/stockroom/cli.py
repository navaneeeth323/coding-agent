"""`python -m stockroom.cli stock.csv 'remove WID-1 2' report`"""

from __future__ import annotations

import argparse
import sys

from .inventory import Inventory
from .models import Item
from .parser import parse_command
from .reports import render_pages, stock_lines
from .storage import load_csv, save_csv


def apply(inv: Inventory, line: str) -> None:
    cmd = parse_command(line)
    if cmd is None:
        return
    if cmd.name == "add":
        inv.add(Item(cmd.sku, cmd.label, cmd.qty, cmd.price_cents))
    elif cmd.name == "remove":
        inv.remove(cmd.sku, cmd.qty)
    elif cmd.name == "restock":
        inv.restock(cmd.sku, cmd.qty)
    elif cmd.name == "price":
        inv.set_price(cmd.sku, cmd.price_cents)
    elif cmd.name == "report":
        for out in render_pages(stock_lines(inv), 20):
            print(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="stockroom")
    ap.add_argument("csv")
    ap.add_argument("commands", nargs="*")
    args = ap.parse_args(argv)
    inv = Inventory()
    for item in load_csv(args.csv):
        inv.add(item)
    for line in args.commands:
        apply(inv, line)
    save_csv(args.csv, inv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
