"""Compact, ranked outline of a repository (in the spirit of Aider's repo map).

Files form a graph: file A links to file B when A calls or imports a name that B
defines (resolved through imports where possible; see resolve.py). PageRank over that graph (optionally biased towards `focus` files) ranks
files; each file's rank flows along its edges to the definitions it uses, which
ranks individual symbols. The most important signatures are rendered until the
token budget runs out.
"""

from __future__ import annotations

import math
import sqlite3
from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

from .resolve import Resolver

DAMPING = 0.85
ITERATIONS = 50
CHARS_PER_TOKEN = 4  # rough, but stable across models; keeps us tokenizer-free
MAX_MAP_SIGNATURE_CHARS = 120


def _pagerank(nodes: List[int], edges: Dict[int, Dict[int, float]], personal: Dict[int, float]) -> Dict[int, float]:
    out_total = {src: sum(dsts.values()) for src, dsts in edges.items()}
    rank = dict(personal)
    for _ in range(ITERATIONS):
        dangling = sum(rank[n] for n in nodes if not out_total.get(n))
        nxt = {n: (1 - DAMPING + DAMPING * dangling) * personal[n] for n in nodes}
        for src, dsts in edges.items():
            share = DAMPING * rank[src] / out_total[src]
            for dst, w in dsts.items():
                nxt[dst] += share * w
        rank = nxt
    return rank


def build_repo_map(conn: sqlite3.Connection, max_tokens: int = 1024, focus: Sequence[str] = ()) -> str:
    files = {r["id"]: r["path"] for r in conn.execute("SELECT id, path FROM files")}
    if not files:
        return ""

    # Definitions worth showing: classes, functions and methods, but not functions
    # nested inside other functions (implementation detail, not API).
    symbols = {}
    for r in conn.execute(
        "SELECT s.id, s.file_id, s.parent_id, s.name, s.kind, s.start_line, s.signature,"
        " p.kind AS parent_kind FROM symbols s LEFT JOIN symbols p ON p.id = s.parent_id"
        " ORDER BY s.id"  # parents first, so the ancestor check below sees them
    ):
        if r["parent_kind"] in ("function", "method"):
            continue
        if r["parent_id"] is not None and r["parent_id"] not in symbols:
            continue  # e.g. a method of a class defined inside a function
        symbols[r["id"]] = r

    # edges[src_file][dst_file] = weight; uses[(src_file, dst_file)] = [(symbol_id, weight)]
    edges: Dict[int, Dict[int, float]] = defaultdict(lambda: defaultdict(float))
    uses: Dict[Tuple[int, int], List[Tuple[int, float]]] = defaultdict(list)
    for ref in Resolver(conn).references():
        if ref.symbol_id not in symbols:
            continue
        dst = symbols[ref.symbol_id]["file_id"]
        # sqrt: a file calling one helper 50 times shouldn't dominate the graph.
        weight = math.sqrt(ref.count) * ref.confidence
        edges[ref.src_file][dst] += weight
        uses[(ref.src_file, dst)].append((ref.symbol_id, weight))

    nodes = list(files)
    focus_ids = {fid for fid, path in files.items() if path in set(focus)}
    if focus_ids:
        personal = {n: (1.0 / len(focus_ids) if n in focus_ids else 0.0) for n in nodes}
    else:
        personal = {n: 1.0 / len(nodes) for n in nodes}
    file_rank = _pagerank(nodes, edges, personal)

    # Spread each file's rank over the definitions it uses.
    sym_rank: Dict[int, float] = defaultdict(float)
    for src, dsts in edges.items():
        total = sum(dsts.values())
        for dst in dsts:
            for sym_id, w in uses[(src, dst)]:
                sym_rank[sym_id] += file_rank[src] * w / total
    # Unreferenced definitions (entry points, tests) still get a sliver of their file's rank.
    per_file = defaultdict(int)
    for s in symbols.values():
        per_file[s["file_id"]] += 1
    for sym_id, s in symbols.items():
        sym_rank[sym_id] += 1e-3 * file_rank[s["file_id"]] / per_file[s["file_id"]]

    return _render(files, symbols, sym_rank, max_tokens * CHARS_PER_TOKEN)


def _signature(sym) -> str:
    sig = sym["signature"]
    return sig if len(sig) <= MAX_MAP_SIGNATURE_CHARS else sig[: MAX_MAP_SIGNATURE_CHARS - 1] + "…"


def _render(files, symbols, sym_rank, budget_chars: int) -> str:
    chosen = set()
    shown_files = set()
    used = 0

    def depth(sym_id: int) -> int:
        d, parent = 0, symbols[sym_id]["parent_id"]
        while parent is not None and parent in symbols:
            d, parent = d + 1, symbols[parent]["parent_id"]
        return d

    def line_cost(sym_id: int) -> int:
        return 2 * (depth(sym_id) + 1) + len(_signature(symbols[sym_id])) + 1

    for sym_id in sorted(symbols, key=lambda s: -sym_rank[s]):
        if sym_id in chosen:
            continue
        # A method is meaningless without its class line, so pay for missing ancestors too.
        needed = [sym_id]
        parent = symbols[sym_id]["parent_id"]
        while parent is not None and parent in symbols and parent not in chosen:
            needed.append(parent)
            parent = symbols[parent]["parent_id"]
        file_id = symbols[sym_id]["file_id"]
        cost = sum(line_cost(s) for s in needed)
        if file_id not in shown_files:
            cost += len(files[file_id]) + 2
        if used + cost > budget_chars:
            continue  # something smaller further down may still fit
        used += cost
        chosen.update(needed)
        shown_files.add(file_id)

    by_file: Dict[int, List[int]] = defaultdict(list)
    for sym_id in chosen:
        by_file[symbols[sym_id]["file_id"]].append(sym_id)

    out = []
    for file_id in sorted(by_file, key=lambda f: files[f]):
        out.append(files[file_id] + ":")
        for sym_id in sorted(by_file[file_id], key=lambda s: symbols[s]["start_line"]):
            out.append("  " * (depth(sym_id) + 1) + _signature(symbols[sym_id]))
    return "\n".join(out)
