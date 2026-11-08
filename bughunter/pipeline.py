"""Pipeline stages. Stage 2: collect candidates from deterministic sources."""

from __future__ import annotations

import hashlib
import logging
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Sequence

from .candidates import store
from .candidates.git import file_history
from .candidates.heuristics import run_heuristics
from .candidates.model import Finding
from .candidates.ruff import run_ruff
from .candidates.scoring import Hotspots, score
from .index import RepoIndex

log = logging.getLogger(__name__)

SOURCES: Dict[str, Callable] = {
    "ruff": run_ruff,
    "heuristic": run_heuristics,
}
MODULE_CONTEXT_LINES = 3  # context hashed around findings outside any function


@dataclass
class CollectStats:
    run_id: int = 0
    findings: int = 0
    added: int = 0
    updated: int = 0
    reopened: int = 0
    by_source: Dict[str, int] = field(default_factory=dict)
    errors: Dict[str, str] = field(default_factory=dict)


def _fingerprints(findings: List[Finding], symbols: List[dict]) -> List[str]:
    """Stable ids that survive line shifts: no line numbers, just an occurrence counter
    to tell apart identical findings (say, two bare excepts) in the same function."""
    seen: Dict[str, int] = defaultdict(int)
    order = sorted(range(len(findings)), key=lambda i: (findings[i].path, findings[i].line, findings[i].col))
    fps = [""] * len(findings)
    for i in order:
        f, sym = findings[i], symbols[i]
        key = "|".join((f.source, f.rule, f.path, sym["qualname"] if sym else "<module>", f.message))
        fps[i] = hashlib.sha1(f"{key}|{seen[key]}".encode()).hexdigest()
        seen[key] += 1
    return fps


def _refresh_llm_candidates(index: RepoIndex, lines_cache: Dict[str, List[str]]) -> None:
    """Scan suspicions aren't re-derived here, so follow their function or retire them.

    A function that moved keeps its suspicion (line numbers shifted); one whose
    code changed loses it until the next scan looks again. Caller owns the transaction.
    """
    conn = index.conn
    for c in conn.execute(
        "SELECT id, path, symbol, line, symbol_start, code_sha1 FROM candidates WHERE source = 'llm' AND active = 1"
    ).fetchall():
        row = conn.execute(
            "SELECT s.start_line, s.end_line FROM symbols s JOIN files f ON f.id = s.file_id"
            " WHERE f.path = ? AND s.qualname = ? ORDER BY s.start_line LIMIT 1",
            (c["path"], c["symbol"]),
        ).fetchone()
        code = None
        if row is not None:
            if c["path"] not in lines_cache:
                lines_cache[c["path"]] = (index.repo / c["path"]).read_text("utf-8", "replace").splitlines()
            code = "\n".join(lines_cache[c["path"]][row["start_line"] - 1 : row["end_line"]])
        if code is None or hashlib.sha1(code.encode()).hexdigest() != c["code_sha1"]:
            conn.execute("UPDATE candidates SET active = 0 WHERE id = ?", (c["id"],))
            continue
        shift = row["start_line"] - c["symbol_start"]
        if shift:
            conn.execute(
                "UPDATE candidates SET line = line + ?, end_line = end_line + ?, symbol_start = ?, symbol_end = ?"
                " WHERE id = ?", (shift, shift, row["start_line"], row["end_line"], c["id"]),
            )


def collect_candidates(index: RepoIndex, sources: Sequence[str] = tuple(SOURCES)) -> CollectStats:
    conn = index.conn
    store.ensure_schema(conn)
    index.update()
    run_id = store.start_run(conn, "candidates")
    stats = CollectStats(run_id=run_id)

    try:
        paths = [r["path"] for r in conn.execute("SELECT path FROM files ORDER BY path")]
        # Only files that still exist, so renamed-away paths don't skew the percentiles.
        history = file_history(index.repo)
        hotspots = Hotspots({p: history[p] for p in paths if p in history})

        findings: List[Finding] = []
        ran = []
        for name in sources:
            try:
                found = SOURCES[name](index.repo, paths)
            except Exception as exc:  # one broken analyzer shouldn't sink the run
                log.warning("source %s failed: %s", name, exc)
                stats.errors[name] = str(exc)
                continue
            ran.append(name)
            stats.by_source[name] = len(found)
            findings.extend(found)
        stats.findings = len(findings)

        symbols = [index.symbol_at(f.path, f.line) for f in findings]
        fingerprints = _fingerprints(findings, symbols)
        lines_cache: Dict[str, List[str]] = {}

        def code_sha1(f: Finding, sym) -> str:
            if f.path not in lines_cache:
                lines_cache[f.path] = (index.repo / f.path).read_text("utf-8", "replace").splitlines()
            lines = lines_cache[f.path]
            if sym:
                start, end = sym["start_line"], sym["end_line"]
            else:
                start, end = f.line - MODULE_CONTEXT_LINES, f.end_line + MODULE_CONTEXT_LINES
            return hashlib.sha1("\n".join(lines[max(0, start - 1) : end]).encode()).hexdigest()

        with conn:
            store.deactivate_sources(conn, ran)
            for f, sym, fp in zip(findings, symbols, fingerprints):
                outcome = store.upsert(conn, run_id, {
                    "fingerprint": fp,
                    **asdict(f),
                    "symbol": sym["qualname"] if sym else None,
                    "symbol_start": sym["start_line"] if sym else None,
                    "symbol_end": sym["end_line"] if sym else None,
                    "code_sha1": code_sha1(f, sym),
                    "score": score(f.rule, f.path, hotspots),
                })
                setattr(stats, outcome, getattr(stats, outcome) + 1)
            _refresh_llm_candidates(index, lines_cache)
    except Exception:
        store.finish_run(conn, run_id, "failed", asdict(stats))
        raise
    store.finish_run(conn, run_id, "ok", asdict(stats))
    return stats
