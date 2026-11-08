"""Candidate state machine, persisted next to the code index.

Each candidate moves through
    new → triaged → investigating → verified → pr_opened
with `rejected` reachable from every non-final state. Every transition is logged
in candidate_events, so runs are resumable and the funnel can be analysed later.
"""

from __future__ import annotations

import json
import sqlite3
import time
from typing import Iterable, List, Optional

from ..index import store as index_store

SCHEMA_VERSION = "1"

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id          INTEGER PRIMARY KEY,
    stage       TEXT NOT NULL,
    started_at  REAL NOT NULL,
    finished_at REAL,
    status      TEXT NOT NULL DEFAULT 'running',  -- running | ok | failed
    stats       TEXT                              -- JSON
);

CREATE TABLE IF NOT EXISTS candidates (
    id             INTEGER PRIMARY KEY,
    fingerprint    TEXT NOT NULL UNIQUE,  -- stable across runs and line shifts
    source         TEXT NOT NULL,         -- ruff | heuristic
    rule           TEXT NOT NULL,
    path           TEXT NOT NULL,
    line           INTEGER NOT NULL,
    col            INTEGER NOT NULL,
    end_line       INTEGER NOT NULL,
    message        TEXT NOT NULL,
    symbol         TEXT,                  -- enclosing qualname; NULL at module level
    symbol_start   INTEGER,
    symbol_end     INTEGER,
    code_sha1      TEXT NOT NULL,         -- hash of the enclosing code, for LLM response caching
    score          REAL NOT NULL,
    state          TEXT NOT NULL DEFAULT 'new',
    active         INTEGER NOT NULL DEFAULT 1,  -- reported by the latest run of its source
    first_seen_run INTEGER NOT NULL REFERENCES runs(id),
    last_seen_run  INTEGER NOT NULL REFERENCES runs(id),
    created_at     REAL NOT NULL,
    updated_at     REAL NOT NULL,
    data           TEXT NOT NULL DEFAULT '{}'  -- JSON scratch space for later stages
);
CREATE INDEX IF NOT EXISTS candidates_queue ON candidates(state, active, score DESC);
CREATE INDEX IF NOT EXISTS candidates_symbol ON candidates(path, symbol);

CREATE TABLE IF NOT EXISTS candidate_events (
    id           INTEGER PRIMARY KEY,
    candidate_id INTEGER NOT NULL REFERENCES candidates(id) ON DELETE CASCADE,
    run_id       INTEGER REFERENCES runs(id),
    at           REAL NOT NULL,
    from_state   TEXT,
    to_state     TEXT NOT NULL,
    note         TEXT
);
CREATE INDEX IF NOT EXISTS candidate_events_candidate ON candidate_events(candidate_id);
"""

STATES = ("new", "triaged", "investigating", "verified", "pr_opened", "rejected")
TRANSITIONS = {
    "new": {"triaged", "rejected"},
    "triaged": {"investigating", "rejected"},
    # Back to triaged lets an interrupted or over-budget investigation be retried.
    "investigating": {"verified", "triaged", "rejected"},
    "verified": {"pr_opened", "rejected"},
    "pr_opened": set(),
    # Reopened when the code under a rejected candidate changes.
    "rejected": {"new"},
}


class InvalidTransition(ValueError):
    pass


def ensure_schema(conn: sqlite3.Connection) -> None:
    version = index_store.get_meta(conn, "candidates_schema")
    if version not in (None, SCHEMA_VERSION):
        raise RuntimeError(f"candidates schema v{version}, expected v{SCHEMA_VERSION}")
    conn.executescript(SCHEMA)
    with conn:
        index_store.set_meta(conn, "candidates_schema", SCHEMA_VERSION)


def start_run(conn: sqlite3.Connection, stage: str) -> int:
    with conn:
        return conn.execute(
            "INSERT INTO runs (stage, started_at) VALUES (?, ?)", (stage, time.time())
        ).lastrowid


def finish_run(conn: sqlite3.Connection, run_id: int, status: str, stats: dict) -> None:
    with conn:
        conn.execute(
            "UPDATE runs SET finished_at = ?, status = ?, stats = ? WHERE id = ?",
            (time.time(), status, json.dumps(stats), run_id),
        )


def _log(conn, candidate_id: int, run_id: Optional[int], from_state, to_state: str, note) -> None:
    conn.execute(
        "INSERT INTO candidate_events (candidate_id, run_id, at, from_state, to_state, note)"
        " VALUES (?, ?, ?, ?, ?, ?)",
        (candidate_id, run_id, time.time(), from_state, to_state, note),
    )


def upsert(conn: sqlite3.Connection, run_id: int, c: dict) -> str:
    """Insert or refresh one candidate. Returns "added", "updated" or "reopened".

    `c` holds every candidates column except the bookkeeping ones. Caller owns the transaction.
    """
    now = time.time()
    row = conn.execute(
        "SELECT id, state, code_sha1 FROM candidates WHERE fingerprint = ?", (c["fingerprint"],)
    ).fetchone()
    if row is None:
        cid = conn.execute(
            "INSERT INTO candidates (fingerprint, source, rule, path, line, col, end_line, message,"
            " symbol, symbol_start, symbol_end, code_sha1, score, first_seen_run, last_seen_run,"
            " created_at, updated_at)"
            " VALUES (:fingerprint, :source, :rule, :path, :line, :col, :end_line, :message,"
            " :symbol, :symbol_start, :symbol_end, :code_sha1, :score, :run, :run, :now, :now)",
            {**c, "run": run_id, "now": now},
        ).lastrowid
        _log(conn, cid, run_id, None, "new", None)
        return "added"

    conn.execute(
        "UPDATE candidates SET line = :line, col = :col, end_line = :end_line, message = :message,"
        " symbol_start = :symbol_start, symbol_end = :symbol_end, code_sha1 = :code_sha1,"
        " score = :score, active = 1, last_seen_run = :run, updated_at = :now WHERE id = :id",
        {**c, "run": run_id, "now": now, "id": row["id"]},
    )
    if row["state"] == "rejected" and row["code_sha1"] != c["code_sha1"]:
        conn.execute("UPDATE candidates SET state = 'new' WHERE id = ?", (row["id"],))
        _log(conn, row["id"], run_id, "rejected", "new", "code changed since rejection")
        return "reopened"
    return "updated"


def deactivate_sources(conn: sqlite3.Connection, sources: Iterable[str]) -> None:
    """Mark candidates from `sources` inactive; the current run re-activates what it still sees."""
    conn.executemany("UPDATE candidates SET active = 0 WHERE source = ?", [(s,) for s in sources])


def transition(
    conn: sqlite3.Connection,
    candidate_id: int,
    to_state: str,
    note: Optional[str] = None,
    run_id: Optional[int] = None,
    data: Optional[dict] = None,
) -> None:
    """Move a candidate to `to_state`, merging `data` into its JSON scratch space."""
    with conn:
        row = conn.execute("SELECT state, data FROM candidates WHERE id = ?", (candidate_id,)).fetchone()
        if row is None:
            raise KeyError(candidate_id)
        if to_state not in TRANSITIONS[row["state"]]:
            raise InvalidTransition(f"candidate {candidate_id}: {row['state']} -> {to_state}")
        merged = {**json.loads(row["data"]), **(data or {})}
        conn.execute(
            "UPDATE candidates SET state = ?, data = ?, updated_at = ? WHERE id = ?",
            (to_state, json.dumps(merged), time.time(), candidate_id),
        )
        _log(conn, candidate_id, run_id, row["state"], to_state, note)


def queue(
    conn: sqlite3.Connection, state: str = "new", limit: int = 50, min_score: float = 0.0
) -> List[dict]:
    """Active candidates in `state`, highest score first: the next stage's work list."""
    rows = conn.execute(
        "SELECT * FROM candidates WHERE state = ? AND active = 1 AND score >= ?"
        " ORDER BY score DESC, path, line LIMIT ?",
        (state, min_score, limit),
    )
    return [dict(r) for r in rows]


def funnel(conn: sqlite3.Connection) -> dict:
    """Active candidate counts per state."""
    counts = {s: 0 for s in STATES}
    for r in conn.execute("SELECT state, COUNT(*) AS n FROM candidates WHERE active = 1 GROUP BY state"):
        counts[r["state"]] = r["n"]
    return counts
