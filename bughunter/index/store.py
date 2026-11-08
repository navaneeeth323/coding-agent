"""SQLite storage for the code index."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterable, List, Optional

from .parser import ParsedFile

SCHEMA_VERSION = 1

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS files (
    id         INTEGER PRIMARY KEY,
    path       TEXT NOT NULL UNIQUE,   -- POSIX path relative to repo root
    sha1       TEXT NOT NULL,
    lines      INTEGER NOT NULL,
    has_error  INTEGER NOT NULL,       -- tree-sitter hit a syntax error
    indexed_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS symbols (
    id         INTEGER PRIMARY KEY,
    file_id    INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    parent_id  INTEGER REFERENCES symbols(id) ON DELETE CASCADE,
    name       TEXT NOT NULL,
    qualname   TEXT NOT NULL,
    kind       TEXT NOT NULL,          -- class | function | method
    start_line INTEGER NOT NULL,
    end_line   INTEGER NOT NULL,
    signature  TEXT NOT NULL,
    docstring  TEXT
);
CREATE INDEX IF NOT EXISTS symbols_name ON symbols(name);
CREATE INDEX IF NOT EXISTS symbols_qualname ON symbols(qualname);
CREATE INDEX IF NOT EXISTS symbols_file ON symbols(file_id, start_line);

CREATE TABLE IF NOT EXISTS calls (
    id          INTEGER PRIMARY KEY,
    file_id     INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    caller_id   INTEGER REFERENCES symbols(id) ON DELETE CASCADE,  -- NULL = module level
    callee_name TEXT NOT NULL,
    callee_expr TEXT NOT NULL,
    line        INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS calls_callee ON calls(callee_name);
CREATE INDEX IF NOT EXISTS calls_file ON calls(file_id);

CREATE TABLE IF NOT EXISTS imports (
    id      INTEGER PRIMARY KEY,
    file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    module  TEXT NOT NULL,
    name    TEXT,
    alias   TEXT,
    line    INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS imports_file ON imports(file_id);
"""


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    if version not in (0, SCHEMA_VERSION):
        raise RuntimeError(
            f"{db_path} has index schema v{version}, expected v{SCHEMA_VERSION}; delete it to rebuild"
        )
    conn.executescript(SCHEMA)
    conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
    return conn


def file_hashes(conn: sqlite3.Connection) -> dict:
    return {row["path"]: row["sha1"] for row in conn.execute("SELECT path, sha1 FROM files")}


def delete_files(conn: sqlite3.Connection, paths: Iterable[str]) -> None:
    conn.executemany("DELETE FROM files WHERE path = ?", [(p,) for p in paths])


def write_file(
    conn: sqlite3.Connection, path: str, sha1: str, lines: int, parsed: ParsedFile, now: float
) -> None:
    """Replace everything indexed for `path`. Caller owns the transaction."""
    conn.execute("DELETE FROM files WHERE path = ?", (path,))
    file_id = conn.execute(
        "INSERT INTO files (path, sha1, lines, has_error, indexed_at) VALUES (?, ?, ?, ?, ?)",
        (path, sha1, lines, int(parsed.has_error), now),
    ).lastrowid

    # Parents always precede children in parsed.symbols, so ids resolve in one pass.
    symbol_ids: List[int] = []
    for sym in parsed.symbols:
        parent_id = symbol_ids[sym.parent] if sym.parent is not None else None
        symbol_ids.append(
            conn.execute(
                "INSERT INTO symbols (file_id, parent_id, name, qualname, kind, start_line,"
                " end_line, signature, docstring) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    file_id,
                    parent_id,
                    sym.name,
                    sym.qualname,
                    sym.kind,
                    sym.start_line,
                    sym.end_line,
                    sym.signature,
                    sym.docstring,
                ),
            ).lastrowid
        )

    conn.executemany(
        "INSERT INTO calls (file_id, caller_id, callee_name, callee_expr, line) VALUES (?, ?, ?, ?, ?)",
        [
            (
                file_id,
                symbol_ids[c.caller] if c.caller is not None else None,
                c.callee_name,
                c.callee_expr,
                c.line,
            )
            for c in parsed.calls
        ],
    )
    conn.executemany(
        "INSERT INTO imports (file_id, module, name, alias, line) VALUES (?, ?, ?, ?, ?)",
        [(file_id, i.module, i.name, i.alias, i.line) for i in parsed.imports],
    )


def get_meta(conn: sqlite3.Connection, key: str) -> Optional[str]:
    row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return row["value"] if row else None


def set_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        "INSERT INTO meta (key, value) VALUES (?, ?)"
        " ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (key, value),
    )
