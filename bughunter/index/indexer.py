"""Build and query the code index for a repository."""

from __future__ import annotations

import hashlib
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

from . import store
from .parser import parse_python
from .repomap import build_repo_map

SKIP_DIRS = {
    ".git", ".hg", ".svn", ".venv", "venv", "env", "node_modules", "__pycache__",
    ".mypy_cache", ".pytest_cache", ".ruff_cache", ".tox", ".nox", "build", "dist",
    ".eggs", "site-packages", ".bughunter",
    # Vendored third-party code: not ours to fix, and it drowns out the real repo map.
    "_vendor", "vendor", "third_party",
}
# Bump when parser output changes, so unchanged files get re-extracted too.
EXTRACTOR_VERSION = "1"
MAX_FILE_BYTES = 1_000_000  # bigger files are almost always generated or vendored
DEFAULT_DB_DIR = ".bughunter"


@dataclass
class UpdateStats:
    added: int = 0
    changed: int = 0
    removed: int = 0
    unchanged: int = 0
    skipped: int = 0  # too large or unreadable
    parse_errors: int = 0
    seconds: float = 0.0


def default_db_path(repo: Path) -> Path:
    return repo / DEFAULT_DB_DIR / "state.db"


def discover_python_files(repo: Path) -> List[str]:
    """Repo-relative POSIX paths of Python files, honouring .gitignore when possible."""
    if (repo / ".git").exists():
        try:
            out = subprocess.run(
                ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard", "--", "*.py"],
                cwd=repo, capture_output=True, check=True,
            ).stdout
            paths = [p for p in out.decode("utf-8", "replace").split("\0") if p]
            return sorted(
                p for p in paths
                if not SKIP_DIRS.intersection(p.split("/")[:-1]) and (repo / p).is_file()
            )
        except (OSError, subprocess.CalledProcessError):
            pass  # fall back to walking the tree

    found = []
    for dirpath, dirnames, filenames in os.walk(repo):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.endswith(".egg-info")]
        for name in filenames:
            if name.endswith(".py"):
                found.append((Path(dirpath) / name).relative_to(repo).as_posix())
    return sorted(found)


class RepoIndex:
    """Symbol index for one repository, persisted in SQLite and updated incrementally."""

    def __init__(self, repo: Path, db_path: Optional[Path] = None):
        self.repo = Path(repo).resolve()
        self.db_path = Path(db_path) if db_path else default_db_path(self.repo)
        if self.db_path.parent.name == DEFAULT_DB_DIR and self.db_path.parent.parent == self.repo:
            # Keep the index out of `git status` (and out of any PR the agent opens).
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            ignore = self.db_path.parent / ".gitignore"
            if not ignore.exists():
                ignore.write_text("*\n")
        self.conn = store.connect(self.db_path)

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "RepoIndex":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ---- building -------------------------------------------------------

    def update(self) -> UpdateStats:
        """Re-index files whose content changed since the last run; drop deleted files."""
        started = time.time()
        stats = UpdateStats()
        if store.get_meta(self.conn, "extractor_version") != EXTRACTOR_VERSION:
            with self.conn:
                self.conn.execute("DELETE FROM files")
        known = store.file_hashes(self.conn)
        current = discover_python_files(self.repo)

        with self.conn:
            stale = set(known) - set(current)
            store.delete_files(self.conn, stale)
            stats.removed = len(stale)

            for rel in current:
                path = self.repo / rel
                try:
                    if path.stat().st_size > MAX_FILE_BYTES:
                        stats.skipped += 1
                        if rel in known:
                            store.delete_files(self.conn, [rel])
                        continue
                    src = path.read_bytes()
                except OSError:
                    stats.skipped += 1
                    continue

                sha1 = hashlib.sha1(src).hexdigest()
                if known.get(rel) == sha1:
                    stats.unchanged += 1
                    continue

                parsed = parse_python(src)
                stats.parse_errors += parsed.has_error
                store.write_file(self.conn, rel, sha1, src.count(b"\n") + 1, parsed, started)
                if rel in known:
                    stats.changed += 1
                else:
                    stats.added += 1

            store.set_meta(self.conn, "repo_root", str(self.repo))
            store.set_meta(self.conn, "extractor_version", EXTRACTOR_VERSION)
            store.set_meta(self.conn, "updated_at", str(started))

        stats.seconds = time.time() - started
        return stats

    # ---- queries --------------------------------------------------------

    def find_symbol(self, name: str, limit: int = 20) -> List[dict]:
        """Definitions matching a bare name ("get") or a qualified name ("Cache.get")."""
        rows = self.conn.execute(
            "SELECT f.path, s.qualname, s.kind, s.start_line, s.end_line, s.signature, s.docstring"
            " FROM symbols s JOIN files f ON f.id = s.file_id"
            " WHERE s.name = ? OR s.qualname = ? OR s.qualname LIKE ?"
            " ORDER BY f.path, s.start_line LIMIT ?",
            (name, name, "%." + name, limit),
        )
        return [dict(r) for r in rows]

    def find_callers(self, name: str, limit: int = 50) -> List[dict]:
        """Call sites whose callee name matches the last component of `name`.

        Matching is by name only (no type inference), so `Cache.get` also matches
        `dict.get` calls. Treat results as leads; `callee_expr` helps disambiguate.
        """
        callee = name.rsplit(".", 1)[-1]
        rows = self.conn.execute(
            "SELECT f.path, c.line, c.callee_expr, s.qualname AS caller"
            " FROM calls c JOIN files f ON f.id = c.file_id"
            " LEFT JOIN symbols s ON s.id = c.caller_id"
            " WHERE c.callee_name = ? ORDER BY f.path, c.line LIMIT ?",
            (callee, limit),
        )
        return [dict(r) for r in rows]

    def symbol_at(self, path: str, line: int) -> Optional[dict]:
        """Innermost class/function enclosing `line` — maps analyzer findings to functions."""
        row = self.conn.execute(
            "SELECT f.path, s.qualname, s.kind, s.start_line, s.end_line, s.signature"
            " FROM symbols s JOIN files f ON f.id = s.file_id"
            " WHERE f.path = ? AND s.start_line <= ? AND s.end_line >= ?"
            " ORDER BY s.end_line - s.start_line LIMIT 1",
            (path, line, line),
        ).fetchone()
        return dict(row) if row else None

    def symbol_source(self, path: str, qualname: str, numbered: bool = True) -> Optional[str]:
        """Source of one symbol, read from disk using the indexed line range."""
        row = self.conn.execute(
            "SELECT f.sha1, s.start_line, s.end_line FROM symbols s JOIN files f ON f.id = s.file_id"
            " WHERE f.path = ? AND s.qualname = ? ORDER BY s.start_line LIMIT 1",
            (path, qualname),
        ).fetchone()
        if row is None:
            return None
        src = (self.repo / path).read_bytes()
        if hashlib.sha1(src).hexdigest() != row["sha1"]:
            raise RuntimeError(f"{path} changed since it was indexed; call update() first")
        lines = src.decode("utf-8", "replace").splitlines()[row["start_line"] - 1 : row["end_line"]]
        if not numbered:
            return "\n".join(lines)
        width = len(str(row["end_line"]))
        return "\n".join(
            f"{n:>{width}}  {text}" for n, text in enumerate(lines, start=row["start_line"])
        )

    def repo_map(self, max_tokens: int = 1024, focus: Sequence[str] = ()) -> str:
        return build_repo_map(self.conn, max_tokens=max_tokens, focus=focus)
