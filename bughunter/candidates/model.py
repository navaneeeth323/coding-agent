from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Finding:
    """One raw lead from a deterministic source, before scoring and storage."""

    source: str  # "ruff" | "heuristic"
    rule: str  # e.g. "B006", "H002"
    path: str  # repo-relative POSIX path
    line: int
    col: int
    end_line: int
    message: str
