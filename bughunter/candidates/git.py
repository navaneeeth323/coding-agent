"""Per-file history signals: churn and how often a file needed fixing."""

from __future__ import annotations

import re
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

FIX_PATTERN = re.compile(r"\b(fix(e[sd])?|bug|regression|crash|broken|hotfix|patch)\b", re.IGNORECASE)
MAX_COMMITS = 2000
SEP = "\x1e"  # record separator; can't appear in commit subjects


@dataclass
class FileHistory:
    commits: int = 0
    fix_commits: int = 0
    last_commit_at: int = 0  # unix seconds


def file_history(repo: Path) -> Dict[str, FileHistory]:
    """History for files under `repo`, keyed by repo-relative path. Empty if not a git checkout."""
    try:
        out = subprocess.run(
            ["git", "log", "--no-merges", "--relative", f"-n{MAX_COMMITS}",
             f"--format={SEP}%ct %s", "--name-only", "--", "."],
            cwd=repo, capture_output=True, text=True, check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return {}

    history: Dict[str, FileHistory] = defaultdict(FileHistory)
    for record in out.split(SEP)[1:]:
        header, _, names = record.partition("\n")
        timestamp, _, subject = header.partition(" ")
        is_fix = bool(FIX_PATTERN.search(subject))
        for name in names.split("\n"):
            if not name.endswith(".py"):
                continue
            h = history[name]
            h.commits += 1
            h.fix_commits += is_fix
            h.last_commit_at = max(h.last_commit_at, int(timestamp))
    return dict(history)
