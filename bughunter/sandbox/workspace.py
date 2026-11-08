"""Throwaway copies of the repo for running untrusted code and trying patches.

The original checkout is never modified: every test run, repro attempt and fix
happens in a copy that is deleted afterwards.
"""

from __future__ import annotations

import hashlib
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Optional

from ..index.indexer import SKIP_DIRS

# State, caches and VCS metadata stay behind; copying them is slow and leaks history.
_IGNORE = shutil.ignore_patterns(*SKIP_DIRS, "*.egg-info", "*.pyc", ".coverage")


def sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()


class Workspace:
    def __init__(self, source: Path, root: Optional[Path] = None):
        self._tmp = tempfile.mkdtemp(prefix="bughunter-ws-", dir=root)
        self.path = Path(self._tmp) / "repo"
        shutil.copytree(source, self.path, ignore=_IGNORE, symlinks=True)

    def read(self, rel: str) -> str:
        return (self.path / rel).read_text("utf-8")

    def write(self, rel: str, text: str) -> None:
        target = self.path / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, "utf-8")

    def apply(self, files: Dict[str, str]) -> None:
        for rel, text in files.items():
            self.write(rel, text)

    def close(self) -> None:
        shutil.rmtree(self._tmp, ignore_errors=True)

    def __enter__(self) -> "Workspace":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
