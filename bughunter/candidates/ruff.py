"""Run ruff with a bug-oriented rule selection and collect its findings."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Sequence

from .model import Finding

# Rules that point at likely defects. Style rules are left out: they cost triage
# calls and never lead to a PR worth opening. --isolated ignores the repo's own
# ruff config, so the selection is the same for every repo; `# noqa` still applies.
SELECT = [
    "F",  # pyflakes: undefined names, unused locals, `is` with literals, ...
    "E711", "E712", "E721", "E722", "E9",
    "B",  # flake8-bugbear
    "PLE", "PLW",  # pylint errors and warnings
    "BLE",  # blind `except Exception`
    "S110", "S112",  # try/except/pass, try/except/continue
    "SIM115",  # open() without a context manager
    "FIX",  # TODO / FIXME / XXX / HACK
    "RUF006",  # dangling asyncio task
    "A001", "A002",  # shadowing builtins
    "ASYNC",
]
IGNORE = [
    "F401",  # unused import
    "F403", "F405",  # star imports
    "B904",  # raise without `from` inside except
    "B905",  # zip() without strict=
    "B028",  # warnings.warn without stacklevel
    "PLW0603",  # `global` statement
]

CHUNK = 400  # files per ruff invocation, to stay under argv limits


def run_ruff(repo: Path, paths: Sequence[str]) -> List[Finding]:
    findings: List[Finding] = []
    # ruff reports resolved paths; resolve the root too, or symlinked temp dirs (macOS /var) don't match.
    root = Path(repo).resolve()
    for i in range(0, len(paths), CHUNK):
        proc = subprocess.run(
            [
                sys.executable, "-m", "ruff", "check", "--isolated", "--no-cache", "--exit-zero",
                "--output-format", "json",
                "--select", ",".join(SELECT), "--ignore", ",".join(IGNORE),
                "--", *paths[i : i + CHUNK],
            ],
            cwd=repo, capture_output=True, text=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"ruff failed ({proc.returncode}): {proc.stderr.strip()[:500]}")
        for d in json.loads(proc.stdout or "[]"):
            path = os.path.relpath(Path(d["filename"]).resolve(), root)
            findings.append(
                Finding(
                    source="ruff",
                    rule=d["code"] or "syntax-error",  # ruff reports parse errors without a code
                    path=Path(path).as_posix(),
                    line=d["location"]["row"],
                    col=d["location"]["column"],
                    end_line=d["end_location"]["row"],
                    message=d["message"],
                )
            )
    return findings
