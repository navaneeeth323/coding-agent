"""Stage 5: the verification gate. No LLM; this is what keeps hallucinated bugs out of PRs.

A proposal from investigation becomes `verified` only if, in fresh copies of the repo:
  1. the new test fails on the original code (every time, so it isn't flaky),
  2. the same test passes with the fix (every time),
  3. the existing suite has no new failures,
  4. ruff (and mypy, if installed) report no more problems in the changed files than before,
     and none in the new test,
  5. it is not the same bug as a fix already verified: neither fix makes the other's test pass.
Investigation already checked most of this, but in a workspace the model's
attempts had been running in; the gate starts from scratch.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
import sys
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional, Set

from .candidates import store
from .candidates.ruff import run_ruff
from .index import RepoIndex
from .sandbox import Sandbox, SandboxError, TestRun, Workspace, run_tests, sha1_text

log = logging.getLogger(__name__)


@dataclass
class VerifyConfig:
    repeat: int = 2  # runs of the new test on each side, to catch flaky tests
    test_timeout: float = 120.0
    suite_timeout: float = 900.0
    type_check: bool = True  # use mypy when it is installed


@dataclass
class VerifyStats:
    run_id: int = 0
    candidates: int = 0
    verified: int = 0
    rejected: int = 0
    stale: int = 0  # code changed since investigation; back to triaged
    errors: List[str] = field(default_factory=list)


@dataclass
class Check:
    name: str
    ok: bool
    detail: str = ""


def _file_ids(run: TestRun, test_path: str) -> Dict[str, str]:
    dotted = test_path[:-3].replace("/", ".")
    return {t: o for t, o in run.outcomes.items() if t == dotted or t.startswith(dotted + "::")}


def _mypy_available() -> bool:
    try:
        return subprocess.run([sys.executable, "-m", "mypy", "--version"], capture_output=True).returncode == 0
    except OSError:
        return False


def _mypy_errors(root, paths: List[str]) -> Counter:
    proc = subprocess.run(
        [sys.executable, "-m", "mypy", "--no-error-summary", "--ignore-missing-imports",
         "--follow-imports=silent", "--no-incremental", *paths],
        cwd=root, capture_output=True, text=True,
    )
    counts: Counter = Counter()
    for line in proc.stdout.splitlines():
        m = re.match(r"^(.+?):\d+: error:", line)
        if m:
            counts[m.group(1)] += 1
    return counts


class Gate:
    def __init__(self, index: RepoIndex, sandbox: Sandbox, config: VerifyConfig):
        self.index, self.sandbox, self.config = index, sandbox, config
        self._baseline: Optional[Set[str]] = None
        self._mypy = config.type_check and _mypy_available()

    def baseline(self) -> Set[str]:
        if self._baseline is None:
            with Workspace(self.index.repo) as ws:
                run = run_tests(self.sandbox, ws.path, timeout=self.config.suite_timeout)
            if run.timed_out:
                raise SandboxError("baseline test suite timed out")
            self._baseline = set(run.broken)
        return self._baseline

    def check(self, inv: dict) -> List[Check]:
        checks: List[Check] = []
        test_path, test_code = inv["test"]["path"], inv["test"]["content"]
        files = {p: f["after"] for p, f in inv["files"].items()}
        repo = self.index.repo

        scope_ok = not (repo / test_path).exists() and test_path not in files
        checks.append(Check("scope", scope_ok, "" if scope_ok else f"{test_path} already exists or is patched"))
        if not scope_ok:
            return checks

        # 1. fails on the original
        with Workspace(repo) as ws:
            ws.write(test_path, test_code)
            runs = [run_tests(self.sandbox, ws.path, [test_path], timeout=self.config.test_timeout)
                    for _ in range(self.config.repeat)]
        fails = [
            r.exit_code == 1 and not r.timed_out and any(o == "failed" for o in _file_ids(r, test_path).values())
            and not any(o == "error" for o in _file_ids(r, test_path).values())
            for r in runs
        ]
        checks.append(Check("fails_before", all(fails), "; ".join(r.summary() for r in runs)))
        if not all(fails):
            return checks

        # 2. passes with the fix, 3. no new failures in the suite
        with Workspace(repo) as ws:
            ws.write(test_path, test_code)
            ws.apply(files)
            runs = [run_tests(self.sandbox, ws.path, [test_path], timeout=self.config.test_timeout)
                    for _ in range(self.config.repeat)]
            passes = [r.exit_code == 0 and _file_ids(r, test_path)
                      and all(o == "passed" for o in _file_ids(r, test_path).values()) for r in runs]
            checks.append(Check("passes_after", all(passes), "; ".join(r.summary() for r in runs)))
            if not all(passes):
                return checks

            suite = run_tests(self.sandbox, ws.path, timeout=self.config.suite_timeout)
            mine = set(_file_ids(suite, test_path))
            new = sorted(set(suite.broken) - self.baseline() - mine)
            suite_ok = not suite.timed_out and not new
            detail = suite.summary() + (f"; new failures: {', '.join(new[:10])}" if new else "")
            checks.append(Check("suite", suite_ok, detail))
            if not suite_ok:
                return checks

            # 4. static checks no worse on the changed files, and clean on the new test
            paths = sorted(files)
            before = Counter(f.path for f in run_ruff(repo, paths))
            after = Counter(f.path for f in run_ruff(ws.path, [*paths, test_path]))
            worse = {p: (before[p], after[p]) for p in [*paths, test_path] if after[p] > before[p]}
            checks.append(Check("ruff", not worse, f"before/after: {worse}" if worse else
                                f"{sum(before.values())} -> {sum(after.values())}"))
            if self._mypy:
                before = _mypy_errors(repo, paths)
                after = _mypy_errors(ws.path, paths)
                worse = {p: (before[p], after[p]) for p in after if after[p] > before[p]}
                checks.append(Check("mypy", not worse, f"before/after: {worse}" if worse else
                                    f"{sum(before.values())} -> {sum(after.values())}"))
        return checks

    def _passes(self, test_path: str, test_code: str, files: Dict[str, str]) -> bool:
        with Workspace(self.index.repo) as ws:
            ws.write(test_path, test_code)
            ws.apply(files)
            run = run_tests(self.sandbox, ws.path, [test_path], timeout=self.config.test_timeout)
        mine = _file_ids(run, test_path)
        return run.exit_code == 0 and bool(mine) and all(o == "passed" for o in mine.values())

    def duplicate_of(self, candidate_id: int, inv: dict) -> Optional[int]:
        """An already verified (or PR'd) candidate fixing the same bug: either fix makes the other's test pass.

        Warnings in different functions can share one cause (a mutable default in __init__
        surfacing in a method), and the function grouping in investigation can't see that.
        """
        rows = self.index.conn.execute(
            "SELECT id, data FROM candidates WHERE state IN ('verified', 'pr_opened') AND id != ? ORDER BY id",
            (candidate_id,),
        ).fetchall()
        mine = {p: f["after"] for p, f in inv["files"].items()}
        for row in rows:
            other = json.loads(row["data"]).get("investigation", {})
            theirs = {p: f["after"] for p, f in other.get("files", {}).items()}
            if not set(theirs) & set(mine) or "test" not in other:
                continue
            if any(sha1_text(self.index.repo.joinpath(p).read_text("utf-8")) != f["before_sha1"]
                   for p, f in other["files"].items() if self.index.repo.joinpath(p).exists()):
                continue  # their fix was made against older code
            if self._passes(inv["test"]["path"], inv["test"]["content"], theirs) or \
                    self._passes(other["test"]["path"], other["test"]["content"], mine):
                return row["id"]
        return None


def verify(
    index: RepoIndex,
    sandbox: Sandbox,
    config: VerifyConfig = VerifyConfig(),
    progress: Optional[Callable[[dict, List[Check]], None]] = None,
) -> VerifyStats:
    conn = index.conn
    store.ensure_schema(conn)
    run_id = store.start_run(conn, "verify")
    stats = VerifyStats(run_id=run_id)
    gate = Gate(index, sandbox, config)

    rows = [c for c in store.queue(conn, "investigating", limit=10_000)
            if json.loads(c["data"]).get("investigation", {}).get("status") == "proposed"]
    stats.candidates = len(rows)
    try:
        for c in rows:
            data = json.loads(c["data"])
            inv = data["investigation"]
            stale = [p for p, f in inv["files"].items()
                     if not (index.repo / p).exists()
                     or sha1_text((index.repo / p).read_text("utf-8")) != f["before_sha1"]]
            if stale:
                stats.stale += 1
                inv["status"] = "stale"
                store.transition(conn, c["id"], "triaged", note=f"code changed since investigation: {stale}",
                                 run_id=run_id, data={"investigation": inv})
                continue
            try:
                checks = gate.check(inv)
            except (SandboxError, OSError, RuntimeError) as exc:
                stats.errors.append(f"#{c['id']}: {exc}")
                log.warning("verification of #%d failed: %s", c["id"], exc)
                continue  # stays proposed; retried next run
            ok = all(ch.ok for ch in checks)
            dup = None
            if ok:
                try:
                    dup = gate.duplicate_of(c["id"], inv)
                except (SandboxError, OSError, RuntimeError) as exc:
                    stats.errors.append(f"#{c['id']}: {exc}")
                    continue
                checks.append(Check("duplicate", dup is None, f"same bug as #{dup}" if dup else "none"))
                ok = dup is None
            record = {"passed": ok, "checks": [asdict(ch) for ch in checks]}
            if dup is not None:
                record["merged_into"] = dup
            if ok:
                stats.verified += 1
                store.transition(conn, c["id"], "verified", run_id=run_id, data={"verification": record})
            else:
                stats.rejected += 1
                failed = next(ch for ch in checks if not ch.ok)
                store.transition(conn, c["id"], "rejected", note=f"verification: {failed.name} {failed.detail}"[:500],
                                 run_id=run_id, data={"verification": record})
            if progress:
                progress(c, checks)
    except BaseException:
        store.finish_run(conn, run_id, "failed", asdict(stats))
        raise
    store.finish_run(conn, run_id, "ok", asdict(stats))
    return stats
