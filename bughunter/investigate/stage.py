"""Stage 4: for each triaged candidate, reproduce the bug with a test, then fix it.

Fixed steps, each bounded:
  1. repro: the model writes a test file; it must fail on the current code
     (and be collectable). Up to `test_attempts` tries, each seeing the last result.
  2. fix: the model rewrites the flagged function; the test must then pass and
     the rest of the suite must not get worse. Up to `fix_attempts` tries.
Every candidate also has a token, call and wall-clock budget.

Outcomes: a proposal (candidate stays `investigating` with a patch, for the
verification gate), `rejected` (model gave up or ran out of attempts), or back
to `triaged` (infrastructure failure, so it is retried next run).
"""

from __future__ import annotations

import ast
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath
from typing import Callable, Dict, List, Optional, Sequence, Set

from ..candidates import store
from ..index import RepoIndex
from ..llm import JSONChat, LLMError, RecordingLLM
from ..sandbox import Sandbox, SandboxError, TestRun, Workspace, run_tests, sha1_text
from .edits import clean_test_code, new_redefinitions, replace_region, strip_fences, unified_diff
from .prompt import Target, fix_messages, fix_schema, repro_messages, repro_schema

RETEST_NOTE = """\
A previous test for this bug could not be satisfied by any fix that follows the analysis, so its \
expected values are probably wrong. That test was:
{test}
Its last failure with a candidate fix:
{failure}
The fixer said: {diagnosis}
Recompute every expected value step by step from what the code is meant to do, then write a new test."""

log = logging.getLogger(__name__)

MODULE_CONTEXT = 5  # lines around a module-level finding that a fix may rewrite
MAX_REGION_LINES = 200  # beyond this a small model can't reliably rewrite the region
KEEP_FEEDBACK_ROUNDS = 2  # only the latest attempts go back to the model; context is precious
REPRO_TOKENS = 2000
FIX_TOKENS_BASE = 800


class BudgetExceeded(Exception):
    pass


@dataclass
class InvestigateConfig:
    test_attempts: int = 3
    fix_attempts: int = 3
    rounds: int = 2  # test -> fix rounds; a new round rewrites a test that no sensible fix could satisfy
    group_tries: int = 2  # candidates tried per function before moving on
    max_tokens: int = 60_000  # prompt + completion, per candidate
    max_calls: int = 10
    wall_seconds: float = 900.0
    test_timeout: float = 120.0
    suite_timeout: float = 900.0
    check_suite: bool = True  # run the whole suite after each fix attempt
    limit: int = 20
    use_cache: bool = True


@dataclass
class InvestigateStats:
    run_id: int = 0
    candidates: int = 0
    proposed: int = 0
    rejected: int = 0
    retry: int = 0  # infrastructure failures, back to triaged
    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    seconds: float = 0.0
    errors: List[str] = field(default_factory=list)


class Budget:
    def __init__(self, config: InvestigateConfig):
        self.config = config
        self.started = time.time()
        self.tokens = 0
        self.calls = 0

    def check(self) -> None:
        c = self.config
        if self.calls >= c.max_calls:
            raise BudgetExceeded(f"{self.calls} LLM calls")
        if self.tokens >= c.max_tokens:
            raise BudgetExceeded(f"{self.tokens} tokens")
        if time.time() - self.started >= c.wall_seconds:
            raise BudgetExceeded(f"{time.time() - self.started:.0f}s wall clock")


def repro_test_path(repo: Path, candidate_id: int) -> str:
    name = f"test_bughunter_{candidate_id}.py"
    return f"tests/{name}" if (repo / "tests").is_dir() else name


def _test_ids(run: TestRun, test_path: str) -> Dict[str, str]:
    """Outcomes of the tests in one file (pytest reports them under its dotted module path)."""
    dotted = str(PurePosixPath(test_path).with_suffix("")).replace("/", ".")
    return {t: o for t, o in run.outcomes.items() if t == dotted or t.startswith(dotted + "::")}


def make_target(index: RepoIndex, c: dict) -> Optional[Target]:
    lines = (index.repo / c["path"]).read_text("utf-8", "replace").splitlines()
    if c["symbol"]:
        start, end = c["symbol_start"], c["symbol_end"]
    else:
        start = max(1, c["line"] - MODULE_CONTEXT)
        end = min(len(lines), c["end_line"] + MODULE_CONTEXT)
    if end - start + 1 > MAX_REGION_LINES:
        return None
    return Target(candidate=c, path=c["path"], symbol=c["symbol"], start=start, end=end, lines=lines)


def check_test_code(code: str, target: Target) -> Optional[str]:
    """Cheap static checks before spending a sandbox run. Returns a problem, or None."""
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return f"The test file has a syntax error: {exc}"
    funcs = {n.name for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    if not any(f.startswith("test") for f in funcs):
        return "The file defines no test function (names must start with `test`)."
    if target.module and target.module not in code:
        return f"The test must import the code under test from `{target.module}`."
    if target.symbol:
        name = target.symbol.rsplit(".", 1)[-1]
        if name in funcs and "." not in target.symbol:
            return f"The test redefines `{name}`; import it from `{target.module}` instead of copying it."
    return None


class Investigator:
    def __init__(self, index: RepoIndex, llm: RecordingLLM, sandbox: Sandbox, config: InvestigateConfig,
                 stats: InvestigateStats):
        self.index, self.llm, self.sandbox, self.config, self.stats = index, llm, sandbox, config, stats
        self._baseline: Optional[Set[str]] = None

    # ---- shared ---------------------------------------------------------

    def _ask(self, budget: Budget, messages, schema, subject: str, max_tokens: int, attempt: int) -> dict:
        budget.check()
        result = self.llm.chat_json(
            messages, schema, temperature=0.0 if attempt == 0 else 0.4, seed=attempt,
            max_tokens=max_tokens, subject=subject,
        )
        budget.calls += 1
        budget.tokens += result.prompt_tokens + result.completion_tokens
        self.stats.llm_calls += 1
        self.stats.prompt_tokens += result.prompt_tokens
        self.stats.completion_tokens += result.completion_tokens
        self.stats.seconds += result.seconds
        return result.data

    def baseline_failures(self) -> Set[str]:
        """Tests already failing on the untouched repo; a fix is judged against these."""
        if self._baseline is None:
            with Workspace(self.index.repo) as ws:
                run = run_tests(self.sandbox, ws.path, timeout=self.config.suite_timeout)
            if run.timed_out:
                raise SandboxError("baseline test suite timed out")
            self._baseline = set(run.broken)
        return self._baseline

    @staticmethod
    def _trim(feedback: List[str]) -> List[str]:
        return feedback[-2 * KEEP_FEEDBACK_ROUNDS:]

    # ---- step 1: reproduce ----------------------------------------------

    def reproduce(self, ws: Workspace, target: Target, triage: dict, test_path: str, budget: Budget,
                  log_: List[dict], note: Optional[str] = None):
        """Returns (test_code, failure_text) or (None, reason)."""
        feedback: List[str] = []
        subject = f"{target.path}::{target.symbol or target.start}#repro"
        for attempt in range(self.config.test_attempts):
            data = self._ask(budget, repro_messages(self.index, target, triage, self._trim(feedback), note),
                             repro_schema(), subject, REPRO_TOKENS, attempt)
            if data.get("give_up"):
                log_.append({"phase": "repro", "attempt": attempt, "result": "gave_up", "plan": data.get("plan")})
                return None, f"model gave up while writing a test: {data.get('plan', '')}"
            code = strip_fences(data.get("test_code", ""))
            problem = check_test_code(code, target)
            if problem is None:
                code = clean_test_code(code)
                ws.write(test_path, code)
                run = run_tests(self.sandbox, ws.path, [test_path], timeout=self.config.test_timeout)
                mine = _test_ids(run, test_path)
                failed = [t for t, o in mine.items() if o == "failed"]
                if run.timed_out:
                    problem = "The test timed out. Make it fast and deterministic."
                elif not mine or any(o == "error" for o in mine.values()) or run.exit_code not in (0, 1):
                    problem = "The test could not run (collection or setup error):\n" + run.failure_text(40)
                elif not failed:
                    problem = ("The test PASSED on the current code, so it does not show the bug. "
                               "Use an input that actually triggers it; if the code is correct, give up.")
                elif "ModuleNotFoundError" in run.output or "ImportError" in run.output:
                    problem = "The test fails only because an import is wrong:\n" + run.failure_text(30)
                else:
                    log_.append({"phase": "repro", "attempt": attempt, "result": "fails_on_original",
                                 "summary": run.summary()})
                    return code, run.failure_text(50)
            log_.append({"phase": "repro", "attempt": attempt, "result": "rejected", "problem": problem[:500]})
            feedback += [json.dumps(data), problem + "\nWrite the corrected complete test file."]
        return None, f"no failing test after {self.config.test_attempts} attempts"

    # ---- step 2: fix ----------------------------------------------------

    def fix(self, ws: Workspace, target: Target, triage: dict, test_path: str, test_code: str, failure: str,
            budget: Budget, log_: List[dict]):
        """Returns (new_file_text, {"diagnosis", "title"}, None) or (None, reason, suspicion).

        `suspicion` is set when the evidence points at the test rather than the code:
        the model said so, or every attempt that ran left the test failing.
        """
        original = "\n".join(target.lines) + "\n"
        feedback: List[str] = []
        still_failing = 0
        last_failure, diagnosis = failure, ""
        subject = f"{target.path}::{target.symbol or target.start}#fix"
        max_tokens = FIX_TOKENS_BASE + len(target.region) // 2
        for attempt in range(self.config.fix_attempts):
            data = self._ask(budget, fix_messages(self.index, target, triage, test_code, failure,
                                                  self._trim(feedback)),
                             fix_schema(), subject, max_tokens, attempt)
            if data.get("give_up"):
                log_.append({"phase": "fix", "attempt": attempt, "result": "gave_up",
                             "diagnosis": data.get("diagnosis")})
                return None, f"model gave up while fixing: {data.get('diagnosis', '')}", {
                    "failure": last_failure, "diagnosis": data.get("diagnosis", "")}
            new_code = data.get("new_code", "")
            problem = None
            patched = replace_region(original, target.start, target.end, new_code) if new_code.strip() else None
            if patched is None or patched == original:
                problem = "You returned the region unchanged (or empty)."
            else:
                try:
                    ast.parse(patched)
                except SyntaxError as exc:
                    problem = f"After pasting your code the file has a syntax error: {exc}"
                else:
                    twice = new_redefinitions(original, patched)
                    if twice:
                        problem = (f"Your code defines {', '.join(f'`{n}`' for n in twice)} a second time, "
                                   "leaving the old definition in place. Change the code in the region instead.")
            if problem is None:
                ws.write(target.path, patched)
                run = run_tests(self.sandbox, ws.path, [test_path], timeout=self.config.test_timeout)
                mine = _test_ids(run, test_path)
                if run.timed_out or not mine or any(o != "passed" for o in mine.values()):
                    problem = "The test still fails with your fix:\n" + run.failure_text(40)
                    still_failing += 1
                    last_failure, diagnosis = run.failure_text(30), data.get("diagnosis", "")
                elif self.config.check_suite:
                    suite = run_tests(self.sandbox, ws.path, timeout=self.config.suite_timeout)
                    new_failures = sorted(set(suite.broken) - self.baseline_failures() - set(mine))
                    if suite.timed_out:
                        problem = "The test suite timed out with your fix (an infinite loop?)."
                    elif new_failures:
                        problem = ("The test passes, but your fix breaks other tests: "
                                   + ", ".join(new_failures[:5]) + "\n" + suite.failure_text(40))
                if problem is None:
                    log_.append({"phase": "fix", "attempt": attempt, "result": "passes"})
                    return patched, {"diagnosis": data.get("diagnosis", ""), "title": data.get("title", "")}, None
                ws.write(target.path, original)
            log_.append({"phase": "fix", "attempt": attempt, "result": "rejected", "problem": problem[:500]})
            feedback += [json.dumps(data), problem + "\nReturn the corrected complete region."]
        suspicion = {"failure": last_failure, "diagnosis": diagnosis} \
            if still_failing == self.config.fix_attempts else None
        return None, f"no passing fix after {self.config.fix_attempts} attempts", suspicion

    # ---- one candidate --------------------------------------------------

    def investigate(self, c: dict, others: Sequence[dict] = ()) -> dict:
        """Returns the `investigation` record; its `status` decides the next state."""
        record: dict = {"status": "rejected", "attempts": []}
        target = make_target(self.index, c)
        if target is not None:
            target.others = list(others)
        if target is None:
            record["reason"] = f"region longer than {MAX_REGION_LINES} lines"
            return record
        triage = json.loads(c["data"]).get("triage", {})
        test_path = repro_test_path(self.index.repo, c["id"])
        budget = Budget(self.config)
        original = "\n".join(target.lines) + "\n"
        try:
            with Workspace(self.index.repo) as ws:
                note = None
                for round_ in range(self.config.rounds):
                    test_code, info = self.reproduce(ws, target, triage, test_path, budget, record["attempts"],
                                                     note)
                    if test_code is None:
                        record["reason"] = info
                        return record
                    record["test"] = {"path": test_path, "content": test_code, "failure": info[-3000:]}
                    patched, info, suspicion = self.fix(ws, target, triage, test_path, test_code, info, budget,
                                                        record["attempts"])
                    if patched is not None:
                        break
                    record["reason"] = info
                    if suspicion is None or round_ == self.config.rounds - 1:
                        return record
                    record["attempts"].append({"phase": "retest", "attempt": round_ + 1,
                                               "result": "test suspected wrong"})
                    note = RETEST_NOTE.format(test=test_code.strip(), failure=suspicion["failure"],
                                              diagnosis=suspicion["diagnosis"] or "(nothing)")
        except BudgetExceeded as exc:
            record["reason"] = f"budget exhausted: {exc}"
            return record
        finally:
            record["budget"] = {"calls": budget.calls, "tokens": budget.tokens,
                                "seconds": round(time.time() - budget.started, 1)}
        record.update({
            "status": "proposed",
            "diagnosis": info["diagnosis"],
            "title": info["title"],
            "files": {target.path: {"before_sha1": sha1_text(original), "after": patched}},
            "diff": unified_diff({target.path: original}, {target.path: patched, test_path: test_code}),
        })
        return record


def reset_stale(conn) -> int:
    """Investigations interrupted mid-way (no proposal stored) go back to the queue."""
    n = 0
    for c in store.queue(conn, "investigating", limit=10_000):
        if json.loads(c["data"]).get("investigation", {}).get("status") != "proposed":
            store.transition(conn, c["id"], "triaged", note="interrupted investigation reset")
            n += 1
    return n


def investigate(
    index: RepoIndex,
    llm: JSONChat,
    sandbox: Sandbox,
    config: InvestigateConfig = InvestigateConfig(),
    progress: Optional[Callable[[dict, dict], None]] = None,
) -> InvestigateStats:
    conn = index.conn
    store.ensure_schema(conn)
    reset_stale(conn)
    run_id = store.start_run(conn, "investigate")
    recorder = RecordingLLM(llm, conn, stage="investigate", run_id=run_id, use_cache=config.use_cache)
    stats = InvestigateStats(run_id=run_id)
    worker = Investigator(index, recorder, sandbox, config, stats)

    # One group per function: several warnings in one function usually point at one bug,
    # and one fix per function per run keeps proposals from colliding. Best P(real) first.
    rows = store.queue(conn, "triaged", limit=10_000)
    rows.sort(key=lambda c: -json.loads(c["data"]).get("triage", {}).get("p_real", 0))
    groups: Dict[tuple, List[dict]] = {}
    for c in rows:
        groups.setdefault((c["path"], c["symbol"]) if c["symbol"] else (c["path"], c["id"]), []).append(c)
    batches = list(groups.values())[: config.limit]
    stats.candidates = sum(len(g) for g in batches)

    try:
        for group in batches:
            winner = None
            for n, c in enumerate(group):
                if winner is not None or n >= config.group_tries:
                    note = f"same function as #{winner}, which has a proposed fix" if winner else \
                        f"same function as {n} candidates investigated without success"
                    stats.rejected += 1
                    store.transition(conn, c["id"], "rejected", note=note, run_id=run_id,
                                     data={"merged_into": winner} if winner else None)
                    continue
                if _investigate_one(conn, worker, stats, run_id, c, [o for o in group if o is not c], progress):
                    winner = c["id"]
    except BaseException:
        store.finish_run(conn, run_id, "failed", asdict(stats))
        raise
    store.finish_run(conn, run_id, "ok", asdict(stats))
    return stats


def _investigate_one(conn, worker: "Investigator", stats: InvestigateStats, run_id: int, c: dict,
                     others: List[dict], progress) -> bool:
    """Investigate one candidate and record the outcome. Returns whether it produced a proposal."""
    store.transition(conn, c["id"], "investigating", run_id=run_id)
    try:
        record = worker.investigate(c, others)
    except (LLMError, SandboxError, OSError) as exc:
        stats.retry += 1
        stats.errors.append(f"#{c['id']}: {exc}")
        log.warning("investigation of #%d failed: %s", c["id"], exc)
        store.transition(conn, c["id"], "triaged", note=f"investigation error: {exc}"[:500], run_id=run_id)
        record = {"status": "error", "reason": str(exc)}
    except BaseException:
        store.transition(conn, c["id"], "triaged", note="investigation interrupted", run_id=run_id)
        raise
    else:
        if record["status"] == "proposed":
            stats.proposed += 1
            # Stays `investigating` until the verification gate passes or fails it.
            with conn:
                data = {**json.loads(c["data"]), "investigation": record}
                conn.execute("UPDATE candidates SET data = ?, updated_at = ? WHERE id = ?",
                             (json.dumps(data), time.time(), c["id"]))
        else:
            stats.rejected += 1
            store.transition(conn, c["id"], "rejected", note=record.get("reason", "")[:500],
                             run_id=run_id, data={"investigation": record})
    if progress:
        progress(c, record)
    return record["status"] == "proposed"
