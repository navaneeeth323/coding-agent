"""Stage 2b: an LLM reads functions one at a time and reports code that contradicts its intent.

Analyzers only see the shape of code. Bugs where valid code does the wrong
thing (a swapped comparison, a missing state update, side effects before a
check, a caller passing grams to a kilograms parameter) need something that
reads the docstring, the class's invariants and the callees' contracts. This
stage does that, for the riskiest files first, and stores at most one
suspicion per function as an ordinary candidate (source "llm"). Triage then
judges it like any analyzer warning, so a second independent call has to agree
before anything is investigated.
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional

from .candidates import store
from .candidates.git import file_history
from .candidates.scoring import TEST_PATH, Hotspots, rule_prior
from .index import RepoIndex
from .llm import JSONChat, LLMError, RecordingLLM
from .triage.prompt import CATEGORIES, TriageGroup, _callees, _callers, _numbered

log = logging.getLogger(__name__)

RULE = "LLM"
MIN_LINES = 3  # one-liners rarely hide a bug worth a call
MAX_LINES = 150
CLASS_CONTEXT_LINES = 25
SCAN_TOKENS = 1200

SYSTEM_PROMPT = """\
You review one Python function at a time and look for a real bug: code that, for some realistic \
input or sequence of calls, does something different from what it is meant to do.

What the code is meant to do comes from its name, docstring and comments, the comments on the class's \
attributes, and the contracts (signatures, parameter names, docstrings) of the functions it calls. \
Look especially for:
- comparisons or bounds that disagree with the docstring (< vs <=, off-by-one, wrong range);
- a wrong value returned, or arguments swapped or in the wrong units for the function they are passed to;
- state changed (or events emitted) before a check that can still fail, so a failed call leaves changes behind;
- a documented invariant the code forgets to maintain (e.g. a counter or cache that must be updated);
- None or missing values not handled although the docstring or a callee says they can occur;
- precision loss, such as truncating floats where rounding is needed.

Most functions are correct. Unusual style is not a bug, and neither is anything a comment says is deliberate. \
Fill the fields in order; keep each one short:
1. intent: what the function is meant to do.
2. concern: the most suspicious thing in it, or "none".
3. line: the line number of that code.
4. failing_input: the smallest concrete call that would expose it.
5. expected: what correct code returns or does for that call.
6. actual: what this code returns or does, worked out step by step.
7. has_bug: true only if actual differs from expected in a way that matters.
8. category, confidence (0.5 unsure to 1.0 certain)."""


def output_schema() -> dict:
    return {
        "type": "object",
        "properties": {
            "intent": {"type": "string"},
            "concern": {"type": "string"},
            "line": {"type": "integer"},
            "failing_input": {"type": "string"},
            "expected": {"type": "string"},
            "actual": {"type": "string"},
            "has_bug": {"type": "boolean"},
            "category": {"type": "string", "enum": CATEGORIES},
            "confidence": {"type": "number", "minimum": 0.5, "maximum": 1.0},
        },
        "required": ["intent", "concern", "line", "failing_input", "expected", "actual",
                     "has_bug", "category", "confidence"],
    }


@dataclass
class ScanConfig:
    max_functions: int = 200  # per run, riskiest files first
    threshold: float = 0.6  # minimum confidence to store a suspicion
    include_tests: bool = False
    use_cache: bool = True


@dataclass
class ScanStats:
    run_id: int = 0
    functions: int = 0
    flagged: int = 0
    cleared: int = 0
    failed: int = 0
    llm_calls: int = 0
    cached_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    seconds: float = 0.0
    errors: List[str] = field(default_factory=list)


def select_functions(index: RepoIndex, config: ScanConfig) -> List[dict]:
    """Functions and methods worth a call, riskiest files first, longest functions first within a file."""
    rows = [dict(r) for r in index.conn.execute(
        "SELECT s.id, f.path, s.qualname, s.kind, s.start_line, s.end_line, s.parent_id"
        " FROM symbols s JOIN files f ON f.id = s.file_id WHERE s.kind IN ('function', 'method')"
    )]
    paths = sorted({r["path"] for r in rows})
    history = file_history(index.repo)
    hotspots = Hotspots({p: history[p] for p in paths if p in history})
    keep = []
    for r in rows:
        n = r["end_line"] - r["start_line"] + 1
        if not MIN_LINES <= n <= MAX_LINES:
            continue
        if not config.include_tests and TEST_PATH.search(r["path"]):
            continue
        r["heat"] = hotspots.multiplier(r["path"])
        keep.append(r)
    keep.sort(key=lambda r: (-r["heat"], r["path"], -(r["end_line"] - r["start_line"]), r["start_line"]))
    return keep[: config.max_functions]


def _class_context(index: RepoIndex, fn: dict, lines: List[str]) -> List[str]:
    """For a method: the class line and its __init__, where attribute invariants are usually documented."""
    if fn["parent_id"] is None:
        return []
    cls = index.conn.execute("SELECT qualname, start_line FROM symbols WHERE id = ?", (fn["parent_id"],)).fetchone()
    init = index.conn.execute(
        "SELECT start_line, end_line FROM symbols WHERE parent_id = ? AND name = '__init__'", (fn["parent_id"],)
    ).fetchone()
    out = [f"Class {cls['qualname']} (line {cls['start_line']}): {lines[cls['start_line'] - 1].strip()}"]
    if init is not None and fn["qualname"] != f"{cls['qualname']}.__init__":
        end = min(init["end_line"], init["start_line"] + CLASS_CONTEXT_LINES - 1)
        out += ["Its __init__:", *_numbered(lines, init["start_line"], end, set())]
    return out


def build_messages(index: RepoIndex, fn: dict, lines: List[str]) -> List[dict]:
    parts = [f"File: {fn['path']}", f"{fn['kind'].capitalize()}: {fn['qualname']}"]
    ctx = _class_context(index, fn, lines)
    if ctx:
        parts += ["", *ctx]
    parts += ["", "Code:", *_numbered(lines, fn["start_line"], fn["end_line"], set())]
    group = TriageGroup(fn["path"], fn["qualname"], fn["start_line"], fn["end_line"], [])
    callees = _callees(index, group, fn["id"])
    if callees:
        parts += ["", "Functions it calls that are defined in this repo:", *callees]
    callers = _callers(index, group, {fn["path"]: lines})
    if callers:
        parts += ["", "Some call sites:", *callers]
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": "\n".join(parts)}]


def _fingerprint(path: str, qualname: str) -> str:
    # One suspicion per function, keyed without the (model-worded) message so reruns update it.
    return hashlib.sha1(f"llm|{RULE}|{path}|{qualname}".encode()).hexdigest()


def _one_line(text: str, limit: int = 300) -> str:
    return re.sub(r"\s+", " ", text).strip()[:limit]


def scan(
    index: RepoIndex,
    llm: JSONChat,
    config: ScanConfig = ScanConfig(),
    progress: Optional[Callable[[dict, Optional[dict]], None]] = None,
) -> ScanStats:
    conn = index.conn
    store.ensure_schema(conn)
    index.update()
    run_id = store.start_run(conn, "scan")
    recorder = RecordingLLM(llm, conn, stage="scan", run_id=run_id, use_cache=config.use_cache)
    stats = ScanStats(run_id=run_id)
    files: Dict[str, List[str]] = {}

    try:
        functions = select_functions(index, config)
        stats.functions = len(functions)
        for fn in functions:
            if fn["path"] not in files:
                files[fn["path"]] = (index.repo / fn["path"]).read_text("utf-8", "replace").splitlines()
            lines = files[fn["path"]]
            try:
                result = recorder.chat_json(build_messages(index, fn, lines), output_schema(),
                                            max_tokens=SCAN_TOKENS, subject=f"{fn['path']}::{fn['qualname']}")
            except LLMError as exc:
                stats.failed += 1
                stats.errors.append(f"{fn['path']}::{fn['qualname']}: {exc}")
                log.warning("scan failed for %s::%s: %s", fn["path"], fn["qualname"], exc)
                if progress:
                    progress(fn, None)
                continue
            stats.llm_calls += 1
            stats.cached_calls += result.cached
            stats.prompt_tokens += result.prompt_tokens
            stats.completion_tokens += result.completion_tokens
            stats.seconds += result.seconds
            v = result.data
            fp = _fingerprint(fn["path"], fn["qualname"])
            flagged = bool(v.get("has_bug")) and float(v.get("confidence", 0)) >= config.threshold
            with conn:
                if flagged:
                    line = v.get("line") if isinstance(v.get("line"), int) else fn["start_line"]
                    line = min(max(line, fn["start_line"]), fn["end_line"])
                    code = "\n".join(lines[fn["start_line"] - 1 : fn["end_line"]])
                    store.upsert(conn, run_id, {
                        "fingerprint": fp, "source": "llm", "rule": RULE, "path": fn["path"],
                        "line": line, "col": 1, "end_line": line,
                        "message": _one_line(f"{v['concern']} (for {v['failing_input']}: expected "
                                             f"{v['expected']}; actual {v['actual']})"),
                        "symbol": fn["qualname"], "symbol_start": fn["start_line"], "symbol_end": fn["end_line"],
                        "code_sha1": hashlib.sha1(code.encode()).hexdigest(),
                        "score": round(min(1.0, rule_prior(RULE) * float(v["confidence"]) * fn["heat"]), 4),
                    })
                    stats.flagged += 1
                else:
                    # A clean bill of health retires an earlier suspicion about this function.
                    conn.execute("UPDATE candidates SET active = 0 WHERE fingerprint = ?", (fp,))
                    stats.cleared += 1
            if progress:
                progress(fn, v)
    except BaseException:
        store.finish_run(conn, run_id, "failed", asdict(stats))
        raise
    store.finish_run(conn, run_id, "ok", asdict(stats))
    return stats
