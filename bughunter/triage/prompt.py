"""Prompt, output schema and code context for the triage stage.

The context is deliberately small (one function, not the file) because the
triage model is a local 9B: every extra token dilutes its attention and slows it down.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from ..index import RepoIndex
from ..index.resolve import BUILTIN_METHODS

MAX_FUNCTION_LINES = 150  # longer functions are windowed around the findings
WINDOW_LINES = 40
MODULE_WINDOW_LINES = 15
MAX_CALLEES = 8
MAX_CALLERS = 3

CATEGORIES = [
    "logic", "off_by_one", "none_handling", "exception_handling", "resource_leak",
    "concurrency", "undefined_name", "mutable_state", "api_misuse", "security",
    "style", "other",
]

SYSTEM_PROMPT = """\
You review warnings from static analyzers on Python code and decide which ones point at real bugs.

A real bug: as written, the code crashes, returns wrong results, corrupts state, leaks a resource, \
or hides an error, for some realistic input or sequence of calls.
Not a real bug: style or readability issues; code that is unusual but correct; deliberate patterns \
(often explained by a nearby comment); problems that cannot happen given how the code is used.

For each warning, fill the fields in order:
1. intent: what the flagged code is meant to do (1-2 sentences).
2. failing_input: the smallest concrete input or call sequence most likely to expose the problem \
(e.g. `f([1, 2])`). Use only what the code shown actually does; do not assume extra statements.
3. trace: run that input through the code statement by statement, with the real values. \
Python ranges exclude their end: range(1, 4) is 1, 2, 3. A for loop builds its iterator before the body runs.
4. expected: what correct code would do for that input: return value, exceptions raised, \
and side effects (files or connections left open, errors hidden, shared state changed).
5. actual: what this code actually does, according to your trace, covering the same points.
6. is_real_bug: true only if actual differs from expected in a way that matters. \
If they are the same, it is not a bug, however odd the code looks.
7. category: the kind of defect ("style" if it is not a bug).
8. confidence: how sure you are of your is_real_bug answer, from 0.5 (unsure) to 1.0 (certain).

Keep every field short. Analyzers are noisy and most warnings are not real bugs. \
Judge the code, not the warning text."""


# What decides whether a warning of this kind is a real bug. General Python facts,
# shown next to the warning; without them a small model pattern-matches the rule name.
RULE_HINTS = {
    "B006": "A mutable default is shared between calls; it is a bug only if the function or something "
            "it calls mutates that default object. Passing an explicit argument never touches the default.",
    "B008": "The default is computed once, at definition time; a bug only if a fresh value per call is needed.",
    "B020": "The inner loop's iterator is created before the name is rebound, so the inner loop itself is fine; "
            "a bug only if the outer variable is used after the inner loop.",
    "B023": "The closure sees the loop variable's final value if it is called after the loop moves on.",
    "SIM115": "A file opened without `with` stays open until garbage collection unless closed on every path, "
              "including when an exception is raised.",
    "E722": "A bare except also catches KeyboardInterrupt and SystemExit; a bug if errors are hidden "
            "from callers that need them, or the code continues in a broken state.",
    "BLE001": "Catching Exception is fine when the error is recorded or reported deliberately.",
    "S110": "Silently ignoring an exception is a bug only if callers need to know about the failure.",
    "F841": "An unused variable is a bug only if the value was meant to be used (e.g. a cleaned or "
            "computed value that the code then ignores in favour of the original).",
    "H001": "range(a, b, step) stops before b. Off-by-one bugs show up only at boundaries, so trace "
            "the boundary inputs, not a typical one: an empty sequence, a length of 1, and a length that "
            "is an exact multiple of the step. Check the last iteration and what it produces.",
    "H003": "An index equal to len(x) is out of range; a slice x[i:] with i == len(x) is fine. "
            "Trace the boundary values: index == len(x), index == len(x) - 1, and an empty sequence.",
    "LLM": "This warning comes from another model reading the function, not from an analyzer, and may be "
           "wrong. Check its claim against the code, the docstring and the callees' contracts yourself.",
}


@dataclass
class TriageGroup:
    """All new candidates inside one function (or one module-level window): one LLM call."""

    path: str
    symbol: Optional[str]
    symbol_start: Optional[int]
    symbol_end: Optional[int]
    candidates: List[dict]

    @property
    def subject(self) -> str:
        return f"{self.path}::{self.symbol or 'line %d' % self.candidates[0]['line']}"


def output_schema(n: int) -> dict:
    return {
        "type": "object",
        "properties": {
            "verdicts": {
                "type": "array",
                "minItems": n,
                "maxItems": n,
                "items": {
                    "type": "object",
                    # Order matters: the model reasons before it commits to a verdict.
                    "properties": {
                        "id": {"type": "integer", "enum": list(range(1, n + 1))},
                        "intent": {"type": "string"},
                        "failing_input": {"type": "string"},
                        "trace": {"type": "string"},
                        "expected": {"type": "string"},
                        "actual": {"type": "string"},
                        "is_real_bug": {"type": "boolean"},
                        "category": {"type": "string", "enum": CATEGORIES},
                        "confidence": {"type": "number", "minimum": 0.5, "maximum": 1.0},
                    },
                    "required": [
                        "id", "intent", "failing_input", "trace", "expected", "actual",
                        "is_real_bug", "category", "confidence",
                    ],
                },
            }
        },
        "required": ["verdicts"],
    }


def _numbered(lines: List[str], start: int, end: int, flagged: set) -> List[str]:
    width = len(str(end))
    return [
        f"{'>>' if n in flagged else '  '} {n:>{width}} | {lines[n - 1]}"
        for n in range(max(1, start), min(len(lines), end) + 1)
    ]


def _code_block(group: TriageGroup, lines: List[str]) -> List[str]:
    flagged = {c["line"] for c in group.candidates}
    first = min(c["line"] for c in group.candidates)
    last = max(c["end_line"] for c in group.candidates)
    if group.symbol is None:
        return _numbered(lines, first - MODULE_WINDOW_LINES, last + MODULE_WINDOW_LINES, flagged)
    start, end = group.symbol_start, group.symbol_end
    if end - start + 1 <= MAX_FUNCTION_LINES:
        return _numbered(lines, start, end, flagged)
    lo, hi = max(start, first - WINDOW_LINES), min(end, last + WINDOW_LINES)
    out = []
    if lo > start:
        out += _numbered(lines, start, start, flagged) + ["   ... (lines omitted)"]
    out += _numbered(lines, lo, hi, flagged)
    if hi < end:
        out.append("   ... (rest of function omitted)")
    return out


def _symbol_row(index: RepoIndex, path: str, qualname: str) -> Optional[dict]:
    row = index.conn.execute(
        "SELECT s.id, s.kind, s.signature, s.parent_id FROM symbols s JOIN files f ON f.id = s.file_id"
        " WHERE f.path = ? AND s.qualname = ? ORDER BY s.start_line LIMIT 1",
        (path, qualname),
    ).fetchone()
    return dict(row) if row else None


def _callees(index: RepoIndex, group: TriageGroup, symbol_id: int) -> List[str]:
    """Signatures of repo functions called from this one, so the model knows their contracts."""
    names = [
        r["callee_name"]
        for r in index.conn.execute(
            "SELECT callee_name, MIN(line) AS first FROM calls WHERE caller_id = ?"
            " GROUP BY callee_name ORDER BY first",
            (symbol_id,),
        )
    ]
    out = []
    for name in names:
        if name in BUILTIN_METHODS:
            continue
        defs = index.find_symbol(name, limit=4)
        defs = [d for d in defs if d["qualname"].rsplit(".", 1)[-1] == name and d["qualname"] != group.symbol]
        if not defs or len(defs) > 3:
            continue  # unknown, or too ambiguous to be helpful
        for d in defs:
            doc = f"  # {d['docstring']}" if d["docstring"] else ""
            out.append(f"{d['path']}:{d['start_line']}  {d['signature']}{doc}")
        if len(out) >= MAX_CALLEES:
            break
    return out[:MAX_CALLEES]


def _callers(index: RepoIndex, group: TriageGroup, file_lines: Dict[str, List[str]]) -> List[str]:
    name = group.symbol.rsplit(".", 1)[-1]
    if name in BUILTIN_METHODS or name.startswith("__"):
        return []
    sites = index.find_callers(name, limit=20)
    if len(sites) > 10:
        return []  # a common name: matches would mostly be other functions
    out = []
    for s in sites[:MAX_CALLERS]:
        if s["path"] not in file_lines:
            file_lines[s["path"]] = (index.repo / s["path"]).read_text("utf-8", "replace").splitlines()
        text = file_lines[s["path"]][s["line"] - 1].strip()
        where = f" in {s['caller']}" if s["caller"] else ""
        out.append(f"{s['path']}:{s['line']}{where}: {text}")
    return out


def build_messages(index: RepoIndex, group: TriageGroup) -> List[dict]:
    file_lines: Dict[str, List[str]] = {}
    lines = (index.repo / group.path).read_text("utf-8", "replace").splitlines()
    file_lines[group.path] = lines

    parts = [f"File: {group.path}"]
    sym = _symbol_row(index, group.path, group.symbol) if group.symbol else None
    if sym is not None:
        parts.append(f"{sym['kind'].capitalize()}: {group.symbol}")
        if sym["parent_id"] is not None:
            parent = index.conn.execute(
                "SELECT kind, signature, start_line FROM symbols WHERE id = ?", (sym["parent_id"],)
            ).fetchone()
            parts.append(f"Defined inside (line {parent['start_line']}): {parent['signature']}")
    else:
        parts.append("Module-level code")

    parts += ["", "Code (flagged lines marked >>):", *_code_block(group, lines)]

    if sym is not None:
        callees = _callees(index, group, sym["id"])
        if callees:
            parts += ["", "Functions it calls that are defined in this repo:", *callees]
        callers = _callers(index, group, file_lines)
        if callers:
            parts += ["", "Some call sites:", *callers]

    parts += ["", "Warnings to assess:"]
    for i, c in enumerate(group.candidates, start=1):
        parts.append(f"[{i}] line {c['line']}, {c['rule']}: {c['message']}")
        hint = RULE_HINTS.get(c["rule"])
        if hint:
            parts.append(f"    Note: {hint}")

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "\n".join(parts)},
    ]
