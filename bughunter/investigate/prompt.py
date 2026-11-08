"""Prompts and schemas for investigation: write a failing test, then a fix.

Two narrow tasks instead of one open-ended agent loop: a 9B model handles
"write one test file" and "rewrite one function" well, and wanders off when
given a toolbox. Tool use is replaced by feeding back real pytest output.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from ..index import RepoIndex
from ..triage.prompt import TriageGroup, _callees, _numbered, _symbol_row
from .edits import module_name, region_text

EXAMPLE_TEST_LINES = 25

TEST_SYSTEM = """\
You write one pytest test file that reproduces a suspected bug in Python code.

The test must FAIL on the code as it is now, because of the bug, and PASS once the bug is fixed. \
Rules:
- Import the code under test from its module; never copy or redefine it in the test.
- Assert the correct behaviour (what the code should do), so the assertion fails today.
- Keep it small: one to three test functions, no network, no sleeping, temporary files only via tmp_path.
- Use only the standard library and pytest.
- If, after looking at the code, you are confident the code is actually correct, set give_up to true \
and leave test_code empty."""

FIX_SYSTEM = """\
You fix a bug in one Python function (or block of code). A test that exposes the bug is given, with its \
failure on the current code.

Rules:
- Return the complete corrected code for the region shown, from its first line to its last, \
ready to paste in place of the original.
- Make the smallest change that fixes the bug. Keep the signature, names, docstring and style.
- Fix the cause, not the symptom: do not special-case the test's inputs, and do not change behaviour \
the test does not need changed.
- Do not add prints, logging or comments explaining the fix.
- If the test itself is wrong rather than the code, set give_up to true and leave new_code empty."""


@dataclass
class Target:
    """What investigation may change: one region of one file."""

    candidate: dict
    path: str
    symbol: Optional[str]
    start: int
    end: int
    lines: List[str]  # the whole file as it was when investigation started
    others: List[dict] = field(default_factory=list)  # other candidates in the same function

    @property
    def region(self) -> str:
        return region_text(self.lines, self.start, self.end)

    @property
    def module(self) -> str:
        return module_name(self.path)


def repro_schema() -> dict:
    return {
        "type": "object",
        "properties": {
            "plan": {"type": "string"},
            "give_up": {"type": "boolean"},
            "test_code": {"type": "string"},
        },
        "required": ["plan", "give_up", "test_code"],
    }


def fix_schema() -> dict:
    return {
        "type": "object",
        "properties": {
            "diagnosis": {"type": "string"},
            "give_up": {"type": "boolean"},
            "new_code": {"type": "string"},
        },
        "required": ["diagnosis", "give_up", "new_code"],
    }


def _sketch(triage: dict) -> List[str]:
    out = []
    for key, label in (("intent", "Intent"), ("failing_input", "Suspected failing input"),
                       ("expected", "Expected"), ("actual", "Actual")):
        if triage.get(key):
            out.append(f"{label}: {triage[key]}")
    return out


def _example_test(index: RepoIndex, target: Target) -> List[str]:
    """The head of an existing test that imports this module, to copy its conventions."""
    mod = target.module
    if not mod:
        return []
    for row in index.conn.execute("SELECT path FROM files ORDER BY path"):
        p = row["path"]
        name = Path(p).name
        if not (name.startswith("test_") or name.endswith("_test.py")):
            continue
        try:
            text = (index.repo / p).read_text("utf-8", "replace")
        except OSError:
            continue
        if f"from {mod} import" in text or f"import {mod}" in text:
            return [f"Existing test {p} (first lines):", *text.splitlines()[:EXAMPLE_TEST_LINES]]
    return []


def _context(index: RepoIndex, target: Target) -> List[str]:
    c = target.candidate
    parts = [f"File: {target.path}  (module `{target.module}`)"]
    if target.symbol:
        parts.append(f"Code under suspicion: {target.symbol}, lines {target.start}-{target.end}")
    parts += ["", "Code (flagged line marked >>):",
              *_numbered(target.lines, target.start, target.end, {c["line"]})]
    if target.others:
        parts += ["", "Other warnings in the same code (may or may not be related):"]
        parts += [f"- line {o['line']} ({o['rule']}): {o['message']}" for o in target.others]
    sym = _symbol_row(index, target.path, target.symbol) if target.symbol else None
    if sym is not None:
        group = TriageGroup(target.path, target.symbol, target.start, target.end, [c])
        callees = _callees(index, group, sym["id"])
        if callees:
            parts += ["", "Functions it calls that are defined in this repo:", *callees]
    return parts


def repro_messages(
    index: RepoIndex, target: Target, triage: dict, feedback: List[str], note: Optional[str] = None,
) -> List[dict]:
    c = target.candidate
    parts = _context(index, target)
    parts += ["", f"Analyzer warning at line {c['line']} ({c['rule']}): {c['message']}"]
    parts += ["", "A reviewer's analysis:", *_sketch(triage)]
    example = _example_test(index, target)
    if example:
        parts += ["", *example]
    if note:
        parts += ["", note]
    parts += ["", f"Write the test file. Import with e.g. `from {target.module} import ...`."]
    messages = [{"role": "system", "content": TEST_SYSTEM}, {"role": "user", "content": "\n".join(parts)}]
    return messages + _feedback_turns(feedback)


def fix_messages(
    index: RepoIndex, target: Target, triage: dict, test_code: str, failure: str, feedback: List[str],
) -> List[dict]:
    parts = _context(index, target)
    parts += ["", "A reviewer's analysis:", *_sketch(triage)]
    parts += ["", "Test that exposes the bug:", test_code.strip(),
              "", "Its failure on the current code:", failure,
              "", f"Region to rewrite (lines {target.start}-{target.end}), exactly as it is now:",
              target.region,
              "", "Return the corrected region."]
    messages = [{"role": "system", "content": FIX_SYSTEM}, {"role": "user", "content": "\n".join(parts)}]
    return messages + _feedback_turns(feedback)


def _feedback_turns(feedback: List[str]) -> List[dict]:
    """Earlier attempts and what went wrong, as alternating turns: the model sees its own output."""
    turns = []
    for i in range(0, len(feedback), 2):
        turns.append({"role": "assistant", "content": feedback[i]})
        turns.append({"role": "user", "content": feedback[i + 1]})
    return turns
