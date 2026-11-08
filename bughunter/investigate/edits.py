"""Small text operations for applying model-written code: regions, indentation, diffs."""

from __future__ import annotations

import difflib
import re
import textwrap
from pathlib import PurePosixPath
from typing import Dict, List, Tuple

_FENCE = re.compile(r"^\s*```[\w+-]*\s*\n(.*?)\n\s*```\s*$", re.S)


def strip_fences(code: str) -> str:
    """Models often wrap code in ``` fences even inside a JSON string."""
    m = _FENCE.match(code)
    return m.group(1) if m else code


def indentation(line: str) -> str:
    return line[: len(line) - len(line.lstrip())]


def region_text(lines: List[str], start: int, end: int) -> str:
    """Lines start..end (1-based, inclusive) as text."""
    return "\n".join(lines[start - 1 : end])


def replace_region(text: str, start: int, end: int, new_code: str) -> str:
    """Replace lines start..end with `new_code`, re-indented to where the original region sat."""
    lines = text.splitlines()
    indent = indentation(lines[start - 1])
    body = textwrap.dedent(strip_fences(new_code).strip("\n"))
    new_lines = [indent + l if l.strip() else "" for l in body.splitlines()]
    out = lines[: start - 1] + new_lines + lines[end:]
    return "\n".join(out) + ("\n" if text.endswith("\n") else "")


def module_name(path: str) -> str:
    """`src/pkg/core.py` -> `pkg.core`; `pkg/__init__.py` -> `pkg`."""
    parts = list(PurePosixPath(path).with_suffix("").parts)
    if parts and parts[0] == "src":
        parts = parts[1:]
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def unified_diff(before: Dict[str, str], after: Dict[str, str]) -> str:
    """git-style diff; a path missing from `before` is a new file."""
    chunks = []
    for path in sorted(after):
        old = before.get(path)
        chunks.extend(difflib.unified_diff(
            [] if old is None else old.splitlines(keepends=True),
            after[path].splitlines(keepends=True),
            fromfile="/dev/null" if old is None else f"a/{path}",
            tofile=f"b/{path}",
        ))
    return "".join(c if c.endswith("\n") else c + "\n" for c in chunks)


def changed_lines(before: str, after: str) -> List[Tuple[int, int]]:
    """Line ranges (1-based, inclusive, in `before`) that the edit touched."""
    sm = difflib.SequenceMatcher(a=before.splitlines(), b=after.splitlines(), autojunk=False)
    return [(i1 + 1, max(i1 + 1, i2)) for tag, i1, i2, _, _ in sm.get_opcodes() if tag != "equal"]
