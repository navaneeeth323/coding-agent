"""Small text operations for applying model-written code: regions, indentation, diffs."""

from __future__ import annotations

import ast
import difflib
import io
import re
import subprocess
import sys
import textwrap
import tokenize
from pathlib import PurePosixPath
from typing import Dict, List, Set, Tuple

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


def _strip_comments(code: str) -> str:
    """Remove `#` comments; lines that held only a comment go entirely. Pragmas (noqa, type:) stay."""
    lines = code.splitlines()
    dropped: Set[int] = set()
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(code).readline))
    except (tokenize.TokenError, SyntaxError):
        return code
    for tok in tokens:
        if tok.type != tokenize.COMMENT or re.match(r"#\s*(noqa|type:|pragma)", tok.string):
            continue
        row, col = tok.start
        lines[row - 1] = lines[row - 1][:col].rstrip()
        if not lines[row - 1].strip():
            dropped.add(row - 1)
    return "\n".join(l for i, l in enumerate(lines) if i not in dropped)


def _squeeze_blank_lines(code: str) -> str:
    """At most one blank line inside a block, two before top-level code."""
    out: List[str] = []
    blanks = 0
    for line in code.splitlines():
        if not line.strip():
            blanks += 1
            continue
        if out:
            out += [""] * min(blanks, 1 if line[:1].isspace() else 2)
        blanks = 0
        out.append(line.rstrip())
    return "\n".join(out) + "\n"


def _remove_unused_imports(code: str) -> str:
    proc = subprocess.run(
        [sys.executable, "-m", "ruff", "check", "--isolated", "--no-cache", "--exit-zero", "--select", "F401",
         "--fix", "--stdin-filename", "test_generated.py", "-"],
        input=code, capture_output=True, text=True,
    )
    return proc.stdout if proc.returncode == 0 and proc.stdout.strip() else code


def clean_test_code(code: str) -> str:
    """Tidy a model-written test before it is run, verified and shown in a PR.

    A small model thinks out loud in comments ("wait, let's check tax"); those, unused
    imports, trailing whitespace and runs of blank lines go. Behaviour is unchanged.
    """
    try:
        ast.parse(code)
    except SyntaxError:
        return code  # leave it for the syntax check to report
    return _remove_unused_imports(_squeeze_blank_lines(_strip_comments(code)))


def _duplicate_defs(tree: ast.AST) -> Set[str]:
    """Functions/classes defined twice in the same module or class body (decorated ones, like setters, aside)."""
    out: Set[str] = set()

    def walk(body, prefix: str) -> None:
        seen: Set[str] = set()
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if node.name in seen and not node.decorator_list:
                    out.add(prefix + node.name)
                seen.add(node.name)
                if isinstance(node, ast.ClassDef):
                    walk(node.body, f"{prefix}{node.name}.")

    walk(tree.body, "")
    return out


def new_redefinitions(before: str, after: str) -> List[str]:
    """Names the edit defines a second time in the same scope, e.g. a fix that pastes in a new `__init__`."""
    return sorted(_duplicate_defs(ast.parse(after)) - _duplicate_defs(ast.parse(before)))
