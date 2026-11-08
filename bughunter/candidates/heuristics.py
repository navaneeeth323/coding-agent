"""Tree-sitter checks for bug-prone patterns that ruff doesn't cover.

These are deliberately cheap and noisy: they flag code worth a second look
(mostly off-by-one territory) and leave the judgement to triage.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, List, Optional, Sequence

from tree_sitter import Node

from ..index.parser import parse_tree
from .model import Finding


def _text(node: Node) -> str:
    return node.text.decode("utf-8", "replace")


def _len_arg(node: Optional[Node]) -> Optional[str]:
    """For `len(x)` return the text of x, else None."""
    if node is None or node.type != "call":
        return None
    fn = node.child_by_field_name("function")
    args = node.child_by_field_name("arguments")
    if fn is None or _text(fn) != "len" or args is None or args.named_child_count != 1:
        return None
    return _text(args.named_children[0])


def _scope(node: Node) -> Node:
    """Innermost enclosing function (or the module)."""
    while node.parent is not None and node.type != "function_definition":
        node = node.parent
    return node


def _indexes(node: Node, seq: str, index: str) -> bool:
    """Whether `seq[index]` appears in the function enclosing `node`."""
    stack = [_scope(node)]
    while stack:
        n = stack.pop()
        if n.type == "subscript":
            value, sub = n.child_by_field_name("value"), n.child_by_field_name("subscript")
            if value is not None and sub is not None and _text(value) == seq and _text(sub) == index:
                return True
        stack.extend(n.children)
    return False


def _finding(rule: str, path: str, node: Node, message: str) -> Finding:
    return Finding(
        source="heuristic",
        rule=rule,
        path=path,
        line=node.start_point[0] + 1,
        col=node.start_point[1] + 1,
        end_line=node.end_point[0] + 1,
        message=message,
    )


def _check(path: str, node: Node) -> Iterator[Finding]:
    if node.type == "call":
        fn = node.child_by_field_name("function")
        args = node.child_by_field_name("arguments")
        if fn is not None and _text(fn) == "range" and args is not None:
            for arg in args.named_children:
                op = arg.child_by_field_name("operator")
                # len(x) - 1 is left alone: pairwise loops (x[i], x[i + 1]) use it on purpose.
                if arg.type == "binary_operator" and op is not None and _text(op) == "+" \
                        and _len_arg(arg.child_by_field_name("left")) is not None:
                    yield _finding("H001", path, node,
                                   f"range() bound `{_text(arg)}` is adjusted from len(); check for off-by-one")

    elif node.type == "subscript":
        value = node.child_by_field_name("value")
        index = node.child_by_field_name("subscript")
        if value is not None and _len_arg(index) == _text(value):
            yield _finding("H002", path, node, f"`{_text(node)}` indexes one past the end; this always raises")

    elif node.type == "comparison_operator":
        children = node.children
        for i in range(1, len(children) - 1):
            op = children[i].type
            left, right = children[i - 1], children[i + 1]
            if op == "<=":
                bound, other = right, left
            elif op == ">=":
                bound, other = left, right
            else:
                continue
            seq = _len_arg(bound)
            # Only when the compared value is then used to index that same sequence:
            # `len(history) >= max_redirects` is a count check, not an index bound.
            if seq is not None and _len_arg(other) is None and _indexes(node, seq, _text(other)):
                yield _finding("H003", path, node,
                               f"`{_text(node)}` admits {_text(other)} == len({seq}), and {seq}[{_text(other)}]"
                               " is used nearby; an index equal to len() is out of range")


def scan_file(path: str, src: bytes) -> List[Finding]:
    found: List[Finding] = []
    stack = [parse_tree(src).root_node]
    while stack:
        node = stack.pop()
        found.extend(_check(path, node))
        stack.extend(node.children)
    return found


def run_heuristics(repo: Path, paths: Sequence[str]) -> List[Finding]:
    found: List[Finding] = []
    for rel in paths:
        try:
            src = (repo / rel).read_bytes()
        except OSError:
            continue
        found.extend(scan_file(rel, src))
    return found
