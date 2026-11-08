"""Extract symbols, call sites and imports from Python source using tree-sitter."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from typing import List, Optional

import tree_sitter_python as tspython
from tree_sitter import Language, Node, Parser, Tree

PY_LANGUAGE = Language(tspython.language())

MAX_SIGNATURE_CHARS = 240
MAX_DOCSTRING_CHARS = 120
MAX_CALL_EXPR_CHARS = 120


@dataclass
class Symbol:
    name: str
    qualname: str  # dotted path within the file, e.g. "Cache.get"
    kind: str  # "class" | "function" | "method"
    start_line: int  # 1-based; includes decorators
    end_line: int
    signature: str
    docstring: Optional[str]
    parent: Optional[int]  # index into ParsedFile.symbols


@dataclass
class Call:
    callee_name: str  # last component: "get" in self.cache.get(k)
    callee_expr: str  # full callee text: "self.cache.get"
    line: int
    caller: Optional[int]  # index into ParsedFile.symbols; None means module level


@dataclass
class Import:
    module: str  # "os.path", ".models", "..utils"
    name: Optional[str]  # None for `import x`; "*" for wildcard
    alias: Optional[str]
    line: int


@dataclass
class ParsedFile:
    symbols: List[Symbol] = field(default_factory=list)
    calls: List[Call] = field(default_factory=list)
    imports: List[Import] = field(default_factory=list)
    has_error: bool = False


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _squash(text: str) -> str:
    """Collapse a possibly multi-line header into one line: `f( a, b, )` -> `f(a, b)`."""
    text = " ".join(text.split())
    text = re.sub(r"([(\[]) ", r"\1", text)
    return re.sub(r",? ([)\]])", r"\1", text)


class _Extractor:
    def __init__(self, src: bytes):
        self.src = src
        self.out = ParsedFile()

    def text(self, node: Node) -> str:
        return self.src[node.start_byte : node.end_byte].decode("utf-8", "replace")

    def run(self, root: Node) -> ParsedFile:
        self.out.has_error = root.has_error
        # Iterative walk: deeply nested expressions can blow Python's recursion limit.
        # Each entry is (node, enclosing symbol index, decorated_definition wrapper).
        stack = [(root, None, None)]
        while stack:
            node, scope, wrapper = stack.pop()
            kind = node.type

            if kind == "decorated_definition":
                definition = node.child_by_field_name("definition")
                # Decorators run in the enclosing scope, so their calls belong to it.
                for child in reversed(node.children):
                    if child != definition:
                        stack.append((child, scope, None))
                if definition is not None:
                    stack.append((definition, scope, node))
                continue

            if kind in ("function_definition", "class_definition"):
                idx = self._add_symbol(node, scope, wrapper)
                for child in reversed(node.children):
                    stack.append((child, idx, None))
                continue

            if kind == "import_statement":
                self._add_import(node)
                continue
            if kind == "import_from_statement":
                self._add_from_import(node)
                continue

            if kind == "call":
                self._add_call(node, scope)

            for child in reversed(node.children):
                stack.append((child, scope, None))
        return self.out

    def _add_symbol(self, node: Node, scope: Optional[int], wrapper: Optional[Node]) -> int:
        name_node = node.child_by_field_name("name")
        name = self.text(name_node) if name_node is not None else "<anonymous>"
        parent = self.out.symbols[scope] if scope is not None else None

        if node.type == "class_definition":
            kind = "class"
        elif parent is not None and parent.kind == "class":
            kind = "method"
        else:
            kind = "function"

        body = node.child_by_field_name("body")
        header_end = body.start_byte if body is not None else node.end_byte
        signature = _squash(self._header(node, header_end)).rstrip(":").rstrip()

        self.out.symbols.append(
            Symbol(
                name=name,
                qualname=f"{parent.qualname}.{name}" if parent else name,
                kind=kind,
                start_line=(wrapper or node).start_point[0] + 1,
                end_line=node.end_point[0] + 1,
                signature=_clip(signature, MAX_SIGNATURE_CHARS),
                docstring=self._docstring(body),
                parent=scope,
            )
        )
        return len(self.out.symbols) - 1

    def _header(self, node: Node, end: int) -> str:
        """Source of node up to byte `end`, minus comments (they'd swallow the squashed line)."""
        comments = []
        stack = [c for c in node.children if c.start_byte < end]
        while stack:
            n = stack.pop()
            if n.type == "comment":
                comments.append((n.start_byte, n.end_byte))
            else:
                stack.extend(n.children)
        pieces, pos = [], node.start_byte
        for start, stop in sorted(comments):
            pieces.append(self.src[pos:start])
            pos = stop
        pieces.append(self.src[pos:end])
        return b"".join(pieces).decode("utf-8", "replace")

    def _docstring(self, body: Optional[Node]) -> Optional[str]:
        if body is None or body.named_child_count == 0:
            return None
        first = body.named_children[0]
        if first.type != "expression_statement" or first.named_child_count != 1:
            return None
        literal = first.named_children[0]
        if literal.type != "string":
            return None
        try:
            value = ast.literal_eval(self.text(literal))
        except (ValueError, SyntaxError):
            return None
        if not isinstance(value, str):
            return None
        for line in value.splitlines():
            if line.strip():
                return _clip(line.strip(), MAX_DOCSTRING_CHARS)
        return None

    def _add_call(self, node: Node, scope: Optional[int]) -> None:
        fn = node.child_by_field_name("function")
        if fn is None:
            return
        if fn.type == "identifier":
            name = self.text(fn)
        elif fn.type == "attribute":
            attr = fn.child_by_field_name("attribute")
            if attr is None:
                return
            name = self.text(attr)
        else:
            # e.g. handlers[key](x) or make()(); nothing nameable to index.
            return
        self.out.calls.append(
            Call(
                callee_name=name,
                callee_expr=_clip(_squash(self.text(fn)), MAX_CALL_EXPR_CHARS),
                line=node.start_point[0] + 1,
                caller=scope,
            )
        )

    def _split_alias(self, node: Node):
        """Return (name, alias) for a dotted_name or aliased_import node."""
        if node.type == "aliased_import":
            name = node.child_by_field_name("name")
            alias = node.child_by_field_name("alias")
            return (
                self.text(name) if name is not None else self.text(node),
                self.text(alias) if alias is not None else None,
            )
        return self.text(node), None

    def _add_import(self, node: Node) -> None:
        line = node.start_point[0] + 1
        for child in node.named_children:
            if child.type in ("dotted_name", "aliased_import"):
                module, alias = self._split_alias(child)
                self.out.imports.append(Import(module=module, name=None, alias=alias, line=line))

    def _add_from_import(self, node: Node) -> None:
        line = node.start_point[0] + 1
        module_node = node.child_by_field_name("module_name")
        module = self.text(module_node) if module_node is not None else ""
        seen_import_kw = False
        for child in node.children:
            if child.type == "import":
                seen_import_kw = True
                continue
            if not seen_import_kw:
                continue
            if child.type == "wildcard_import":
                self.out.imports.append(Import(module=module, name="*", alias=None, line=line))
            elif child.type in ("dotted_name", "aliased_import"):
                name, alias = self._split_alias(child)
                self.out.imports.append(Import(module=module, name=name, alias=alias, line=line))


_parser = Parser(PY_LANGUAGE)


def parse_tree(src: bytes) -> Tree:
    return _parser.parse(src)


def parse_python(src: bytes) -> ParsedFile:
    return _Extractor(src).run(_parser.parse(src).root_node)
