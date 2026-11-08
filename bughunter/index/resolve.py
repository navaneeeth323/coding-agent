"""Best-effort resolution of call sites and imports to definitions in the repo.

No type inference: a reference is resolved through the importing file's imports
when possible, and otherwise matched by name with a weight that reflects how
ambiguous the match is. Calls into external packages and builtins are dropped,
which is what keeps `str(x)` from looking like a call to some `Matcher.str`.
"""

from __future__ import annotations

import io
import logging
import pathlib
import re
import sqlite3
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Set

# Method names of builtin and everyday stdlib types: `obj.get(...)` is far more
# likely dict.get than ours, `m.group()` far more likely re.Match.group.
_COMMON_TYPES = (
    object, str, bytes, list, dict, set, frozenset, tuple, int, float,
    io.IOBase, io.TextIOBase, logging.Logger, pathlib.Path, re.Match, re.Pattern, subprocess.Popen,
)
BUILTIN_METHODS: Set[str] = {name for t in _COMMON_TYPES for name in dir(t)}

# Confidence of an unresolved `obj.method()` match relative to an import-resolved one.
UNRESOLVED_ATTR_CONFIDENCE = 0.2


@dataclass(frozen=True)
class Reference:
    src_file: int
    symbol_id: int
    count: int  # occurrences in src_file
    confidence: float  # 1.0 when resolved through an import; lower for name-only guesses


def module_names(paths: List[str], root_name: Optional[str] = None) -> Dict[str, List[str]]:
    """Importable module names for each file path; the first is the path-derived one.

    "src/pkg/a.py" is importable as "pkg.a" when src/ has no __init__.py, so each
    file also gets its package-derived name. If the repo root is itself a package,
    its directory name (`root_name`) prefixes the package-derived names.
    """
    package_dirs = {p.rsplit("/", 1)[0] for p in paths if p.endswith("/__init__.py")}
    root_is_package = "__init__.py" in paths
    names: Dict[str, List[str]] = {}
    for path in paths:
        parts = path[: -len(".py")].split("/")
        if parts[-1] == "__init__":
            parts = parts[:-1]
        # Climb while the containing directory is a package.
        dirs = path.split("/")[:-1]
        top = len(dirs)
        while top > 0 and "/".join(dirs[:top]) in package_dirs:
            top -= 1
        package_parts = parts[top:]
        if top == 0 and root_is_package and root_name:
            package_parts = [root_name] + package_parts
        path_name, package_name = ".".join(parts), ".".join(package_parts)
        names[path] = [path_name] + ([package_name] if package_name not in ("", path_name) else [])
    return names


class Resolver:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
        self.paths = {r["id"]: r["path"] for r in conn.execute("SELECT id, path FROM files")}
        self.module_of: Dict[int, str] = {}  # canonical (path-derived) module name
        self.file_by_module: Dict[str, int] = {}
        root = conn.execute("SELECT value FROM meta WHERE key = 'repo_root'").fetchone()
        root_name = root["value"].rstrip("/").rsplit("/", 1)[-1] if root else None
        names_by_path = module_names(list(self.paths.values()), root_name)
        for fid, path in self.paths.items():
            names = names_by_path[path]
            self.module_of[fid] = names[0]
            for name in names:
                self.file_by_module.setdefault(name, fid)

        self.top_level: Dict[int, Dict[str, int]] = defaultdict(dict)  # file -> name -> symbol
        self.top_level_by_name: Dict[str, List[int]] = defaultdict(list)
        self.methods_by_name: Dict[str, List[int]] = defaultdict(list)
        self.symbol_file: Dict[int, int] = {}
        for r in conn.execute("SELECT id, file_id, parent_id, name, kind FROM symbols"):
            self.symbol_file[r["id"]] = r["file_id"]
            if r["parent_id"] is None:
                self.top_level[r["file_id"]].setdefault(r["name"], r["id"])
                self.top_level_by_name[r["name"]].append(r["id"])
            elif r["kind"] == "method":
                self.methods_by_name[r["name"]].append(r["id"])

    def resolve_module(self, module: str, importer: int) -> Optional[int]:
        if module.startswith("."):
            level = len(module) - len(module.lstrip("."))
            base = self.module_of[importer].split(".")
            if not self.paths[importer].endswith("__init__.py"):
                base = base[:-1]
            base = base[: len(base) - (level - 1)] if level > 1 else base
            rest = module.lstrip(".")
            module = ".".join(base + ([rest] if rest else []))
        return self.file_by_module.get(module)

    def references(self) -> Iterator[Reference]:
        imports = defaultdict(list)
        for r in self.conn.execute("SELECT file_id, module, name, alias FROM imports"):
            imports[r["file_id"]].append(r)

        # Per file: local name -> (file it comes from, original name or None for a module).
        bindings: Dict[int, Dict[str, tuple]] = defaultdict(dict)
        star_sources: Dict[int, List[int]] = defaultdict(list)
        for fid, rows in imports.items():
            for r in rows:
                if r["name"] is None:  # import a.b.c [as x]
                    target = self.resolve_module(r["module"], fid)
                    if target is not None:
                        # Key by the receiver text calls will use: `x.f()` or `a.b.f()`.
                        bindings[fid][r["alias"] or r["module"]] = (target, None)
                    continue
                source = self.resolve_module(r["module"], fid)
                if r["name"] == "*":
                    if source is not None:
                        star_sources[fid].append(source)
                    continue
                local = r["alias"] or r["name"]
                # `from pkg import mod` imports a submodule if one exists.
                sep = "" if r["module"].endswith(".") else "."
                submodule = self.resolve_module(r["module"] + sep + r["name"], fid)
                if submodule is not None:
                    bindings[fid][local] = (submodule, None)
                elif source is not None:
                    bindings[fid][local] = (source, r["name"])
                    # The import itself is a use (type hints, subclassing, re-exports).
                    yield from self._resolved(fid, source, r["name"], 1)

        for r in self.conn.execute(
            "SELECT file_id, callee_name, callee_expr, COUNT(*) AS n FROM calls GROUP BY 1, 2, 3"
        ):
            fid, name, expr, n = r["file_id"], r["callee_name"], r["callee_expr"], r["n"]
            if expr == name:
                yield from self._bare_call(fid, name, n, bindings[fid], star_sources[fid])
                continue
            receiver = expr[: -len(name) - 1]
            if receiver in ("self", "cls", "super()"):
                continue  # same class or its bases; not worth guessing
            bound = bindings[fid].get(receiver)
            if bound is not None and bound[1] is None:  # module.func()
                yield from self._resolved(fid, bound[0], name, n)
                continue
            if name in BUILTIN_METHODS:
                continue
            targets = self.methods_by_name.get(name, [])
            for sym in targets:
                if self.symbol_file[sym] != fid:
                    yield Reference(fid, sym, n, UNRESOLVED_ATTR_CONFIDENCE / len(targets))

    def _bare_call(self, fid, name, n, bindings, star_sources) -> Iterator[Reference]:
        if name in self.top_level[fid]:
            return  # defined in this file
        bound = bindings.get(name)
        if bound is not None:
            if bound[1] is not None:
                yield from self._resolved(fid, bound[0], bound[1], n)
            return
        for source in star_sources:
            if name in self.top_level[source]:
                yield Reference(fid, self.top_level[source][name], n, 1.0)
                return
        # Otherwise a builtin, or a name imported from outside the repo.

    def _resolved(self, fid, module_file, name, n) -> Iterator[Reference]:
        sym = self.top_level[module_file].get(name)
        if sym is not None:
            if module_file != fid:
                yield Reference(fid, sym, n, 1.0)
            return
        # Re-exported through a package __init__: fall back to any top-level def by name.
        targets = [s for s in self.top_level_by_name.get(name, []) if self.symbol_file[s] != fid]
        for sym in targets:
            yield Reference(fid, sym, n, 1.0 / len(targets))
