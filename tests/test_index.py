import textwrap
from pathlib import Path

import pytest

from bughunter.index import RepoIndex, parse_python


def write(root: Path, rel: str, src: str) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(src).lstrip())


@pytest.fixture
def repo(tmp_path):
    write(tmp_path, "shop/cache.py", '''
        class Cache:
            """In-memory cache.

            Longer description.
            """

            def __init__(self):
                self.data = {}

            def get(self, key, default=None):
                return self.data.get(key, default)

            @property
            def size(self) -> int:
                return len(self.data)
        ''')
    write(tmp_path, "shop/orders.py", '''
        from .cache import Cache as C
        import os.path

        CACHE = C()

        def total(order, tax_rate: float = 0.2) -> float:
            def line(item):
                return item.price * item.qty
            subtotal = sum(line(i) for i in order.items)
            return subtotal * (1 + tax_rate)

        async def fetch(order_id):
            return CACHE.get(order_id)
        ''')
    write(tmp_path, "shop/cli.py", '''
        from shop.orders import total, fetch

        def main():
            print(total(fetch(1)))
        ''')
    write(tmp_path, ".venv/lib/ignored.py", "def ignored(): pass\n")
    return tmp_path


def test_parser_extracts_symbols_calls_imports():
    parsed = parse_python(textwrap.dedent('''
        import os.path as p, sys
        from ..pkg import a as b, c
        from x import *

        @app.route("/")
        async def handler(req, *, retries: int = 3) -> "Resp":
            """Handle it."""
            return helper(req).send()

        class A(Base, metaclass=M):
            def m(self):
                class Inner:
                    pass
        ''').encode())

    names = {s.qualname: s for s in parsed.symbols}
    assert set(names) == {"handler", "A", "A.m", "A.m.Inner"}
    assert names["handler"].signature == 'async def handler(req, *, retries: int = 3) -> "Resp"'
    assert names["handler"].start_line == 6  # decorator line (source starts with a blank line)
    assert names["handler"].docstring == "Handle it."
    assert names["A"].signature == "class A(Base, metaclass=M)"
    assert names["A.m"].kind == "method"

    calls = {(c.callee_name, c.callee_expr, c.caller) for c in parsed.calls}
    handler_idx = parsed.symbols.index(names["handler"])
    assert ("route", "app.route", None) in calls  # decorator runs in the enclosing scope
    assert ("helper", "helper", handler_idx) in calls
    assert ("send", "helper(req).send", handler_idx) in calls

    imports = {(i.module, i.name, i.alias) for i in parsed.imports}
    assert imports == {
        ("os.path", None, "p"), ("sys", None, None),
        ("..pkg", "a", "b"), ("..pkg", "c", None), ("x", "*", None),
    }


def test_parser_flags_syntax_errors_but_keeps_going():
    parsed = parse_python(b"def ok():\n    pass\n\ndef broken(:\n    pass\n")
    assert parsed.has_error
    assert "ok" in {s.name for s in parsed.symbols}


def test_index_queries(repo):
    with RepoIndex(repo) as idx:
        stats = idx.update()
        assert (stats.added, stats.changed) == (3, 0)

        [get] = idx.find_symbol("Cache.get")
        assert get["path"] == "shop/cache.py"
        assert get["signature"] == "def get(self, key, default=None)"
        assert idx.find_symbol("ignored") == []

        callers = {(c["path"], c["caller"]) for c in idx.find_callers("total")}
        assert callers == {("shop/cli.py", "main")}
        # Name-based matching: dict.get inside Cache.get is reported too.
        exprs = {c["callee_expr"] for c in idx.find_callers("Cache.get")}
        assert exprs == {"CACHE.get", "self.data.get"}

        assert idx.symbol_at("shop/orders.py", 8)["qualname"] == "total.line"
        assert idx.symbol_at("shop/orders.py", 9)["qualname"] == "total"
        assert idx.symbol_at("shop/orders.py", 1) is None

        src = idx.symbol_source("shop/cache.py", "Cache.size")
        assert src.splitlines()[0].split() == ["13", "@property"]


def test_incremental_update(repo):
    with RepoIndex(repo) as idx:
        idx.update()
        assert idx.update().unchanged == 3

        write(repo, "shop/cli.py", "def main():\n    pass\n")
        (repo / "shop/cache.py").unlink()
        stats = idx.update()
        assert (stats.changed, stats.removed, stats.unchanged) == (1, 1, 1)
        assert idx.find_symbol("Cache") == []
        assert idx.find_callers("total") == []  # rows from the old cli.py are gone


def test_stale_source_is_refused(repo):
    with RepoIndex(repo) as idx:
        idx.update()
        write(repo, "shop/cache.py", "# shifted\n" + (repo / "shop/cache.py").read_text())
        with pytest.raises(RuntimeError):
            idx.symbol_source("shop/cache.py", "Cache.get")


def test_repo_map_ranks_and_fits_budget(repo):
    with RepoIndex(repo) as idx:
        idx.update()
        full = idx.repo_map(max_tokens=10_000)
        assert "shop/cache.py:\n  class Cache\n    def __init__(self)\n    def get(self, key, default=None)" in full
        assert "def line(item)" not in full  # nested helpers are not API

        tiny = idx.repo_map(max_tokens=30)
        assert 0 < len(tiny) <= 30 * 4
        # Cache is used from orders.py, orders from cli.py: the ranked core wins.
        assert "Cache" in tiny or "total" in tiny

        assert (repo / ".bughunter/.gitignore").read_text() == "*\n"


def test_resolver_follows_imports_and_ignores_builtins(tmp_path):
    from bughunter.index.resolve import Resolver, module_names

    write(tmp_path, "__init__.py", "")
    write(tmp_path, "util.py", "def helper():\n    pass\n")
    write(tmp_path, "matcher.py", '''
        class Matcher:
            def str(self):
                return "m"

            def frobnicate(self):
                pass
        ''')
    write(tmp_path, "sub/__init__.py", "")
    write(tmp_path, "sub/user.py", '''
        from .. import util
        from ..util import helper as h

        def run(m):
            str(m)        # builtin, not Matcher.str
            h()
            util.helper()
            m.frobnicate()  # unresolved method: low confidence
            m.get("k")      # dict-like, ignored
        ''')
    write(tmp_path, "vendor/lib.py", "def helper():\n    pass\n")

    root = tmp_path.name
    names = module_names(["__init__.py", "util.py", "sub/__init__.py", "sub/user.py"], root)
    assert names["sub/user.py"] == ["sub.user", f"{root}.sub.user"]
    assert names["__init__.py"] == ["", root]

    with RepoIndex(tmp_path) as idx:
        idx.update()
        assert [r["path"] for r in idx.find_symbol("helper")] == ["util.py"]  # vendor/ skipped
        refs = {}
        for ref in Resolver(idx.conn).references():
            sym = idx.conn.execute("SELECT qualname FROM symbols WHERE id = ?", (ref.symbol_id,)).fetchone()
            refs[sym["qualname"]] = refs.get(sym["qualname"], 0) + ref.count * ref.confidence
        assert "Matcher.str" not in refs
        assert refs["helper"] == 3  # the import itself, h(), util.helper()
        assert 0 < refs["Matcher.frobnicate"] < 1


def test_extractor_version_change_forces_reindex(repo, monkeypatch):
    from bughunter.index import indexer

    with RepoIndex(repo) as idx:
        idx.update()
        monkeypatch.setattr(indexer, "EXTRACTOR_VERSION", "test-bump")
        assert idx.update().added == 3
