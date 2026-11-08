import subprocess
import textwrap
from pathlib import Path

import pytest

from bughunter.candidates import store
from bughunter.candidates.git import file_history
from bughunter.candidates.heuristics import scan_file
from bughunter.candidates.scoring import Hotspots, rule_prior, score
from bughunter.candidates.git import FileHistory
from bughunter.index import RepoIndex
from bughunter.pipeline import collect_candidates


def write(root: Path, rel: str, src: str) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(src).lstrip())


def git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", *args],
        cwd=repo, check=True, capture_output=True,
    )


def rules(src: str):
    return [(f.rule, f.line) for f in scan_file("x.py", textwrap.dedent(src).lstrip().encode())]


def test_heuristics():
    assert rules("""
        def a(xs):
            return xs[len(xs)]
        """) == [("H002", 2)]
    assert rules("""
        def b(xs):
            for i in range(len(xs) + 1):
                pass
            for i in range(len(xs) - 1):  # pairwise: fine
                pass
        """) == [("H001", 2)]
    assert rules("""
        def c(xs, i):
            if i <= len(xs):
                return xs[i]
        def d(resp, limit):
            if len(resp.history) >= limit:  # count check, no indexing
                raise ValueError
        """) == [("H003", 2)]


def test_scoring():
    assert rule_prior("F821") == 0.9
    assert rule_prior("B999") == rule_prior("B")
    hot = Hotspots({"hot.py": FileHistory(20, 10, 0), "cold.py": FileHistory(1, 0, 0)})
    assert hot.multiplier("cold.py") == 1.0
    assert 1.0 < hot.multiplier("hot.py") <= 1.5
    assert score("F821", "tests/test_x.py", hot) < score("F821", "pkg/x.py", hot)


@pytest.fixture
def repo(tmp_path):
    write(tmp_path, "pkg/core.py", """
        def load(path, cache={}):
            try:
                return cache[path]
            except:
                return missing_name
        """)
    write(tmp_path, "pkg/other.py", "def fine():\n    return 1\n")
    git(tmp_path, "init", "-q")
    git(tmp_path, "add", ".")
    git(tmp_path, "commit", "-qm", "initial")
    write(tmp_path, "pkg/core.py", (tmp_path / "pkg/core.py").read_text() + "\n")
    git(tmp_path, "commit", "-qam", "fix crash in load")
    return tmp_path


def test_git_history(repo):
    h = file_history(repo)
    assert (h["pkg/core.py"].commits, h["pkg/core.py"].fix_commits) == (2, 1)
    assert (h["pkg/other.py"].commits, h["pkg/other.py"].fix_commits) == (1, 0)


def test_collect_end_to_end(repo):
    with RepoIndex(repo) as idx:
        stats = collect_candidates(idx)
        assert not stats.errors
        queue = store.queue(idx.conn)
        found = {(c["rule"], c["symbol"]) for c in queue}
        assert {("B006", "load"), ("E722", "load"), ("F821", "load")} <= found
        assert queue[0]["rule"] == "F821"  # undefined name ranks first

        # Shifting lines keeps identities; nothing is duplicated.
        core = repo / "pkg/core.py"
        core.write_text("\n\n" + core.read_text())
        stats = collect_candidates(idx)
        assert (stats.added, stats.updated) == (0, len(queue))
        f821 = next(c for c in store.queue(idx.conn) if c["rule"] == "F821")
        assert f821["line"] == queue[0]["line"] + 2

        # State machine: legal moves pass, illegal ones raise, history is logged.
        store.transition(idx.conn, f821["id"], "rejected", note="decoy", data={"triage": {"is_real_bug": False}})
        with pytest.raises(store.InvalidTransition):
            store.transition(idx.conn, f821["id"], "verified")
        assert f821["id"] not in {c["id"] for c in store.queue(idx.conn)}

        # Re-running on unchanged code keeps the rejection...
        collect_candidates(idx)
        assert store.funnel(idx.conn)["rejected"] == 1
        # ...but editing the function reopens it for another look.
        write(repo, "pkg/core.py", (repo / "pkg/core.py").read_text().replace("return cache[path]", "return cache.get(path)"))
        assert collect_candidates(idx).reopened == 1
        events = [r["to_state"] for r in idx.conn.execute(
            "SELECT to_state FROM candidate_events WHERE candidate_id = ? ORDER BY id", (f821["id"],))]
        assert events == ["new", "rejected", "new"]

        # Fixing the code makes the finding disappear from the active queue.
        write(repo, "pkg/core.py", "def load(path):\n    return path\n")
        collect_candidates(idx)
        assert store.queue(idx.conn) == []
        assert sum(store.funnel(idx.conn).values()) == 0
