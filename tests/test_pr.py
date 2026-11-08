import json
import subprocess

import pytest

from bughunter.candidates import store
from bughunter.index import RepoIndex
from bughunter.pr import PRError, open_pr, parse_github_remote
from bughunter.sandbox import sha1_text


def git(cwd, *args):
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout


class FakeGitHub:
    token = "t"

    def __init__(self, open_items=()):
        self.open_items = list(open_items)
        self.calls = []

    def request(self, method, path, body=None):
        self.calls.append((method, path, body))
        if method == "GET":
            return {"default_branch": "main"}
        return {"html_url": "https://github.com/me/proj/pull/7"}

    def search_open(self, owner, repo, terms):
        self.calls.append(("SEARCH", terms))
        return self.open_items


ORIGINAL = "def total(items):\n    return totl\n"
FIXED = "def total(items):\n    return sum(items)\n"
TEST = "from pkg.core import total\n\ndef test_total():\n    assert total([1]) == 1\n"


@pytest.fixture
def repo(tmp_path):
    bare = tmp_path / "origin.git"
    git(tmp_path, "init", "-q", "--bare", "-b", "main", str(bare))
    root = tmp_path / "repo"
    (root / "pkg").mkdir(parents=True)
    (root / "pkg/core.py").write_text(ORIGINAL)
    (root / "tests").mkdir()
    git(root, "init", "-q", "-b", "main")
    git(root, "config", "user.email", "t@example.com")
    git(root, "config", "user.name", "t")
    git(root, "remote", "add", "origin", "https://github.com/me/proj.git")
    # Pushes to the GitHub URL land in the local bare repo.
    git(root, "config", f"url.{bare}.insteadOf", "https://github.com/me/proj.git")
    git(root, "add", ".")
    git(root, "commit", "-q", "-m", "init")
    return root


def _verified(idx):
    store.ensure_schema(idx.conn)
    run = store.start_run(idx.conn, "test")
    with idx.conn:
        store.upsert(idx.conn, run, {
            "fingerprint": "f", "source": "ruff", "rule": "F821", "path": "pkg/core.py", "line": 2, "col": 12,
            "end_line": 2, "message": "Undefined name `totl`", "symbol": "total", "symbol_start": 1,
            "symbol_end": 2, "code_sha1": "x", "score": 0.9,
        })
    cid = idx.conn.execute("SELECT id FROM candidates").fetchone()[0]
    with idx.conn:
        idx.conn.execute("UPDATE candidates SET state = 'verified', data = ? WHERE id = ?", (json.dumps({
            "triage": {"p_real": 0.9, "category": "undefined_name", "model": "m"},
            "investigation": {
                "status": "proposed", "diagnosis": "typo in return", "title": "return the sum, not `totl`.",
                "files": {"pkg/core.py": {"before_sha1": sha1_text(ORIGINAL), "after": FIXED}},
                "test": {"path": "tests/test_bughunter_1.py", "content": TEST,
                         "failure": "    def test_total():\n>       assert total([1]) == 1\nE   NameError: name 'totl'"},
                "diff": "--- a/pkg/core.py\n+++ b/pkg/core.py\n-    return totl\n+    return sum(items)\n"
                        "--- /dev/null\n+++ b/tests/test_bughunter_1.py\n+from pkg.core import total\n"},
            "verification": {"passed": True, "checks": [
                {"name": "scope", "ok": True, "detail": ""}, {"name": "suite", "ok": True, "detail": "3 passed"}]},
        }), cid))
    return cid


def test_parse_remote():
    assert parse_github_remote("git@github.com:me/proj.git") == ("me", "proj")
    assert parse_github_remote("https://github.com/me/proj") == ("me", "proj")
    assert parse_github_remote("https://x-access-token:abc@github.com/me/p.git") == ("me", "p")
    assert parse_github_remote("/srv/git/proj.git") is None


def test_dry_run_then_push(repo):
    with RepoIndex(repo) as idx:
        cid = _verified(idx)
        gh = FakeGitHub()
        plan = open_pr(idx, cid, github=gh)
        assert plan.url is None and plan.branch == f"bughunter/{cid}-total"
        assert plan.title == "Return the sum, not `totl`"
        assert "```python\nfrom pkg.core import total" in plan.body and "confidence that this is a real bug: 0.9" in plan.body
        assert "fails with:\n\n```\nE   NameError: name 'totl'\n```" in plan.body
        fix = plan.body.split("## Fix")[1]
        assert "+    return sum(items)" in fix and "test_bughunter_1" not in fix
        assert "✅ Existing test suite has no new failures (3 passed)" in plan.body and "scope" not in plan.body
        assert not git(repo, "ls-remote", "--heads", "origin").strip()  # nothing pushed

        plan = open_pr(idx, cid, push=True, github=gh)
        assert plan.url == "https://github.com/me/proj/pull/7"
        post = [c for c in gh.calls if c[0] == "POST"][0]
        assert post[1] == "/repos/me/proj/pulls" and post[2]["draft"] and post[2]["base"] == "main"
        assert plan.branch in git(repo, "ls-remote", "--heads", "origin")
        shown = git(repo, "show", f"{plan.branch}:pkg/core.py")
        assert shown == FIXED and git(repo, "show", f"{plan.branch}:tests/test_bughunter_1.py") == TEST
        # The user's checkout is untouched.
        assert (repo / "pkg/core.py").read_text() == ORIGINAL and git(repo, "branch", "--show-current") == "main\n"
        assert idx.conn.execute("SELECT state FROM candidates WHERE id = ?", (cid,)).fetchone()[0] == "pr_opened"


def test_duplicates_block_push(repo):
    with RepoIndex(repo) as idx:
        cid = _verified(idx)
        gh = FakeGitHub(open_items=["https://github.com/me/proj/issues/3"])
        plan = open_pr(idx, cid, push=True, github=gh)
        assert plan.url is None and plan.duplicates == ["https://github.com/me/proj/issues/3"]
        assert ("SEARCH", ["total", "core.py"]) in gh.calls


def test_refuses_unverified_or_moved_head(repo):
    with RepoIndex(repo) as idx:
        cid = _verified(idx)
        (repo / "pkg/core.py").write_text("def total(items):\n    return 0\n")
        git(repo, "commit", "-qam", "change")
        with pytest.raises(PRError, match="differs from the code that was verified"):
            open_pr(idx, cid, github=FakeGitHub())
        with idx.conn:
            idx.conn.execute("UPDATE candidates SET state = 'triaged'")
        with pytest.raises(PRError, match="only verified"):
            open_pr(idx, cid, github=FakeGitHub())
