import json
import textwrap

import pytest

from bughunter.candidates import store
from bughunter.index import RepoIndex
from bughunter.investigate import InvestigateConfig, investigate
from bughunter.investigate.edits import changed_lines, module_name, replace_region, strip_fences
from bughunter.llm import ChatResult
from bughunter.pipeline import collect_candidates
from bughunter.sandbox import LocalSandbox
from bughunter.verify import verify


class ScriptedLLM:
    """Answers repro and fix requests from queues; records the prompts it saw."""

    model = "scripted"

    def __init__(self, repro, fix):
        self.repro, self.fix = list(repro), list(fix)
        self.prompts = []

    def chat_json(self, messages, schema, *, temperature=0.0, seed=None, max_tokens=None):
        self.prompts.append(messages)
        props = schema["properties"]
        if "test_code" in props:
            data = self.repro.pop(0)
        else:
            data = self.fix.pop(0)
        return ChatResult(data=data, text=json.dumps(data), model=self.model, prompt_tokens=100, completion_tokens=50)


REPRO_OK = {"plan": "sum two numbers", "give_up": False, "test_code": textwrap.dedent("""
    from pkg.core import total

    def test_total():
        assert total([1, 2]) == 3
    """)}
REPRO_PASSES = {"plan": "p", "give_up": False,
                "test_code": "from pkg.core import total\n\ndef test_nothing():\n    assert True\n"}
FIX_OK = {"diagnosis": "returns a misspelled name", "give_up": False, "new_code": textwrap.dedent("""
    def total(items):
        s = 0
        for x in items:
            s += x
        return s
    """)}
# Passes the repro test but clobbers a neighbour that the existing suite covers.
FIX_BREAKS_SUITE = {"diagnosis": "d", "give_up": False,
                    "new_code": "def total(items):\n    return sum(items)\n\n\ndouble = None\n"}


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "repo"
    (root / "pkg").mkdir(parents=True)
    (root / "pkg/__init__.py").write_text("")
    (root / "pkg/core.py").write_text(textwrap.dedent("""
        def double(x):
            return x * 2


        def total(items):
            s = 0
            for x in items:
                s += x
            return totl
        """).lstrip())
    (root / "tests").mkdir()
    (root / "tests/test_core.py").write_text(textwrap.dedent("""
        from pkg.core import double


        def test_double():
            assert double(2) == 4
        """).lstrip())
    return root


def _triaged(idx):
    collect_candidates(idx)
    [c] = [c for c in store.queue(idx.conn) if c["rule"] == "F821"]
    store.transition(idx.conn, c["id"], "triaged", data={"triage": {
        "p_real": 0.9, "intent": "sum items", "failing_input": "total([1, 2])",
        "expected": "3", "actual": "NameError"}})
    return c["id"]


@pytest.fixture
def sandbox():
    return LocalSandbox(allow_unsafe=True)


def test_edit_helpers():
    text = "class A:\n    def f(self):\n        return 1\n\nx = 1\n"
    out = replace_region(text, 2, 3, "```python\ndef f(self):\n    return 2\n```")
    assert out == "class A:\n    def f(self):\n        return 2\n\nx = 1\n"
    assert changed_lines(text, out) == [(3, 3)]
    assert module_name("src/pkg/mod.py") == "pkg.mod" and module_name("pkg/__init__.py") == "pkg"
    assert strip_fences("plain") == "plain"


def test_investigate_then_verify(repo, sandbox):
    with RepoIndex(repo) as idx:
        cid = _triaged(idx)
        llm = ScriptedLLM(repro=[REPRO_PASSES, REPRO_OK], fix=[FIX_BREAKS_SUITE, FIX_OK])
        stats = investigate(idx, llm, sandbox, InvestigateConfig(use_cache=False))
        assert stats.proposed == 1, stats.errors

        row = idx.conn.execute("SELECT state, data FROM candidates WHERE id = ?", (cid,)).fetchone()
        inv = json.loads(row["data"])["investigation"]
        assert row["state"] == "investigating" and inv["status"] == "proposed"
        assert [a["result"] for a in inv["attempts"]] == ["rejected", "fails_on_original", "rejected", "passes"]
        assert "PASSED on the current code" in inv["attempts"][0]["problem"]
        assert "breaks other tests" in inv["attempts"][2]["problem"]
        assert inv["test"]["path"] == f"tests/test_bughunter_{cid}.py"
        assert "+    return s" in inv["diff"] and "b/tests/test_bughunter_" in inv["diff"]
        # Feedback from the failed attempt reached the model.
        assert "PASSED on the current code" in llm.prompts[1][-1]["content"]
        assert "return totl" in (repo / "pkg/core.py").read_text()  # original untouched

        vstats = verify(idx, sandbox)
        assert vstats.verified == 1, idx.conn.execute("SELECT note FROM candidate_events").fetchall()
        row = idx.conn.execute("SELECT state, data FROM candidates WHERE id = ?", (cid,)).fetchone()
        assert row["state"] == "verified"
        checks = {c["name"]: c["ok"] for c in json.loads(row["data"])["verification"]["checks"]}
        assert checks == {**checks, "fails_before": True, "passes_after": True, "suite": True, "ruff": True}


def test_give_up_rejects(repo, sandbox):
    with RepoIndex(repo) as idx:
        cid = _triaged(idx)
        llm = ScriptedLLM(repro=[{"plan": "code is fine", "give_up": True, "test_code": ""}], fix=[])
        stats = investigate(idx, llm, sandbox, InvestigateConfig(use_cache=False))
        assert stats.rejected == 1
        row = idx.conn.execute("SELECT state FROM candidates WHERE id = ?", (cid,)).fetchone()
        assert row["state"] == "rejected"


def test_budget_rejects(repo, sandbox):
    with RepoIndex(repo) as idx:
        _triaged(idx)
        llm = ScriptedLLM(repro=[REPRO_PASSES] * 5, fix=[])
        stats = investigate(idx, llm, sandbox, InvestigateConfig(use_cache=False, max_calls=2))
        assert stats.rejected == 1 and len(llm.prompts) == 2


def test_verify_rejects_bogus_and_stale(repo, sandbox):
    with RepoIndex(repo) as idx:
        cid = _triaged(idx)
        investigate(idx, ScriptedLLM(repro=[REPRO_OK], fix=[FIX_OK]), sandbox,
                    InvestigateConfig(use_cache=False))
        # Tamper: a "fix" that doesn't fix anything must not pass the gate.
        data = json.loads(idx.conn.execute("SELECT data FROM candidates WHERE id = ?", (cid,)).fetchone()[0])
        good = data["investigation"]["files"]["pkg/core.py"]["after"]
        data["investigation"]["files"]["pkg/core.py"]["after"] = good.replace("return s", "return 0")
        with idx.conn:
            idx.conn.execute("UPDATE candidates SET data = ? WHERE id = ?", (json.dumps(data), cid))
        stats = verify(idx, sandbox)
        assert stats.rejected == 1
        note = idx.conn.execute(
            "SELECT note FROM candidate_events WHERE candidate_id = ? ORDER BY id DESC", (cid,)).fetchone()[0]
        assert note.startswith("verification: passes_after")

        # Stale: the file changed after investigation.
        with idx.conn:
            idx.conn.execute("UPDATE candidates SET state = 'investigating' WHERE id = ?", (cid,))
        (repo / "pkg/core.py").write_text((repo / "pkg/core.py").read_text() + "\n# edited\n")
        stats = verify(idx, sandbox)
        assert stats.stale == 1


def test_wrong_test_gets_rewritten(repo, sandbox):
    # The first test expects the wrong total, so no correct fix can pass it; after
    # `fix_attempts` failures investigation goes back and asks for a new test.
    wrong = {**REPRO_OK, "test_code": REPRO_OK["test_code"].replace("== 3", "== 4")}
    with RepoIndex(repo) as idx:
        cid = _triaged(idx)
        llm = ScriptedLLM(repro=[wrong, REPRO_OK], fix=[FIX_OK] * 4)
        stats = investigate(idx, llm, sandbox, InvestigateConfig(use_cache=False))
        assert stats.proposed == 1
        inv = json.loads(idx.conn.execute("SELECT data FROM candidates WHERE id = ?", (cid,)).fetchone()[0])
        phases = [(a["phase"], a["result"]) for a in inv["investigation"]["attempts"]]
        assert phases == [("repro", "fails_on_original")] + [("fix", "rejected")] * 3 + [
            ("retest", "test suspected wrong"), ("repro", "fails_on_original"), ("fix", "passes")]
        retest_prompt = llm.prompts[4][1]["content"]
        assert "expected values are probably wrong" in retest_prompt and "== 4" in retest_prompt
        assert "== 3" in inv["investigation"]["test"]["content"]


def test_one_investigation_per_function(repo, sandbox):
    with RepoIndex(repo) as idx:
        cid = _triaged(idx)
        run = store.start_run(idx.conn, "test")
        lead = idx.conn.execute("SELECT * FROM candidates WHERE id = ?", (cid,)).fetchone()
        with idx.conn:
            store.upsert(idx.conn, run, {**{k: lead[k] for k in (
                "path", "line", "col", "end_line", "symbol", "symbol_start", "symbol_end", "code_sha1")},
                "fingerprint": "second", "source": "llm", "rule": "LLM", "message": "returns a misspelled name",
                "score": 0.5})
        sibling = idx.conn.execute("SELECT id FROM candidates WHERE fingerprint = 'second'").fetchone()[0]
        store.transition(idx.conn, sibling, "triaged", data={"triage": {"p_real": 0.8}})

        llm = ScriptedLLM(repro=[REPRO_OK], fix=[FIX_OK])
        stats = investigate(idx, llm, sandbox, InvestigateConfig(use_cache=False))
        assert stats.proposed == 1 and stats.rejected == 1 and len(llm.prompts) == 2
        assert "line 9 (LLM): returns a misspelled name" in llm.prompts[0][1]["content"]  # sibling shown as context
        row = idx.conn.execute("SELECT state, data FROM candidates WHERE id = ?", (sibling,)).fetchone()
        assert row["state"] == "rejected" and json.loads(row["data"])["merged_into"] == cid
