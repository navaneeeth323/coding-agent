import json
import re
import textwrap
from pathlib import Path

import pytest

from bughunter.candidates import store
from bughunter.index import RepoIndex
from bughunter.llm import ChatResult, LLMError
from bughunter.pipeline import collect_candidates
from bughunter.triage import TriageConfig, triage
from bughunter.triage.prompt import build_messages


class FakeLLM:
    """Judges warnings by rule; records prompts. Real if the rule is in `real`."""

    model = "fake"

    def __init__(self, real=(), fail_subjects=()):
        self.real = set(real)
        self.fail = set(fail_subjects)
        self.prompts = []

    def chat_json(self, messages, schema, *, temperature=0.0, seed=None, max_tokens=None):
        user = messages[-1]["content"]
        self.prompts.append(user)
        if any(s in user for s in self.fail):
            raise LLMError("boom")
        warnings = re.findall(r"^\[(\d+)\] line \d+, (\S+):", user, re.M)
        verdicts = [
            {"id": int(i), "intent": "i", "failing_input": "f", "trace": "t", "expected": "e", "actual": "a",
             "is_real_bug": rule in self.real,
             "category": "logic", "confidence": 0.9}
            for i, rule in warnings
        ]
        return ChatResult(data={"verdicts": verdicts}, text=json.dumps({"verdicts": verdicts}), model=self.model,
                          prompt_tokens=10, completion_tokens=5)


@pytest.fixture
def repo(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg/util.py").write_text(textwrap.dedent('''
        def normalize(name):
            """Collapse whitespace and title-case."""
            return " ".join(name.split()).title()
        ''').lstrip())
    (tmp_path / "pkg/core.py").write_text(textwrap.dedent('''
        from pkg.util import normalize


        def load(path, cache={}):
            try:
                return normalize(cache[path])
            except:
                return missing_name


        def other():
            return undefined_too
        ''').lstrip())
    return tmp_path


def test_prompt_context(repo):
    with RepoIndex(repo) as idx:
        collect_candidates(idx)
        rows = [c for c in store.queue(idx.conn) if c["symbol"] == "load"]
        from bughunter.triage import group_candidates
        [group] = group_candidates(rows)
        user = build_messages(idx, group)[1]["content"]
        assert "Function: load" in user
        assert ">> 4 | def load(path, cache={}):" in user  # flagged line marked
        assert "pkg/util.py:1  def normalize(name)  # Collapse whitespace and title-case." in user
        assert re.search(r"\[1\] line 4, B006", user) and "[3] line 8, F821" in user


def test_triage_transitions_and_cache(repo):
    with RepoIndex(repo) as idx:
        collect_candidates(idx)
        llm = FakeLLM(real={"F821"})
        stats = triage(idx, llm, TriageConfig(min_score=0.0))
        assert stats.groups == 2 and stats.llm_calls == 2  # one call per function
        assert stats.kept == 2 and stats.rejected == stats.candidates - 2
        kept = store.queue(idx.conn, "triaged")
        assert {c["symbol"] for c in kept} == {"load", "other"}
        t = json.loads(kept[0]["data"])["triage"]
        assert t["p_real"] == 0.9 and t["keep"] and t["model"] == "fake"

        # Identical prompts are served from the cache.
        with idx.conn:
            idx.conn.execute("UPDATE candidates SET state = 'new'")
        stats = triage(idx, llm, TriageConfig(min_score=0.0))
        assert stats.cached_calls == 2 and len(llm.prompts) == 2
        calls = idx.conn.execute("SELECT COUNT(*) FROM llm_calls WHERE stage = 'triage'").fetchone()[0]
        assert calls == 4  # every call traced, hits included


def test_failed_calls_leave_candidates_new(repo):
    with RepoIndex(repo) as idx:
        collect_candidates(idx)
        stats = triage(idx, FakeLLM(fail_subjects={"Function: other"}), TriageConfig(min_score=0.0))
        assert stats.failed == 1 and len(stats.errors) == 2  # first try + retry
        assert [c["symbol"] for c in store.queue(idx.conn, "new")] == ["other"]


def test_votes_average(repo):
    class Flaky(FakeLLM):
        def chat_json(self, messages, schema, **kw):
            self.real = {"F821"} if kw["seed"] % 2 == 0 else set()
            return super().chat_json(messages, schema, **kw)

    with RepoIndex(repo) as idx:
        collect_candidates(idx)
        triage(idx, Flaky(), TriageConfig(min_score=0.0, samples=3, threshold=0.5))
        [other] = [c for c in store.queue(idx.conn, "triaged") if c["symbol"] == "other"]
        # Votes: real 0.9, not-real (0.1), real 0.9 -> mean 0.633
        assert json.loads(other["data"])["triage"]["p_real"] == pytest.approx(0.633, abs=1e-3)
