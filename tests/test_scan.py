import json
import textwrap

import pytest

from bughunter.candidates import store
from bughunter.index import RepoIndex
from bughunter.llm import ChatResult
from bughunter.pipeline import collect_candidates
from bughunter.scan import ScanConfig, scan
from bughunter.triage import group_candidates
from bughunter.triage.prompt import build_messages


class FakeScanner:
    """Flags the functions named in `buggy`; records the prompts it saw."""

    model = "fake"

    def __init__(self, buggy=(), confidence=0.9):
        self.buggy, self.confidence = set(buggy), confidence
        self.prompts = {}

    def chat_json(self, messages, schema, *, temperature=0.0, seed=None, max_tokens=None):
        user = messages[-1]["content"]
        name = next(l.split(": ", 1)[1] for l in user.splitlines() if l.startswith(("Function:", "Method:")))
        self.prompts[name] = user
        bug = name in self.buggy
        data = {"intent": "i", "concern": "uses < but the docstring says or fewer" if bug else "none",
                "line": 999, "failing_input": "low([3], 3)", "expected": "[3]", "actual": "[]",
                "has_bug": bug, "category": "off_by_one", "confidence": self.confidence}
        return ChatResult(data=data, text=json.dumps(data), model=self.model, prompt_tokens=10, completion_tokens=5)


SRC = textwrap.dedent('''
    class Stock:
        def __init__(self):
            self.items = {}
            # Bumped on every change, so caches know to refresh.
            self.version = 0

        def low(self, threshold):
            """Names with `threshold` units or fewer."""
            return [n for n, q in self.items.items() if q < threshold]


    def tiny(x):
        return x


    def helper(a, b):
        total = a + b
        return total * 2
    ''').lstrip()


@pytest.fixture
def repo(tmp_path):
    (tmp_path / "stock.py").write_text(SRC)
    return tmp_path


def test_scan_stores_suspicions(repo):
    with RepoIndex(repo) as idx:
        llm = FakeScanner(buggy={"Stock.low"})
        stats = scan(idx, llm, ScanConfig())
        # tiny() is too short to scan; __init__, low and helper are scanned.
        assert stats.functions == 3 and stats.flagged == 1 and stats.cleared == 2
        assert set(llm.prompts) == {"Stock.__init__", "Stock.low", "helper"}
        # Methods see their class's __init__, where invariants are documented.
        assert "Bumped on every change" in llm.prompts["Stock.low"]

        [c] = store.queue(idx.conn, "new")
        assert (c["source"], c["rule"], c["symbol"]) == ("llm", "LLM", "Stock.low")
        assert c["line"] == c["symbol_end"] == 9  # out-of-range line clamped into the function
        assert "docstring says or fewer" in c["message"] and c["score"] == pytest.approx(0.54)

        # Triage judges it like any warning, with a note that it is a model's claim.
        [group] = group_candidates([c])
        user = build_messages(idx, group)[1]["content"]
        assert "LLM: uses < but the docstring" in user and "another model" in user


def test_rescan_and_code_changes_retire_suspicions(repo):
    with RepoIndex(repo) as idx:
        scan(idx, FakeScanner(buggy={"Stock.low"}))
        collect_candidates(idx)  # unchanged code: suspicion survives the analyzer run
        assert [c["rule"] for c in store.queue(idx.conn, "new")] == ["LLM"]

        # Code moved down two lines: suspicion follows the function.
        (repo / "stock.py").write_text("import os\n\n" + SRC)
        collect_candidates(idx)
        [c] = [c for c in store.queue(idx.conn, "new") if c["rule"] == "LLM"]
        assert (c["line"], c["symbol_start"]) == (11, 9)

        # Code changed: retired until the next scan looks again.
        (repo / "stock.py").write_text(SRC.replace("q < threshold", "q <= threshold"))
        collect_candidates(idx)
        assert not [c for c in store.queue(idx.conn, "new") if c["rule"] == "LLM"]

        # A rescan that finds nothing keeps it retired; one below threshold stores nothing.
        scan(idx, FakeScanner(buggy={"Stock.low"}, confidence=0.55))
        assert not [c for c in store.queue(idx.conn, "new") if c["rule"] == "LLM"]
