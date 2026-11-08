"""`run_tests`: pytest inside a sandbox, with per-test outcomes from its JUnit report."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence

from .runner import RunResult, Sandbox

JUNIT_FILE = ".bughunter-junit.xml"

# pytest exit codes
EXIT_OK, EXIT_FAILED, EXIT_INTERRUPTED, EXIT_INTERNAL, EXIT_USAGE, EXIT_NO_TESTS = 0, 1, 2, 3, 4, 5


@dataclass
class TestRun:
    __test__ = False  # not a pytest test class

    exit_code: int
    timed_out: bool
    seconds: float
    output: str
    outcomes: Dict[str, str] = field(default_factory=dict)  # test id -> passed | failed | error | skipped

    def ids(self, outcome: str) -> List[str]:
        return sorted(t for t, o in self.outcomes.items() if o == outcome)

    @property
    def broken(self) -> List[str]:
        """Tests that failed or errored (collection errors included)."""
        return sorted(t for t, o in self.outcomes.items() if o in ("failed", "error"))

    @property
    def collection_ok(self) -> bool:
        return not self.timed_out and self.exit_code in (EXIT_OK, EXIT_FAILED) and not self.ids("error")

    def summary(self) -> str:
        counts: Dict[str, int] = {}
        for o in self.outcomes.values():
            counts[o] = counts.get(o, 0) + 1
        state = "timed out" if self.timed_out else f"exit {self.exit_code}"
        return f"{state}: " + (", ".join(f"{n} {o}" for o, n in sorted(counts.items())) or "no tests ran")

    def failure_text(self, max_lines: int = 60) -> str:
        """The end of pytest's output (where tracebacks and the summary are), for a model to read."""
        lines = [l for l in self.output.splitlines() if l.strip()]
        if len(lines) > max_lines:
            lines = ["..."] + lines[-max_lines:]
        return "\n".join(lines)


def parse_junit(xml_text: str) -> Dict[str, str]:
    outcomes: Dict[str, str] = {}
    root = ET.fromstring(xml_text)
    for case in root.iter("testcase"):
        cls, name = case.get("classname", ""), case.get("name", "")
        # A collection error has an empty classname and the module path as its name.
        tid = f"{cls}::{name}" if cls else name
        if case.find("error") is not None:
            outcome = "error"
        elif case.find("failure") is not None:
            outcome = "failed"
        elif case.find("skipped") is not None:
            outcome = "skipped"
        else:
            outcome = "passed"
        # A test can be reported twice (failure in the call, error in teardown): keep the worst.
        if outcomes.get(tid) not in ("error", "failed"):
            outcomes[tid] = outcome
    return outcomes


def run_tests(
    sandbox: Sandbox, workdir: Path, selectors: Sequence[str] = (), timeout: float = 300.0,
) -> TestRun:
    junit = workdir / JUNIT_FILE
    junit.unlink(missing_ok=True)
    cmd = [
        sandbox.python, "-m", "pytest", "-q", "--tb=short", "-p", "no:cacheprovider",
        f"--junitxml={JUNIT_FILE}", "-o", "junit_family=xunit1", *selectors,
    ]
    result: RunResult = sandbox.run(workdir, cmd, timeout=timeout)
    outcomes: Dict[str, str] = {}
    if junit.exists():
        try:
            outcomes = parse_junit(junit.read_text("utf-8", "replace"))
        except ET.ParseError:
            pass  # killed mid-write; the exit code and output still tell the story
        junit.unlink()
    return TestRun(
        exit_code=result.exit_code, timed_out=result.timed_out, seconds=result.seconds,
        output=result.output, outcomes=outcomes,
    )


def run_snippet(sandbox: Sandbox, workdir: Path, code: str, timeout: float = 60.0) -> RunResult:
    script = workdir / ".bughunter_snippet.py"
    script.write_text(code, "utf-8")
    try:
        return sandbox.run(workdir, [sandbox.python, script.name], timeout=timeout)
    finally:
        script.unlink()
