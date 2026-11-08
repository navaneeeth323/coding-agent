"""Turn a finding into a priority score in [0, 1] for the triage queue.

score = rule prior × hotspot multiplier × location factor, clipped to 1. The
priors are hand-set guesses; once triage and verification produce outcomes,
replace them with per-rule precision measured from the candidates table.
"""

from __future__ import annotations

import bisect
import re
from typing import Dict, List

from .git import FileHistory

# Longest matching prefix wins.
RULE_PRIORS = {
    "syntax-error": 0.5,
    "F821": 0.9,  # undefined name
    "F822": 0.8,  # undefined name in __all__
    "F823": 0.9,  # local referenced before assignment
    "F632": 0.8,  # `is` with a literal
    "F811": 0.5,  # redefinition of unused name
    "F841": 0.3,  # local assigned but never used
    "F": 0.4,
    "E9": 0.9,
    "E711": 0.2, "E712": 0.2, "E721": 0.3,
    "E722": 0.35,  # bare except
    "B006": 0.6,  # mutable default argument
    "B008": 0.4,  # call in default argument
    "B015": 0.6,  # pointless comparison
    "B018": 0.4,  # useless expression
    "B020": 0.7,  # loop variable overrides the iterable
    "B023": 0.7,  # closure captures loop variable
    "B007": 0.2,  # unused loop variable
    "B009": 0.1, "B010": 0.1,  # getattr/setattr with a constant: style
    "B": 0.45,
    "PLE": 0.8,
    "PLW2901": 0.2,  # loop variable overwritten in body
    "PLW0127": 0.3,  # self-assignment
    "PLW1641": 0.3,  # __eq__ without __hash__
    "PLW": 0.35,
    "BLE": 0.3,
    "S110": 0.3, "S112": 0.3,
    "SIM115": 0.35,
    "FIX001": 0.35, "FIX003": 0.3, "FIX004": 0.3,  # FIXME, XXX, HACK
    "FIX002": 0.15,  # TODO
    "RUF006": 0.6,
    "A": 0.2,
    "ASYNC": 0.5,
    "H001": 0.3,  # range(len(x) + k)
    "H002": 0.85,  # x[len(x)]
    "H003": 0.5,  # i <= len(x), then x[i]
    "LLM": 0.6,  # function scan suspicion; scaled by the scan's confidence
}
DEFAULT_PRIOR = 0.3

TEST_PATH = re.compile(r"(^|/)(tests?|testing)/|(^|/)test_[^/]*\.py$|_test\.py$|(^|/)conftest\.py$")
TEST_FACTOR = 0.3  # bugs in tests matter, but rarely as much as bugs in the code under test


def rule_prior(rule: str) -> float:
    for end in range(len(rule), 0, -1):
        prior = RULE_PRIORS.get(rule[:end])
        if prior is not None:
            return prior
    return DEFAULT_PRIOR


class Hotspots:
    """Ranks files against the rest of the repo by how often they needed fixing.

    Relative, not absolute: in an old project every core file has dozens of fix
    commits, and an absolute boost would just inflate everything equally.
    """

    MAX_BOOST = 0.5

    def __init__(self, history: Dict[str, FileHistory]):
        self.history = history
        self._fixes: List[int] = sorted(h.fix_commits for h in history.values())
        self._commits: List[int] = sorted(h.commits for h in history.values())

    @staticmethod
    def _percentile(values: List[int], x: int) -> float:
        return bisect.bisect_left(values, x) / len(values) if values else 0.0

    def multiplier(self, path: str) -> float:
        """1.0 for files with no history, up to 1.5 for the repo's most-fixed files."""
        h = self.history.get(path)
        if h is None:
            return 1.0
        heat = 0.75 * self._percentile(self._fixes, h.fix_commits) + 0.25 * self._percentile(self._commits, h.commits)
        return 1.0 + self.MAX_BOOST * heat


def score(rule: str, path: str, hotspots: Hotspots) -> float:
    s = rule_prior(rule) * hotspots.multiplier(path)
    if TEST_PATH.search(path):
        s *= TEST_FACTOR
    return round(min(1.0, s), 4)
