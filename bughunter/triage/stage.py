"""Stage 3: triage new candidates with a cheap model; keep the likely bugs.

Each function's candidates are judged together in one call. With `samples` > 1
the call is repeated at a non-zero temperature and the verdicts are averaged
(self-consistency), which gives a better-calibrated probability than the
model's own confidence.
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional

from ..candidates import store
from ..index import RepoIndex
from ..llm import JSONChat, LLMError, RecordingLLM
from .prompt import TriageGroup, build_messages, output_schema

log = logging.getLogger(__name__)

SAMPLE_TEMPERATURE = 0.7
RETRY_TEMPERATURE = 0.3  # a failed greedy decode is retried once with a little randomness
TOKENS_BASE, TOKENS_PER_FINDING = 300, 800


@dataclass
class TriageConfig:
    threshold: float = 0.5  # minimum P(real bug) to move on to investigation
    samples: int = 1
    min_score: float = 0.2  # skip low-priority candidates (TODOs, style-ish rules)
    limit: int = 500  # max candidates per run
    use_cache: bool = True


@dataclass
class TriageStats:
    run_id: int = 0
    groups: int = 0
    candidates: int = 0
    kept: int = 0
    rejected: int = 0
    failed: int = 0  # left in `new`; retried next run
    llm_calls: int = 0
    cached_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    seconds: float = 0.0
    errors: List[str] = field(default_factory=list)


def group_candidates(rows: List[dict]) -> List[TriageGroup]:
    """One group per function; module-level findings are judged one at a time."""
    groups: Dict[tuple, TriageGroup] = {}
    for c in rows:
        key = (c["path"], c["symbol"]) if c["symbol"] else (c["path"], None, c["id"])
        if key not in groups:
            groups[key] = TriageGroup(c["path"], c["symbol"], c["symbol_start"], c["symbol_end"], [])
        groups[key].candidates.append(c)
    for g in groups.values():
        g.candidates.sort(key=lambda c: (c["line"], c["col"]))
    # Rows arrive best-first, so dict order already puts the most promising groups first.
    return list(groups.values())


def _p_real(verdict: dict) -> float:
    conf = min(1.0, max(0.5, float(verdict["confidence"])))
    return conf if verdict["is_real_bug"] else 1.0 - conf


def triage(
    index: RepoIndex,
    llm: JSONChat,
    config: TriageConfig = TriageConfig(),
    progress: Optional[Callable[[TriageGroup, Dict[int, dict]], None]] = None,
) -> TriageStats:
    conn = index.conn
    store.ensure_schema(conn)
    run_id = store.start_run(conn, "triage")
    recorder = RecordingLLM(llm, conn, stage="triage", run_id=run_id, use_cache=config.use_cache)
    stats = TriageStats(run_id=run_id)

    rows = store.queue(conn, "new", limit=config.limit, min_score=config.min_score)
    groups = group_candidates(rows)
    stats.groups, stats.candidates = len(groups), len(rows)

    try:
        for group in groups:
            messages = build_messages(index, group)
            schema = output_schema(len(group.candidates))
            votes: Dict[int, List[dict]] = defaultdict(list)  # local id -> verdicts
            max_tokens = TOKENS_BASE + TOKENS_PER_FINDING * len(group.candidates)
            for sample in range(config.samples):
                temperature = 0.0 if config.samples == 1 else SAMPLE_TEMPERATURE
                result = None
                for attempt in range(2):
                    try:
                        result = recorder.chat_json(
                            messages, schema, temperature=temperature if attempt == 0 else max(temperature, RETRY_TEMPERATURE),
                            seed=sample + 1000 * attempt, max_tokens=max_tokens, subject=group.subject,
                        )
                        break
                    except LLMError as exc:
                        stats.errors.append(f"{group.subject}: {exc}")
                        log.warning("triage call failed for %s (attempt %d): %s", group.subject, attempt + 1, exc)
                if result is None:
                    continue
                stats.llm_calls += 1
                stats.cached_calls += result.cached
                stats.prompt_tokens += result.prompt_tokens
                stats.completion_tokens += result.completion_tokens
                stats.seconds += result.seconds
                for v in result.data.get("verdicts", []):
                    if isinstance(v.get("id"), int) and 1 <= v["id"] <= len(group.candidates):
                        votes[v["id"]].append(v)

            decisions = {}
            for local_id, cand in enumerate(group.candidates, start=1):
                vs = votes.get(local_id)
                if not vs:
                    stats.failed += 1
                    continue
                p_real = sum(_p_real(v) for v in vs) / len(vs)
                keep = p_real >= config.threshold
                # Report the reasoning of a sample that agrees with the decision.
                lead = next((v for v in vs if v["is_real_bug"] == keep), vs[0])
                decision = {
                    "p_real": round(p_real, 3),
                    "keep": keep,
                    "category": Counter(v["category"] for v in vs).most_common(1)[0][0],
                    # The repro sketch the investigation stage starts from.
                    "intent": lead["intent"],
                    "failing_input": lead["failing_input"],
                    "trace": lead["trace"],
                    "expected": lead["expected"],
                    "actual": lead["actual"],
                    "votes": [{"is_real_bug": v["is_real_bug"], "confidence": v["confidence"]} for v in vs],
                    "model": llm.model,
                }
                store.transition(
                    conn, cand["id"], "triaged" if keep else "rejected",
                    note=f"triage p_real={p_real:.2f}", run_id=run_id, data={"triage": decision},
                )
                if keep:
                    stats.kept += 1
                else:
                    stats.rejected += 1
                decisions[cand["id"]] = decision
            if progress:
                progress(group, decisions)
    except BaseException:
        store.finish_run(conn, run_id, "failed", asdict(stats))
        raise
    store.finish_run(conn, run_id, "ok", asdict(stats))
    return stats
