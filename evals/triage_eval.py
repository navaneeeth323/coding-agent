"""Score the triage stage on labelled cases.

    .venv/bin/python -m evals.triage_eval [--samples N] [--think] [--model qwen3.5:9b]

Runs the real pipeline (index -> ruff/heuristics -> triage) on a temporary repo
built from evals/triage_cases.py, then reports accuracy, precision and recall for
real bugs, the false-positive rate on decoys, and the time and tokens spent.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

from bughunter.candidates import store
from bughunter.index import RepoIndex
from bughunter.llm import OllamaChat
from bughunter.pipeline import collect_candidates
from bughunter.triage import TriageConfig, triage

from .triage_cases import CASES


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3.5:9b")
    ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--think", action="store_true")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--db", help="state db to reuse (keeps the LLM cache between eval runs)")
    ap.add_argument("--only", nargs="*", help="case names to run")
    ap.add_argument("--verbose", "-v", action="store_true")
    args = ap.parse_args(argv)

    cases = [c for c in CASES if not args.only or c["name"] in args.only]
    with tempfile.TemporaryDirectory() as tmp:
        repo = Path(tmp) / "evalrepo"
        (repo / "cases").mkdir(parents=True)
        for c in cases:
            (repo / "cases" / f"{c['name']}.py").write_text(c["source"].lstrip())

        db = Path(args.db) if args.db else repo / ".bughunter" / "state.db"
        with RepoIndex(repo, db) as idx:
            # Start every eval from `new`; only the LLM cache carries over between runs
            # (prompts don't contain the temp dir, so cached answers still apply).
            store.ensure_schema(idx.conn)
            with idx.conn:
                idx.conn.execute("DELETE FROM candidates")
            collect_candidates(idx)
            # Score every labelled finding, whatever its static priority.
            config = TriageConfig(threshold=args.threshold, samples=args.samples, min_score=0.0)
            llm = OllamaChat(model=args.model, think=args.think)
            started = time.time()
            stats = triage(idx, llm, config)
            wall = time.time() - started

            got = {}
            for r in idx.conn.execute(
                "SELECT path, symbol, rule, state, data FROM candidates WHERE active = 1"
            ):
                got[(Path(r["path"]).stem, r["rule"], r["symbol"])] = (r["state"], json.loads(r["data"]))

    tp = fp = tn = fn = missing = 0
    rows = []
    for c in cases:
        for rule, symbol, real in c["expect"]:
            hit = got.get((c["name"], rule, symbol))
            if hit is None or hit[0] == "new":
                missing += 1
                rows.append(("MISSING", c["name"], rule, symbol, None))
                continue
            kept = hit[0] == "triaged"
            triage_data = hit[1].get("triage", {})
            outcome = {(True, True): "TP", (True, False): "FN", (False, True): "FP", (False, False): "TN"}[(real, kept)]
            tp += outcome == "TP"; fn += outcome == "FN"; fp += outcome == "FP"; tn += outcome == "TN"
            rows.append((outcome, c["name"], rule, symbol, triage_data))
        # Unlabelled findings in a case file are reported but not scored, except
        # decoy-only files with no expectations at all, where anything kept is a false positive.
        if not c["expect"]:
            for (name, rule, symbol), (state, data) in got.items():
                if name == c["name"] and state == "triaged":
                    fp += 1
                    rows.append(("FP", name, rule, symbol, data.get("triage", {})))
                elif name == c["name"] and state == "rejected":
                    tn += 1
                    rows.append(("TN", name, rule, symbol, data.get("triage", {})))

    for outcome, name, rule, symbol, t in rows:
        p = f"p={t['p_real']:.2f}" if t else "      "
        print(f"{outcome:<8} {p}  {name:<40} {rule:<8} {symbol}")
        if args.verbose and t and outcome in ("FP", "FN"):
            for k in ("intent", "failing_input", "trace", "expected", "actual"):
                print(f"           {k + ':':<15}{t.get(k)}")

    # Thresholds are applied to stored P(real), so sweeping them costs no extra calls.
    labelled = []
    for c in cases:
        for rule, symbol, real in c["expect"]:
            hit = got.get((c["name"], rule, symbol))
            p = hit and hit[1].get("triage", {}).get("p_real")
            if p is not None:
                labelled.append((real, p))
    if labelled:
        print("\nthreshold  precision  recall  decoy-FP-rate")
        for th in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7):
            s_tp = sum(1 for real, p in labelled if real and p >= th)
            s_fp = sum(1 for real, p in labelled if not real and p >= th)
            n_real = sum(1 for real, _ in labelled if real)
            n_decoy = len(labelled) - n_real
            prec = s_tp / (s_tp + s_fp) if s_tp + s_fp else 0.0
            print(f"  {th:.1f}       {prec:.2f}      {s_tp / n_real if n_real else 0:.2f}    {s_fp / n_decoy if n_decoy else 0:.2f}")

    scored = tp + fp + tn + fn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    print(
        f"\naccuracy {(tp + tn) / scored if scored else 0:.2f}  precision {precision:.2f}  recall {recall:.2f}"
        f"  decoy FP rate {fp / (fp + tn) if fp + tn else 0:.2f}  (TP {tp} FP {fp} TN {tn} FN {fn}, missing {missing})"
    )
    print(
        f"{stats.llm_calls} calls ({stats.cached_calls} cached), {stats.prompt_tokens} prompt +"
        f" {stats.completion_tokens} completion tokens, {wall:.1f}s wall"
        + (f", errors: {stats.errors}" if stats.errors else "")
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
