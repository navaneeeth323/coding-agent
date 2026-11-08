"""End-to-end benchmark: run the whole pipeline on the toy project and score it.

    .venv/bin/python -m evals.benchmark_eval --check        # validate the benchmark itself (no LLM)
    .venv/bin/python -m evals.benchmark_eval [--stop-after triage] [--db PATH] [--model ...]

The agent only ever sees a temporary copy of evals/benchmark/project; the answer
key (bugs.json) and hidden tests stay outside it. A verified proposal counts as
a true positive only if it makes a planted bug's hidden test pass; the
verification gate's own test proves the agent found *a* bug, the hidden test
proves it was the planted one.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from bughunter.candidates import store
from bughunter.index import RepoIndex
from bughunter.investigate import InvestigateConfig, investigate
from bughunter.investigate.edits import changed_lines
from bughunter.llm import OllamaChat
from bughunter.pipeline import collect_candidates
from bughunter.scan import ScanConfig, scan
from bughunter.sandbox import LocalSandbox, Sandbox, Workspace, make_sandbox, run_tests
from bughunter.triage import TriageConfig, triage
from bughunter.verify import verify

BENCH = Path(__file__).parent / "benchmark"
HIDDEN_DEST = "tests/test_zz_hidden.py"  # where hidden tests go in scoring-only copies
STAGES = ["candidates", "scan", "triage", "investigate", "verify"]


def load_key() -> dict:
    key = json.loads((BENCH / "bugs.json").read_text())
    project = BENCH / key["project"]
    for entry in key["bugs"] + key["decoys"]:
        text = (project / entry["file"]).read_text()
        if text.count(entry["anchor"]) != 1:
            raise SystemExit(f"{entry['id']}: anchor must occur exactly once in {entry['file']}")
        entry["line"] = text[: text.index(entry["anchor"])].count("\n") + 1
    return key


def hidden_outcomes(sandbox: Sandbox, project: Path, files: Dict[str, str], key: dict) -> Dict[str, str]:
    """Outcome of each bug's hidden test with `files` applied to a fresh copy of the project."""
    with Workspace(project) as ws:
        ws.apply(files)
        ws.write(HIDDEN_DEST, (BENCH / key["hidden_tests"]).read_text())
        run = run_tests(sandbox, ws.path, [HIDDEN_DEST], timeout=300)
    by_name = {t.split("::")[-1]: o for t, o in run.outcomes.items()}
    return {b["id"]: by_name.get(b["hidden_test"], "missing") for b in key["bugs"]}


def _patched(project: Path, fixes: List[dict], file: str) -> str:
    text = (project / file).read_text()
    for f in fixes:
        if text.count(f["old"]) != 1:
            raise ValueError(f"fix text must occur exactly once in {file}: {f['old'][:60]!r}")
        text = text.replace(f["old"], f["new"])
    return text


def check(key: dict) -> int:
    """The benchmark's own invariants: visible suite green, each hidden test red until its fix."""
    project = BENCH / key["project"]
    sandbox = LocalSandbox(allow_unsafe=True)  # our own code
    ok = True
    with Workspace(project) as ws:
        run = run_tests(sandbox, ws.path)
    print(f"visible suite on buggy code: {run.summary()}")
    ok &= run.exit_code == 0

    before = hidden_outcomes(sandbox, project, {}, key)
    for b in key["bugs"]:
        files = {b["file"]: _patched(project, b["fix"], b["file"])}
        after = hidden_outcomes(sandbox, project, files, key)
        with Workspace(project) as ws:
            ws.apply(files)
            suite = run_tests(sandbox, ws.path)
        others = sorted(k for k, o in after.items() if k != b["id"] and o != before[k])
        good = before[b["id"]] == "failed" and after[b["id"]] == "passed" and suite.exit_code == 0 and not others
        ok &= good
        print(f"{'ok  ' if good else 'BAD '} {b['id']:<3} {b['difficulty']:<6} hidden test {before[b['id']]} -> "
              f"{after[b['id']]}; suite with fix: {suite.summary()}" + (f"; also changed {others}" if others else ""))

    # Static reach: which bugs any deterministic source flags at all (the ceiling for recall).
    with tempfile.TemporaryDirectory() as tmp:
        repo = Path(tmp) / "project"
        shutil.copytree(project, repo)
        with RepoIndex(repo) as idx:
            collect_candidates(idx)
            rows = store.queue(idx.conn, "new", limit=10_000)
    print("\nflagged by candidate sources:")
    for entry in key["bugs"] + key["decoys"]:
        hits = sorted({r["rule"] for r in rows if _locates(r, entry)})
        print(f"  {entry['id']:<3} {entry.get('difficulty', 'decoy'):<6} {', '.join(hits) or '-'}")
    print("\nbenchmark OK" if ok else "\nbenchmark BROKEN")
    return 0 if ok else 1


def _locates(c: dict, entry: dict) -> bool:
    files = {entry["file"], *entry.get("also_files", [])}
    if c["path"] not in files:
        return False
    if c["symbol"] and c["symbol"] in entry["symbols"]:
        return True
    return c["path"] == entry["file"] and abs(c["line"] - entry["line"]) <= 2


def _touches(inv: dict, entry: dict, project: Path) -> bool:
    for path, f in inv.get("files", {}).items():
        if path == entry["file"]:
            before = (project / path).read_text()
            if any(lo - 3 <= entry["line"] <= hi + 3 for lo, hi in changed_lines(before, f["after"])):
                return True
    return False


def run(args, key: dict) -> int:
    project = BENCH / key["project"]
    stop = STAGES.index(args.stop_after)
    started = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        repo = Path(tmp) / "project"
        shutil.copytree(project, repo)
        db = Path(args.db) if args.db else repo / ".bughunter" / "state.db"
        sandbox = make_sandbox("local" if args.sandbox == "local-unsafe" else "docker", image=args.image)
        with RepoIndex(repo, db) as idx:
            store.ensure_schema(idx.conn)
            with idx.conn:  # fresh funnel each run; only the LLM cache carries over
                idx.conn.execute("DELETE FROM candidates")
            collect_candidates(idx)
            llm = OllamaChat(model=args.model, think=args.think, num_ctx=args.num_ctx)
            stats = {}
            if stop >= 1 and not args.no_scan:
                stats["scan"] = scan(idx, llm, ScanConfig(threshold=args.scan_threshold))
                s = stats["scan"]
                print(f"scan: {s.functions} functions -> {s.flagged} suspicious, {s.failed} failed", flush=True)
            if stop >= 2:
                stats["triage"] = triage(idx, llm, TriageConfig(threshold=args.threshold, samples=args.samples,
                                                                min_score=args.min_score))
                t = stats["triage"]
                print(f"triage: kept {t.kept}, rejected {t.rejected}, failed {t.failed}", flush=True)
                if t.failed and not (t.kept or t.rejected):
                    print(f"every triage call failed; is the model up? first error: {t.errors[0]}", file=sys.stderr)
                    return 1
            if stop >= 3:
                def progress(c, record):
                    print(f"  investigate #{c['id']} {c['path']}:{c['line']} ({c['symbol']}): {record['status']}"
                          + (f" -- {record.get('reason', '')[:100]}" if record.get("reason") else ""), flush=True)
                stats["investigate"] = investigate(idx, llm, sandbox, InvestigateConfig(limit=args.limit),
                                                   progress=progress)
            if stop >= 4:
                stats["verify"] = verify(idx, sandbox)
            rows = [dict(r) for r in idx.conn.execute("SELECT * FROM candidates WHERE active = 1")]

    # ---- score ------------------------------------------------------------
    for r in rows:
        r["d"] = json.loads(r["data"])
    funnel = {}
    for b in key["bugs"]:
        mine = [r for r in rows if _locates(r, b)]
        funnel[b["id"]] = {
            "flagged": bool(mine),
            "kept": any(r["d"].get("triage", {}).get("keep") for r in mine),
            "proposed": any(r["d"].get("investigation", {}).get("status") == "proposed" for r in mine),
            "verified": any(r["state"] == "verified" for r in mine),
            "fixed": False,
        }

    verified = [r for r in rows if r["state"] == "verified"]
    sandbox = LocalSandbox(allow_unsafe=True)
    baseline = hidden_outcomes(sandbox, project, {}, key) if verified else {}
    tp = decoy_fp = other_fp = 0
    print("\nverified proposals:")
    for r in verified:
        inv = r["d"]["investigation"]
        after = hidden_outcomes(sandbox, project, {p: f["after"] for p, f in inv["files"].items()}, key)
        fixed = [k for k, o in after.items() if o == "passed" and baseline[k] != "passed"]
        decoys = [d["id"] for d in key["decoys"] if _touches(inv, d, project)]
        if fixed:
            tp += 1
            verdict = f"TP fixes {', '.join(fixed)}"
            for k in fixed:
                funnel[k]["fixed"] = True
        elif decoys:
            decoy_fp += 1
            verdict = f"FP on decoy {', '.join(decoys)}"
        else:
            other_fp += 1
            verdict = "FP? fixes no planted bug (review: may be a real unplanted bug)"
        print(f"  #{r['id']:<4} {r['rule']:<6} {r['path']}:{r['line']} ({r['symbol']}): {verdict}")

    print("\nper bug        flagged kept proposed verified fixed")
    by_diff: Dict[str, List[bool]] = defaultdict(list)
    for b in key["bugs"]:
        f = funnel[b["id"]]
        by_diff[b["difficulty"]].append(f["fixed"])
        print(f"  {b['id']:<3} {b['difficulty']:<7}  " + "  ".join(
            f"{'yes' if f[k] else ' - ':>6}" for k in ("flagged", "kept", "proposed", "verified", "fixed")))

    decoy_kept = sorted({d["id"] for d in key["decoys"] for r in rows
                         if _locates(r, d) and r["d"].get("triage", {}).get("keep")})
    n_fixed = sum(f["fixed"] for f in funnel.values())
    print(
        f"\nprecision {tp / len(verified) if verified else 0:.2f} ({tp}/{len(verified)} verified fix a planted bug)"
        f"  recall {n_fixed / len(key['bugs']):.2f} ({n_fixed}/{len(key['bugs'])})"
        f"  decoy FPs: kept by triage {decoy_kept or 'none'}, verified {decoy_fp}; other FPs {other_fp}"
    )
    print("recall by difficulty: " + ", ".join(
        f"{d} {sum(v)}/{len(v)}" for d, v in sorted(by_diff.items())))
    calls = sum(getattr(s, "llm_calls", 0) for s in stats.values())
    tokens = sum(getattr(s, "prompt_tokens", 0) + getattr(s, "completion_tokens", 0) for s in stats.values())
    print(f"{calls} LLM calls, {tokens} tokens, {time.time() - started:.0f}s wall"
          + (f", {tokens // n_fixed} tokens per fixed bug" if n_fixed else ""))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="validate the benchmark (no LLM)")
    ap.add_argument("--stop-after", default="verify", choices=STAGES)
    ap.add_argument("--model", default="qwen3.5:9b")
    ap.add_argument("--think", action="store_true")
    ap.add_argument("--num-ctx", type=int, default=16384)
    ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--min-score", type=float, default=0.2)
    ap.add_argument("--limit", type=int, default=20, help="max candidates to investigate")
    ap.add_argument("--no-scan", action="store_true", help="analyzer candidates only (no LLM function scan)")
    ap.add_argument("--scan-threshold", type=float, default=0.6)
    ap.add_argument("--db", help="state db to reuse (keeps the LLM cache between runs)")
    ap.add_argument("--sandbox", default="local-unsafe", choices=["local-unsafe", "docker"],
                    help="the benchmark is our own code, so local is acceptable here")
    ap.add_argument("--image", default="bughunter-sandbox:latest")
    args = ap.parse_args(argv)
    key = load_key()
    return check(key) if args.check else run(args, key)


if __name__ == "__main__":
    sys.exit(main())
