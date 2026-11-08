"""Command-line entry point: `python -m bughunter <command> ...`."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .candidates import store as candidate_store
from .index import RepoIndex
from .investigate import InvestigateConfig, investigate
from .llm import OllamaChat
from .pipeline import collect_candidates
from .pr import PRError, open_pr
from .scan import ScanConfig, scan
from .sandbox import SandboxError, Workspace, build_image, make_sandbox, run_tests
from .triage import TriageConfig, triage
from .verify import VerifyConfig, verify


def _open(args) -> RepoIndex:
    return RepoIndex(Path(args.repo), Path(args.db) if args.db else None)


def cmd_index(args) -> int:
    with _open(args) as idx:
        s = idx.update()
        print(
            f"indexed {idx.repo}: +{s.added} ~{s.changed} -{s.removed} ={s.unchanged}"
            f" (skipped {s.skipped}, syntax errors {s.parse_errors}) in {s.seconds:.2f}s"
            f"\ndb: {idx.db_path}"
        )
    return 0


def cmd_map(args) -> int:
    with _open(args) as idx:
        idx.update()
        print(idx.repo_map(max_tokens=args.tokens, focus=args.focus))
    return 0


def _print_rows(rows) -> int:
    for row in rows:
        print(json.dumps(row))
    return 0 if rows else 1


def cmd_find(args) -> int:
    with _open(args) as idx:
        idx.update()
        return _print_rows(idx.find_symbol(args.name))


def cmd_callers(args) -> int:
    with _open(args) as idx:
        idx.update()
        return _print_rows(idx.find_callers(args.name))


def cmd_at(args) -> int:
    with _open(args) as idx:
        idx.update()
        row = idx.symbol_at(args.path, args.line)
        return _print_rows([row] if row else [])


def cmd_show(args) -> int:
    with _open(args) as idx:
        idx.update()
        src = idx.symbol_source(args.path, args.qualname)
        if src is None:
            return 1
        print(src)
    return 0


def cmd_candidates(args) -> int:
    with _open(args) as idx:
        s = collect_candidates(idx)
        print(
            f"run {s.run_id}: {s.findings} findings {s.by_source} ->"
            f" +{s.added} new, {s.updated} updated, {s.reopened} reopened"
        )
        for source, err in s.errors.items():
            print(f"  {source} FAILED: {err}", file=sys.stderr)
        _print_queue(candidate_store.queue(idx.conn, "new", limit=args.top))
    return 1 if s.errors else 0


def _print_queue(rows) -> None:
    for r in rows:
        where = f"{r['path']}:{r['line']}" + (f" ({r['symbol']})" if r["symbol"] else "")
        print(f"{r['score']:.2f}  #{r['id']:<5} {r['rule']:<8} {where}  {r['message']}")


def cmd_queue(args) -> int:
    with _open(args) as idx:
        candidate_store.ensure_schema(idx.conn)
        _print_queue(candidate_store.queue(idx.conn, args.state, limit=args.limit, min_score=args.min_score))
        print(json.dumps(candidate_store.funnel(idx.conn)), file=sys.stderr)
    return 0


def cmd_scan(args) -> int:
    with _open(args) as idx:
        collect_candidates(idx)  # keeps existing scan suspicions in step with the code
        llm = OllamaChat(model=args.model, host=args.host, think=args.think, num_ctx=args.num_ctx)
        config = ScanConfig(max_functions=args.max_functions, threshold=args.threshold,
                            include_tests=args.include_tests, use_cache=not args.no_cache)

        def progress(fn, v):
            verdict = "FAILED " if v is None else (
                "SUSPECT" if v["has_bug"] and v["confidence"] >= args.threshold else "ok     ")
            extra = f"  {v['concern'][:100]}" if v and verdict == "SUSPECT" else ""
            print(f"{verdict} {fn['path']}::{fn['qualname']}{extra}", flush=True)

        s = scan(idx, llm, config, progress=progress)
        print(
            f"run {s.run_id}: {s.functions} functions -> {s.flagged} suspicious, {s.cleared} clean,"
            f" {s.failed} failed; {s.llm_calls} calls ({s.cached_calls} cached),"
            f" {s.prompt_tokens}+{s.completion_tokens} tokens, {s.seconds:.0f}s in the model"
        )
    return 1 if s.failed else 0


def cmd_triage(args) -> int:
    with _open(args) as idx:
        collect_candidates(idx)  # cheap; guarantees line numbers match the code on disk
        llm = OllamaChat(model=args.model, host=args.host, think=args.think)
        config = TriageConfig(
            threshold=args.threshold, samples=args.samples, min_score=args.min_score,
            limit=args.limit, use_cache=not args.no_cache,
        )

        def progress(group, decisions):
            for c in group.candidates:
                d = decisions.get(c["id"])
                verdict = "FAILED " if d is None else ("KEEP   " if d["keep"] else "reject ")
                p = f"p={d['p_real']:.2f}" if d else "      "
                print(f"{verdict}{p}  #{c['id']:<5} {c['rule']:<8} {c['path']}:{c['line']} ({c['symbol']})", flush=True)

        s = triage(idx, llm, config, progress=progress)
        print(
            f"run {s.run_id}: {s.candidates} candidates in {s.groups} groups -> kept {s.kept},"
            f" rejected {s.rejected}, failed {s.failed}; {s.llm_calls} calls ({s.cached_calls} cached),"
            f" {s.prompt_tokens}+{s.completion_tokens} tokens, {s.seconds:.0f}s in the model"
        )
    return 1 if s.failed else 0


def _sandbox(args):
    if args.sandbox == "local-unsafe":
        print("warning: running repo code WITHOUT isolation (local-unsafe sandbox)", file=sys.stderr)
        return make_sandbox("local", python=args.python)
    return make_sandbox("docker", image=args.image)


def cmd_sandbox_build(args) -> int:
    build_image(Path(args.repo).resolve(), image=args.image, python_version=args.python_version)
    print(f"built {args.image}")
    return 0


def cmd_test(args) -> int:
    sandbox = _sandbox(args)
    with Workspace(Path(args.repo).resolve()) as ws:
        run = run_tests(sandbox, ws.path, args.selectors, timeout=args.timeout)
    print(run.failure_text(args.lines))
    print(run.summary())
    return 0 if run.exit_code == 0 else 1


def cmd_investigate(args) -> int:
    with _open(args) as idx:
        llm = OllamaChat(model=args.model, host=args.host, think=args.think, num_ctx=args.num_ctx)
        config = InvestigateConfig(
            test_attempts=args.test_attempts, fix_attempts=args.fix_attempts, max_tokens=args.max_tokens,
            wall_seconds=args.wall_seconds, check_suite=not args.no_suite, limit=args.limit,
            use_cache=not args.no_cache,
        )

        def progress(c, record):
            outcome = {"proposed": "PROPOSED", "rejected": "rejected", "error": "ERROR   "}[record["status"]]
            b = record.get("budget", {})
            print(f"{outcome} #{c['id']:<5} {c['rule']:<8} {c['path']}:{c['line']} ({c['symbol']})"
                  f"  {b.get('calls', 0)} calls {b.get('tokens', 0)} tok {b.get('seconds', 0)}s"
                  + (f"  -- {record['reason'][:120]}" if record.get("reason") else ""), flush=True)

        s = investigate(idx, llm, _sandbox(args), config, progress=progress)
        print(
            f"run {s.run_id}: {s.candidates} candidates -> {s.proposed} proposed, {s.rejected} rejected,"
            f" {s.retry} to retry; {s.llm_calls} calls, {s.prompt_tokens}+{s.completion_tokens} tokens,"
            f" {s.seconds:.0f}s in the model"
        )
    return 1 if s.retry else 0


def cmd_verify(args) -> int:
    with _open(args) as idx:
        def progress(c, checks):
            ok = all(ch.ok for ch in checks)
            print(f"{'VERIFIED' if ok else 'rejected'} #{c['id']:<5} {c['path']}:{c['line']} ({c['symbol']})  "
                  + "  ".join(f"{ch.name}={'ok' if ch.ok else 'FAIL'}" for ch in checks), flush=True)

        s = verify(idx, _sandbox(args), VerifyConfig(repeat=args.repeat), progress=progress)
        print(f"run {s.run_id}: {s.candidates} proposals -> {s.verified} verified, {s.rejected} rejected,"
              f" {s.stale} stale" + (f", errors: {s.errors}" if s.errors else ""))
    return 1 if s.errors else 0


def cmd_pr(args) -> int:
    with _open(args) as idx:
        ids = args.ids or [c["id"] for c in candidate_store.queue(idx.conn, "verified", limit=1000)]
        status = 0
        for cid in ids:
            try:
                plan = open_pr(idx, cid, push=args.push, base=args.base, remote=args.remote)
            except PRError as exc:
                print(f"#{cid}: {exc}", file=sys.stderr)
                status = 1
                continue
            if plan.duplicates:
                print(f"#{cid}: skipped, possible duplicates: {', '.join(plan.duplicates)}")
            elif plan.url:
                print(f"#{cid}: opened draft PR {plan.url}")
            else:
                print(f"#{cid}: would open draft PR on branch {plan.branch} (dry run; pass --push)\n")
                print(f"# {plan.title}\n\n{plan.body}")
    return status


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="bughunter")
    parser.add_argument("--repo", default=".", help="repository root (default: .)")
    parser.add_argument("--db", help="state database (default: <repo>/.bughunter/state.db)")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("index", help="build or incrementally update the index").set_defaults(fn=cmd_index)

    p = sub.add_parser("map", help="print the ranked repo map")
    p.add_argument("--tokens", type=int, default=1024)
    p.add_argument("--focus", nargs="*", default=[], help="repo-relative files to centre the map on")
    p.set_defaults(fn=cmd_map)

    p = sub.add_parser("find", help="find definitions by name or qualified name")
    p.add_argument("name")
    p.set_defaults(fn=cmd_find)

    p = sub.add_parser("callers", help="find call sites of a name")
    p.add_argument("name")
    p.set_defaults(fn=cmd_callers)

    p = sub.add_parser("at", help="innermost symbol enclosing path:line")
    p.add_argument("path")
    p.add_argument("line", type=int)
    p.set_defaults(fn=cmd_at)

    p = sub.add_parser("show", help="print a symbol's source with line numbers")
    p.add_argument("path")
    p.add_argument("qualname")
    p.set_defaults(fn=cmd_show)

    p = sub.add_parser("candidates", help="run analyzers and heuristics; store ranked candidates")
    p.add_argument("--top", type=int, default=20, help="how many new candidates to print")
    p.set_defaults(fn=cmd_candidates)

    p = sub.add_parser("queue", help="list stored candidates in a state, best first")
    p.add_argument("--state", default="new", choices=candidate_store.STATES)
    p.add_argument("--limit", type=int, default=50)
    p.add_argument("--min-score", type=float, default=0.0)
    p.set_defaults(fn=cmd_queue)

    p = sub.add_parser("triage", help="judge new candidates with the local model")
    p.add_argument("--model", default="qwen3.5:9b")
    p.add_argument("--host", default="http://localhost:11434")
    p.add_argument("--think", action="store_true", help="let the model think first (slower)")
    p.add_argument("--samples", type=int, default=1, help="votes per candidate (self-consistency)")
    p.add_argument("--threshold", type=float, default=0.5, help="min P(real bug) to keep")
    p.add_argument("--min-score", type=float, default=0.2)
    p.add_argument("--limit", type=int, default=500)
    p.add_argument("--no-cache", action="store_true")
    p.set_defaults(fn=cmd_triage)

    def sandbox_opts(p):
        p.add_argument("--sandbox", default="docker", choices=["docker", "local-unsafe"],
                       help="where repo code runs (local-unsafe: no isolation, for machines without Docker)")
        p.add_argument("--image", default="bughunter-sandbox:latest", help="docker image (see sandbox-build)")
        p.add_argument("--python", help="interpreter for local-unsafe (default: this one)")

    def llm_opts(p):
        p.add_argument("--model", default="qwen3.5:9b")
        p.add_argument("--host", default="http://localhost:11434")
        p.add_argument("--think", action="store_true", help="let the model think first (slower)")
        p.add_argument("--num-ctx", type=int, default=16384)
        p.add_argument("--no-cache", action="store_true")

    p = sub.add_parser("scan", help="LLM reads functions (riskiest files first) and stores suspicions as candidates")
    llm_opts(p)
    p.add_argument("--max-functions", type=int, default=200)
    p.add_argument("--threshold", type=float, default=0.6, help="min confidence to store a suspicion")
    p.add_argument("--include-tests", action="store_true")
    p.set_defaults(fn=cmd_scan)

    p = sub.add_parser("sandbox-build", help="build the docker sandbox image with the repo's dependencies")
    p.add_argument("--image", default="bughunter-sandbox:latest")
    p.add_argument("--python-version", default="3.12")
    p.set_defaults(fn=cmd_sandbox_build)

    p = sub.add_parser("test", help="run the repo's tests in the sandbox")
    p.add_argument("selectors", nargs="*")
    p.add_argument("--timeout", type=float, default=900)
    p.add_argument("--lines", type=int, default=40)
    sandbox_opts(p)
    p.set_defaults(fn=cmd_test)

    p = sub.add_parser("investigate", help="write a failing test and a fix for triaged candidates")
    llm_opts(p)
    sandbox_opts(p)
    p.add_argument("--test-attempts", type=int, default=3)
    p.add_argument("--fix-attempts", type=int, default=3)
    p.add_argument("--max-tokens", type=int, default=60_000, help="token budget per candidate")
    p.add_argument("--wall-seconds", type=float, default=900, help="time budget per candidate")
    p.add_argument("--no-suite", action="store_true", help="skip the full suite during attempts (the gate still runs it)")
    p.add_argument("--limit", type=int, default=20)
    p.set_defaults(fn=cmd_investigate)

    p = sub.add_parser("verify", help="run the verification gate on proposed fixes")
    sandbox_opts(p)
    p.add_argument("--repeat", type=int, default=2, help="runs of each new test per side (flakiness)")
    p.set_defaults(fn=cmd_verify)

    p = sub.add_parser("pr", help="open draft PRs for verified candidates (dry run unless --push)")
    p.add_argument("ids", nargs="*", type=int, help="candidate ids (default: all verified)")
    p.add_argument("--push", action="store_true", help="really push the branch and open the PR")
    p.add_argument("--base", help="base branch (default: the repo's default branch)")
    p.add_argument("--remote", default="origin")
    p.set_defaults(fn=cmd_pr)

    args = parser.parse_args(argv)
    try:
        return args.fn(args)
    except SandboxError as exc:
        print(f"sandbox error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
