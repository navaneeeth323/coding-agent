# bughunter

An agent that goes through a Python codebase, finds real bugs, and opens draft pull requests that fix them. It is built to run on a small local model (by default `qwen3.5:9b` through Ollama).

## Core idea

A weak model with good engineering around it beats a strong model with none. Two rules drive the design:

1. **Spend LLM tokens only where cheaper deterministic tools can't do the job.**
2. **Never open a PR that hasn't been verified by running code.**

## How it works: a funnel, not a chatbot

```
 repo ──► Index ──► Candidates ──► Scan ──► Triage ──► Investigate ──► Verify ──► PR
         (no LLM)   (no LLM)      (LLM)     (LLM)      (LLM + tests)  (no LLM)  (no LLM)
```

Each stage cuts down how many items reach the next, so the expensive stages only see a few promising leads.

1. **Index.** Parses the repo with tree-sitter into a symbol map (functions, classes, signatures, call sites) stored in SQLite, plus a compact ranked repo map.
2. **Candidates.** Bug leads from cheap sources: ruff, tree-sitter heuristics (bare `except`, off-by-one-prone ranges, unchecked `None`, and so on) and git history hotspots.
3. **Scan.** The model reads functions one at a time, riskiest files first, with the docstring, the class's `__init__` and the contracts of the functions it calls. It catches bugs where valid code contradicts its intent, which analyzers can't see. Each suspicion becomes an ordinary candidate.
4. **Triage.** A separate structured-output call judges each candidate: is it a real bug, for what input, with what confidence. Most leads are dropped here.
5. **Investigate.** For the survivors, a fixed two-step pipeline: write a test that fails on the current code, then rewrite the flagged function so it passes. Test output is fed back on retries, and each candidate has a hard budget (calls, tokens, wall clock).
6. **Verify.** On fresh copies of the repo, the fix is accepted only if:
   - the new test fails before the fix and passes after it (each checked twice);
   - the existing test suite has no new failures;
   - ruff (and mypy, if installed) report nothing new.
7. **PR.** Commits the fix and test on a new `bughunter/...` branch in a separate git worktree and opens a **draft** PR through the GitHub REST API. It never pushes to the default branch, checks for duplicate issues and PRs first, and is a dry run unless you pass `--push`.

Every candidate moves through a persisted state machine (`new → triaged → investigating → verified → pr_opened`, or `rejected`), so runs are resumable. Every model call is cached and logged, so you can debug prompts by reading traces.

## Requirements

- Python 3.9+
- [Ollama](https://ollama.com) with a model pulled, for example `ollama pull qwen3.5:9b`
- Docker, for the default sandbox (recommended for any code you don't trust)
- A GitHub token (`GITHUB_TOKEN` or `GH_TOKEN`) with Contents and Pull requests write access, only for opening PRs

## Install

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'
```

## Usage

Run the stages in order against a target repository:

```bash
bughunter --repo path/to/repo index          # build the symbol index
bughunter --repo path/to/repo candidates     # analyzers, heuristics, git hotspots
bughunter --repo path/to/repo scan           # LLM reads functions, adds suspicions
bughunter --repo path/to/repo triage         # LLM judges each candidate
bughunter --repo path/to/repo sandbox-build  # bake the repo's dependencies into the Docker image
bughunter --repo path/to/repo investigate    # failing test + fix
bughunter --repo path/to/repo verify         # the verification gate
bughunter --repo path/to/repo pr             # dry run: show the PR it would open
bughunter --repo path/to/repo pr --push      # push the branch and open a draft PR
```

- Use `bughunter queue` to list candidates in a given state.
- Use `--model` and `--host` to point at a different Ollama model or server.
- Without Docker, pass `--sandbox local-unsafe` to run tests in a subprocess with a scrubbed environment. That's only appropriate for code you trust.

## Evaluation

`evals/benchmark/` is a toy project (`stockroom`) with 15 planted bugs (easy, medium and hard) and 7 decoys: code that looks buggy but is correct. The answer key (`bugs.json`) and hidden tests are kept outside the project the agent sees.

```bash
python -m evals.benchmark_eval --check   # check the benchmark itself is sound
python -m evals.benchmark_eval           # run the full pipeline and score it
```

The run reports precision (verified fixes that fix a planted bug), recall by difficulty, false positives on decoys, and cost per fixed bug.

## Tests

```bash
.venv/bin/pytest
```
