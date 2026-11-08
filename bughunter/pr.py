"""Stage 6: open a draft PR for a verified candidate.

Uses plain git plus the GitHub REST API (token from GITHUB_TOKEN or GH_TOKEN),
so `gh` isn't required. The fix is committed on a new `bughunter/...` branch
created from HEAD in a temporary worktree; the user's checkout is never touched
and nothing is pushed to the default branch. Without `push=True` it only
reports what it would do.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

from .candidates import store
from .index import RepoIndex
from .sandbox import sha1_text

API = "https://api.github.com"
BRANCH_PREFIX = "bughunter/"


class PRError(RuntimeError):
    pass


@dataclass
class PRPlan:
    candidate_id: int
    branch: str
    title: str
    body: str
    files: dict  # repo-relative path -> new content (fix and test)
    duplicates: List[str] = field(default_factory=list)
    url: Optional[str] = None


def _git(cwd: Path, *args: str, check: bool = True) -> str:
    proc = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    if check and proc.returncode != 0:
        raise PRError(f"git {' '.join(args)} failed: {proc.stderr.strip()[:500]}")
    return proc.stdout


def parse_github_remote(url: str) -> Optional[Tuple[str, str]]:
    m = re.match(r"^(?:https://(?:[^@/]+@)?github\.com/|git@github\.com:|ssh://git@github\.com/)"
                 r"([^/]+)/(.+?)(?:\.git)?/?$", url.strip())
    return (m.group(1), m.group(2)) if m else None


class GitHub:
    def __init__(self, token: Optional[str] = None):
        self.token = token or os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")

    def request(self, method: str, path: str, body: Optional[dict] = None) -> dict:
        if not self.token:
            raise PRError("no GitHub token: set GITHUB_TOKEN (or GH_TOKEN)")
        req = urllib.request.Request(
            API + path, method=method, data=json.dumps(body).encode() if body is not None else None,
            headers={"Authorization": f"Bearer {self.token}", "Accept": "application/vnd.github+json",
                     "X-GitHub-Api-Version": "2022-11-28", "User-Agent": "bughunter"},
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as exc:
            raise PRError(f"GitHub {method} {path}: HTTP {exc.code} {exc.read().decode(errors='replace')[:300]}") from exc
        except (urllib.error.URLError, OSError) as exc:
            raise PRError(f"GitHub {method} {path}: {exc}") from exc

    def search_open(self, owner: str, repo: str, terms: List[str]) -> List[str]:
        q = " ".join([f"repo:{owner}/{repo}", "is:open", *(f'"{t}"' for t in terms)])
        res = self.request("GET", "/search/issues?" + urllib.parse.urlencode({"q": q, "per_page": 10}))
        return [item["html_url"] for item in res.get("items", [])]


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:40] or "fix"


def build_plan(c: dict) -> PRPlan:
    data = json.loads(c["data"])
    t, inv, ver = data.get("triage", {}), data["investigation"], data.get("verification", {})
    where = c["symbol"] or f"line {c['line']}"
    files = {p: f["after"] for p, f in inv["files"].items()}
    files[inv["test"]["path"]] = inv["test"]["content"]

    summary = inv.get("diagnosis") or t.get("intent") or c["message"]
    title = f"Fix {t.get('category', 'bug').replace('_', ' ')} in {where} ({c['path']})"
    checks = "\n".join(f"- {'✅' if ch['ok'] else '❌'} {ch['name']}: {ch['detail']}" for ch in ver.get("checks", []))
    body = f"""\
## Bug

{summary}

- Where: `{c['path']}:{c['line']}` in `{where}`
- Flagged by: {c['source']} `{c['rule']}` ({c['message']})
- Input: {t.get('failing_input', 'see test')}
- Expected: {t.get('expected', '')}
- Actual: {t.get('actual', '')}

## Reproduction

`{inv['test']['path']}` fails on the current code and passes with this fix:

```python
{inv['test']['content'].strip()}
```

## Fix

```diff
{inv['diff'].strip()}
```

## Confidence

- Triage P(real bug): {t.get('p_real', 'n/a')} (model `{t.get('model', '?')}`)
- Verification gate (fresh checkouts, sandboxed):
{checks}

---
Opened as a draft by bughunter, an automated bug-finding agent. Please review before merging.
"""
    return PRPlan(candidate_id=c["id"], branch=f"{BRANCH_PREFIX}{c['id']}-{_slug(where)}",
                  title=title, body=body, files=files)


def local_duplicates(conn, c: dict) -> List[str]:
    rows = conn.execute(
        "SELECT id, data FROM candidates WHERE state = 'pr_opened' AND path = ? AND symbol IS ? AND id != ?",
        (c["path"], c["symbol"], c["id"]),
    ).fetchall()
    return [json.loads(r["data"]).get("pr", {}).get("url") or f"candidate #{r['id']}" for r in rows]


def open_pr(
    index: RepoIndex, candidate_id: int, push: bool = False, base: Optional[str] = None,
    remote: str = "origin", github: Optional[GitHub] = None,
) -> PRPlan:
    conn = index.conn
    row = conn.execute("SELECT * FROM candidates WHERE id = ?", (candidate_id,)).fetchone()
    if row is None:
        raise PRError(f"no candidate #{candidate_id}")
    c = dict(row)
    if c["state"] != "verified":
        raise PRError(f"candidate #{candidate_id} is {c['state']}; only verified candidates get PRs")
    plan = build_plan(c)
    inv = json.loads(c["data"])["investigation"]

    top = Path(_git(index.repo, "rev-parse", "--show-toplevel").strip())
    prefix = index.repo.relative_to(top.resolve()).as_posix()
    prefix = "" if prefix == "." else prefix + "/"
    # The patch was made against the working tree; the PR branch starts from HEAD.
    for path, f in inv["files"].items():
        head = _git(top, "show", f"HEAD:{prefix}{path}", check=False)
        if sha1_text(head) != f["before_sha1"]:
            raise PRError(f"{path} at HEAD differs from the code that was verified; commit or re-run")

    plan.duplicates = local_duplicates(conn, c)
    gh_repo = parse_github_remote(_git(top, "config", "--get", f"remote.{remote}.url", check=False))
    github = github or GitHub()
    if gh_repo and github.token:
        name = (c["symbol"] or "").rsplit(".", 1)[-1]
        plan.duplicates += github.search_open(*gh_repo, [t for t in (name, Path(c["path"]).name) if t])
    if _git(top, "ls-remote", "--heads", remote, plan.branch, check=False).strip():
        plan.duplicates.append(f"branch {plan.branch} already exists on {remote}")
    if not push or plan.duplicates:
        return plan

    if gh_repo is None:
        raise PRError(f"remote {remote!r} is not a GitHub repository")
    base = base or github.request("GET", f"/repos/{gh_repo[0]}/{gh_repo[1]}")["default_branch"]
    if plan.branch == base:
        raise PRError("refusing to push to the base branch")

    tmp = tempfile.mkdtemp(prefix="bughunter-pr-")
    wt = Path(tmp) / "wt"
    _git(top, "worktree", "add", "-q", "-b", plan.branch, str(wt), "HEAD")
    try:
        for path, text in plan.files.items():
            target = wt / prefix / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text, "utf-8")
            _git(wt, "add", "--", f"{prefix}{path}")
        _git(wt, "commit", "-q", "-m", plan.title, "-m", f"Found and verified by bughunter (candidate #{c['id']}).")
        _git(wt, "push", "-q", remote, f"{plan.branch}:{plan.branch}")
    except BaseException:
        _git(top, "worktree", "remove", "--force", str(wt), check=False)
        _git(top, "branch", "-D", plan.branch, check=False)  # so a retry can recreate it
        raise
    finally:
        _git(top, "worktree", "remove", "--force", str(wt), check=False)
        shutil.rmtree(tmp, ignore_errors=True)

    pr = github.request("POST", f"/repos/{gh_repo[0]}/{gh_repo[1]}/pulls", {
        "title": plan.title, "head": plan.branch, "base": base, "body": plan.body, "draft": True,
    })
    plan.url = pr["html_url"]
    store.transition(conn, c["id"], "pr_opened", note=plan.url,
                     data={"pr": {"url": plan.url, "branch": plan.branch, "base": base}})
    return plan
