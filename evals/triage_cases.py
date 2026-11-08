"""Labelled triage cases: real bugs and decoys (flagged but correct).

Each case is one file. `expect` lists (rule, enclosing symbol, is_real_bug) for
the findings that are scored; any other findings in the file are ignored.
Keep comments neutral: no "BUG" hints, but decoys may carry the kind of comment
a real developer would leave for intentional code.
"""

CASES = [
    # ---------------------------------------------------------------- real bugs
    {
        "name": "mutable_default_accumulates",
        "source": '''
def add_tag(tag, tags=[]):
    """Return a list of tags including `tag`."""
    tags.append(tag.lower())
    return tags


def label(post):
    post["tags"] = add_tag(post["category"])
    return post
''',
        "expect": [("B006", "add_tag", True)],
    },
    {
        "name": "undefined_name_typo",
        "source": '''
import json


def parse_response(raw):
    try:
        response = json.loads(raw)
    except ValueError:
        return None
    if "error" in response:
        raise RuntimeError(reponse["error"])
    return response["data"]
''',
        "expect": [("F821", "parse_response", True)],
    },
    {
        "name": "closure_captures_loop_variable",
        "source": '''
def make_multipliers(factors):
    funcs = []
    for f in factors:
        funcs.append(lambda x: x * f)
    return funcs


def apply_all(value, factors):
    return [fn(value) for fn in make_multipliers(factors)]
''',
        "expect": [("B023", "make_multipliers", True)],
    },
    {
        "name": "index_at_len",
        "source": '''
def last_reading(readings):
    """Most recent reading, or None if there are none."""
    if not readings:
        return None
    return readings[len(readings)]
''',
        "expect": [("H002", "last_reading", True)],
    },
    {
        "name": "le_len_bound",
        "source": '''
def checksum(data):
    total = 0
    i = 0
    while i <= len(data):
        total = (total + data[i]) % 256
        i += 1
    return total
''',
        "expect": [("H003", "checksum", True)],
    },
    {
        "name": "loop_var_overrides_iterable",
        "source": '''
def flatten(groups):
    out = []
    for items in groups:
        for items in items:
            out.append(items)
    return out


def member_counts(groups):
    counts = {}
    for group in groups:
        for group in group.members:
            counts.setdefault("_sizes", []).append(group.size)
        counts[group.name] = len(group.members)
    return counts
''',
        # flatten() rebinds too, but the inner iterator already exists, so it is correct.
        "expect": [("B020", "flatten", False), ("B020", "member_counts", True)],
    },
    {
        "name": "result_discarded",
        "source": '''
def normalize(name):
    return " ".join(name.split()).title()


def register(users, name):
    cleaned = normalize(name)
    if name in users:
        raise ValueError(f"duplicate user {name}")
    users.add(name)
    return name
''',
        "expect": [("F841", "register", True)],
    },
    {
        "name": "bare_except_swallows_and_returns_none",
        "source": '''
import time


def fetch_with_retry(client, url, attempts=3):
    for attempt in range(attempts):
        try:
            return client.get(url).json()
        except:
            time.sleep(2 ** attempt)


def get_user_count(client):
    return fetch_with_retry(client, "/users/count")["count"]
''',
        "expect": [("E722", "fetch_with_retry", True)],
    },
    {
        "name": "file_handle_leak",
        "source": '''
def read_config_lines(path):
    lines = []
    fh = open(path)
    for line in fh:
        if line.strip() and not line.startswith("#"):
            lines.append(line.strip())
    return lines


def load_all(paths):
    return {p: read_config_lines(p) for p in paths}
''',
        "expect": [("SIM115", "read_config_lines", True)],
    },
    {
        "name": "comparison_instead_of_assignment",
        "source": '''
class Counter:
    def __init__(self):
        self.value = 0


def bump(counter, limit=100):
    counter.value += 1
    if counter.value > limit:
        counter.value == 0
    return counter.value
''',
        "expect": [("B015", "bump", True)],
    },
    {
        "name": "is_with_string_literal",
        "source": '''
def is_done(job):
    status = job.status.strip().lower()
    return status is "done"
''',
        "expect": [("F632", "is_done", True)],
    },
    {
        "name": "dangling_task",
        "source": '''
import asyncio


async def send_metrics(client, batch):
    await client.post("/metrics", json=batch)


async def handle(client, request):
    asyncio.create_task(send_metrics(client, request.metrics))
    return {"ok": True}
''',
        "expect": [("RUF006", "handle", True)],
    },
    {
        "name": "builtin_shadowed_then_called",
        "source": '''
def describe(obj, type="item"):
    if type(obj) is dict:
        return f"{type} with {len(obj)} fields"
    return f"{type}: {obj!r}"
''',
        "expect": [("A002", "describe", True)],
    },
    {
        "name": "logging_too_few_args",
        "source": '''
import logging

log = logging.getLogger(__name__)


def report_transfer(src, dst, nbytes):
    log.info("copied %d bytes from %s to %s", nbytes, src)
    return nbytes
''',
        "expect": [("PLE1206", "report_transfer", True)],
    },
    # ------------------------------------------------------------------ decoys
    {
        "name": "decoy_consume_body",
        "source": '''
class Response:
    def __init__(self, raw):
        self.raw = raw
        self._content = None

    @property
    def content(self):
        if self._content is None:
            self._content = self.raw.read()
        return self._content

    def close(self):
        self.raw.close()


def release(resp):
    # Read the body so the underlying connection can go back to the pool.
    resp.content
    resp.close()
''',
        "expect": [("B018", "release", False)],
    },
    {
        "name": "decoy_readonly_mutable_default",
        "source": '''
DEFAULTS = {"timeout": 10}


def get_option(name, overrides={}):
    if name in overrides:
        return overrides[name]
    return DEFAULTS.get(name)
''',
        "expect": [("B006", "get_option", False)],
    },
    {
        "name": "decoy_worker_collects_errors",
        "source": '''
def run_jobs(jobs):
    """Run every job; failures are collected and reported, never fatal."""
    results, errors = [], []
    for job in jobs:
        try:
            results.append(job.run())
        except Exception as exc:
            errors.append((job, exc))
    return results, errors
''',
        "expect": [("BLE001", "run_jobs", False)],
    },
    {
        "name": "decoy_todo",
        "source": '''
def list_projects(api):
    # TODO: support pagination once the API returns more than 100 projects.
    return api.get("/projects")["items"]
''',
        "expect": [("FIX002", "list_projects", False)],
    },
    {
        "name": "decoy_best_effort_cleanup",
        "source": '''
import os


def remove_quietly(path):
    try:
        os.remove(path)
    except Exception:
        pass  # best effort: the file may already be gone, and callers don't care
''',
        "expect": [("S110", "remove_quietly", False)],
    },
    {
        "name": "decoy_prefix_sums",
        "source": '''
def prefix_sums(xs):
    sums = [0] * (len(xs) + 1)
    for i in range(1, len(xs) + 1):
        sums[i] = sums[i - 1] + xs[i - 1]
    return sums
''',
        "expect": [("H001", "prefix_sums", False)],
    },
    {
        "name": "decoy_setattr_constant",
        "source": '''
class Settings:
    pass


def configure(settings, debug):
    setattr(settings, "debug", bool(debug))
    return settings
''',
        "expect": [("B010", "configure", False)],
    },
    {
        "name": "decoy_loop_var_overwritten",
        "source": '''
def read_names(lines):
    names = []
    for line in lines:
        line = line.strip()
        if line:
            names.append(line)
    return names
''',
        "expect": [("PLW2901", "read_names", False)],
    },
    {
        "name": "decoy_le_len_slice",
        "source": '''
def take(xs, n):
    """First n items; n may equal len(xs)."""
    if 0 <= n <= len(xs):
        return xs[:n]
    raise ValueError("n out of range")


def chunk(xs, i):
    if i <= len(xs):
        return xs[i:]
    return []
''',
        "expect": [],
    },
    {
        "name": "decoy_unused_exception_variable",
        "source": '''
import logging

log = logging.getLogger(__name__)


def safe_ratio(a, b):
    try:
        return a / b
    except ZeroDivisionError as exc:
        log.warning("division by zero, returning 0")
        return 0
''',
        "expect": [("F841", "safe_ratio", False)],
    },
    {
        "name": "decoy_eq_none",
        "source": '''
def or_default(value, default):
    if value == None:
        return default
    return value
''',
        "expect": [("E711", "or_default", False)],
    },
    {
        "name": "decoy_shadow_id_unused",
        "source": '''
def fetch_user(db, id):
    row = db.get("users", id)
    return row["name"] if row else None
''',
        "expect": [("A002", "fetch_user", False)],
    },
    {
        "name": "decoy_exact_type_eq",
        "source": '''
class Point:
    def __init__(self, x, y):
        self.x, self.y = x, y

    def __eq__(self, other):
        # Subclasses compare unequal on purpose.
        if type(other) != type(self):
            return NotImplemented
        return (self.x, self.y) == (other.x, other.y)

    def __hash__(self):
        return hash((self.x, self.y))
''',
        "expect": [("E721", "Point.__eq__", False)],
    },
    {
        "name": "decoy_fixme_performance",
        "source": '''
def find_duplicates(items):
    # FIXME: quadratic; fine for the small lists we get today.
    dupes = []
    for i, a in enumerate(items):
        if a in items[i + 1:] and a not in dupes:
            dupes.append(a)
    return dupes
''',
        "expect": [("FIX001", "find_duplicates", False)],
    },
    {
        "name": "decoy_le_len_with_guard",
        "source": '''
def window(xs, start, size):
    end = start + size
    if end <= len(xs):
        return xs[start:end]
    return xs[start:]
''',
        "expect": [],
    },
]
