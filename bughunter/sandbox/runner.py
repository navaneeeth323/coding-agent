"""Run commands against a workspace: in Docker (default) or, if explicitly allowed, locally.

Docker runs with no network, no capabilities, capped memory/CPU/processes and
only the workspace mounted, so repo code can't reach credentials or the
internet. The local runner only strips the environment and enforces timeouts;
it exists for machines without Docker and must be opted into.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Protocol, Sequence

MAX_OUTPUT_CHARS = 200_000  # keep the tail; failures are reported last
DEFAULT_IMAGE = "bughunter-sandbox:latest"


class SandboxError(RuntimeError):
    """The sandbox itself failed (not the code under test)."""


@dataclass
class RunResult:
    exit_code: int
    output: str  # stdout and stderr interleaved
    seconds: float
    timed_out: bool = False


def _tail(text: str) -> str:
    return text if len(text) <= MAX_OUTPUT_CHARS else "...(truncated)\n" + text[-MAX_OUTPUT_CHARS:]


class Sandbox(Protocol):
    python: str  # interpreter to put first in commands

    def run(self, workdir: Path, cmd: Sequence[str], timeout: float = 300.0) -> RunResult: ...


class LocalSandbox:
    """Subprocess in the workspace with a scrubbed environment. NOT isolation."""

    def __init__(self, python: str = sys.executable, allow_unsafe: bool = False):
        if not allow_unsafe:
            raise SandboxError("the local sandbox runs repo code unisolated; pass allow_unsafe=True to use it")
        self.python = python

    def _env(self, workdir: Path) -> dict:
        # Nothing from the caller's environment except what's needed to find programs:
        # no tokens, no cloud credentials, no SSH agent.
        return {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "HOME": str(workdir.parent),
            "LANG": os.environ.get("LANG", "C.UTF-8"),
            "PYTHONPATH": os.pathsep.join(str(p) for p in (workdir, workdir / "src") if p.exists()),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
        }

    def run(self, workdir: Path, cmd: Sequence[str], timeout: float = 300.0) -> RunResult:
        started = time.time()
        proc = subprocess.Popen(
            list(cmd), cwd=workdir, env=self._env(workdir), stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True,
        )
        try:
            out, _ = proc.communicate(timeout=timeout)
            timed_out = False
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)  # the whole group: pytest may have spawned children
            out, _ = proc.communicate()
            timed_out = True
        return RunResult(
            exit_code=proc.returncode, output=_tail(out.decode("utf-8", "replace")),
            seconds=time.time() - started, timed_out=timed_out,
        )


class DockerSandbox:
    def __init__(
        self,
        image: str = DEFAULT_IMAGE,
        memory: str = "2g",
        cpus: str = "2",
        pids: int = 512,
        docker: str = "docker",
    ):
        if shutil.which(docker) is None:
            raise SandboxError(f"`{docker}` not found; install Docker or use the local sandbox")
        self.image, self.memory, self.cpus, self.pids, self.docker = image, memory, cpus, pids, docker
        self.python = "python"

    def command(self, workdir: Path, cmd: Sequence[str], name: str) -> List[str]:
        user = [] if not hasattr(os, "getuid") else ["--user", f"{os.getuid()}:{os.getgid()}"]
        return [
            self.docker, "run", "--rm", "--name", name,
            "--network", "none",
            "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
            "--memory", self.memory, "--cpus", self.cpus, "--pids-limit", str(self.pids),
            *user,
            "-e", "HOME=/tmp", "-e", "PYTHONPATH=/work:/work/src",
            "-e", "PYTHONDONTWRITEBYTECODE=1", "-e", "PYTHONHASHSEED=0",
            "-v", f"{workdir.resolve()}:/work", "-w", "/work",
            self.image, *cmd,
        ]

    def run(self, workdir: Path, cmd: Sequence[str], timeout: float = 300.0) -> RunResult:
        name = f"bughunter-{uuid.uuid4().hex[:12]}"
        started = time.time()
        try:
            proc = subprocess.run(
                self.command(workdir, cmd, name), stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            subprocess.run([self.docker, "kill", name], capture_output=True)
            out = (exc.stdout or b"").decode("utf-8", "replace")
            return RunResult(exit_code=-9, output=_tail(out), seconds=time.time() - started, timed_out=True)
        out = proc.stdout.decode("utf-8", "replace")
        # 125-127 come from docker itself (bad flags, missing image), not from the command.
        if proc.returncode in (125, 126, 127) and ("docker:" in out or "Unable to find image" in out):
            raise SandboxError(f"docker run failed ({proc.returncode}): {out.strip()[:500]}")
        return RunResult(exit_code=proc.returncode, output=_tail(out), seconds=time.time() - started)


DOCKERFILE = """\
FROM python:{python_version}-slim
RUN pip install --no-cache-dir pytest
COPY . /src
# Dependencies are installed at build time (network on); tests later run with
# --network none. The repo itself comes from the mounted workspace via PYTHONPATH.
RUN set -e; cd /src; \\
    for f in requirements.txt requirements-dev.txt requirements-test.txt; do \\
        if [ -f "$f" ]; then pip install --no-cache-dir -r "$f"; fi; done; \\
    if [ -f pyproject.toml ] || [ -f setup.py ]; then pip install --no-cache-dir ".[test]" || pip install --no-cache-dir . || true; fi
WORKDIR /work
"""


def build_image(repo: Path, image: str = DEFAULT_IMAGE, python_version: str = "3.12", docker: str = "docker") -> None:
    """Build the sandbox image with the repo's dependencies preinstalled."""
    if shutil.which(docker) is None:
        raise SandboxError(f"`{docker}` not found")
    from .workspace import Workspace  # build from a clean copy: no .git, venvs or state

    with Workspace(repo) as ws:
        (ws.path / ".bughunter.Dockerfile").write_text(DOCKERFILE.format(python_version=python_version))
        proc = subprocess.run(
            [docker, "build", "-t", image, "-f", ".bughunter.Dockerfile", "."], cwd=ws.path,
        )
    if proc.returncode != 0:
        raise SandboxError(f"docker build failed ({proc.returncode})")


def make_sandbox(kind: str = "docker", image: str = DEFAULT_IMAGE, python: Optional[str] = None) -> Sandbox:
    if kind == "docker":
        return DockerSandbox(image=image)
    if kind == "local":
        return LocalSandbox(python=python or sys.executable, allow_unsafe=True)
    raise ValueError(f"unknown sandbox {kind!r}")
