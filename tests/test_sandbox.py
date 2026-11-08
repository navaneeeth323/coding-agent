import os
import textwrap
from pathlib import Path

import pytest

from bughunter.sandbox import DockerSandbox, LocalSandbox, SandboxError, Workspace, run_snippet, run_tests


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "src_repo"
    (root / "pkg").mkdir(parents=True)
    (root / "pkg/__init__.py").write_text("")
    (root / "pkg/calc.py").write_text("def add(a, b):\n    return a + b\n")
    (root / "tests").mkdir()
    (root / "tests/test_calc.py").write_text(textwrap.dedent("""
        import pytest
        from pkg.calc import add

        def test_add():
            assert add(1, 2) == 3

        @pytest.mark.parametrize("a", [1, 2])
        def test_param(a):
            assert add(a, 0) == a
        """))
    (root / ".bughunter").mkdir()
    (root / ".bughunter/state.db").write_text("x")
    return root


@pytest.fixture
def sandbox():
    return LocalSandbox(allow_unsafe=True)


def test_local_requires_opt_in():
    with pytest.raises(SandboxError):
        LocalSandbox()


def test_workspace_is_a_copy(repo):
    with Workspace(repo) as ws:
        assert not (ws.path / ".bughunter").exists()  # state stays behind
        ws.write("pkg/calc.py", "def add(a, b):\n    return a - b\n")
        assert "a + b" in (repo / "pkg/calc.py").read_text()
        path = ws.path
    assert not path.exists()


def test_run_tests_outcomes(repo, sandbox):
    with Workspace(repo) as ws:
        run = run_tests(sandbox, ws.path)
        assert run.exit_code == 0 and run.collection_ok
        assert run.ids("passed") == [
            "tests.test_calc::test_add", "tests.test_calc::test_param[1]", "tests.test_calc::test_param[2]",
        ]
        assert not (ws.path / ".bughunter-junit.xml").exists()

        ws.write("pkg/calc.py", "def add(a, b):\n    return a - b\n")
        run = run_tests(sandbox, ws.path, ["tests/test_calc.py::test_add"])
        assert run.exit_code == 1 and run.broken == ["tests.test_calc::test_add"]
        assert "assert -1 == 3" in run.failure_text()

        ws.write("tests/test_broken.py", "import nonexistent_module\n")
        run = run_tests(sandbox, ws.path)
        assert not run.collection_ok and run.ids("error") == ["tests.test_broken"]


def test_workspace_code_shadows_installed_package(repo, sandbox):
    # The copy's modules must win even if the original is installed (e.g. editable) in the venv.
    with Workspace(repo) as ws:
        r = run_snippet(sandbox, ws.path, "import pkg, os; print(os.path.dirname(pkg.__file__))")
        assert r.exit_code == 0 and Path(r.output.strip()).resolve() == (ws.path / "pkg").resolve()


def test_timeout_and_scrubbed_env(repo, sandbox, monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "secret")
    with Workspace(repo) as ws:
        r = run_snippet(sandbox, ws.path, "import os; print(os.environ.get('GITHUB_TOKEN'))")
        assert r.output.strip() == "None"
        r = run_snippet(sandbox, ws.path, "import time; time.sleep(30)", timeout=0.5)
        assert r.timed_out and r.seconds < 10


def test_docker_command_is_locked_down(tmp_path, monkeypatch):
    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/docker")
    cmd = DockerSandbox().command(tmp_path, ["python", "-V"], "n")
    joined = " ".join(cmd)
    assert "--network none" in joined and "--cap-drop ALL" in joined and "--pids-limit" in joined
    assert f"{tmp_path.resolve()}:/work" in joined and cmd[-2:] == ["python", "-V"]
    assert not any("TOKEN" in c for c in cmd) and os.environ.get("HOME") not in joined
