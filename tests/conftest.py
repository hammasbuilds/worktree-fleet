"""Fixtures: small throwaway git repositories built commit by commit."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from worktree_fleet.gitops import Git
from worktree_fleet.suite import SuiteConfig

PADDING = "".join(f"# filler line {i}\n" for i in range(12))


class RepoBuilder:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.mkdir(parents=True, exist_ok=True)
        self._git("init", "-q", "-b", "main")
        self._git("config", "user.name", "Fixture")
        self._git("config", "user.email", "fixture@example.invalid")
        self._git("config", "commit.gpgsign", "false")
        self._git("config", "core.autocrlf", "false")
        self.git = Git(path)

    def _git(self, *args: str) -> str:
        proc = subprocess.run(
            ["git", *args], cwd=self.path, capture_output=True, text=True, check=True
        )
        return proc.stdout.strip()

    def commit(self, message: str, files: dict[str, str | None]) -> str:
        """Write (or delete, for None) files and commit them. Returns the commit id."""
        for name, content in files.items():
            target = self.path / name
            if content is None:
                target.unlink()
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content.encode("utf-8"))
        self._git("add", "-A")
        self._git("commit", "-q", "--allow-empty", "-m", message)
        return self._git("rev-parse", "HEAD")

    def checkout(self, ref: str) -> None:
        self._git("checkout", "-q", ref)

    def show(self, rev: str, path: str) -> str | None:
        """A file's content at `rev`, or None if it does not exist there."""
        proc = subprocess.run(["git", "show", f"{rev}:{path}"], cwd=self.path, capture_output=True)
        return proc.stdout.decode("utf-8") if proc.returncode == 0 else None


@pytest.fixture
def repo(tmp_path: Path) -> RepoBuilder:
    return RepoBuilder(tmp_path / "repo")


@pytest.fixture
def suite_config() -> SuiteConfig:
    """Run a fixture repo's tests with this interpreter (pytest is a dev dependency)."""
    return SuiteConfig(python=sys.executable, args=["-q"], pythonpath=["."], timeout=120)


@pytest.fixture
def semantic_history(repo: RepoBuilder) -> tuple[RepoBuilder, str, list[str]]:
    """Two changes written independently from the same base.

    c1 renames greet() to hello(); c2 adds a caller of greet() far down the file. Each
    passes its tests alone; together they merge cleanly and break.
    """
    base = repo.commit(
        "base",
        {
            "mod.py": 'def greet(name):\n    return "hi " + name\n' + PADDING,
            "test_mod.py": "from mod import greet\n\n\ndef test_greet():\n"
            '    assert greet("a") == "hi a"\n',
        },
    )
    c1 = repo.commit(
        "rename greet to hello",
        {
            "mod.py": 'def hello(name):\n    return "hi " + name\n' + PADDING,
            "test_mod.py": "from mod import hello\n\n\ndef test_hello():\n"
            '    assert hello("a") == "hi a"\n',
        },
    )
    repo.checkout(base)
    c2 = repo.commit(
        "add shout",
        {
            "mod.py": 'def greet(name):\n    return "hi " + name\n'
            + PADDING
            + "\n\ndef shout(name):\n    return greet(name).upper()\n",
            "test_shout.py": "from mod import shout\n\n\ndef test_shout():\n"
            '    assert shout("a") == "HI A"\n',
        },
    )
    return repo, base, [c1, c2]
