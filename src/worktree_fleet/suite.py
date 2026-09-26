"""Run a repository's test suite on a tree and record per-test outcomes.

Results are cached by *tree id*: two merge candidates with identical content share one run,
which is what makes testing every step of every simulated queue affordable. A tree that
shows failures is run a second time; a test that fails once and passes once is recorded as
flaky rather than as a failure, so a flaky test can never be blamed on a merge.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
import time
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from .gitops import Git

# Pseudo-test ids for failures that are not attributable to one test.
SUITE_ERROR = "<suite-error>"
SUITE_TIMEOUT = "<suite-timeout>"
PSEUDO_IDS = frozenset({SUITE_ERROR, SUITE_TIMEOUT})


@dataclass
class SuiteConfig:
    """How to run one repository's tests."""

    python: str
    args: list[str] = field(default_factory=list)
    pythonpath: list[str] = field(default_factory=list)
    timeout: float = 600.0
    env: dict[str, str] = field(default_factory=dict)


@dataclass
class SuiteResult:
    """Per-test outcome of one tree, after the flaky-confirmation rerun."""

    tree: str
    failed: set[str]
    flaky: set[str]
    passed: int
    duration: float
    runs: int
    detail: str = ""

    @property
    def green(self) -> bool:
        return not self.failed

    def to_json(self) -> dict[str, object]:
        return {
            "tree": self.tree,
            "failed": sorted(self.failed),
            "flaky": sorted(self.flaky),
            "passed": self.passed,
            "duration": round(self.duration, 3),
            "runs": self.runs,
            "detail": self.detail,
        }

    @classmethod
    def from_json(cls, data: dict[str, object]) -> SuiteResult:
        return cls(
            tree=str(data["tree"]),
            failed=set(data["failed"]),  # type: ignore[arg-type]
            flaky=set(data["flaky"]),  # type: ignore[arg-type]
            passed=int(data["passed"]),  # type: ignore[arg-type]
            duration=float(data["duration"]),  # type: ignore[arg-type]
            runs=int(data["runs"]),  # type: ignore[arg-type]
            detail=str(data.get("detail", "")),
        )


def environment_fingerprint(python: str) -> str:
    """A short hash of the packages installed next to `python`.

    Part of the cache key: a result recorded under pytest 9 must not be reused after the
    environment is rebuilt with pytest 8.
    """
    proc = subprocess.run(
        [python, "-c", _LIST_DISTS], capture_output=True, text=True, encoding="utf-8"
    )
    if proc.returncode != 0:
        raise RuntimeError(f"cannot run {python}: {proc.stderr.strip()[:300]}")
    return hashlib.sha256(proc.stdout.encode()).hexdigest()[:10]


_LIST_DISTS = (
    "import importlib.metadata as m, sys;"
    "print(sys.version.split()[0]);"
    "print('\\n'.join(sorted(f\"{d.metadata['Name']}=={d.version}\" for d in m.distributions())))"
)


def has_pytest_config(root: Path) -> bool:
    """Whether `root` carries its own pytest configuration."""
    markers = {
        "pytest.ini": None,
        "pyproject.toml": "[tool.pytest",
        "tox.ini": "[pytest]",
        "setup.cfg": "[tool:pytest]",
    }
    for name, needle in markers.items():
        path = root / name
        if path.is_file():
            if needle is None or needle in path.read_text(encoding="utf-8", errors="replace"):
                return True
    return False


def parse_junit(path: Path) -> tuple[set[str], int]:
    """Return (failing test ids, number of passing tests) from a junit XML report."""
    root = ET.parse(path).getroot()
    failed: set[str] = set()
    passed = 0
    for case in root.iter("testcase"):
        test_id = f"{case.get('classname', '')}::{case.get('name', '')}"
        tags = {child.tag for child in case}
        if tags & {"failure", "error"}:
            failed.add(test_id)
        elif "skipped" not in tags:
            passed += 1
    return failed, passed


class SuiteRunner:
    """Runs a test suite in a dedicated worktree, with a tree-keyed on-disk cache."""

    def __init__(self, git: Git, config: SuiteConfig, cache_dir: Path | None) -> None:
        self.git = git
        self.config = config
        self.cache_dir = cache_dir
        self.fresh_runs = 0
        # Tests already confirmed failing by a rerun. A tree whose only failures are these
        # is not rerun again - otherwise one permanently broken test doubles every run.
        self.confirmed: set[str] = set()
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, tree: str) -> Path | None:
        return None if self.cache_dir is None else self.cache_dir / f"{tree}.json"

    def cached(self, tree: str) -> SuiteResult | None:
        path = self._cache_path(tree)
        if path is None or not path.exists():
            return None
        return SuiteResult.from_json(json.loads(path.read_text(encoding="utf-8")))

    def result(self, commit: str, worktree: Path) -> SuiteResult:
        """Test `commit`'s tree, checking it out into `worktree` only on a cache miss."""
        tree = self.git.tree_of(commit)
        hit = self.cached(tree)
        if hit is not None:
            return hit
        self.git.checkout_tree(worktree, commit)
        started = time.perf_counter()
        first, passed, detail = self.run_once_detailed(worktree)
        runs = 1
        # A run that did not complete says nothing about any test. Check out afresh and try
        # again before believing it: under heavy load a run can die for reasons that have
        # nothing to do with the tree.
        while first <= PSEUDO_IDS and first and runs < 3:
            self.git.checkout_tree(worktree, commit)
            first, passed, detail = self.run_once_detailed(worktree)
            runs += 1
        failed, flaky = first, set()
        if first - self.confirmed - PSEUDO_IDS:
            second, passed2, _ = self.run_once_detailed(worktree)
            runs += 1
            failed = first & second
            flaky = first ^ second
            passed = min(passed, passed2)
            self.confirmed |= failed - PSEUDO_IDS
        result = SuiteResult(tree, failed, flaky, passed, time.perf_counter() - started, runs)
        if failed & PSEUDO_IDS:
            result.detail = detail
        self.fresh_runs += runs
        path = self._cache_path(tree)
        if path is not None:
            tmp = path.with_suffix(f".{os.getpid()}.tmp")
            tmp.write_text(json.dumps(result.to_json()), encoding="utf-8")
            os.replace(tmp, path)
        return result

    def run_once(self, worktree: Path) -> tuple[set[str], int]:
        """One pytest invocation. Returns (failing ids, passing count)."""
        failed, passed, _ = self.run_once_detailed(worktree)
        return failed, passed

    def run_once_detailed(self, worktree: Path) -> tuple[set[str], int, str]:
        """One pytest invocation: (failing ids, passing count, tail of its output)."""
        env = dict(os.environ)
        env.pop("VIRTUAL_ENV", None)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["PYTHONHASHSEED"] = "0"
        paths = [str(worktree / p) for p in self.config.pythonpath]
        env["PYTHONPATH"] = os.pathsep.join(paths)
        env.update(self.config.env)
        with tempfile.TemporaryDirectory(prefix="fleet-junit-") as tmp:
            report = Path(tmp) / "report.xml"
            # Pin pytest's rootdir to the worktree. Without it, a repository with no pytest
            # config of its own inherits the nearest one above it, and test ids then embed
            # the worktree's path - so the same test gets a different id in every worktree.
            isolation = [f"--rootdir={worktree}"]
            if not has_pytest_config(worktree):
                blank = Path(tmp) / "pytest.ini"
                blank.write_text("[pytest]\n", encoding="utf-8")
                isolation += ["-c", str(blank)]
            cmd: Sequence[str] = [
                self.config.python,
                "-m",
                "pytest",
                *isolation,
                "-q",
                "-p",
                "no:cacheprovider",
                "-o",
                "addopts=",
                # Without this a single import error in one test module aborts the whole
                # session, and every other test's outcome is lost.
                "--continue-on-collection-errors",
                f"--junitxml={report}",
                *self.config.args,
            ]
            try:
                proc = subprocess.run(
                    cmd,
                    cwd=worktree,
                    env=env,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=self.config.timeout,
                )
            except subprocess.TimeoutExpired:
                return {SUITE_TIMEOUT}, 0, f"no result after {self.config.timeout:.0f}s"
            lines = (proc.stdout + proc.stderr).strip().splitlines()[-15:]
            tail = "\n".join([f"exit {proc.returncode}", *lines])
            if not report.exists():
                return {SUITE_ERROR}, 0, tail
            failed, passed = parse_junit(report)
            # Exit codes 2-4 are interrupted/internal/usage errors: the suite did not run
            # to completion, so a partial pass count must not look like a green tree.
            if proc.returncode in (2, 3, 4) and not failed:
                failed = {SUITE_ERROR}
            if proc.returncode == 5:  # no tests collected at all
                failed = {SUITE_ERROR}
            return failed, passed, tail
