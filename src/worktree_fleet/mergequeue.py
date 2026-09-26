"""The merge queue: integrate branches one at a time, test every candidate, keep main green.

A candidate is the three-way merge of main and a task branch, built in memory. A conflict
rejects it without touching a working tree. A clean candidate is checked out into the
queue's own worktree and tested; it is accepted only if no test fails that was passing on
main. That second gate is what catches a *semantic* conflict: a merge git calls clean whose
result is broken.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .gitops import Git
from .suite import SuiteRunner

ACCEPTED = "accepted"
TEXTUAL = "textual"
SEMANTIC = "semantic"
AGENT_FAILED = "agent-failed"


@dataclass
class Integration:
    """The outcome of offering one branch to the queue."""

    outcome: str
    candidate: str | None = None
    conflicted: list[str] = field(default_factory=list)
    new_failures: list[str] = field(default_factory=list)


class MergeQueue:
    """Serialises integration into a single main line.

    `ignore` holds test ids that must never block a merge - tests already known to be broken
    or flaky. Failures present on main before a candidate are never blamed on it.
    """

    def __init__(
        self,
        git: Git,
        main: str,
        ref: str,
        runner: SuiteRunner | None = None,
        worktree: Path | None = None,
        ignore: set[str] | None = None,
    ) -> None:
        if runner is not None and worktree is None:
            raise ValueError("a queue that runs tests needs a worktree to run them in")
        self.git = git
        self.main = main
        self.ref = ref
        self.runner = runner
        self.worktree = worktree
        self.ignore = set(ignore or ())
        self.flaky: set[str] = set()
        self.main_failures: set[str] = set()
        self.test_runs = 0
        git.update_ref(ref, main)
        if runner is not None and worktree is not None:
            baseline = runner.result(main, worktree)
            self.main_failures = set(baseline.failed)
            self.flaky |= baseline.flaky

    def integrate(self, branch: str, label: str) -> Integration:
        merged = self.git.merge(self.main, branch)
        if not merged.clean:
            return Integration(TEXTUAL, conflicted=merged.conflicted)
        candidate = self.git.commit_tree(merged.tree, [self.main, branch], f"fleet: merge {label}")
        failures: set[str] = set()
        if self.runner is not None and self.worktree is not None:
            result = self.runner.result(candidate, self.worktree)
            self.test_runs += 1
            self.flaky |= result.flaky
            failures = set(result.failed)
            new = failures - self.main_failures - self.ignore - self.flaky
            if new:
                return Integration(SEMANTIC, candidate=candidate, new_failures=sorted(new))
        self.main = candidate
        self.main_failures = failures
        self.git.update_ref(self.ref, candidate)
        return Integration(ACCEPTED, candidate=candidate)
