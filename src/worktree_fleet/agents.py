"""Agents: whatever edits the files in a task's worktree.

The fleet only needs `work(task, worktree, git)`: change files in `worktree`, report whether
it managed to. Committing, merging and testing are the fleet's job, so any agent - a replay of
real history, an LLM, a shell script - plugs in the same way.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

from .gitops import Git
from .tasks import Task


@dataclass
class AgentResult:
    """What an agent reports. When `ok` is false, `conflict` says why: True means the work
    collides with the base it was given (a conflict, attributable to parallelism); False means
    the agent itself failed (an error, attributable to the agent)."""

    ok: bool
    note: str = ""
    conflicted: list[str] = field(default_factory=list)
    conflict: bool = False


class Agent(Protocol):
    name: str

    def work(self, task: Task, worktree: Path, git: Git) -> AgentResult: ...


@runtime_checkable
class InMemoryAgent(Protocol):
    """An agent that can also produce its result as a tree, without a checkout."""

    name: str

    def work(self, task: Task, worktree: Path, git: Git) -> AgentResult: ...

    def work_on_tree(self, task: Task, start: str, git: Git) -> tuple[str | None, AgentResult]: ...


class ReplayAgent:
    """Re-applies the real commit that resolved a task, on whatever base it is given.

    The change is replayed with a three-way merge against the commit's own parent, which is
    exactly what `git cherry-pick` does. On the commit's real parent that reproduces history
    byte for byte; on an older base it fails precisely when the change was written on top of
    lines the base does not have yet.
    """

    name = "replay"

    def work(self, task: Task, worktree: Path, git: Git) -> AgentResult:
        head = git.out("rev-parse", "HEAD", cwd=worktree)
        tree, result = self.work_on_tree(task, head, git)
        if tree is not None:
            git.run("read-tree", "--reset", "-u", tree, cwd=worktree)
        return result

    def work_on_tree(self, task: Task, start: str, git: Git) -> tuple[str | None, AgentResult]:
        if task.commit is None:
            return None, AgentResult(False, f"task {task.id} has no commit to replay")
        result = git.replay(task.commit, start)
        if not result.clean:
            return None, AgentResult(
                False,
                "the change does not apply to this base: it edits lines this base lacks",
                result.conflicted,
                conflict=True,
            )
        return result.tree, AgentResult(True)
