"""Agents: whatever edits the files in a task's worktree.

The fleet only needs `work(task, worktree, git)`: change files in `worktree`, report whether
it managed to. Committing, merging and testing are the fleet's job, so any agent - a replay of
real history, an LLM, a shell script - plugs in the same way.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from .gitops import Git
from .tasks import Task


@dataclass
class AgentResult:
    ok: bool
    note: str = ""
    conflicted: list[str] = field(default_factory=list)


class Agent(Protocol):
    name: str

    def work(self, task: Task, worktree: Path, git: Git) -> AgentResult: ...


class ReplayAgent:
    """Re-applies the real commit that resolved a task, on whatever base it is given.

    The change is replayed with a three-way merge against the commit's own parent, which is
    exactly what `git cherry-pick` does. On the commit's real parent that reproduces history
    byte for byte; on an older base it fails precisely when the change was written on top of
    lines the base does not have yet.
    """

    name = "replay"

    def work(self, task: Task, worktree: Path, git: Git) -> AgentResult:
        if task.commit is None:
            return AgentResult(False, f"task {task.id} has no commit to replay")
        head = git.out("rev-parse", "HEAD", cwd=worktree)
        result = git.replay(task.commit, head)
        if not result.clean:
            return AgentResult(
                False,
                "the change does not apply to this base: it edits lines this base lacks",
                result.conflicted,
            )
        git.run("read-tree", "--reset", "-u", result.tree, cwd=worktree)
        return AgentResult(True)
