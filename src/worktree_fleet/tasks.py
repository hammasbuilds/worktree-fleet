"""Tasks and footprints: what a unit of work is, and which lines it touches."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .gitops import Hunk


@dataclass(frozen=True)
class Task:
    """One unit of work for one agent.

    `description` is all a real agent (or a pre-start predictor) gets to see. `commit` is set
    only for replayed history: the real change that resolved this task.
    """

    id: str
    description: str
    commit: str | None = None


def load_tasks(path: Path) -> list[Task]:
    """Read tasks from a JSON list of {"id", "description", "commit"?} objects."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"{path}: expected a JSON list of tasks")
    tasks = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict) or "description" not in item:
            raise ValueError(f"{path}: task {i} needs at least a 'description'")
        tasks.append(
            Task(
                id=str(item.get("id", f"task-{i + 1}")),
                description=str(item["description"]),
                commit=item.get("commit"),
            )
        )
    ids = [t.id for t in tasks]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path}: task ids must be unique")
    return tasks


@dataclass
class Footprint:
    """The files, and optionally the line ranges, a task is expected to touch.

    A file with no hunk list is claimed whole: it overlaps anything else touching it.
    """

    files: set[str] = field(default_factory=set)
    hunks: dict[str, list[Hunk]] = field(default_factory=dict)

    def overlaps(self, other: Footprint, margin: int = 1) -> bool:
        return bool(self.overlapping_files(other, margin))

    def overlapping_files(self, other: Footprint, margin: int = 1) -> set[str]:
        """Shared files whose claimed regions come within `margin` lines of each other.

        `margin=1` mirrors git: two edits to adjacent lines conflict even if they do not
        share a line.
        """
        shared = self.files & other.files
        hits = set()
        for path in shared:
            mine = self.hunks.get(path)
            theirs = other.hunks.get(path)
            if mine is None or theirs is None:
                hits.add(path)
            elif any(_near(a, b, margin) for a in mine for b in theirs):
                hits.add(path)
        return hits


def _near(a: Hunk, b: Hunk, margin: int) -> bool:
    a_lo, a_hi = a.span()
    b_lo, b_hi = b.span()
    return a_lo <= b_hi + margin and b_lo <= a_hi + margin
