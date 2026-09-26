"""The fleet: plan waves, run agents in parallel worktrees, integrate through the queue.

Three policies:

* `serial` - one task at a time, each starting from the latest main. Nothing can collide;
  the cost is that N tasks take N rounds.
* `parallel` - every task starts from the same base at once; the queue sorts out collisions
  afterwards. One round, plus whatever has to be redone.
* `predicted` - a predictor guesses which tasks will collide; tasks that are predicted to
  collide go in different waves, and each wave starts from main as the previous wave left it.

A task whose integration fails (the agent could not apply its work, the merge conflicts, or
the merged result breaks tests) is retried: the agent redoes it from the current main. A
mechanical `git rebase` would not help - rebasing a branch onto main produces the same tree as
merging it - so the retry is a redo, and the first attempt counts as wasted work.
"""

from __future__ import annotations

import random
import re
import shutil
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

from .agents import Agent
from .gitops import Git
from .mergequeue import ACCEPTED, AGENT_FAILED, Integration, MergeQueue
from .predict import Predictor, predicted_conflicts
from .tasks import Task

SERIAL = "serial"
PARALLEL = "parallel"
PREDICTED = "predicted"
POLICIES = (SERIAL, PARALLEL, PREDICTED)

# Integration order inside a wave: as agents finish (a live fleet), the order the tasks were
# listed in, or a seeded shuffle (a reproducible stand-in for "whoever finishes first").
ORDERS = ("completion", "listed", "shuffled")

NOOP = "no-op"
REJECTED = "rejected"


@dataclass
class Attempt:
    number: int
    base: str
    outcome: str
    branch: str | None = None
    candidate: str | None = None
    conflicted: list[str] = field(default_factory=list)
    new_failures: list[str] = field(default_factory=list)
    note: str = ""


@dataclass
class TaskRecord:
    task: Task
    wave: int
    attempts: list[Attempt] = field(default_factory=list)

    @property
    def final(self) -> str:
        last = self.attempts[-1].outcome if self.attempts else REJECTED
        return last if last in (ACCEPTED, NOOP) else REJECTED

    @property
    def first(self) -> Attempt:
        return self.attempts[0]


@dataclass
class FleetReport:
    policy: str
    predictor: str | None
    waves: list[list[int]]
    records: list[TaskRecord]
    main: str
    test_runs: int
    integration_order: list[int] = field(default_factory=list)

    @property
    def agent_runs(self) -> int:
        return sum(len(r.attempts) for r in self.records)

    @property
    def redos(self) -> int:
        return sum(len(r.attempts) - 1 for r in self.records)

    @property
    def makespan(self) -> int:
        """Rounds of agent work on the critical path: one per wave, plus one per redo.

        A redo blocks the queue behind it, so it adds a full round.
        """
        return len(self.waves) + self.redos

    def to_json(self) -> dict[str, object]:
        return {
            "policy": self.policy,
            "predictor": self.predictor,
            "waves": [[self.records[i].task.id for i in wave] for wave in self.waves],
            "integration_order": [self.records[i].task.id for i in self.integration_order],
            "main": self.main,
            "agent_runs": self.agent_runs,
            "redos": self.redos,
            "makespan": self.makespan,
            "test_runs": self.test_runs,
            "tasks": [
                {
                    "id": r.task.id,
                    "wave": r.wave,
                    "final": r.final,
                    "attempts": [a.__dict__ for a in r.attempts],
                }
                for r in self.records
            ],
        }


def plan_waves(n: int, conflicts: set[tuple[int, int]]) -> list[list[int]]:
    """Order-preserving wave assignment.

    Task j goes one wave after the latest earlier task it is predicted to conflict with, so
    conflicting tasks never run side by side and still integrate in their original order.
    """
    wave_of: list[int] = []
    for j in range(n):
        wave = 0
        for i in range(j):
            if (i, j) in conflicts:
                wave = max(wave, wave_of[i] + 1)
        wave_of.append(wave)
    waves: list[list[int]] = [[] for _ in range(max(wave_of, default=-1) + 1)]
    for j, wave in enumerate(wave_of):
        waves[wave].append(j)
    return waves


QueueFactory = Callable[[str, str], MergeQueue]


class Fleet:
    """Runs tasks with one agent per git worktree and integrates them through a queue."""

    def __init__(
        self,
        git: Git,
        agent: Agent,
        workdir: Path,
        queue_factory: QueueFactory,
        max_workers: int = 4,
        retries: int = 1,
        keep_worktrees: bool = False,
        order: str = "completion",
        seed: int = 0,
    ) -> None:
        if order not in ORDERS:
            raise ValueError(f"unknown order {order!r}; choose one of {', '.join(ORDERS)}")
        self.git = git
        self.agent = agent
        self.workdir = Path(workdir).resolve()
        self.queue_factory = queue_factory
        self.max_workers = max_workers
        self.retries = retries
        self.keep_worktrees = keep_worktrees
        self.order = order
        self._rng = random.Random(seed)
        self._worktree_lock = threading.Lock()

    def waves_for(
        self, tasks: list[Task], base: str, policy: str, predictor: Predictor | None
    ) -> list[list[int]]:
        if policy == SERIAL:
            return [[i] for i in range(len(tasks))]
        if policy == PARALLEL:
            return [list(range(len(tasks)))] if tasks else []
        if policy == PREDICTED:
            if predictor is None:
                raise ValueError("the 'predicted' policy needs a predictor")
            return plan_waves(len(tasks), set(predicted_conflicts(predictor, tasks, base)))
        raise ValueError(f"unknown policy {policy!r}; choose one of {', '.join(POLICIES)}")

    def run(
        self,
        tasks: list[Task],
        base: str,
        policy: str,
        predictor: Predictor | None = None,
        run_id: str = "run",
    ) -> FleetReport:
        base = self.git.rev_parse(base)
        waves = self.waves_for(tasks, base, policy, predictor)
        queue = self.queue_factory(base, f"refs/fleet/{run_id}/main")
        records = [TaskRecord(task, wave=-1) for task in tasks]
        integrated: list[int] = []
        for w, wave in enumerate(waves):
            start = queue.main
            with ThreadPoolExecutor(max_workers=max(1, self.max_workers)) as pool:
                futures = {pool.submit(self._attempt, tasks[i], start, 1, run_id): i for i in wave}
                if self.order == "completion":
                    # Integrate each branch the moment its agent finishes, as a live fleet does.
                    for future in as_completed(futures):
                        self._settle(queue, records[futures[future]], future.result(), w, run_id)
                        integrated.append(futures[future])
                else:
                    done = {i: f.result() for f, i in futures.items()}
                    sequence = list(wave)
                    if self.order == "shuffled":
                        self._rng.shuffle(sequence)
                    for i in sequence:
                        self._settle(queue, records[i], done[i], w, run_id)
                        integrated.append(i)
        return FleetReport(
            policy=policy,
            predictor=predictor.name if predictor is not None and policy == PREDICTED else None,
            waves=waves,
            records=records,
            main=queue.main,
            test_runs=queue.test_runs,
            integration_order=integrated,
        )

    def _settle(
        self, queue: MergeQueue, record: TaskRecord, attempt: Attempt, wave: int, run_id: str
    ) -> None:
        """Integrate a first attempt, then redo from the current main until accepted."""
        record.wave = wave
        self._integrate(queue, attempt, record.task)
        record.attempts.append(attempt)
        while record.attempts[-1].outcome not in (ACCEPTED, NOOP) and (
            len(record.attempts) <= self.retries
        ):
            redo = self._attempt(record.task, queue.main, len(record.attempts) + 1, run_id)
            self._integrate(queue, redo, record.task)
            record.attempts.append(redo)

    def _integrate(self, queue: MergeQueue, attempt: Attempt, task: Task) -> None:
        if attempt.outcome != "pending" or attempt.branch is None:
            return
        result: Integration = queue.integrate(attempt.branch, task.id)
        attempt.outcome = result.outcome
        attempt.candidate = result.candidate
        attempt.conflicted = result.conflicted
        attempt.new_failures = result.new_failures

    def _attempt(self, task: Task, start: str, number: int, run_id: str) -> Attempt:
        """Give the agent a fresh worktree at `start` and commit whatever it produces."""
        # Flat, run-unique directory names: git names worktree metadata after the basename,
        # and two fleets running at once must never pick the same one.
        path = self.workdir / f"{_slug(run_id)}-{_slug(task.id)}-{number}"
        with self._worktree_lock:
            if path.exists():
                self.git.worktree_remove(path)
                shutil.rmtree(path, ignore_errors=True)
            path.parent.mkdir(parents=True, exist_ok=True)
            self.git.worktree_add(path, start)
        try:
            result = self.agent.work(task, path, self.git)
            if not result.ok:
                return Attempt(
                    number, start, AGENT_FAILED, conflicted=result.conflicted, note=result.note
                )
            branch = self.git.commit_worktree(path, f"{task.id}: {task.description[:60]}")
            if branch is None:
                return Attempt(number, start, NOOP, note="the agent changed nothing")
            self.git.update_ref(f"refs/fleet/{run_id}/tasks/{_slug(task.id)}-{number}", branch)
            return Attempt(number, start, "pending", branch=branch, note=result.note)
        finally:
            if not self.keep_worktrees:
                with self._worktree_lock:
                    self.git.worktree_remove(path)
                    shutil.rmtree(path, ignore_errors=True)


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-") or "task"
