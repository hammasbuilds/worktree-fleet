"""Replay real history as if N agents had started at once, and measure what breaks.

Tasks are the first-parent commits of a real repository that did real work: empty diffs and
housekeeping (branch-sync merges, bot and dependency bumps, version bumps) are left out by a
fixed rule, `housekeeping_reason`. Each task is one agent; the `ReplayAgent` produces the
change that really happened, replayed onto whatever base the agent was given. Every policy
runs the same tasks through a real merge queue that runs the repository's own tests.

Two ways to pick the N tasks of a window, because they answer different questions:

* `consecutive` - N consecutive tasks c1..cN from base B = parent(c1), as if a fleet took the
  next N items off the backlog. A later change was often written on top of an earlier one; its
  replay onto B then fails before any merge (`base-conflict`). That rate is how often a task
  depends on, or edits the same lines as, work still in flight.
* `independent` - N tasks after B whose changes each apply to B on their own (scanning at most
  `lookahead * N` tasks ahead). Every agent could really have produced its change from B, so
  whatever fails here fails in the merge queue itself: a textual conflict between two applicable
  changes, or a clean merge that breaks a test.

Integration order: integrated in history's own order, a consecutive replay can only reproduce
history, and each intermediate main is byte-identical to a real, passing commit. Agents do not
finish in history's order, so the headline runs integrate each wave in a seeded shuffle.

Attribution. A test only counts against a merge if it is not already known to be broken or
flaky: the ignore set of a window is every test failing at B or at any real commit in the
window, plus every test seen to flip on a rerun. Every semantic failure - first attempt or redo
- is then re-tested on the task's branch alone: failing there too is `stale-base` (the change
needed in-flight work it did not have), passing there is an `interaction`.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from .agents import Agent, ReplayAgent
from .fleet import PARALLEL, PREDICTED, SERIAL, Attempt, Fleet, FleetReport
from .gitops import Git
from .mergequeue import BASE_CONFLICT, SEMANTIC, TEXTUAL, MergeQueue
from .predict import PREDICTOR_NAMES, ExcludingPredictor, Predictor, build_predictor
from .suite import (
    PSEUDO_IDS,
    SUITE_ERROR,
    SUITE_TIMEOUT,
    SuiteResult,
    SuiteRunner,
    environment_fingerprint,
)
from .targets import Target
from .tasks import Task

CONSECUTIVE = "consecutive"
INDEPENDENT = "independent"
MODES = (CONSECUTIVE, INDEPENDENT)

# (policy, predictor, integration order within a wave, changelogs merged with `union`)
Spec = tuple[str, str | None, str, bool]
POLICY_SPECS: list[Spec] = [
    (SERIAL, None, "listed", False),
    (PARALLEL, None, "listed", False),
    (PARALLEL, None, "shuffled", False),
    (PARALLEL, None, "shuffled", True),
    (PREDICTED, "description", "shuffled", False),
    (PREDICTED, "description", "shuffled", True),
    (PREDICTED, "oracle-files", "shuffled", False),
    (PREDICTED, "oracle-hunks", "shuffled", False),
]

# git's built-in union driver keeps both sides' lines instead of conflicting: the usual fix
# for changelogs, where every change appends an entry at the same spot.
CHANGELOG_UNION = "".join(
    f"{pattern} merge=union\n" for pattern in ("CHANGES*", "CHANGELOG*", "HISTORY*", "NEWS*")
)

# The housekeeping rule. A commit is housekeeping - not a task anyone would hand an agent -
# if its author is a bot, or its subject is a merge of a release/main branch into another
# (a branch sync, not new work), a dependency or tooling bump, or a version bump. Merges of
# feature branches and of pull requests are tasks, unless the pull request merges a release
# branch.
_BOT_AUTHOR = re.compile(r"\[bot\]|dependabot|pre-commit-ci", re.I)
_SYNC_BRANCH = r"(\S+/)?(stable|main|master|\d+(\.\d+)*\.x|release[\w./-]*|maintenance[\w./-]*)"
_HOUSEKEEPING_SUBJECT = re.compile(
    r"^(merge (remote-tracking )?branch '" + _SYNC_BRANCH + r"'"
    r"|merge pull request #\d+ from \S+/(stable|\d+(\.\d+)*\.x)$"
    r"|bump |\[pre-commit\.ci\]|pre-commit autoupdate"
    r"|(update|upgrade|pin) (dev |development |test |ci )?(dependencies|deps|requirements|pins)"
    r"|(release|prepare|start|bump) (version|release)|release \d|version \d|v?\d+\.\d+(\.\d+)*$)",
    re.I,
)


def housekeeping_reason(author: str, subject: str) -> str | None:
    """Why a commit is housekeeping rather than a task, or None if it is a task."""
    if _BOT_AUTHOR.search(author):
        return "bot author"
    match = _HOUSEKEEPING_SUBJECT.search(subject.strip())
    return f"subject: {match.group(0)!r}" if match else None


def spec_label(spec: Spec) -> str:
    """The name a spec goes by on the command line and in the report."""
    policy, predictor, order, union = spec
    if policy == PREDICTED:
        label = f"predicted:{predictor}"
    elif policy == PARALLEL and order == "listed":
        label = "parallel:history-order"
    else:
        label = policy
    return f"{label}+changelog-union" if union else label


SPECS_BY_LABEL: dict[str, Spec] = {spec_label(s): s for s in POLICY_SPECS}


def parse_specs(text: str) -> list[Spec]:
    """Comma-separated labels, e.g. "serial,parallel,predicted:description"."""
    specs = []
    for label in (part.strip() for part in text.split(",") if part.strip()):
        if label not in SPECS_BY_LABEL:
            raise ValueError(f"unknown policy {label!r}; choose from {', '.join(SPECS_BY_LABEL)}")
        specs.append(SPECS_BY_LABEL[label])
    if not specs:
        raise ValueError("no policies given")
    return specs


@dataclass(frozen=True)
class Window:
    mode: str
    size: int
    index: int
    base: str
    commits: tuple[str, ...]
    scanned: int = 0  # independent mode: tasks looked at to find these N
    dependent: int = 0  # ...of which did not apply to the base


def make_agent(spec: dict | None) -> Agent:
    """`None` or {"kind": "replay"} replays history; {"kind": "ollama", ...} asks a model."""
    if spec is None or spec.get("kind") == "replay":
        return ReplayAgent()
    if spec.get("kind") == "ollama":
        from .llm import OllamaAgent, OllamaClient

        client = OllamaClient(spec["url"], cache_dir=Path(spec["cache"]))
        return OllamaAgent(client, spec["model"])
    raise ValueError(f"unknown agent spec {spec!r}")


@dataclass
class History:
    commits: list[str]  # every first-parent commit with a non-empty change, oldest first
    tasks: list[str]  # ...minus housekeeping
    housekeeping: dict[str, str]  # commit -> reason


def history_tasks(git: Git, ref: str, limit: int) -> History:
    """The last `limit` first-parent commits of `ref`, split into tasks and housekeeping.

    A commit whose diff against its first parent is empty (a merge of already-merged work)
    gives an agent nothing to do and is dropped entirely.
    """
    chain = git.first_parent_chain(ref, limit)
    text = git.out("log", "--no-walk=unsorted", "--format=%H%x00%an%x00%s", *chain)
    meta = {}
    for line in text.splitlines():
        sha, author, subject = line.split("\x00", 2)
        meta[sha] = (author, subject)
    commits, tasks, housekeeping = [], [], {}
    for commit in chain:
        parents = git.parents(commit)
        if not parents or not git.changed_files(parents[0], commit):
            continue
        commits.append(commit)
        reason = housekeeping_reason(*meta.get(commit, ("", "")))
        if reason:
            housekeeping[commit] = reason
        else:
            tasks.append(commit)
    return History(commits, tasks, housekeeping)


def healthy(result: SuiteResult | None) -> bool:
    """A commit whose suite ran to completion with only a handful of failures."""
    if result is None:
        return False
    if SUITE_ERROR in result.failed or SUITE_TIMEOUT in result.failed:
        return False
    return len(result.failed) <= max(5, int(0.02 * result.passed))


def make_windows(
    git: Git,
    tasks: list[str],
    sizes: Iterable[int],
    results: dict[str, SuiteResult],
    mode: str = CONSECUTIVE,
    lookahead: int = 3,
) -> list[Window]:
    """Non-overlapping windows whose base and task commits all test healthy."""
    windows = []
    for size in sizes:
        if mode == CONSECUTIVE:
            for index in range(len(tasks) // size):
                chunk = tuple(tasks[index * size : (index + 1) * size])
                base = git.parents(chunk[0])[0]
                if all(healthy(results.get(c)) for c in (base, *chunk)):
                    windows.append(Window(mode, size, index, base, chunk))
        elif mode == INDEPENDENT:
            windows.extend(_independent_windows(git, tasks, size, results, lookahead))
        else:
            raise ValueError(f"unknown mode {mode!r}; choose one of {', '.join(MODES)}")
    return windows


def _independent_windows(
    git: Git, tasks: list[str], size: int, results: dict[str, SuiteResult], lookahead: int
) -> list[Window]:
    windows: list[Window] = []
    start = 0
    while start < len(tasks):
        base = git.parents(tasks[start])[0]
        if not healthy(results.get(base)):
            start += 1
            continue
        picked: list[str] = []
        dependent = 0
        j = start
        while j < len(tasks) and len(picked) < size and j - start < lookahead * size:
            commit = tasks[j]
            j += 1
            if not healthy(results.get(commit)):
                continue
            if git.replay(commit, base).clean:
                picked.append(commit)
            else:
                dependent += 1
        if len(picked) == size:
            windows.append(
                Window(INDEPENDENT, size, len(windows), base, tuple(picked), j - start, dependent)
            )
        start = j
    return windows


# --- workers (top-level so they pickle on Windows) --------------------------------------


def _test_commits(target: Target, commits: list[str], cache: Path, work: Path) -> dict:
    git = Git(target.path)
    runner = SuiteRunner(git, target.suite(), cache)
    worktree = work / f"pre-{os.getpid()}"
    _fresh_worktree(git, worktree, commits[0])
    try:
        # Failures at real commits are only used to *exclude* tests, so a flaky failure here
        # can only make the measurement more conservative: no confirming rerun is needed.
        return {c: runner.result(c, worktree, expected=None).to_json() for c in commits}
    finally:
        _drop_worktree(git, worktree)


def _run_window(
    target: Target,
    window: Window,
    ignore: list[str],
    cache: Path,
    work: Path,
    specs: list[Spec],
    agent_spec: dict | None = None,
) -> list[dict]:
    git = Git(target.path)
    runner = SuiteRunner(git, target.suite(), cache)
    tasks = [Task(id=c[:10], description=git.message(c), commit=c) for c in window.commits]
    tag = f"{window.mode[0]}{window.size}-w{window.index}-{os.getpid()}"
    qtree = work / f"q-{tag}"
    _fresh_worktree(git, qtree, window.base)
    predictors = {name: build_predictor(name, git) for name in PREDICTOR_NAMES}
    records = []
    try:
        window_info = _window_info(git, tasks, window, predictors)
        seed = int(window.base[:8], 16)
        union_file = work / f"changelog-union-{os.getpid()}.gitattributes"
        union_file.write_text(CHANGELOG_UNION, encoding="utf-8")
        for policy, pred_name, order, union in specs:
            run_id = f"{tag}-{policy}-{pred_name or 'none'}-{order}{'-union' if union else ''}"
            run_git = Git(target.path, attributes_file=union_file) if union else git

            def factory(base: str, ref: str, run_git: Git = run_git) -> MergeQueue:
                return MergeQueue(run_git, base, ref, runner, qtree, set(ignore))

            fleet = Fleet(
                run_git,
                make_agent(agent_spec),
                work / "agents",
                factory,
                max_workers=4,
                order=order,
                seed=seed,
                in_memory=True,
            )
            predictor = predictors[pred_name] if pred_name else None
            if predictor is not None and union:
                predictor = ExcludingPredictor(predictor)
            report = fleet.run(tasks, window.base, policy, predictor, run_id=run_id)
            record = _record(target, window, report, runner, qtree, window_info)
            record["order"] = order
            record["changelog_union"] = union
            record["label"] = spec_label((policy, pred_name, order, union))
            record["agent"] = (agent_spec or {}).get("kind", "replay")
            records.append(record)
            _delete_refs(git, f"refs/fleet/{run_id}/")
    finally:
        _drop_worktree(git, qtree)
    return records


def _window_info(
    git: Git, tasks: list[Task], window: Window, predictors: dict[str, Predictor]
) -> dict:
    """Per-window facts shared by every policy: each task's files and every prediction."""
    files = []
    for task in tasks:
        assert task.commit is not None
        parent = git.parents(task.commit)[0]
        files.append(sorted(git.changed_files(parent, task.commit)))
    predicted = {}
    for name, predictor in predictors.items():
        prints = [predictor.footprint(t, window.base) for t in tasks]
        pairs = []
        for j in range(len(tasks)):
            for i in range(j):
                if prints[i].overlaps(prints[j], predictor.margin):
                    pairs.append([i, j])
        predicted[name] = {"pairs": pairs, "files": [sorted(p.files) for p in prints]}
    return {"files": files, "predicted": predicted}


def classify(attempt: Attempt, runner: SuiteRunner, qtree: Path) -> str | None:
    """Why an attempt failed: base / merge / stale-base / interaction (None otherwise)."""
    if attempt.outcome == BASE_CONFLICT:
        return "base"
    if attempt.outcome == TEXTUAL:
        return "merge"
    if attempt.outcome == SEMANTIC and attempt.branch is not None:
        alone = runner.result(attempt.branch, qtree)
        # A branch whose suite cannot even run alone is broken on its own, whatever the
        # merged tree then fails on.
        broken_alone = alone.failed & (set(attempt.new_failures) | PSEUDO_IDS)
        return "stale-base" if broken_alone else "interaction"
    return None


def _record(
    target: Target,
    window: Window,
    report: FleetReport,
    runner: SuiteRunner,
    qtree: Path,
    info: dict,
) -> dict:
    data = report.to_json()
    for rec, task_json in zip(report.records, data["tasks"], strict=True):  # type: ignore[arg-type]
        for attempt, attempt_json in zip(rec.attempts, task_json["attempts"], strict=True):
            attempt_json["detail"] = classify(attempt, runner, qtree)
        task_json["first_outcome"] = rec.first.outcome
        task_json["first_detail"] = task_json["attempts"][0]["detail"]
    return {
        "target": target.name,
        "mode": window.mode,
        "size": window.size,
        "window": window.index,
        "base": window.base,
        "commits": list(window.commits),
        "scanned": window.scanned,
        "dependent": window.dependent,
        "info": info,
        **data,
    }


# --- driver -----------------------------------------------------------------------------


def precompute(
    target: Target, commits: list[str], cache: Path, work: Path, workers: int
) -> dict[str, SuiteResult]:
    """Test every commit (cached by tree), spread over `workers` processes."""
    todo = list(dict.fromkeys(commits))
    chunks = [todo[i::workers] for i in range(workers) if todo[i::workers]]
    results: dict[str, SuiteResult] = {}
    with ProcessPoolExecutor(max_workers=len(chunks) or 1) as pool:
        futures = [pool.submit(_test_commits, target, chunk, cache, work) for chunk in chunks]
        for future in as_completed(futures):
            for commit, data in future.result().items():
                results[commit] = SuiteResult.from_json(data)
    return results


def run_experiment(
    target: Target,
    sizes: list[int],
    out: Path,
    cache_root: Path,
    workers: int = 4,
    max_windows: int | None = None,
    specs: list[Spec] | None = None,
    agent_spec: dict | None = None,
    log=None,
    dry_run: bool = False,
    modes: Iterable[str] = MODES,
) -> Path:
    """Run every window of every size and mode for one target; append records to `out`.

    Windows already present in `out` are skipped, so an interrupted run resumes. With
    `dry_run`, nothing in the repository or the cache is touched: windows are planned from
    cached test results alone, and the job list and an agent-call estimate are printed.
    """
    log = log or _log
    modes = list(modes)
    for mode in modes:
        if mode not in MODES:
            raise ValueError(f"unknown mode {mode!r}; choose one of {', '.join(MODES)}")
    specs = specs or POLICY_SPECS
    git = Git(target.path)
    cache_root = cache_root.resolve()
    cache = cache_root / f"{target.name}-{environment_fingerprint(target.python)}"
    work = cache_root / "work" / target.name
    head = git.rev_parse(target.ref)
    history = history_tasks(git, head, target.history)
    bases = [git.parents(c)[0] for c in history.tasks]
    needed = list(dict.fromkeys(bases + history.tasks))
    log(
        f"[{target.name}] {len(history.tasks)} tasks "
        f"({len(history.housekeeping)} housekeeping commits left out)"
    )
    if dry_run:
        results = _cached_results(git, target, cache, needed) if cache.exists() else {}
        log(
            f"[{target.name}] dry run: {len(results)}/{len(needed)} commits have cached "
            "test results; windows needing the others are left out"
        )
        _plan(target, git, history, sizes, results, max_windows, specs, modes, out, log, True)
        return out
    with _Lock(work):
        git.run("config", "gc.auto", "0")
        _drop_stale_worktrees(git, work)
        log(f"[{target.name}] testing {len(needed)} commits (cached by tree)")
        results = precompute(target, needed, cache, work, workers)
        todo = _plan(target, git, history, sizes, results, max_windows, specs, modes, out, log)
        flaky = set().union(*(r.flaky for r in results.values())) if results else set()
        out.parent.mkdir(parents=True, exist_ok=True)
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {}
            for w, missing in todo:
                ignore = set(results[w.base].failed) | flaky
                for c in w.commits:
                    ignore |= results[c].failed
                future = pool.submit(
                    _run_window, target, w, sorted(ignore), cache, work, missing, agent_spec
                )
                futures[future] = w
            for n, future in enumerate(as_completed(futures), 1):
                w = futures[future]
                records = future.result()
                with out.open("a", encoding="utf-8") as fh:
                    for record in records:
                        fh.write(json.dumps(record) + "\n")
                log(f"[{target.name}] {w.mode} N={w.size} #{w.index} done ({n}/{len(todo)})")
    facts = {
        c: {"healthy": healthy(results.get(c)), **results[c].to_json()}
        for c in needed
        if c in results
    }
    history_path = out.with_name(f"{target.name}-history.json")
    history_path.write_text(
        json.dumps(
            {
                "head": head,
                "commits": history.tasks,
                "housekeeping": history.housekeeping,
                "results": facts,
            }
        )
    )
    return out


def _plan(
    target: Target,
    git: Git,
    history: History,
    sizes: list[int],
    results: dict[str, SuiteResult],
    max_windows: int | None,
    specs: list[Spec],
    modes: list[str],
    out: Path,
    log,
    dry_run: bool = False,
) -> list[tuple[Window, list[Spec]]]:
    healthy_count = sum(healthy(results.get(c)) for c in history.tasks)
    log(f"[{target.name}] {healthy_count}/{len(history.tasks)} task commits test healthy")
    windows = []
    for mode in modes:
        windows.extend(make_windows(git, history.tasks, sizes, results, mode))
    if max_windows is not None:
        windows = spread(windows, max_windows)
    done = _done_labels(out)
    todo = []
    for w in windows:
        have = done.get((target.name, w.mode, w.size, w.index), set())
        missing = [s for s in specs if spec_label(s) not in have]
        if missing:
            todo.append((w, missing))
    log(f"[{target.name}] {len(windows)} eligible windows, {len(todo)} with runs still to do")
    if dry_run:
        calls = 0
        for w, missing in todo:
            calls += w.size * len(missing)
            log(
                f"  job: {w.mode} N={w.size} window #{w.index} base {w.base[:10]}, "
                f"{len(missing)} policies"
            )
        log(
            f"[{target.name}] agent calls: {calls} first attempts, at most "
            f"{2 * calls} with one redo each"
        )
    return todo


def _cached_results(
    git: Git, target: Target, cache: Path, commits: list[str]
) -> dict[str, SuiteResult]:
    runner = SuiteRunner(git, target.suite(), cache)
    found = {}
    for commit in commits:
        hit = runner.cached(git.tree_of(commit))
        if hit is not None:
            found[commit] = hit
    return found


def spread(windows: list[Window], per_size: int) -> list[Window]:
    """At most `per_size` windows of each (mode, size), evenly spaced through history."""
    chosen = []
    for key in sorted({(w.mode, w.size) for w in windows}):
        pool = [w for w in windows if (w.mode, w.size) == key]
        if len(pool) <= per_size:
            chosen.extend(pool)
            continue
        step = len(pool) / per_size
        chosen.extend(pool[int(k * step)] for k in range(per_size))
    return chosen


def _log(message: str) -> None:
    print(message, flush=True)


def _done_labels(out: Path) -> dict[tuple[str, str, int, int], set[str]]:
    """Which policy runs each window already has in `out`."""
    done: dict[tuple[str, str, int, int], set[str]] = {}
    if not out.exists():
        return done
    for line in out.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            key = (rec["target"], rec["mode"], rec["size"], rec["window"])
            done.setdefault(key, set()).add(rec["label"])
    return done


class _Lock:
    """One experiment per target at a time: a second one would remove the first one's
    worktrees as stale. The lock is a file; a crash leaves it behind, with a clear message."""

    def __init__(self, work: Path) -> None:
        self.path = work / "experiment.lock"

    def __enter__(self) -> _Lock:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            raise RuntimeError(
                f"another experiment holds {self.path}; if none is running, delete the file"
            ) from None
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        return self

    def __exit__(self, *exc: object) -> None:
        self.path.unlink(missing_ok=True)


def _drop_stale_worktrees(git: Git, work: Path) -> None:
    """Remove worktrees an interrupted earlier run left under `work` (and only there)."""
    listing = git.out("worktree", "list", "--porcelain")
    root = str(work.resolve()).replace("\\", "/").lower()
    for line in listing.splitlines():
        if line.startswith("worktree "):
            path = line.removeprefix("worktree ").strip()
            if path.replace("\\", "/").lower().startswith(root):
                _drop_worktree(git, Path(path))
    git.worktree_prune()


def _fresh_worktree(git: Git, path: Path, commit: str) -> None:
    if path.exists():
        git.worktree_remove(path)
        shutil.rmtree(path, ignore_errors=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    git.worktree_add(path, commit)


def _drop_worktree(git: Git, path: Path) -> None:
    git.worktree_remove(path)
    shutil.rmtree(path, ignore_errors=True)


def _delete_refs(git: Git, prefix: str) -> None:
    refs = git.out("for-each-ref", "--format=%(refname)", prefix).splitlines()
    if refs:
        git.run("update-ref", "--stdin", input_="".join(f"delete {r}\n" for r in refs))
