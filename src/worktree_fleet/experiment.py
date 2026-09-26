"""Replay real history as if N agents had started at once, and measure what breaks.

For a window of N consecutive first-parent commits c1..cN with base B = parent(c1), each
commit becomes one task. Every policy runs the same N tasks from B with the `ReplayAgent`,
through a real merge queue that runs the repository's own test suite on every candidate.

Integration order matters. Integrated in history's own order, a replay can only reproduce
history: every change was written on top of the ones before it, so once the merges are clean
each intermediate main is byte-identical to a real commit. Parallel agents do not finish in
history's order, so the headline runs integrate each wave in a seeded shuffle (one seed per
window, shared by every policy); history's order is kept as a separate, optimistic run.

Attribution. A test only counts against a merge if it is not already known to be broken or
flaky: the ignore set of a window is every test failing at B or at any real commit c1..cN
(history broke it on its own), plus every test seen to flip on a rerun. A semantic conflict
is then split by testing the task's branch alone: if the branch alone already fails the
test, the change needed in-flight work it did not have ("stale-base"); if the branch alone
passes and main passes but their merge fails, it is a true "interaction".
"""

from __future__ import annotations

import json
import os
import shutil
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from .agents import Agent, ReplayAgent
from .fleet import PARALLEL, PREDICTED, SERIAL, Fleet, FleetReport
from .gitops import Git
from .mergequeue import AGENT_FAILED, SEMANTIC, TEXTUAL, MergeQueue
from .predict import PREDICTOR_NAMES, ExcludingPredictor, Predictor, build_predictor
from .suite import (
    SUITE_ERROR,
    SUITE_TIMEOUT,
    SuiteResult,
    SuiteRunner,
    environment_fingerprint,
)
from .targets import Target
from .tasks import Task

# (policy, predictor, integration order within a wave)
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


SPECS_BY_LABEL: dict[str, Spec] = {
    spec_label(s): s for s in [*POLICY_SPECS, (PREDICTED, "description-spans", "shuffled", False)]
}


def parse_specs(text: str) -> list[Spec]:
    """Comma-separated labels, e.g. "serial,parallel,predicted:description"."""
    specs = []
    for label in (part.strip() for part in text.split(",") if part.strip()):
        if label not in SPECS_BY_LABEL:
            raise ValueError(f"unknown policy {label!r}; choose from {', '.join(SPECS_BY_LABEL)}")
        specs.append(SPECS_BY_LABEL[label])
    return specs


@dataclass(frozen=True)
class Window:
    size: int
    index: int
    base: str
    commits: tuple[str, ...]


def make_agent(spec: dict | None) -> Agent:
    """`None` or {"kind": "replay"} replays history; {"kind": "ollama", ...} asks a model."""
    if spec is None or spec.get("kind") == "replay":
        return ReplayAgent()
    if spec.get("kind") == "ollama":
        from .llm import OllamaAgent, OllamaClient

        client = OllamaClient(spec["url"], cache_dir=Path(spec["cache"]))
        return OllamaAgent(client, spec["model"])
    raise ValueError(f"unknown agent spec {spec!r}")


def history_commits(git: Git, ref: str, limit: int) -> list[str]:
    """The last `limit` first-parent commits of `ref`, oldest first, minus empty changes.

    A commit whose diff against its first parent is empty (a merge of already-merged work)
    gives an agent nothing to do, so it is not a task.
    """
    chain = git.first_parent_chain(ref, limit)
    keep = []
    for commit in chain:
        parents = git.parents(commit)
        if parents and git.changed_files(parents[0], commit):
            keep.append(commit)
    return keep


def healthy(result: SuiteResult) -> bool:
    """A commit whose suite ran to completion with only a handful of failures."""
    if SUITE_ERROR in result.failed or SUITE_TIMEOUT in result.failed:
        return False
    return len(result.failed) <= max(5, int(0.02 * result.passed))


def make_windows(
    git: Git, commits: list[str], sizes: Iterable[int], results: dict[str, SuiteResult]
) -> list[Window]:
    """Non-overlapping windows of consecutive tasks whose base and commits all test healthy."""
    windows = []
    for size in sizes:
        for index in range(len(commits) // size):
            chunk = tuple(commits[index * size : (index + 1) * size])
            base = git.parents(chunk[0])[0]
            needed = (base, *chunk)
            if all(c in results and healthy(results[c]) for c in needed):
                windows.append(Window(size, index, base, chunk))
    return windows


# --- workers (top-level so they pickle on Windows) --------------------------------------


def _test_commits(target: Target, commits: list[str], cache: Path, work: Path) -> dict:
    git = Git(target.path)
    runner = SuiteRunner(git, target.suite(), cache)
    worktree = work / f"pre-{os.getpid()}"
    _fresh_worktree(git, worktree, commits[0])
    try:
        return {c: runner.result(c, worktree).to_json() for c in commits}
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
    tag = f"n{window.size}-w{window.index}-{os.getpid()}"
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
    """Per-window facts shared by every policy: footprints, predictions, pairwise truth."""
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
    # Pairwise textual truth: do tasks i and j, each replayed alone onto the base, merge?
    replays: list[str | None] = []
    for task in tasks:
        assert task.commit is not None
        result = git.replay(task.commit, window.base)
        replays.append(
            git.commit_tree(result.tree, [window.base], task.id) if result.clean else None
        )
    truth = []
    for j in range(len(tasks)):
        for i in range(j):
            a, b = replays[i], replays[j]
            if a is None or b is None:
                truth.append([i, j, None])
            else:
                truth.append([i, j, not git.merge(a, b, base=window.base).clean])
    return {"files": files, "predicted": predicted, "pair_truth": truth, "replays": replays}


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
        first = rec.first
        kind = first.outcome
        detail = None
        if kind == AGENT_FAILED:
            detail = "base"
        elif kind == TEXTUAL:
            detail = "merge"
        elif kind == SEMANTIC and first.branch is not None:
            alone = runner.result(first.branch, qtree)
            blamed = set(first.new_failures)
            detail = "stale-base" if alone.failed & blamed else "interaction"
        task_json["first_outcome"] = kind
        task_json["first_detail"] = detail
    return {
        "target": target.name,
        "size": window.size,
        "window": window.index,
        "base": window.base,
        "commits": list(window.commits),
        "info": {k: v for k, v in info.items() if k != "replays"},
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
) -> Path:
    """Run every window of every size for one target; append records to `out` (JSONL).

    Windows already present in `out` are skipped, so an interrupted run resumes. With
    `dry_run`, nothing is tested or run: windows are planned from cached test results only
    and the job list and an agent-call estimate are printed.
    """
    log = log or _log
    git = Git(target.path)
    git.run("config", "gc.auto", "0")
    cache_root = cache_root.resolve()
    cache = cache_root / f"{target.name}-{environment_fingerprint(target.python)}"
    work = cache_root / "work" / target.name
    work.mkdir(parents=True, exist_ok=True)
    _drop_stale_worktrees(git, work)
    head = git.rev_parse(target.ref)
    commits = history_commits(git, head, target.history)
    bases = [git.parents(c)[0] for c in commits]
    if dry_run:
        runner = SuiteRunner(git, target.suite(), cache)
        results = {c: r for c in bases + commits if (r := runner.cached(git.tree_of(c)))}
        log(
            f"[{target.name}] dry run: {len(results)}/{len(set(bases + commits))} commits "
            "have cached test results (run the replay experiment first to fill the rest)"
        )
    else:
        log(f"[{target.name}] {len(commits)} task commits; testing each (cached by tree)")
        results = precompute(target, bases + commits, cache, work, workers)
    healthy_count = sum(healthy(results[c]) for c in commits)
    log(f"[{target.name}] {healthy_count}/{len(commits)} commits test healthy")
    windows = make_windows(git, commits, sizes, results)
    if max_windows is not None:
        windows = spread(windows, max_windows)
    specs = specs or POLICY_SPECS
    done = _done_labels(out)
    todo = []
    for w in windows:
        have = done.get((target.name, w.size, w.index), set())
        missing = [s for s in specs if spec_label(s) not in have]
        if missing:
            todo.append((w, missing))
    log(f"[{target.name}] {len(windows)} eligible windows, {len(todo)} with runs still to do")
    if dry_run:
        calls = 0
        for w, missing in todo:
            calls += w.size * len(missing)
            log(f"  job: N={w.size} window #{w.index} base {w.base[:10]}, {len(missing)} policies")
        log(
            f"[{target.name}] agent calls: {calls} first attempts, at most "
            f"{2 * calls} with one redo each"
        )
        return out
    flaky = set().union(*(r.flaky for r in results.values())) if results else set()
    out.parent.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {}
        for w, missing in todo:
            ignore = set(results[w.base].failed) | flaky
            for c in w.commits:
                ignore |= results[c].failed
            future = pool.submit(
                _run_window,
                target,
                w,
                sorted(ignore),
                cache,
                work,
                missing,
                agent_spec,
            )
            futures[future] = w
        for n, future in enumerate(as_completed(futures), 1):
            w = futures[future]
            records = future.result()
            with out.open("a", encoding="utf-8") as fh:
                for record in records:
                    fh.write(json.dumps(record) + "\n")
            log(f"[{target.name}] window N={w.size} #{w.index} done ({n}/{len(todo)})")
    history = {c: {"healthy": healthy(results[c]), **results[c].to_json()} for c in bases + commits}
    history_path = out.with_name(f"{target.name}-history.json")
    history_path.write_text(json.dumps({"head": head, "commits": commits, "results": history}))
    return out


def spread(windows: list[Window], per_size: int) -> list[Window]:
    """At most `per_size` windows of each size, evenly spaced through history."""
    chosen = []
    for size in sorted({w.size for w in windows}):
        pool = [w for w in windows if w.size == size]
        if len(pool) <= per_size:
            chosen.extend(pool)
            continue
        step = len(pool) / per_size
        chosen.extend(pool[int(k * step)] for k in range(per_size))
    return chosen


def _log(message: str) -> None:
    print(message, flush=True)


def _done_labels(out: Path) -> dict[tuple[str, int, int], set[str]]:
    """Which policy runs each window already has in `out`."""
    done: dict[tuple[str, int, int], set[str]] = {}
    if not out.exists():
        return done
    for line in out.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            key = (rec["target"], rec["size"], rec["window"])
            done.setdefault(key, set()).add(record_label(rec))
    return done


def record_label(rec: dict) -> str:
    """A record's policy label (records written before `label` existed carry no union flag)."""
    if "label" in rec:
        return rec["label"]
    spec = (rec["policy"], rec.get("predictor"), rec.get("order", "shuffled"), False)
    return spec_label(spec)


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
