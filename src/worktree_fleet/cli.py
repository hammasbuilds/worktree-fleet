"""Command line: `fleet plan`, `fleet run`, `fleet experiment`, `fleet report`."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from .agents import Agent, ReplayAgent
from .fleet import ORDERS, POLICIES, PREDICTED, Fleet, plan_waves
from .gitops import Git, GitError
from .mergequeue import MergeQueue
from .predict import PREDICTOR_NAMES, build_predictor, predicted_conflicts
from .suite import SuiteConfig, SuiteRunner
from .tasks import Task, load_tasks

AGENTS = ("replay", "ollama")


class UsageError(Exception):
    """A problem with the command line or its inputs, reported as one line."""


def _repo(path: str) -> Git:
    root = Path(path)
    if not root.is_dir():
        raise UsageError(f"--repo {path!r} is not a directory")
    git = Git(root)
    if git.run("rev-parse", "--git-dir", check=False).returncode != 0:
        raise UsageError(f"--repo {path!r} is not a git repository")
    return git


def _tasks_from_args(args: argparse.Namespace, git: Git) -> list[Task]:
    if args.tasks:
        tasks = load_tasks(Path(args.tasks))
    elif args.commits is not None:
        names = [c.strip() for c in args.commits.split(",") if c.strip()]
        if not names:
            raise UsageError("--commits is empty; give one or more commits, comma-separated")
        tasks = []
        for name in names:
            try:
                sha = git.rev_parse(name)
            except GitError:
                raise UsageError(f"--commits: {name!r} is not a commit in {git.repo}") from None
            tasks.append(Task(sha[:10], git.message(sha), sha))
    else:
        raise UsageError("give tasks with --tasks FILE.json or --commits SHA,SHA,...")
    if not tasks:
        raise UsageError("no tasks to run")
    return tasks


def _agent(args: argparse.Namespace, tasks: list[Task]) -> Agent:
    if args.agent == "replay":
        missing = [t.id for t in tasks if t.commit is None]
        if missing:
            raise UsageError(
                f"the replay agent re-applies each task's real commit, and task(s) "
                f'{", ".join(missing)} have none; give them a "commit" or use --agent ollama'
            )
        return ReplayAgent()
    from .llm import OllamaAgent, OllamaClient

    return OllamaAgent(OllamaClient(args.ollama_url, cache_dir=Path(args.llm_cache)), args.model)


def _base(git: Git, ref: str) -> str:
    try:
        return git.rev_parse(ref)
    except GitError:
        raise UsageError(f"--base {ref!r} is not a commit in {git.repo}") from None


def cmd_plan(args: argparse.Namespace) -> int:
    git = _repo(args.repo)
    tasks = _tasks_from_args(args, git)
    base = _base(git, args.base)
    predictor = build_predictor(args.predictor, git)
    pairs = predicted_conflicts(predictor, tasks, base)
    waves = plan_waves(len(tasks), set(pairs))
    print(f"{len(tasks)} tasks, predictor {predictor.name}: {len(waves)} wave(s)")
    for w, wave in enumerate(waves, 1):
        print(f"  wave {w}: " + ", ".join(tasks[i].id for i in wave))
    for (i, j), files in sorted(pairs.items()):
        print(f"  {tasks[i].id} x {tasks[j].id}: {', '.join(sorted(files))}")
    return 0


def _usable_python(python: str) -> str:
    """An absolute path to a python that can run pytest, or a clear error.

    Tests run with each worktree as the working directory, so a relative path that works
    from here would not be found there.
    """
    candidate = Path(python)
    resolved = str(candidate.resolve()) if candidate.exists() else python
    try:
        proc = subprocess.run(
            [resolved, "-m", "pytest", "--version"], capture_output=True, text=True, timeout=120
        )
    except OSError as exc:
        raise UsageError(f"cannot run --python {python!r}: {exc}") from exc
    if proc.returncode != 0:
        raise UsageError(f"{python!r} runs, but has no pytest: {proc.stderr.strip()[:200]}")
    return resolved


def cmd_run(args: argparse.Namespace) -> int:
    git = _repo(args.repo)
    tasks = _tasks_from_args(args, git)
    base = _base(git, args.base)
    agent = _agent(args, tasks)
    runner = None
    if args.python:
        config = SuiteConfig(
            python=_usable_python(args.python),
            args=args.pytest_args.split() if args.pytest_args else [],
            pythonpath=args.pythonpath.split(",") if args.pythonpath else [],
            timeout=args.timeout,
        )
        runner = SuiteRunner(git, config, Path(args.cache) if args.cache else None)
    workdir = Path(args.workdir or tempfile.mkdtemp(prefix="fleet-"))
    qtree = (workdir / f"{args.run_id}-queue").resolve()
    if runner is not None:
        git.worktree_add(qtree, base)

    def factory(start: str, ref: str) -> MergeQueue:
        return MergeQueue(git, start, ref, runner, qtree if runner else None)

    fleet = Fleet(
        git,
        agent,
        workdir,
        factory,
        max_workers=args.workers,
        retries=args.retries,
        keep_worktrees=args.keep_worktrees,
        order=args.order,
        seed=args.seed,
    )
    predictor = build_predictor(args.predictor, git) if args.policy == PREDICTED else None
    try:
        report = fleet.run(tasks, base, args.policy, predictor, run_id=args.run_id)
    finally:
        if runner is not None:
            git.worktree_remove(qtree)
    print(
        f"policy {report.policy}: {len(report.waves)} wave(s), makespan {report.makespan}, "
        f"{report.agent_runs} agent run(s), {report.test_runs} test run(s)"
    )
    for rec in report.records:
        trail = " -> ".join(a.outcome for a in rec.attempts)
        print(f"  [{rec.final:>8}] {rec.task.id} (wave {rec.wave + 1}): {trail}")
        for n, attempt in enumerate(rec.attempts, 1):
            why = []
            if attempt.conflicted:
                why.append(f"conflicts: {', '.join(attempt.conflicted)}")
            if attempt.new_failures:
                why.append(f"broke: {', '.join(attempt.new_failures[:3])}")
            if attempt.note and not attempt.conflicted:
                why.append(attempt.note)
            if why:
                print(f"      attempt {n}: {'; '.join(why)}")
    print(f"main is now {report.main[:12]} (ref refs/fleet/{args.run_id}/main)")
    if args.json:
        Path(args.json).write_text(json.dumps(report.to_json(), indent=2), encoding="utf-8")
    return 0 if all(r.final != "rejected" for r in report.records) else 1


def cmd_experiment(args: argparse.Namespace) -> int:
    from .experiment import MODES, parse_specs, run_experiment
    from .targets import load_targets

    specs = parse_specs(args.policies) if args.policies else None
    sizes = _int_list(args.sizes, "--sizes")
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    for mode in modes:
        if mode not in MODES:
            raise UsageError(f"--modes: unknown mode {mode!r}; choose from {', '.join(MODES)}")
    agent_spec = None
    if args.agent == "ollama":
        agent_spec = {
            "kind": "ollama",
            "url": args.ollama_url,
            "model": args.model,
            "cache": str(Path(args.llm_cache).resolve()),
        }
    targets = load_targets(Path(args.targets))
    names = args.target or sorted(targets)
    for name in names:
        if name not in targets:
            raise UsageError(f"unknown target {name!r}; known: {', '.join(sorted(targets))}")
        target = targets[name]
        if not (target.path / ".git").exists():
            raise UsageError(f"{target.path} is not a git repository; run scripts/fetch_targets.sh")
        run_experiment(
            target,
            sizes,
            Path(args.out) / f"{name}.jsonl",
            Path(args.cache),
            workers=args.workers,
            max_windows=args.max_windows,
            specs=specs,
            agent_spec=agent_spec,
            dry_run=args.dry_run,
            modes=modes,
        )
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    from .report import build_report, format_table
    from .targets import load_targets

    repos = {}
    if Path(args.targets).exists():
        repos = {name: t.path for name, t in load_targets(Path(args.targets)).items()}
    summary = build_report(Path(args.runs), seed=args.seed, resamples=args.resamples, repos=repos)
    out = Path(args.out)
    if summary["repos_missing"]:
        print(
            "fleet: warning: no git repository for "
            f"{', '.join(summary['repos_missing'])}; novel-state counts are left out and broken "
            "states are counted by commit, not by tree (run scripts/fetch_targets.sh)",
            file=sys.stderr,
        )
        if out.exists() and not args.force:
            raise UsageError(
                f"not overwriting {out} with a partial report; pass --out elsewhere or --force"
            )
    print(format_table(summary))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    return 0


def _int_list(text: str, flag: str) -> list[int]:
    try:
        values = [int(s) for s in text.split(",") if s.strip()]
    except ValueError:
        raise UsageError(f"{flag} must be comma-separated integers, got {text!r}") from None
    if not values or any(v < 1 for v in values):
        raise UsageError(f"{flag} needs one or more integers >= 1, got {text!r}")
    return values


def _at_least(minimum: int):
    def parse(text: str) -> int:
        try:
            value = int(text)
        except ValueError:
            raise argparse.ArgumentTypeError(f"expected an integer, got {text!r}") from None
        if value < minimum:
            raise argparse.ArgumentTypeError(f"must be at least {minimum}, got {value}")
        return value

    return parse


def _add_task_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--repo", default=".", help="repository to work in")
    p.add_argument("--base", default="HEAD", help="commit every task starts from")
    p.add_argument("--tasks", help='JSON list of {"id", "description", "commit"?}')
    p.add_argument("--commits", help="comma-separated commits to replay as tasks")
    p.add_argument(
        "--predictor",
        default="description",
        choices=PREDICTOR_NAMES,
        help="how collisions are predicted (the oracles need tasks with commits)",
    )


def parser() -> argparse.ArgumentParser:
    fmt = argparse.ArgumentDefaultsHelpFormatter
    ap = argparse.ArgumentParser(
        prog="fleet",
        description="Run coding tasks in parallel git worktrees and integrate them through a "
        "tested merge queue.",
        formatter_class=fmt,
    )
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser(
        "plan", help="predict which tasks collide and print the waves", formatter_class=fmt
    )
    _add_task_args(p)
    p.set_defaults(func=cmd_plan)

    p = sub.add_parser(
        "run", help="run tasks with agents in worktrees and integrate them", formatter_class=fmt
    )
    _add_task_args(p)
    p.add_argument(
        "--policy",
        default="predicted",
        choices=POLICIES,
        help="serial: one at a time; parallel: all at once; predicted: waves from --predictor",
    )
    p.add_argument(
        "--agent",
        default="replay",
        choices=AGENTS,
        help="replay: re-apply each task's commit; ollama: ask a local model",
    )
    p.add_argument("--model", default="qwen2.5-coder:14b", help="Ollama model (--agent ollama)")
    p.add_argument("--ollama-url", default="http://127.0.0.1:11434", help="Ollama server")
    p.add_argument("--llm-cache", default=".fleet-llm-cache", help="where generations are cached")
    p.add_argument("--python", help="python with pytest installed; enables the test gate")
    p.add_argument("--pytest-args", default="", help="e.g. 'tests -x'")
    p.add_argument("--pythonpath", default="", help="comma-separated dirs, e.g. src")
    p.add_argument("--timeout", type=float, default=600.0, help="seconds per test run")
    p.add_argument("--cache", help="directory for tree-keyed test results (default: none)")
    p.add_argument("--workdir", help="where agent worktrees go (default: a temp dir)")
    p.add_argument("--workers", type=_at_least(1), default=4, help="agents running at once")
    p.add_argument("--retries", type=_at_least(0), default=1, help="redos per failed task")
    p.add_argument(
        "--order",
        default="completion",
        choices=ORDERS,
        help="integration order within a wave: as agents finish, as listed, or a seeded shuffle",
    )
    p.add_argument("--seed", type=int, default=0, help="seed for --order shuffled")
    p.add_argument("--run-id", default="run", help="names refs/fleet/<run-id>/...")
    p.add_argument("--keep-worktrees", action="store_true", help="leave agent worktrees on disk")
    p.add_argument("--json", help="also write the full report here")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser(
        "experiment", help="replay real history at fleet sizes N", formatter_class=fmt
    )
    p.add_argument("--targets", default="targets.toml", help="target definitions")
    p.add_argument("--target", action="append", help="target name, repeatable (default: all)")
    p.add_argument("--sizes", default="2,4,8,16", help="fleet sizes N, comma-separated")
    p.add_argument(
        "--modes",
        default="consecutive,independent",
        help="how windows pick their N tasks: consecutive tasks, or tasks that each apply to "
        "the common base",
    )
    p.add_argument("--workers", type=_at_least(1), default=4, help="windows run at once")
    p.add_argument("--max-windows", type=_at_least(1), help="cap windows per mode and size")
    p.add_argument("--out", default="results/runs", help="directory for <target>.jsonl records")
    p.add_argument("--cache", default="targets/.fleet", help="test cache and scratch worktrees")
    p.add_argument(
        "--policies",
        help="comma-separated subset, e.g. serial,parallel,predicted:description "
        "(default: all eight)",
    )
    p.add_argument("--agent", default="replay", choices=AGENTS, help="who writes each change")
    p.add_argument("--model", default="qwen2.5-coder:14b", help="Ollama model (--agent ollama)")
    p.add_argument("--ollama-url", default="http://127.0.0.1:11434", help="Ollama server")
    p.add_argument("--llm-cache", default="results/llm-cache", help="where generations are cached")
    p.add_argument(
        "--dry-run", action="store_true", help="print the job list; touch nothing, run nothing"
    )
    p.set_defaults(func=cmd_experiment)

    p = sub.add_parser(
        "report", help="aggregate experiment runs into a summary", formatter_class=fmt
    )
    p.add_argument("--runs", default="results/runs", help="directory of experiment records")
    p.add_argument("--out", default="results/summary.json", help="summary file to write")
    p.add_argument("--seed", type=int, default=0, help="bootstrap seed")
    p.add_argument("--resamples", type=_at_least(1), default=2000, help="bootstrap resamples")
    p.add_argument("--targets", default="targets.toml", help="to look states up in git")
    p.add_argument(
        "--force", action="store_true", help="overwrite --out even with target repos missing"
    )
    p.set_defaults(func=cmd_report)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        return args.func(args)
    except ConnectionError as exc:
        print(f"fleet: {exc} (is `ollama serve` running?)", file=sys.stderr)
        return 2
    except (UsageError, GitError, ValueError, FileNotFoundError, RuntimeError) as exc:
        print(f"fleet: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
