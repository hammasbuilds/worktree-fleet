"""Command line: `fleet run`, `fleet plan`, `fleet experiment`, `fleet report`."""

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


def _tasks_from_args(args: argparse.Namespace, git: Git) -> list[Task]:
    if args.tasks:
        return load_tasks(Path(args.tasks))
    if args.commits:
        commits = [git.rev_parse(c) for c in args.commits.split(",")]
        return [Task(c[:10], git.message(c), c) for c in commits]
    raise SystemExit("give tasks with --tasks FILE.json or --commits SHA,SHA,...")


def _agent(name: str, args: argparse.Namespace) -> Agent:
    if name == "replay":
        return ReplayAgent()
    from .llm import OllamaAgent, OllamaClient

    return OllamaAgent(OllamaClient(args.ollama_url, cache_dir=Path(args.llm_cache)), args.model)


def cmd_plan(args: argparse.Namespace) -> int:
    git = Git(args.repo)
    tasks = _tasks_from_args(args, git)
    base = git.rev_parse(args.base)
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
        raise SystemExit(f"fleet: cannot run --python {python!r}: {exc}") from exc
    if proc.returncode != 0:
        raise SystemExit(f"fleet: {python!r} runs, but has no pytest: {proc.stderr.strip()[:200]}")
    return resolved


def cmd_run(args: argparse.Namespace) -> int:
    git = Git(args.repo)
    tasks = _tasks_from_args(args, git)
    base = git.rev_parse(args.base)
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
        _agent(args.agent, args),
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
        detail = ""
        first = rec.first
        if first.conflicted:
            detail = f"  conflicts: {', '.join(first.conflicted)}"
        elif first.new_failures:
            detail = f"  broke: {', '.join(first.new_failures[:3])}"
        print(f"  [{rec.final:>8}] {rec.task.id} (wave {rec.wave + 1}): {trail}{detail}")
    print(f"main is now {report.main[:12]} (ref refs/fleet/{args.run_id}/main)")
    if args.json:
        Path(args.json).write_text(json.dumps(report.to_json(), indent=2), encoding="utf-8")
    return 0 if all(r.final != "rejected" for r in report.records) else 1


def cmd_experiment(args: argparse.Namespace) -> int:
    from .experiment import parse_specs, run_experiment
    from .targets import load_targets

    specs = parse_specs(args.policies) if args.policies else None
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
            raise SystemExit(f"unknown target {name!r}; known: {', '.join(sorted(targets))}")
        target = targets[name]
        if not (target.path / ".git").exists():
            raise SystemExit(f"{target.path} is not a git repository; run scripts/fetch_targets.sh")
        out = Path(args.out) / f"{name}.jsonl"
        run_experiment(
            target,
            [int(s) for s in args.sizes.split(",")],
            out,
            Path(args.cache),
            workers=args.workers,
            max_windows=args.max_windows,
            specs=specs,
            agent_spec=agent_spec,
            dry_run=args.dry_run,
        )
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    from .report import build_report, format_table
    from .targets import load_targets

    repos = {}
    if Path(args.targets).exists():
        repos = {name: t.path for name, t in load_targets(Path(args.targets)).items()}
    summary = build_report(Path(args.runs), seed=args.seed, resamples=args.resamples, repos=repos)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(format_table(summary))
    print(f"wrote {args.out}")
    return 0


def _add_task_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--repo", default=".", help="repository to work in (default: .)")
    p.add_argument("--base", default="HEAD", help="commit every task starts from")
    p.add_argument("--tasks", help="JSON list of {id, description, commit?}")
    p.add_argument("--commits", help="comma-separated commits to replay as tasks")
    p.add_argument("--predictor", default="description", choices=PREDICTOR_NAMES)


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="fleet",
        description="Run coding tasks in parallel git worktrees and integrate them through a "
        "tested merge queue.",
    )
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("plan", help="predict which tasks collide and print the waves")
    _add_task_args(p)
    p.set_defaults(func=cmd_plan)

    p = sub.add_parser("run", help="run tasks with agents in worktrees and integrate them")
    _add_task_args(p)
    p.add_argument("--policy", default="predicted", choices=POLICIES)
    p.add_argument("--agent", default="replay", choices=("replay", "ollama"))
    p.add_argument("--model", default="qwen2.5-coder:14b")
    p.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    p.add_argument("--llm-cache", default=".fleet-llm-cache")
    p.add_argument("--python", help="python with pytest installed; enables the test gate")
    p.add_argument("--pytest-args", default="", help="e.g. 'tests -x'")
    p.add_argument("--pythonpath", default="", help="comma-separated dirs, e.g. src")
    p.add_argument("--timeout", type=float, default=600.0, help="seconds per test run")
    p.add_argument("--cache", help="directory for tree-keyed test results")
    p.add_argument("--workdir", help="where agent worktrees go (default: a temp dir)")
    p.add_argument("--workers", type=int, default=4, help="agents running at once")
    p.add_argument("--retries", type=int, default=1, help="redos per failed task")
    p.add_argument(
        "--order",
        default="completion",
        choices=ORDERS,
        help="integration order within a wave: as agents finish (default), as listed, or a "
        "seeded shuffle",
    )
    p.add_argument("--seed", type=int, default=0, help="seed for --order shuffled")
    p.add_argument("--run-id", default="run", help="names refs/fleet/<run-id>/...")
    p.add_argument("--keep-worktrees", action="store_true")
    p.add_argument("--json", help="also write the full report here")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("experiment", help="replay real history at fleet sizes N")
    p.add_argument("--targets", default="targets.toml")
    p.add_argument("--target", action="append", help="target name (repeatable)")
    p.add_argument("--sizes", default="2,4,8,16")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--max-windows", type=int, help="cap windows per size (for a quick look)")
    p.add_argument("--out", default="results/runs")
    p.add_argument("--cache", default="targets/.fleet")
    p.add_argument(
        "--policies",
        help="comma-separated subset, e.g. serial,parallel,predicted:description "
        "(default: all eight headline policies)",
    )
    p.add_argument("--agent", default="replay", choices=("replay", "ollama"))
    p.add_argument("--model", default="qwen2.5-coder:14b")
    p.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    p.add_argument("--llm-cache", default="results/llm-cache")
    p.add_argument("--dry-run", action="store_true", help="print the job list, run nothing")
    p.set_defaults(func=cmd_experiment)

    p = sub.add_parser("report", help="aggregate experiment runs into results/summary.json")
    p.add_argument("--runs", default="results/runs")
    p.add_argument("--out", default="results/summary.json")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resamples", type=int, default=2000)
    p.add_argument("--targets", default="targets.toml", help="to look up states in git")
    p.set_defaults(func=cmd_report)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        return args.func(args)
    except GitError as exc:
        print(f"fleet: {exc}", file=sys.stderr)
        return 2
    except (ValueError, FileNotFoundError) as exc:
        print(f"fleet: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
