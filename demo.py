"""Four tasks, one base, and a known right answer for each.

Builds a throwaway repository with four independent changes written from the same base:

  rename   renames greet() to hello() (and its test)
  shout    adds shout(), which calls greet(), far down the same file
  docs-a   rewrites the README's first line
  docs-b   rewrites the README's first line differently

Known answers: docs-a/docs-b collide textually; rename/shout merge cleanly but break a test
(a semantic conflict). The fleet runs them three ways and prints what happened.

    uv run python demo.py
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

from worktree_fleet.agents import ReplayAgent
from worktree_fleet.fleet import Fleet
from worktree_fleet.gitops import Git
from worktree_fleet.mergequeue import MergeQueue
from worktree_fleet.predict import HunkPredictor, predicted_conflicts
from worktree_fleet.suite import SuiteConfig, SuiteRunner
from worktree_fleet.tasks import Task

PAD = "".join(f"# line {i}\n" for i in range(12))
BASE = {
    "mod.py": 'def greet(name):\n    return "hi " + name\n' + PAD,
    "test_mod.py": "from mod import greet\n\n\ndef test_greet():\n"
    '    assert greet("a") == "hi a"\n',
    "README.md": "# demo\n\nA tiny project.\n",
}
CHANGES = {
    "rename": {
        "mod.py": 'def hello(name):\n    return "hi " + name\n' + PAD,
        "test_mod.py": "from mod import hello\n\n\ndef test_hello():\n"
        '    assert hello("a") == "hi a"\n',
    },
    "shout": {
        "mod.py": BASE["mod.py"] + "\n\ndef shout(name):\n    return greet(name).upper()\n",
        "test_shout.py": "from mod import shout\n\n\ndef test_shout():\n"
        '    assert shout("a") == "HI A"\n',
    },
    "docs-a": {"README.md": "# demo (fleet edition)\n\nA tiny project.\n"},
    "docs-b": {"README.md": "# demo, now with agents\n\nA tiny project.\n"},
}


def sh(cwd: Path, *args: str) -> str:
    return subprocess.run(args, cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def commit(repo: Path, files: dict[str, str], message: str) -> str:
    for name, text in files.items():
        (repo / name).write_bytes(text.encode())
    sh(repo, "git", "add", "-A")
    sh(
        repo,
        "git",
        "-c",
        "user.name=demo",
        "-c",
        "user.email=demo@example.invalid",
        "-c",
        "commit.gpgsign=false",
        "commit",
        "-qm",
        message,
    )
    return sh(repo, "git", "rev-parse", "HEAD")


def main() -> None:
    root = Path(tempfile.mkdtemp(prefix="fleet-demo-"))
    repo = root / "repo"
    repo.mkdir()
    sh(repo, "git", "init", "-q", "-b", "main")
    sh(repo, "git", "config", "core.autocrlf", "false")
    base = commit(repo, BASE, "base")
    tasks = []
    for name, files in CHANGES.items():
        sh(repo, "git", "checkout", "-q", base)
        tasks.append(Task(name, f"task {name}", commit(repo, files, name)))

    git = Git(repo)
    runner = SuiteRunner(git, SuiteConfig(sys.executable, ["-q"], ["."], 120), root / "cache")
    predictor = HunkPredictor(git)
    print(f"repository: {repo}")
    print("predicted collisions (exact-hunk predictor):")
    for (i, j), files in predicted_conflicts(predictor, tasks, base).items():
        print(f"  {tasks[i].id} x {tasks[j].id}: {', '.join(sorted(files))}")

    for n, (policy, pred) in enumerate(
        [("serial", None), ("parallel", None), ("predicted", predictor)]
    ):
        qtree = root / f"queue-{n}"
        git.worktree_add(qtree, base)

        def factory(start: str, ref: str, qtree: Path = qtree) -> MergeQueue:
            return MergeQueue(git, start, ref, runner, qtree)

        fleet = Fleet(git, ReplayAgent(), root / "agents", factory, order="listed")
        report = fleet.run(tasks, base, policy, pred, run_id=f"demo-{policy}")
        git.worktree_remove(qtree)
        print(
            f"\n{policy:>9}: {len(report.waves)} wave(s), makespan {report.makespan}, "
            f"{report.agent_runs} agent runs"
        )
        for rec in report.records:
            first = rec.first
            why = ""
            if first.conflicted:
                why = f"conflict in {', '.join(first.conflicted)}"
            elif first.new_failures:
                why = f"merged cleanly, broke {', '.join(first.new_failures)}"
            trail = " -> ".join(a.outcome for a in rec.attempts)
            print(f"   {rec.task.id:<7} {trail:<32} {why}")


if __name__ == "__main__":
    main()
