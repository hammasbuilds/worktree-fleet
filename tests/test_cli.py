import json

import pytest

from worktree_fleet.cli import main, parser


def _two_tasks(repo):
    base = repo.commit("base", {"a.txt": "1\n2\n3\n4\n5\n", "b.txt": "x\n"})
    a = repo.commit("a", {"a.txt": "one\n2\n3\n4\n5\n"})
    b = repo.commit("b", {"a.txt": "one\n2\n3\n4\nfive\n", "b.txt": "y\n"})
    return base, a, b


def test_help_lists_every_command(capsys):
    with pytest.raises(SystemExit):
        parser().parse_args(["--help"])
    out = capsys.readouterr().out
    for command in ("plan", "run", "experiment", "report"):
        assert command in out


def test_plan_prints_waves(repo, capsys):
    base, a, b = _two_tasks(repo)
    code = main(
        [
            "plan",
            "--repo",
            str(repo.path),
            "--base",
            base,
            "--commits",
            f"{a},{b}",
            "--predictor",
            "oracle-files",
        ]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "2 wave(s)" in out and "a.txt" in out


def test_run_replays_and_writes_json(repo, tmp_path, capsys):
    base, a, b = _two_tasks(repo)
    report = tmp_path / "report.json"
    code = main(
        [
            "run",
            "--repo",
            str(repo.path),
            "--base",
            base,
            "--commits",
            f"{a},{b}",
            "--policy",
            "parallel",
            "--workdir",
            str(tmp_path / "w"),
            "--json",
            str(report),
        ]
    )
    assert code == 0
    data = json.loads(report.read_text())
    assert [t["final"] for t in data["tasks"]] == ["accepted", "accepted"]
    assert "main is now" in capsys.readouterr().out


def test_run_with_tasks_file_and_test_gate(repo, tmp_path, capsys, suite_config):
    base = repo.commit("base", {"test_t.py": "def test_t():\n    assert True\n"})
    bad = repo.commit("break it", {"test_t.py": "def test_t():\n    assert False\n"})
    tasks = tmp_path / "tasks.json"
    tasks.write_text(json.dumps([{"id": "breaker", "description": "x", "commit": bad}]))
    code = main(
        [
            "run",
            "--repo",
            str(repo.path),
            "--base",
            base,
            "--tasks",
            str(tasks),
            "--policy",
            "serial",
            "--python",
            suite_config.python,
            "--pythonpath",
            ".",
            "--workdir",
            str(tmp_path / "w"),
            "--retries",
            "0",
        ]
    )
    out = capsys.readouterr().out
    assert code == 1
    assert "semantic" in out and "broke: test_t::test_t" in out


def test_errors_are_one_line_messages(repo, tmp_path, capsys):
    repo.commit("base", {"a": "1"})
    assert main(["plan", "--repo", str(repo.path), "--commits", "nope"]) == 2
    assert "fleet:" in capsys.readouterr().err
    bad = tmp_path / "bad.json"
    bad.write_text("{}")
    assert main(["plan", "--repo", str(repo.path), "--tasks", str(bad)]) == 2
    assert "expected a JSON list" in capsys.readouterr().err
    with pytest.raises(SystemExit, match="--tasks"):
        main(["plan", "--repo", str(repo.path)])


def test_experiment_rejects_unknown_target(tmp_path):
    (tmp_path / "t.toml").write_text(
        '[targets.a]\nurl = "u"\npath = "a"\nref = "r"\nhistory = 5\nvenv = "v"\n'
    )
    with pytest.raises(SystemExit, match="unknown target"):
        main(["experiment", "--targets", str(tmp_path / "t.toml"), "--target", "zzz"])
    with pytest.raises(SystemExit, match="fetch_targets"):
        main(["experiment", "--targets", str(tmp_path / "t.toml"), "--target", "a"])
