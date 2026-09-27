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


def test_experiment_rejects_unknown_target(tmp_path):
    (tmp_path / "t.toml").write_text(
        '[targets.a]\nurl = "u"\npath = "a"\nref = "r"\nhistory = 5\nvenv = "v"\n'
    )
    assert main(["experiment", "--targets", str(tmp_path / "t.toml"), "--target", "zzz"]) == 2
    assert main(["experiment", "--targets", str(tmp_path / "t.toml"), "--target", "a"]) == 2


def test_run_rejects_a_python_without_pytest(repo, tmp_path, capsys):
    base = repo.commit("base", {"a": "1"})
    argv = ["run", "--repo", str(repo.path), "--base", base, "--commits", base]
    assert main([*argv, "--python", str(tmp_path / "no-such-python")]) == 2
    assert "cannot run --python" in capsys.readouterr().err


def test_run_listed_order_requeues_the_dependent_task(repo, tmp_path, capsys):
    base = repo.commit("base", {"f.txt": "1\n2\n3\n4\n"})
    c1 = repo.commit("c1", {"f.txt": "one\n2\n3\n4\n"})
    c2 = repo.commit("c2", {"f.txt": "one\ntwo\n3\n4\n"})
    code = main(
        [
            "run",
            "--repo",
            str(repo.path),
            "--base",
            base,
            "--commits",
            f"{c2},{c1}",
            "--policy",
            "parallel",
            "--order",
            "listed",
            "--workdir",
            str(tmp_path / "w"),
        ]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "base-conflict -> accepted" in out


def test_bad_inputs_get_one_line_errors(repo, tmp_path, capsys):
    base = repo.commit("base", {"a": "1"})
    cases = [
        (["plan", "--repo", str(tmp_path / "nowhere"), "--commits", base], "not a directory"),
        (["plan", "--repo", str(tmp_path), "--commits", base], "not a git repository"),
        (["plan", "--repo", str(repo.path), "--commits", ","], "--commits is empty"),
        (["plan", "--repo", str(repo.path), "--commits", "nope"], "'nope' is not a commit"),
        (["plan", "--repo", str(repo.path), "--commits", base, "--base", "zz"], "--base"),
        (["plan", "--repo", str(repo.path)], "--tasks FILE.json"),
    ]
    for argv, message in cases:
        assert main(argv) == 2, argv
        err = capsys.readouterr().err
        assert err.startswith("fleet: ") and message in err, (argv, err)
        assert "Traceback" not in err


@pytest.mark.parametrize(("flag", "value"), [("--workers", "0"), ("--retries", "-1")])
def test_nonsense_counts_are_refused(repo, flag, value, capsys):
    base = repo.commit("base", {"a": "1"})
    with pytest.raises(SystemExit):
        main(["run", "--repo", str(repo.path), "--commits", base, flag, value])
    assert "must be at least" in capsys.readouterr().err


def test_tasks_without_commits_need_a_real_agent(repo, tmp_path, capsys):
    base = repo.commit("base", {"a": "1"})
    tasks = tmp_path / "tasks.json"
    tasks.write_text(json.dumps([{"id": "t1", "description": "add a feature"}]))
    assert main(["run", "--repo", str(repo.path), "--base", base, "--tasks", str(tasks)]) == 2
    err = capsys.readouterr().err
    assert "t1" in err and "--agent ollama" in err


def test_ollama_down_is_a_clean_error(repo, tmp_path, capsys):
    base = repo.commit("base", {"a.py": "x = 1\n"})
    tasks = tmp_path / "tasks.json"
    tasks.write_text(json.dumps([{"id": "t1", "description": "change x"}]))
    code = main(
        [
            "run",
            "--repo",
            str(repo.path),
            "--base",
            base,
            "--tasks",
            str(tasks),
            "--agent",
            "ollama",
            "--ollama-url",
            "http://127.0.0.1:9",
            "--llm-cache",
            str(tmp_path / "c"),
            "--workdir",
            str(tmp_path / "w"),
        ]
    )
    err = capsys.readouterr().err
    assert code == 2 and "not reachable" in err and "Traceback" not in err


def test_run_prints_why_an_attempt_failed(repo, tmp_path, capsys):
    base = repo.commit("base", {"f.txt": "1\n2\n"})
    c1 = repo.commit("c1", {"f.txt": "one\n2\n"})
    c2 = repo.commit("c2", {"f.txt": "uno\n2\n"})
    main(
        [
            "run",
            "--repo",
            str(repo.path),
            "--base",
            base,
            "--commits",
            f"{c2},{c1}",
            "--policy",
            "parallel",
            "--order",
            "listed",
            "--retries",
            "0",
            "--workdir",
            str(tmp_path / "w"),
        ]
    )
    out = capsys.readouterr().out
    assert "attempt 1: conflicts: f.txt" in out


def test_report_refuses_to_clobber_with_a_partial_summary(tmp_path, capsys):
    runs = tmp_path / "runs"
    runs.mkdir()
    record = {
        "target": "gone",
        "mode": "consecutive",
        "size": 1,
        "window": 0,
        "label": "serial",
        "agent_runs": 1,
        "redos": 0,
        "makespan": 1,
        "info": {"predicted": {}},
        "tasks": [
            {
                "id": "t",
                "first_outcome": "accepted",
                "first_detail": None,
                "final": "accepted",
                "attempts": [
                    {"outcome": "accepted", "detail": None, "conflicted": [], "candidate": "abc"}
                ],
            }
        ],
    }
    (runs / "gone.jsonl").write_text(json.dumps(record) + "\n")
    out = tmp_path / "summary.json"
    out.write_text("{}")
    argv = [
        "report",
        "--runs",
        str(runs),
        "--out",
        str(out),
        "--resamples",
        "5",
        "--targets",
        str(tmp_path / "none.toml"),
    ]
    assert main(argv) == 2
    err = capsys.readouterr().err
    assert "warning: no git repository for gone" in err and "not overwriting" in err
    assert out.read_text() == "{}"
    assert main([*argv, "--force"]) == 0
    assert json.loads(out.read_text())["repos_missing"] == ["gone"]


def test_help_shows_defaults(capsys):
    with pytest.raises(SystemExit):
        parser().parse_args(["experiment", "--help"])
    out = " ".join(capsys.readouterr().out.split())
    assert "(default: 2,4,8,16)" in out and "(default: results/runs)" in out
