import json
import random
import sys

from worktree_fleet.experiment import (
    POLICY_SPECS,
    Window,
    healthy,
    history_commits,
    make_windows,
    run_experiment,
    spread,
)
from worktree_fleet.report import bootstrap, build_report, policy_key, window_counts
from worktree_fleet.suite import SUITE_ERROR, SuiteResult
from worktree_fleet.targets import Target, load_targets


def _result(failed=(), passed=100):
    return SuiteResult("t", set(failed), set(), passed, 1.0, 1)


def test_healthy_thresholds():
    assert healthy(_result())
    assert healthy(_result(failed=["a"] * 1))
    assert not healthy(_result(failed=[SUITE_ERROR]))
    assert not healthy(_result(failed=[f"t{i}" for i in range(6)]))


def test_history_skips_empty_changes_and_windows_need_healthy_commits(repo):
    c = [repo.commit("base", {"a": "0"})]
    for i in range(1, 6):
        c.append(repo.commit(f"c{i}", {"a": str(i)}))
    c.append(repo.commit("empty", {}))
    commits = history_commits(repo.git, c[-1], 10)
    assert commits == c[1:6]  # the root has no parent, the empty commit changes nothing
    results = {x: _result() for x in c}
    windows = make_windows(repo.git, commits, [2], results)
    assert [w.commits for w in windows] == [tuple(commits[0:2]), tuple(commits[2:4])]
    results[commits[3]] = _result(failed=[SUITE_ERROR])
    assert len(make_windows(repo.git, commits, [2], results)) == 1


def test_spread_picks_evenly():
    windows = [Window(2, i, "b", ()) for i in range(10)] + [Window(4, 0, "b", ())]
    picked = spread(windows, 3)
    assert [w.index for w in picked if w.size == 2] == [0, 3, 6]
    assert [w.size for w in picked].count(4) == 1


def _history_repo(repo):
    """A tiny project with real tests and a history containing one overlapping pair."""
    lib = "".join(f"def f{i}():\n    return {i}\n\n\n" for i in range(8))
    tests = "from lib import *\n\n\ndef test_f0():\n    assert f0() == 0\n"
    repo.commit("base", {"lib.py": lib, "test_lib.py": tests})
    for i in range(1, 8):
        lib = lib.replace(f"return {i}\n", f"return {i}  # tuned\n")
        repo.commit(f"tune f{i}", {"lib.py": lib})
    # Two consecutive edits of the same line: they cannot both start from one base.
    lib = lib.replace("return 7  # tuned\n", "return 7  # retuned\n")
    return repo.commit("retune f7", {"lib.py": lib})


def test_run_experiment_end_to_end(repo, tmp_path):
    head = _history_repo(repo)
    target = Target(
        name="tiny",
        path=repo.path,
        url="local",
        ref=head,
        history=8,
        python=sys.executable,
        pytest_args=["-q"],
        pythonpath=["."],
        timeout=120,
    )
    out = tmp_path / "runs" / "tiny.jsonl"
    run_experiment(target, [2, 4], out, tmp_path / "cache", workers=2, log=lambda m: None)
    records = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(records) == (4 + 2) * len(POLICY_SPECS)
    serial = [r for r in records if r["policy"] == "serial"]
    assert all(t["first_outcome"] == "accepted" for r in serial for t in r["tasks"])
    # f6/f7 edits sit on adjacent lines, and "retune f7" rewrites f7's line again.
    parallel = [r for r in records if policy_key(r) == "parallel" and r["size"] == 4]
    firsts = [t["first_outcome"] for r in parallel for t in r["tasks"]]
    assert "accepted" in firsts and ("agent-failed" in firsts or "textual" in firsts)
    assert (tmp_path / "runs" / "tiny-history.json").exists()
    # Rerunning resumes: nothing left to do, nothing appended.
    run_experiment(target, [2, 4], out, tmp_path / "cache", workers=2, log=lambda m: None)
    assert len(out.read_text().splitlines()) == len(records)
    summary = build_report(tmp_path / "runs", resamples=50, repos={"tiny": repo.path})
    states = summary["novel_states"]["tiny"]
    # Shuffled integration reaches states history never had; replayed history never breaks.
    assert 0 < states["novel_states_tested"] <= states["distinct_candidates_tested"]
    assert states["novel_states_that_broke_a_test"] == 0
    rows = {(r["scope"], r["size"], r["policy"]): r for r in summary["policies"]["rows"]}
    assert rows[("tiny", 2, "serial")]["first_attempt_failure"]["value"] == 0.0
    assert rows[("all", 4, "serial")]["makespan_vs_serial"]["value"] == 1.0
    assert rows[("all", 4, "parallel")]["makespan_vs_serial"]["value"] < 1.0
    assert "oracle-hunks" in summary["predictors"]


def _record(outcomes, size=None, policy="parallel", order="shuffled"):
    tasks = []
    for kind, detail, final, conflicted in outcomes:
        tasks.append(
            {
                "first_outcome": kind,
                "first_detail": detail,
                "final": final,
                "attempts": [{"conflicted": conflicted}],
            }
        )
    n = size or len(tasks)
    redos = sum(1 for t in tasks if t["first_outcome"] != "accepted")
    return {
        "target": "x",
        "size": n,
        "window": 0,
        "policy": policy,
        "predictor": None,
        "order": order,
        "tasks": tasks,
        "agent_runs": n + redos,
        "redos": redos,
        "makespan": 1 + redos,
    }


def test_window_counts_classify_every_outcome():
    c = window_counts(
        _record(
            [
                ("accepted", None, "accepted", []),
                ("textual", "merge", "accepted", ["CHANGES.rst"]),
                ("agent-failed", "base", "rejected", ["src/a.py"]),
                ("semantic", "interaction", "accepted", []),
            ]
        )
    )
    assert c["first_fail"] == 3 and c["textual"] == 2 and c["semantic"] == 1
    assert c["textual_files_changelog"] == 1 and c["textual_files_code"] == 1
    assert c["textual_base"] == 1
    assert c["semantic_interaction"] == 1 and c["rejected"] == 1
    assert c["window_failed"] == 1 and c["redos"] == 3 and c["landed"] == 3


def test_policy_labels():
    assert policy_key(_record([], policy="parallel", order="listed")) == "parallel:history-order"
    assert policy_key(_record([], policy="parallel")) == "parallel"
    assert policy_key({"policy": "predicted", "predictor": "oracle-files"}) == (
        "predicted:oracle-files"
    )


def test_bootstrap_interval_brackets_the_estimate_and_is_seeded():
    items = [{"hit": float(i % 3 == 0), "n": 1.0} for i in range(60)]
    strata = {"x": items}

    def stat(total):
        return total["hit"] / total["n"]

    a = bootstrap(strata, stat, random.Random(1), 300)
    b = bootstrap(strata, stat, random.Random(1), 300)
    assert a == b
    assert a["lo"] <= a["value"] <= a["hi"]
    assert a["hi"] - a["lo"] > 0.05


def test_load_targets(tmp_path):
    (tmp_path / "targets.toml").write_text(
        '[targets.demo]\nurl = "u"\npath = "t/demo"\nref = "abc"\nhistory = 50\n'
        'venv = "t/.venvs/demo"\npythonpath = ["src"]\npytest_args = ["tests"]\n'
    )
    target = load_targets(tmp_path / "targets.toml")["demo"]
    assert target.path == tmp_path / "t" / "demo"
    assert target.suite().pythonpath == ["src"] and target.history == 50


def test_conflict_kind():
    from worktree_fleet.report import conflict_kind

    assert conflict_kind(["CHANGES.rst", "src/a.py"]) == "code"
    assert conflict_kind(["CHANGES.rst", "docs/CHANGELOG.md"]) == "changelog"
    assert conflict_kind([".github/workflows/tests.yaml", "uv.lock"]) == "other"
    assert conflict_kind([]) == "other"
