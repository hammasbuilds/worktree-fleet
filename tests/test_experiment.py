import json
import random
import subprocess
import sys

import pytest

from worktree_fleet.experiment import (
    CONSECUTIVE,
    INDEPENDENT,
    POLICY_SPECS,
    Window,
    _Lock,
    healthy,
    history_tasks,
    housekeeping_reason,
    make_windows,
    parse_specs,
    run_experiment,
    spread,
)
from worktree_fleet.report import (
    bootstrap_many,
    build_report,
    conflict_kind,
    semantic_facts,
    window_counts,
)
from worktree_fleet.suite import SUITE_ERROR, SuiteResult
from worktree_fleet.targets import Target, load_targets


def _result(failed=(), passed=100):
    return SuiteResult("t", set(failed), set(), passed, 1.0, 1)


def test_healthy_thresholds():
    assert healthy(_result())
    assert healthy(_result(failed=["a"]))
    assert not healthy(None)
    assert not healthy(_result(failed=[SUITE_ERROR]))
    assert not healthy(_result(failed=[f"t{i}" for i in range(6)]))


@pytest.mark.parametrize(
    ("author", "subject", "housekeeping"),
    [
        ("dependabot[bot]", "Bump actions/checkout from 3 to 4", True),
        ("Jane", "Merge branch 'stable'", True),
        ("Jane", "Merge branch '2.3.x'", True),
        ("Jane", "Merge remote-tracking branch 'origin/main'", True),
        ("Jane", "Merge pull request #12 from pallets/stable", True),
        ("Jane", "update dev dependencies", True),
        ("Jane", "Release version 8.1.2", True),
        ("Jane", "start version 3.2.0", True),
        ("Jane", "Merge branch 'valtron-add-chunked-even'", False),
        ("Jane", "Merge pull request #539 from jane/master", False),
        ("Jane", "fix echo_via_pager on Windows", False),
    ],
)
def test_housekeeping_rule(author, subject, housekeeping):
    assert (housekeeping_reason(author, subject) is not None) is housekeeping


def test_history_skips_empty_changes_and_housekeeping(repo):
    c = [repo.commit("base", {"a": "0"})]
    for i in range(1, 6):
        c.append(repo.commit(f"c{i}", {"a": str(i)}))
    bump = repo.commit("Bump version to 1.0", {"v": "1"})
    empty = repo.commit("empty", {})
    history = history_tasks(repo.git, empty, 10)
    assert history.tasks == c[1:6]  # root has no parent; bump is housekeeping; empty is nothing
    assert list(history.housekeeping) == [bump]
    results = {x: _result() for x in c}
    windows = make_windows(repo.git, history.tasks, [2], results)
    assert [w.commits for w in windows] == [tuple(c[1:3]), tuple(c[3:5])]
    results[c[4]] = _result(failed=[SUITE_ERROR])
    assert len(make_windows(repo.git, history.tasks, [2], results)) == 1
    # A commit with no cached result is simply not eligible - never a KeyError.
    del results[c[1]]
    assert len(make_windows(repo.git, history.tasks, [2], results)) == 0


def test_independent_windows_skip_changes_that_need_in_flight_work(repo):
    lines = [f"{i}\n" for i in range(20)]
    base = repo.commit("base", {"f.txt": "".join(lines)})
    commits = []
    for i, text in [(0, "a"), (0, "a2"), (10, "b"), (15, "c")]:
        lines[i] = f"{text}\n"
        commits.append(repo.commit(f"edit {text}", {"f.txt": "".join(lines)}))
    results = {x: _result() for x in [base, *commits]}
    windows = make_windows(repo.git, commits, [3], results, mode=INDEPENDENT)
    # "a2" rewrites the line "a" wrote, so it cannot start from the base; the window skips it.
    assert [w.commits for w in windows] == [(commits[0], commits[2], commits[3])]
    assert windows[0].scanned == 4 and windows[0].dependent == 1
    with pytest.raises(ValueError, match="mode"):
        make_windows(repo.git, commits, [3], results, mode="random")


def test_spread_picks_evenly_per_mode_and_size():
    windows = [Window(CONSECUTIVE, 2, i, "b", ()) for i in range(10)]
    windows += [Window(INDEPENDENT, 2, i, "b", ()) for i in range(2)]
    picked = spread(windows, 3)
    assert [w.index for w in picked if w.mode == CONSECUTIVE] == [0, 3, 6]
    assert [w.index for w in picked if w.mode == INDEPENDENT] == [0, 1]


def test_parse_specs():
    assert parse_specs("serial, parallel") == [POLICY_SPECS[0], POLICY_SPECS[2]]
    with pytest.raises(ValueError, match="unknown policy"):
        parse_specs("parallel,magic")
    with pytest.raises(ValueError, match="no policies"):
        parse_specs(" , ")


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


def _tiny_target(repo, head):
    return Target(
        name="tiny",
        path=repo.path,
        url="local",
        ref=head,
        history=9,
        python=sys.executable,
        pytest_args=["-q"],
        pythonpath=["."],
        timeout=120,
    )


def test_run_experiment_end_to_end(repo, tmp_path):
    head = _history_repo(repo)
    target = _tiny_target(repo, head)
    out = tmp_path / "runs" / "tiny.jsonl"
    quiet = {"log": lambda m: None}
    run_experiment(target, [2, 4], out, tmp_path / "cache", workers=2, **quiet)
    records = [json.loads(line) for line in out.read_text().splitlines()]
    consecutive = [r for r in records if r["mode"] == CONSECUTIVE]
    assert len(consecutive) == (4 + 2) * len(POLICY_SPECS)
    serial = [r for r in consecutive if r["label"] == "serial"]
    assert all(t["first_outcome"] == "accepted" for r in serial for t in r["tasks"])
    parallel = [r for r in consecutive if r["label"] == "parallel" and r["size"] == 4]
    firsts = [t["first_outcome"] for r in parallel for t in r["tasks"]]
    # "retune f7" rewrites f7's line again: from a base without "tune f7" it cannot apply.
    assert "accepted" in firsts and "base-conflict" in firsts
    # In independent windows that change is never picked, so nothing conflicts at the base.
    independent = [r for r in records if r["mode"] == INDEPENDENT]
    assert independent
    assert all(
        t["first_outcome"] != "base-conflict"
        for r in independent
        if r["label"] == "parallel"
        for t in r["tasks"]
    )
    history = json.loads((tmp_path / "runs" / "tiny-history.json").read_text())
    assert len(history["commits"]) == 8
    # Rerunning resumes: nothing left to do, nothing appended.
    run_experiment(target, [2, 4], out, tmp_path / "cache", workers=2, **quiet)
    assert len(out.read_text().splitlines()) == len(records)
    summary = build_report(tmp_path / "runs", resamples=50, repos={"tiny": repo.path})
    assert summary["repos_missing"] == []
    states = summary["semantic"]["tiny"][CONSECUTIVE]
    assert 0 < states["novel_states_tested"] <= states["distinct_candidates_tested"]
    assert states["semantic_attempts"] == 0
    rows = {(r["scope"], r["mode"], r["size"], r["policy"]): r for r in summary["policies"]["rows"]}
    serial4 = rows[("all", CONSECUTIVE, 4, "serial")]
    assert serial4["makespan_vs_serial"]["value"] == 1.0 and serial4["landed"]["value"] == 1.0
    assert rows[("all", CONSECUTIVE, 4, "parallel")]["base_conflict"]["value"] > 0
    predictors = {(p["predictor"], p["mode"], p["size"], p["scope"]) for p in summary["predictors"]}
    assert ("oracle-hunks", CONSECUTIVE, 4, "all") in predictors
    assert ("oracle-hunks", CONSECUTIVE, "all", "tiny") in predictors


def test_dry_run_touches_nothing_and_survives_a_partial_cache(repo, tmp_path):
    head = _history_repo(repo)
    target = _tiny_target(repo, head)
    cache = tmp_path / "cache"
    # Another experiment is "running": its worktree and its lock must both survive.
    live = cache / "work" / "tiny" / "q-live"
    repo.git.worktree_add(live, head)
    lock = cache / "work" / "tiny" / "experiment.lock"
    lock.write_text("123")
    lines = []
    run_experiment(target, [2], tmp_path / "out.jsonl", cache, dry_run=True, log=lines.append)
    assert live.exists() and lock.exists()
    assert not (tmp_path / "out.jsonl").exists()
    config = subprocess.run(
        ["git", "config", "--get", "gc.auto"], cwd=repo.path, capture_output=True, text=True
    )
    assert config.stdout.strip() == ""
    assert any("0/" in line and "cached" in line for line in lines)
    # The real run refuses to start while the lock is held.
    with pytest.raises(RuntimeError, match="another experiment"):
        run_experiment(target, [2], tmp_path / "out.jsonl", cache, log=lines.append)


def test_lock_is_released(tmp_path):
    with _Lock(tmp_path):
        assert (tmp_path / "experiment.lock").exists()
    assert not (tmp_path / "experiment.lock").exists()


def _record(outcomes, mode=CONSECUTIVE, label="parallel"):
    tasks = []
    for i, (kind, detail, final, conflicted) in enumerate(outcomes):
        attempts = [
            {"outcome": kind, "detail": detail, "conflicted": conflicted, "candidate": f"cand{i}"}
        ]
        if final == "accepted" and kind != "accepted":
            attempts.append(
                {"outcome": "accepted", "detail": None, "conflicted": [], "candidate": f"redo{i}"}
            )
        tasks.append(
            {
                "id": f"t{i}",
                "first_outcome": kind,
                "first_detail": detail,
                "final": final,
                "attempts": attempts,
            }
        )
    n = len(tasks)
    redos = sum(len(t["attempts"]) - 1 for t in tasks)
    return {
        "target": "x",
        "mode": mode,
        "size": n,
        "window": 0,
        "label": label,
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
                ("base-conflict", "base", "rejected", ["src/a.pyi"]),
                ("semantic", "interaction", "accepted", []),
                ("agent-error", None, "rejected", []),
            ]
        )
    )
    assert c["first_fail"] == 4
    assert c["base-conflict"] == 1 and c["textual"] == 1 and c["semantic"] == 1
    # An agent that failed on its own is not a conflict of any kind.
    assert c["agent_error"] == 1
    assert c["conflict_files_changelog"] == 1 and c["conflict_files_code"] == 1
    assert c["semantic_interaction"] == 1 and c["rejected"] == 2
    assert c["landed"] == 3


def test_semantic_facts_count_redo_failures_and_changes(tmp_path):
    rec = _record([("accepted", None, "accepted", [])])
    rec["tasks"][0]["attempts"] = [
        {"outcome": "base-conflict", "detail": "base", "conflicted": [], "candidate": None},
        {"outcome": "semantic", "detail": "stale-base", "conflicted": [], "candidate": "c1"},
    ]
    other = _record([("semantic", "interaction", "accepted", [])])
    other["tasks"][0]["id"] = "t9"
    facts = semantic_facts([rec, other], {}, tmp_path)["x"][CONSECUTIVE]
    assert facts["semantic_attempts"] == 2
    assert facts["change_counts"] == {"interaction": 1, "stale-base": 1}
    assert facts["distinct_changes"] == 2


def test_bootstrap_interval_brackets_the_estimate_and_is_seeded():
    items = [{"hit": float(i % 3 == 0), "n": 1.0} for i in range(60)]

    def stat(total):
        return total["hit"] / total["n"]

    a = bootstrap_many({"x": items}, {"s": stat}, random.Random(1), 300)["s"]
    b = bootstrap_many({"x": items}, {"s": stat}, random.Random(1), 300)["s"]
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
    assert conflict_kind(["CHANGES.rst", "src/a.py"]) == "code"
    assert conflict_kind(["src/a.pyi"]) == "code"
    assert conflict_kind(["CHANGES.rst", "docs/CHANGELOG.md"]) == "changelog"
    assert conflict_kind([".github/workflows/tests.yaml", "uv.lock"]) == "other"
    assert conflict_kind([]) == "other"
