import pytest

from worktree_fleet.agents import AgentResult, ReplayAgent
from worktree_fleet.fleet import NOOP, Fleet, plan_waves
from worktree_fleet.mergequeue import ACCEPTED, AGENT_FAILED, SEMANTIC, TEXTUAL, MergeQueue
from worktree_fleet.predict import FilePredictor
from worktree_fleet.suite import SuiteRunner
from worktree_fleet.tasks import Task


def test_plan_waves_keeps_order_and_separates_conflicts():
    assert plan_waves(4, set()) == [[0, 1, 2, 3]]
    assert plan_waves(4, {(0, 2), (2, 3)}) == [[0, 1], [2], [3]]
    assert plan_waves(3, {(0, 1), (0, 2), (1, 2)}) == [[0], [1], [2]]
    assert plan_waves(0, set()) == []


def _fleet(repo, tmp_path, runner=None, **kw):
    qtree = None
    if runner is not None:
        qtree = tmp_path / "queue"

    def factory(base, ref):
        if runner is not None and not qtree.exists():
            repo.git.worktree_add(qtree, base)
        return MergeQueue(repo.git, base, ref, runner, qtree)

    return Fleet(repo.git, ReplayAgent(), tmp_path / "agents", factory, **kw)


def _independent_edits(repo):
    """Three tasks from one base: a and c edit the same line, b edits another file."""
    base = repo.commit("base", {"a.txt": "1\n2\n3\n", "b.txt": "x\n"})
    a = repo.commit("a", {"a.txt": "one\n2\n3\n"})
    repo.checkout(base)
    b = repo.commit("b", {"b.txt": "y\n"})
    repo.checkout(base)
    c = repo.commit("c", {"a.txt": "uno\n2\n3\n"})
    return base, [Task("a", "edit a", a), Task("b", "edit b", b), Task("c", "edit a again", c)]


def test_parallel_fleet_hits_a_textual_conflict_and_redo_cannot_fix_it(repo, tmp_path):
    base, tasks = _independent_edits(repo)
    report = _fleet(repo, tmp_path, order="listed").run(tasks, base, "parallel", run_id="p")
    outcomes = [r.first.outcome for r in report.records]
    assert outcomes == [ACCEPTED, ACCEPTED, TEXTUAL]
    # The redo replays c onto the new main; c still rewrites a line a already changed.
    assert [a.outcome for a in report.records[2].attempts] == [TEXTUAL, AGENT_FAILED]
    assert report.records[2].final == "rejected"
    assert report.makespan == 1 + 1 and report.agent_runs == 4
    assert repo.git.show_file(report.main, "a.txt") == "one\n2\n3\n"
    assert repo.git.show_file(report.main, "b.txt") == "y\n"


def test_serial_fleet_replays_real_history_exactly(repo, tmp_path):
    base = repo.commit("base", {"f.txt": "1\n2\n3\n4\n5\n6\n"})
    c1 = repo.commit("c1", {"f.txt": "one\n2\n3\n4\n5\n6\n"})
    c2 = repo.commit("c2", {"f.txt": "one\ntwo\n3\n4\n5\n6\n"})
    tasks = [Task("c1", "", c1), Task("c2", "", c2)]
    report = _fleet(repo, tmp_path).run(tasks, base, "serial", run_id="s")
    assert all(r.final == ACCEPTED for r in report.records)
    assert repo.git.tree_of(report.main) == repo.git.tree_of(c2)
    assert report.makespan == 2
    # In parallel, c2 was written on top of c1's line: it cannot even apply to the base.
    par = _fleet(repo, tmp_path, order="listed").run(tasks, base, "parallel", run_id="p")
    assert par.records[1].first.outcome == AGENT_FAILED
    assert par.records[1].attempts[1].outcome == ACCEPTED  # the redo on the new main works
    assert repo.git.tree_of(par.main) == repo.git.tree_of(c2)


def test_predicted_waves_avoid_the_conflict(repo, tmp_path):
    base, tasks = _independent_edits(repo)
    predictor = FilePredictor(repo.git)
    report = _fleet(repo, tmp_path).run(tasks, base, "predicted", predictor, run_id="w")
    assert [[report.records[i].task.id for i in w] for w in report.waves] == [["a", "b"], ["c"]]
    # c now starts from a main that already has a, and its replay conflicts at the base.
    assert report.records[2].first.outcome == AGENT_FAILED


def test_semantic_conflict_is_caught_by_the_test_gate(semantic_history, tmp_path, suite_config):
    repo, base, (c1, c2) = semantic_history
    runner = SuiteRunner(repo.git, suite_config, tmp_path / "cache")
    tasks = [Task("rename", "rename greet", c1), Task("shout", "add shout", c2)]
    report = _fleet(repo, tmp_path, runner, order="listed").run(
        tasks, base, "parallel", run_id="sem"
    )
    first = report.records[1].first
    assert first.outcome == SEMANTIC
    assert first.new_failures == ["test_shout::test_shout"]
    # Main never took the broken merge.
    assert repo.git.show_file(report.main, "test_shout.py") is None


def test_queue_does_not_blame_a_failure_main_already_had(repo, tmp_path, suite_config):
    base = repo.commit(
        "base",
        {"test_old.py": "def test_old():\n    assert False\n", "a.txt": "1\n"},
    )
    c = repo.commit("c", {"a.txt": "2\n"})
    runner = SuiteRunner(repo.git, suite_config, None)
    repo.git.worktree_add(tmp_path / "q", base)
    queue = MergeQueue(repo.git, base, "refs/fleet/t/main", runner, tmp_path / "q")
    assert queue.integrate(c, "c").outcome == ACCEPTED


def test_queue_honours_the_ignore_set(repo, tmp_path, suite_config):
    base = repo.commit("base", {"test_t.py": "def test_t():\n    assert True\n"})
    c = repo.commit("c", {"test_t.py": "def test_t():\n    assert False\n"})
    runner = SuiteRunner(repo.git, suite_config, None)
    repo.git.worktree_add(tmp_path / "q", base)
    strict = MergeQueue(repo.git, base, "refs/fleet/a/main", runner, tmp_path / "q")
    assert strict.integrate(c, "c").outcome == SEMANTIC
    lenient = MergeQueue(
        repo.git, base, "refs/fleet/b/main", runner, tmp_path / "q", ignore={"test_t::test_t"}
    )
    assert lenient.integrate(c, "c").outcome == ACCEPTED


def test_queue_requires_a_worktree_to_test(repo, suite_config):
    base = repo.commit("base", {"a": "1"})
    with pytest.raises(ValueError, match="worktree"):
        MergeQueue(repo.git, base, "refs/fleet/x/main", SuiteRunner(repo.git, suite_config, None))


class _NoChange:
    name = "idle"

    def work(self, task, worktree, git):
        return AgentResult(True)


def test_agent_that_changes_nothing_is_a_noop(repo, tmp_path):
    base = repo.commit("base", {"a": "1"})

    def factory(start, ref):
        return MergeQueue(repo.git, start, ref)

    fleet = Fleet(repo.git, _NoChange(), tmp_path / "w", factory)
    report = fleet.run([Task("t", "nothing")], base, "parallel", run_id="n")
    assert report.records[0].final == NOOP and report.redos == 0


def test_completion_order_integrates_everything_and_cleans_worktrees(repo, tmp_path):
    base, tasks = _independent_edits(repo)
    report = _fleet(repo, tmp_path).run(tasks[:2], base, "parallel", run_id="live")
    assert sorted(report.integration_order) == [0, 1]
    assert all(r.final == ACCEPTED for r in report.records)
    assert not any((tmp_path / "agents").iterdir())


def test_unknown_policy_and_order_are_rejected(repo, tmp_path):
    base, tasks = _independent_edits(repo)
    with pytest.raises(ValueError, match="order"):
        _fleet(repo, tmp_path, order="alphabetical")
    with pytest.raises(ValueError, match="policy"):
        _fleet(repo, tmp_path).run(tasks, base, "yolo")
    with pytest.raises(ValueError, match="predictor"):
        _fleet(repo, tmp_path).run(tasks, base, "predicted")


def test_failed_task_is_requeued_behind_the_rest_of_its_wave(repo, tmp_path):
    """The dependent change arrives first; its redo waits until its dependency has landed."""
    base = repo.commit("base", {"f.txt": "1\n2\n3\n4\n5\n6\n"})
    c1 = repo.commit("c1", {"f.txt": "one\n2\n3\n4\n5\n6\n"})
    c2 = repo.commit("c2", {"f.txt": "one\ntwo\n3\n4\n5\n6\n"})
    tasks = [Task("c2", "", c2), Task("c1", "", c1)]
    report = _fleet(repo, tmp_path, order="listed").run(tasks, base, "parallel", run_id="rq")
    assert [a.outcome for a in report.records[0].attempts] == [AGENT_FAILED, ACCEPTED]
    assert report.records[1].final == ACCEPTED
    assert repo.git.tree_of(report.main) == repo.git.tree_of(c2)


@pytest.mark.parametrize("policy", ["serial", "parallel"])
def test_in_memory_replay_matches_the_worktree_path(repo, tmp_path, policy):
    base, tasks = _independent_edits(repo)
    live = _fleet(repo, tmp_path / "a", order="listed").run(tasks, base, policy, run_id="a")
    fast = _fleet(repo, tmp_path / "b", order="listed", in_memory=True).run(
        tasks, base, policy, run_id="b"
    )
    assert repo.git.tree_of(live.main) == repo.git.tree_of(fast.main)
    for x, y in zip(live.records, fast.records, strict=True):
        assert [a.outcome for a in x.attempts] == [a.outcome for a in y.attempts]


class _Crashing:
    name = "crashing"

    def work(self, task, worktree, git):
        raise RuntimeError("boom")


def test_a_crashing_agent_fails_its_task_not_the_fleet(repo, tmp_path):
    base = repo.commit("base", {"a": "1"})

    def factory(start, ref):
        return MergeQueue(repo.git, start, ref)

    fleet = Fleet(repo.git, _Crashing(), tmp_path / "w", factory, retries=0)
    report = fleet.run([Task("t", "x")], base, "parallel", run_id="crash")
    assert report.records[0].first.outcome == AGENT_FAILED
    assert "boom" in report.records[0].first.note
