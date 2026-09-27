import os

from worktree_fleet.suite import (
    SUITE_ERROR,
    SUITE_TIMEOUT,
    SuiteConfig,
    SuiteRunner,
    environment_fingerprint,
    parse_junit,
)

JUNIT = """<?xml version="1.0" encoding="utf-8"?>
<testsuites><testsuite name="pytest">
<testcase classname="tests.test_a" name="test_ok" />
<testcase classname="tests.test_a" name="test_bad"><failure message="x"/></testcase>
<testcase classname="tests.test_a" name="test_skip"><skipped message="s"/></testcase>
<testcase classname="" name="tests.test_b"><error message="collection failure"/></testcase>
</testsuite></testsuites>
"""


def test_parse_junit(tmp_path):
    path = tmp_path / "r.xml"
    path.write_text(JUNIT)
    failed, passed = parse_junit(path)
    assert failed == {"tests.test_a::test_bad", "::tests.test_b"}
    assert passed == 1


def _worktree(repo, tmp_path, commit):
    return repo.git.worktree_add(tmp_path / "wt", commit)


def test_runner_records_failures_and_caches_by_tree(repo, tmp_path, suite_config):
    good = repo.commit("good", {"test_x.py": "def test_x():\n    assert True\n"})
    bad = repo.commit("bad", {"test_x.py": "def test_x():\n    assert False\n"})
    wt = _worktree(repo, tmp_path, good)
    runner = SuiteRunner(repo.git, suite_config, tmp_path / "cache")
    assert not runner.result(good, wt).failed
    red = runner.result(bad, wt)
    assert red.failed == {"test_x::test_x"} and red.runs == 2
    runs = runner.fresh_runs
    # Same tree under a different commit id: served from the cache, no new run.
    twin = repo.git.commit_tree(repo.git.tree_of(bad), [good], "same tree")
    assert runner.result(twin, wt).failed == {"test_x::test_x"}
    assert runner.fresh_runs == runs


def test_flaky_test_is_not_a_failure(repo, tmp_path, suite_config):
    counter = tmp_path / "count.txt"
    counter.write_text("0")
    flaky = (
        "import os, pathlib\n"
        "def test_flip():\n"
        "    p = pathlib.Path(os.environ['FLIP_FILE'])\n"
        "    n = int(p.read_text()) + 1\n"
        "    p.write_text(str(n))\n"
        "    assert n % 2 == 0\n"
    )
    c = repo.commit("flaky", {"test_f.py": flaky})
    wt = _worktree(repo, tmp_path, c)
    config = SuiteConfig(**{**suite_config.__dict__, "env": {"FLIP_FILE": str(counter)}})
    result = SuiteRunner(repo.git, config, None).result(c, wt)
    assert result.failed == set()
    assert result.flaky == {"test_f::test_flip"}


FLIP = (
    "import os, pathlib\n"
    "def test_t():\n"
    "    p = pathlib.Path(os.environ['FLIP_FILE'])\n"
    "    n = int(p.read_text()) + 1\n"
    "    p.write_text(str(n))\n"
    "    assert n % 2 == 0\n"
)


def _flip_config(suite_config, tmp_path):
    counter = tmp_path / "count.txt"
    counter.write_text("0")
    return SuiteConfig(**{**suite_config.__dict__, "env": {"FLIP_FILE": str(counter)}})


def test_a_failure_confirmed_on_one_tree_is_still_rechecked_on_another(
    repo, tmp_path, suite_config
):
    """test_t fails for real on tree a and is flaky on tree b: b must come out flaky, not
    failing - a failure confirmed elsewhere is no evidence about this tree."""
    a = repo.commit("a", {"test_t.py": "def test_t():\n    assert False\n"})
    b = repo.commit("b", {"test_t.py": FLIP})
    wt = _worktree(repo, tmp_path, a)
    runner = SuiteRunner(repo.git, _flip_config(suite_config, tmp_path), tmp_path / "cache")
    assert runner.result(a, wt).failed == {"test_t::test_t"}
    flaky = runner.result(b, wt)
    assert flaky.failed == set() and flaky.flaky == {"test_t::test_t"}


def test_expected_failures_skip_the_rerun_but_only_for_that_caller(repo, tmp_path, suite_config):
    a = repo.commit("a", {"test_x.py": "def test_x():\n    assert False\n"})
    wt = _worktree(repo, tmp_path, a)
    runner = SuiteRunner(repo.git, suite_config, tmp_path / "cache")
    quick = runner.result(a, wt, expected={"test_x::test_x"})
    assert quick.runs == 1 and not quick.confirmed
    # A caller that did not expect the failure gets a fresh, confirmed run, not the cache.
    strict = runner.result(a, wt)
    assert strict.runs == 2 and strict.confirmed
    assert runner.cached(repo.git.tree_of(a), expected=frozenset()).confirmed


def test_broken_collection_and_empty_suite_are_errors(repo, tmp_path, suite_config):
    broken = repo.commit("syntax", {"test_s.py": "def test_s(:\n"})
    wt = _worktree(repo, tmp_path, broken)
    runner = SuiteRunner(repo.git, suite_config, None)
    assert runner.result(broken, wt).failed  # the collection error is a failure
    empty = repo.commit("empty", {"test_s.py": None, "readme.txt": "no tests\n"})
    nothing = runner.result(empty, wt)
    assert nothing.failed == {SUITE_ERROR}
    # An incomplete run is retried from a fresh checkout before it is believed, and the
    # reason is kept for whoever reads the result.
    assert nothing.runs == 3 and nothing.detail.startswith("exit 5")


def test_timeout_is_reported(repo, tmp_path, suite_config):
    c = repo.commit("slow", {"test_slow.py": "import time\ndef test_s():\n    time.sleep(30)\n"})
    wt = _worktree(repo, tmp_path, c)
    config = SuiteConfig(**{**suite_config.__dict__, "timeout": 3})
    assert SuiteRunner(repo.git, config, None).result(c, wt).failed == {SUITE_TIMEOUT}


def test_environment_fingerprint_is_stable(suite_config):
    first = environment_fingerprint(suite_config.python)
    assert first == environment_fingerprint(suite_config.python)
    assert len(first) == 10
    assert os.path.exists(suite_config.python)


def test_test_ids_do_not_depend_on_where_the_worktree_lives(tmp_path, suite_config):
    """A repo with no pytest config, checked out under a project that has one."""
    from conftest import RepoBuilder

    outer = tmp_path / "outer"
    outer.mkdir()
    (outer / "pyproject.toml").write_text('[tool.pytest.ini_options]\ntestpaths = ["nowhere"]\n')
    repo = RepoBuilder(tmp_path / "repo")
    c = repo.commit("c", {"tests/test_a.py": "def test_a():\n    assert False\n"})
    runner = SuiteRunner(repo.git, suite_config, None)
    ids = []
    for name in ("one", "two"):
        wt = repo.git.worktree_add(outer / name / "deeper", c)
        ids.append(runner.run_once(wt)[0])
    assert ids[0] == ids[1] == {"tests.test_a::test_a"}
