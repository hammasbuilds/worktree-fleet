import json
import random

import pytest

from worktree_fleet.gitops import Hunk
from worktree_fleet.predict import (
    DescriptionPredictor,
    FilePredictor,
    HunkPredictor,
    build_predictor,
    predicted_conflicts,
)
from worktree_fleet.tasks import Footprint, Task, load_tasks


def test_footprint_whole_file_claims_overlap_anything():
    a = Footprint(files={"x.py"})
    b = Footprint(files={"x.py"}, hunks={"x.py": [Hunk(100, 1)]})
    assert a.overlaps(b)
    assert not a.overlaps(Footprint(files={"y.py"}))


@pytest.mark.parametrize(
    ("a", "b", "margin", "expected"),
    [
        (Hunk(5, 1), Hunk(6, 1), 1, True),  # adjacent lines: git conflicts
        (Hunk(5, 1), Hunk(7, 1), 1, False),  # one untouched line between
        (Hunk(5, 1), Hunk(7, 1), 2, True),
        (Hunk(5, 0), Hunk(6, 1), 1, True),  # insertion right before an edit
        (Hunk(5, 0), Hunk(5, 0), 0, True),  # two insertions at the same point
    ],
)
def test_hunk_overlap_margin(a, b, margin, expected):
    fa = Footprint(files={"f"}, hunks={"f": [a]})
    fb = Footprint(files={"f"}, hunks={"f": [b]})
    assert fa.overlaps(fb, margin) is expected


def test_load_tasks_validates(tmp_path):
    good = tmp_path / "t.json"
    good.write_text(json.dumps([{"id": "a", "description": "do x"}, {"description": "y"}]))
    tasks = load_tasks(good)
    assert [t.id for t in tasks] == ["a", "task-2"]
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps([{"id": "a"}]))
    with pytest.raises(ValueError, match="description"):
        load_tasks(bad)
    dup = tmp_path / "dup.json"
    dup.write_text(json.dumps([{"id": "a", "description": "x"}, {"id": "a", "description": "y"}]))
    with pytest.raises(ValueError, match="unique"):
        load_tasks(dup)


def _lines(n: int, mark: dict[int, str] | None = None) -> str:
    mark = mark or {}
    return "".join(mark.get(i, f"line {i}") + "\n" for i in range(1, n + 1))


def test_hunk_predictor_never_misses_a_git_conflict(repo):
    """Random edits to one file: whenever git refuses the merge, the predictor flagged it."""
    rng = random.Random(7)
    base = repo.commit("base", {"f.txt": _lines(40)})
    git = repo.git
    predictor = HunkPredictor(git)
    misses = conflicts = 0
    for trial in range(25):
        repo.checkout(base)
        i = rng.randint(1, 40)
        a = repo.commit(f"a{trial}", {"f.txt": _lines(40, {i: f"A{trial}"})})
        repo.checkout(base)
        j = rng.randint(1, 40)
        b = repo.commit(f"b{trial}", {"f.txt": _lines(40, {j: f"B{trial}"})})
        truth = not git.merge(a, b, base=base).clean
        fa = predictor.footprint(Task("a", "", a), base)
        fb = predictor.footprint(Task("b", "", b), base)
        conflicts += truth
        misses += truth and not fa.overlaps(fb, predictor.margin)
    assert conflicts > 0
    assert misses == 0


def test_file_predictor_uses_the_real_change(repo):
    repo.commit("base", {"a.py": "1\n", "b.py": "2\n"})
    c = repo.commit("c", {"b.py": "3\n"})
    assert FilePredictor(repo.git).footprint(Task("c", "", c), c).files == {"b.py"}


def test_hunk_predictor_falls_back_to_files_when_replay_conflicts(repo):
    base = repo.commit("base", {"a.py": "x\n"})
    repo.commit("c1", {"a.py": "y\n"})
    c2 = repo.commit("c2", {"a.py": "z\n"})
    fp = HunkPredictor(repo.git).footprint(Task("c2", "", c2), base)
    assert fp.files == {"a.py"} and fp.hunks == {}


def _description_repo(repo):
    repo.commit(
        "base",
        {
            "pkg/termui.py": "def echo_via_pager(x):\n    return x\n\n\ndef prompt(x):\n"
            "    return x\n",
            "pkg/parser.py": "class OptionParser:\n    def parse_args(self):\n        return 1\n",
            "tests/test_termui.py": "def test_pager():\n    pass\n",
            "tests/test_parser.py": "def test_parse():\n    pass\n",
            "CHANGES.rst": "changes\n",
        },
    )
    for n in range(4):
        repo.commit(
            f"pager {n}",
            {"pkg/termui.py": f"def echo_via_pager(x):\n    return x  # {n}\n\n\ndef prompt(x):\n"
             "    return x\n", "tests/test_termui.py": f"def test_pager():\n    pass  # {n}\n",
             "CHANGES.rst": f"changes {n}\n"},
        )
    return repo.commit("parser", {"pkg/parser.py": "class OptionParser:\n    def parse_args("
                                  "self):\n        return 2\n"})


def test_description_predictor_signals(repo):
    head = _description_repo(repo)
    predictor = DescriptionPredictor(repo.git, hot_share=0.6, cochange_share=0.5)
    fp = predictor.footprint(Task("t", "Fix `echo_via_pager` hanging on Windows"), head)
    # the symbol's file, its co-changed test, and the hot changelog
    assert {"pkg/termui.py", "tests/test_termui.py", "CHANGES.rst"} <= fp.files
    assert "pkg/parser.py" not in fp.files
    stem = predictor.footprint(Task("t", "parser: accept empty values"), head)
    assert "pkg/parser.py" in stem.files


def test_description_spans_claim_only_the_named_symbol(repo):
    head = _description_repo(repo)
    spans = DescriptionPredictor(repo.git, hot_share=1.1, symbol_spans=True)
    a = spans.footprint(Task("a", "change echo_via_pager"), head)
    b = spans.footprint(Task("b", "change prompt"), head)
    assert a.hunks["pkg/termui.py"] == [Hunk(1, 2)]
    assert b.hunks["pkg/termui.py"] == [Hunk(5, 2)]
    # Same file, different functions: the source file no longer counts as a collision
    # (the co-changed test module and changelog still do - they are claimed whole).
    assert "pkg/termui.py" not in a.overlapping_files(b, margin=1)


def test_description_predictor_ignores_commit_trailers(repo):
    head = _description_repo(repo)
    predictor = DescriptionPredictor(repo.git, hot_share=1.1)
    fp = predictor.footprint(Task("t", "tidy\n\nCo-authored-by: echo_via_pager <x@y>"), head)
    assert fp.files == set()


def test_predicted_conflicts_and_builder(repo):
    base = repo.commit("base", {"a.py": "1\n", "b.py": "1\n"})
    c1 = repo.commit("c1", {"a.py": "2\n"})
    c2 = repo.commit("c2", {"b.py": "2\n"})
    c3 = repo.commit("c3", {"a.py": "3\n"})
    tasks = [Task(c[:7], "", c) for c in (c1, c2, c3)]
    pairs = predicted_conflicts(build_predictor("oracle-files", repo.git), tasks, base)
    assert pairs == {(0, 2): {"a.py"}}
    with pytest.raises(ValueError, match="unknown predictor"):
        build_predictor("psychic", repo.git)
    with pytest.raises(ValueError, match="no commit"):
        FilePredictor(repo.git).footprint(Task("x", "no commit"), base)
