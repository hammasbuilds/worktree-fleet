"""Overlap predictors: guess, before tasks run, which of them will collide.

Three levels of information, from realistic to oracle:

* `DescriptionPredictor` sees only what exists before an agent starts - the task text, the
  repository at the base commit, and the history *before* the base. This is the one a real
  fleet can use.
* `FilePredictor` knows the exact set of files each task's real change touches (as if every
  agent declared its files up front and kept to them).
* `HunkPredictor` knows the exact lines each change touches relative to the shared base - an
  upper bound for any overlap-based scheduler.

A predictor turns a task into a `Footprint`; two tasks are predicted to conflict when their
footprints overlap.
"""

from __future__ import annotations

import ast
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Protocol

from .gitops import Git, Hunk
from .tasks import Footprint, Task


class Predictor(Protocol):
    name: str
    margin: int

    def footprint(self, task: Task, base: str) -> Footprint: ...


def predicted_conflicts(
    predictor: Predictor, tasks: list[Task], base: str
) -> dict[tuple[int, int], set[str]]:
    """Every pair (i, j), i < j, the predictor expects to conflict, with the shared files."""
    prints = [predictor.footprint(t, base) for t in tasks]
    pairs: dict[tuple[int, int], set[str]] = {}
    for j in range(len(tasks)):
        for i in range(j):
            shared = prints[i].overlapping_files(prints[j], predictor.margin)
            if shared:
                pairs[(i, j)] = shared
    return pairs


class FilePredictor:
    """Oracle: the exact file set of each task's real change."""

    name = "oracle-files"
    margin = 1

    def __init__(self, git: Git) -> None:
        self.git = git

    def footprint(self, task: Task, base: str) -> Footprint:
        commit = _require_commit(task)
        parent = self.git.parents(commit)[0]
        return Footprint(files=set(self.git.changed_files(parent, commit)))


class HunkPredictor:
    """Oracle: the exact lines each change touches, expressed against the shared base.

    The change is replayed onto the base in memory and diffed from there, so every task's
    hunks share one coordinate system. If the replay itself conflicts (the change was
    written on top of work the base does not have), the files are claimed whole.
    """

    name = "oracle-hunks"

    def __init__(self, git: Git, margin: int = 1) -> None:
        self.git = git
        self.margin = margin

    def footprint(self, task: Task, base: str) -> Footprint:
        commit = _require_commit(task)
        parent = self.git.parents(commit)[0]
        files = set(self.git.changed_files(parent, commit))
        replayed = self.git.replay(commit, base)
        if not replayed.clean:
            return Footprint(files=files)
        head = self.git.commit_tree(replayed.tree, [base], f"replay {task.id}")
        hunks = self.git.diff_hunks(base, head)
        return Footprint(files=files | set(hunks), hunks=hunks)


# --- the realistic predictor -----------------------------------------------------------

_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*")
_TRAILER = re.compile(r"^(co-authored-by|signed-off-by|reviewed-by)\s*:.*$", re.I | re.M)
_GENERIC = {
    "test",
    "tests",
    "src",
    "init",
    "main",
    "utils",
    "util",
    "core",
    "docs",
    "doc",
    "index",
    "setup",
    "conftest",
    "config",
    "readme",
    "the",
    "and",
    "for",
    "with",
    "from",
    "into",
    "fix",
    "add",
    "use",
    "remove",
    "update",
    "support",
    "when",
    "that",
    "this",
    "not",
    "merge",
    "pull",
    "request",
    "branch",
    "none",
    "true",
    "false",
    "self",
    "return",
}


@dataclass(frozen=True)
class RankedFile:
    path: str
    score: float
    spans: list[Hunk]


@dataclass
class _RepoIndex:
    symbols: dict[str, set[str]] = field(default_factory=dict)
    stems: dict[str, set[str]] = field(default_factory=dict)
    spans: dict[tuple[str, str], list[Hunk]] = field(default_factory=dict)
    hot: set[str] = field(default_factory=set)
    cochange: dict[str, Counter[str]] = field(default_factory=dict)
    touched: Counter[str] = field(default_factory=Counter)


class DescriptionPredictor:
    """Predict a task's files from its description and the repository before it starts.

    Signals, all computed at the base commit or from history strictly before it:

    1. code identifiers in the description that name a top-level function, class or method
       defined in exactly one or two files at the base;
    2. words that match a module's file stem (`termui` -> `src/click/termui.py`);
    3. "hot" files - changed by at least `hot_share` of the last `history` commits (a
       changelog is the typical one);
    4. co-change: for every predicted source file, the files that changed alongside it in at
       least `cochange_share` of the commits that touched it (usually its test module).

    `ranked` orders the same files by how directly the description points at them, and
    carries the line spans of any named symbols - what an LLM agent is shown first.
    """

    def __init__(
        self,
        git: Git,
        history: int = 300,
        hot_share: float = 0.25,
        cochange_share: float = 0.4,
        margin: int = 1,
    ) -> None:
        self.git = git
        self.history = history
        self.hot_share = hot_share
        self.cochange_share = cochange_share
        self.margin = margin
        self.name = "description"
        self._indexes: dict[str, _RepoIndex] = {}

    def footprint(self, task: Task, base: str) -> Footprint:
        return Footprint(files={r.path for r in self.ranked(task, base)})

    def ranked(self, task: Task, base: str) -> list[RankedFile]:
        """Predicted files, most directly named first.

        Score: 3 per named symbol the file defines (shared between the files defining it),
        2 for a file-stem match, the co-change share for a partner of a named file; "hot"
        files (changed by a large share of all commits) sort last unless the description
        names them.
        """
        index = self._index(base)
        scores: Counter[str] = Counter()
        spans: dict[str, list[Hunk]] = defaultdict(list)
        for word in _description_tokens(task.description):
            for name in {word, word.split(".")[-1]}:
                owners = index.symbols.get(name)
                if owners and len(owners) <= 2:
                    for path in owners:
                        scores[path] += 3 / len(owners)
                        spans[path].extend(index.spans.get((path, name), []))
            owners = index.stems.get(word.lower())
            if owners and len(owners) <= 2:
                for path in owners:
                    scores[path] += 2
        named = set(scores)
        for path in named:
            base_count = index.touched[path]
            if base_count < 2:
                continue
            for other, count in index.cochange.get(path, Counter()).items():
                share = count / base_count
                if share >= self.cochange_share:
                    scores[other] = max(scores[other], share)
        for path in index.hot:
            scores[path] = max(scores[path], 0.1)
        # A hot file (a changelog) is almost always edited and almost never the point of
        # the task: it goes last whatever else points at it.
        order = sorted(scores, key=lambda p: (p in index.hot and p not in named, -scores[p], p))
        return [
            RankedFile(p, scores[p], sorted(set(spans.get(p, [])), key=lambda h: h.start))
            for p in order
        ]

    def _index(self, base: str) -> _RepoIndex:
        if base in self._indexes:
            return self._indexes[base]
        index = _RepoIndex()
        paths = self.git.ls_files(base)
        for path in paths:
            stem = PurePosixPath(path).stem.lower()
            if stem not in _GENERIC and len(stem) >= 4:
                index.stems.setdefault(stem, set()).add(path)
        sources = self.git.read_blobs(base, [p for p in paths if p.endswith(".py")])
        for path, source in sources.items():
            for name, span in _defined_symbols(source):
                index.symbols.setdefault(name, set()).add(path)
                index.spans.setdefault((path, name), []).append(span)
        # History up to and including the base: none of it is in the future of any task.
        commits = self.git.history_changes(base, self.history)
        for _, changed in commits:
            for path in changed:
                index.touched[path] += 1
                partners = index.cochange.setdefault(path, Counter())
                for other in changed - {path}:
                    partners[other] += 1
        if commits:
            index.hot = {p for p, n in index.touched.items() if n / len(commits) >= self.hot_share}
        self._indexes[base] = index
        return index


def _description_tokens(text: str) -> set[str]:
    text = _TRAILER.sub("", text)
    text = re.sub(r"https?://\S+", " ", text)
    tokens = set()
    for match in _IDENT.finditer(text):
        token = match.group(0).strip(".")
        if len(token) >= 3 and token.lower() not in _GENERIC:
            tokens.add(token)
    return tokens


def _defined_symbols(source: str) -> list[tuple[str, Hunk]]:
    """Top-level functions/classes and their methods, with their line spans."""
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return []
    found: list[tuple[str, Hunk]] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            found.append((node.name, _span(node)))
            if isinstance(node, ast.ClassDef):
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        if not (item.name.startswith("__") and item.name.endswith("__")):
                            found.append((item.name, _span(item)))
    return found


def _span(node: ast.AST) -> Hunk:
    start = getattr(node, "lineno", 1)
    decorators = getattr(node, "decorator_list", [])
    if decorators:
        start = min(start, *(d.lineno for d in decorators))
    end = getattr(node, "end_lineno", None) or start
    return Hunk(start, end - start + 1)


# Changelog-style files: every change appends an entry at the same spot.
CHANGELOG = re.compile(r"(^|/)(changes|changelog|history|news)(\.[a-z]+)?$", re.I)


class ExcludingPredictor:
    """Wraps a predictor and drops files a merge driver makes conflict-free (changelogs
    under `merge=union`), so they no longer serialise the tasks that touch them."""

    def __init__(self, inner: Predictor, pattern: re.Pattern[str] = CHANGELOG) -> None:
        self.inner = inner
        self.pattern = pattern
        self.name = f"{inner.name}+changelog-union"
        self.margin = inner.margin

    def footprint(self, task: Task, base: str) -> Footprint:
        fp = self.inner.footprint(task, base)
        keep = {f for f in fp.files if not self.pattern.search(f)}
        return Footprint(files=keep, hunks={f: h for f, h in fp.hunks.items() if f in keep})


PREDICTOR_NAMES = ("description", "oracle-files", "oracle-hunks")


def build_predictor(name: str, git: Git) -> Predictor:
    """Construct a predictor by its CLI name."""
    if name == "description":
        return DescriptionPredictor(git)
    if name == "oracle-files":
        return FilePredictor(git)
    if name == "oracle-hunks":
        return HunkPredictor(git)
    raise ValueError(f"unknown predictor {name!r}; choose one of {', '.join(PREDICTOR_NAMES)}")


def _require_commit(task: Task) -> str:
    if task.commit is None:
        raise ValueError(f"task {task.id} has no commit; oracle predictors need real history")
    return task.commit
