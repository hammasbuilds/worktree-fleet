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

    When `symbol_spans` is on, a file reached only through named symbols is claimed as just
    those symbols' line spans instead of whole.
    """

    def __init__(
        self,
        git: Git,
        history: int = 300,
        hot_share: float = 0.25,
        cochange_share: float = 0.4,
        symbol_spans: bool = False,
        margin: int = 1,
    ) -> None:
        self.git = git
        self.history = history
        self.hot_share = hot_share
        self.cochange_share = cochange_share
        self.symbol_spans = symbol_spans
        self.margin = margin
        self.name = "description-spans" if symbol_spans else "description"
        self._indexes: dict[str, _RepoIndex] = {}

    def footprint(self, task: Task, base: str) -> Footprint:
        index = self._index(base)
        words = _description_tokens(task.description)
        via_symbol: dict[str, list[Hunk]] = defaultdict(list)
        whole: set[str] = set()
        for word in words:
            for name in {word, word.split(".")[-1]}:
                owners = index.symbols.get(name)
                if owners and len(owners) <= 2:
                    for path in owners:
                        via_symbol[path].extend(index.spans.get((path, name), []))
            owners = index.stems.get(word.lower())
            if owners and len(owners) <= 2:
                whole |= owners
        seeds = set(via_symbol) | whole
        files = seeds | index.hot
        for path in seeds:
            partners = index.cochange.get(path)
            if not partners:
                continue
            base_count = index.touched[path]
            for other, count in partners.items():
                if base_count >= 2 and count / base_count >= self.cochange_share:
                    files.add(other)
        hunks: dict[str, list[Hunk]] = {}
        if self.symbol_spans:
            for path, spans in via_symbol.items():
                if path not in whole and path not in index.hot and spans:
                    hunks[path] = spans
        return Footprint(files=files, hunks=hunks)

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


PREDICTOR_NAMES = ("description", "description-spans", "oracle-files", "oracle-hunks")


def build_predictor(name: str, git: Git) -> Predictor:
    """Construct a predictor by its CLI name."""
    if name == "description":
        return DescriptionPredictor(git)
    if name == "description-spans":
        return DescriptionPredictor(git, symbol_spans=True)
    if name == "oracle-files":
        return FilePredictor(git)
    if name == "oracle-hunks":
        return HunkPredictor(git)
    raise ValueError(f"unknown predictor {name!r}; choose one of {', '.join(PREDICTOR_NAMES)}")


def _require_commit(task: Task) -> str:
    if task.commit is None:
        raise ValueError(f"task {task.id} has no commit; oracle predictors need real history")
    return task.commit
