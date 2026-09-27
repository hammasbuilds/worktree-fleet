"""A thin, typed layer over the git CLI.

Everything the fleet does to a repository goes through here: in-memory three-way merges
(`git merge-tree --write-tree`), commits built from trees, worktrees, and zero-context diffs.
Merges never touch a working directory, so a merge queue can try a candidate in
milliseconds and only check out the trees it actually needs to test.
"""

from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

# Synthetic commits (agent results, merge candidates) get a fixed identity and date so the
# same inputs always produce the same commit id. The user's own git identity is never used.
_FLEET_ENV = {
    "GIT_AUTHOR_NAME": "worktree-fleet",
    "GIT_AUTHOR_EMAIL": "fleet@localhost",
    "GIT_COMMITTER_NAME": "worktree-fleet",
    "GIT_COMMITTER_EMAIL": "fleet@localhost",
    "GIT_AUTHOR_DATE": "2000-01-01T00:00:00+0000",
    "GIT_COMMITTER_DATE": "2000-01-01T00:00:00+0000",
}

_HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


class GitError(RuntimeError):
    """A git command failed. Carries the command and its stderr."""

    def __init__(self, args: list[str], code: int, stderr: str) -> None:
        super().__init__(f"git {' '.join(args)} exited {code}: {stderr.strip()[:500]}")
        self.args_ = args
        self.code = code
        self.stderr = stderr

    def __reduce__(self) -> tuple[type[GitError], tuple[list[str], int, str]]:
        # Needed to cross a process boundary: the default pickling replays only str(self).
        return (GitError, (self.args_, self.code, self.stderr))


@dataclass(frozen=True)
class Hunk:
    """One changed region, in the coordinates of the *old* side of a diff.

    `start` is 1-based. A pure insertion has `length == 0` and sits after line `start`.
    """

    start: int
    length: int

    def span(self) -> tuple[float, float]:
        """Closed interval on the old side. Insertions become a half-line point."""
        if self.length == 0:
            return (self.start + 0.5, self.start + 0.5)
        return (float(self.start), float(self.start + self.length - 1))


@dataclass
class MergeResult:
    """Outcome of an in-memory three-way merge."""

    tree: str
    clean: bool
    conflicted: list[str] = field(default_factory=list)


class Git:
    """Run git against one repository (or one of its worktrees)."""

    def __init__(self, repo: Path | str, attributes_file: Path | None = None) -> None:
        self.repo = Path(repo).resolve()
        # Extra gitattributes applied to every command - how a run opts into a merge driver
        # (e.g. `CHANGES.rst merge=union`) without committing anything to the repository.
        self.attributes_file = attributes_file

    def run(
        self,
        *args: str,
        check: bool = True,
        input_: str | None = None,
        cwd: Path | None = None,
        fleet_identity: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        env = dict(os.environ)
        env["GIT_TERMINAL_PROMPT"] = "0"
        if fleet_identity:
            env.update(_FLEET_ENV)
        cmd = ["git", "-c", "core.quotepath=false", "-c", "core.autocrlf=false"]
        if self.attributes_file is not None:
            cmd += ["-c", f"core.attributesFile={self.attributes_file}"]
        cmd += args
        # Bytes, not text mode: text mode on Windows turns "\n" in stdin into "\r\n", which
        # corrupts `update-ref --stdin` and commit messages.
        raw = subprocess.run(
            cmd,
            cwd=cwd or self.repo,
            input=input_.encode("utf-8") if input_ is not None else None,
            capture_output=True,
            env=env,
        )
        proc = subprocess.CompletedProcess(
            raw.args,
            raw.returncode,
            raw.stdout.decode("utf-8", errors="replace"),
            raw.stderr.decode("utf-8", errors="replace"),
        )
        if check and proc.returncode != 0:
            raise GitError(list(args), proc.returncode, proc.stderr)
        return proc

    def out(self, *args: str, cwd: Path | None = None) -> str:
        return self.run(*args, cwd=cwd).stdout.strip()

    # --- objects --------------------------------------------------------------------

    def rev_parse(self, ref: str) -> str:
        return self.out("rev-parse", "--verify", f"{ref}^{{commit}}")

    def tree_of(self, commit: str) -> str:
        return self.out("rev-parse", "--verify", f"{commit}^{{tree}}")

    def parents(self, commit: str) -> list[str]:
        line = self.out("rev-list", "--parents", "-n", "1", commit)
        return line.split()[1:]

    def first_parent_chain(self, ref: str, limit: int) -> list[str]:
        """The last `limit` first-parent commits ending at `ref`, oldest first."""
        text = self.out("rev-list", "--first-parent", f"--max-count={limit}", ref)
        return list(reversed(text.split()))

    def message(self, commit: str) -> str:
        return self.out("log", "-1", "--format=%B", commit)

    def commit_tree(self, tree: str, parents: list[str], message: str) -> str:
        args = ["commit-tree", tree]
        for parent in parents:
            args += ["-p", parent]
        return self.run(*args, input_=message, fleet_identity=True).stdout.strip()

    def update_ref(self, ref: str, commit: str) -> None:
        self.run("update-ref", ref, commit)

    # --- merging --------------------------------------------------------------------

    def merge(self, ours: str, theirs: str, base: str | None = None) -> MergeResult:
        """Three-way merge of two commits without touching any working tree.

        With `base` given this is exactly a cherry-pick of `theirs` onto `ours` when `base`
        is the parent of `theirs`.
        """
        args = ["merge-tree", "--write-tree", "--name-only", "--no-messages"]
        if base is not None:
            args.append(f"--merge-base={base}")
        args += [ours, theirs]
        proc = self.run(*args, check=False)
        if proc.returncode not in (0, 1):
            raise GitError(args, proc.returncode, proc.stderr)
        lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
        tree = lines[0].strip()
        conflicted = sorted(set(lines[1:])) if proc.returncode == 1 else []
        return MergeResult(tree=tree, clean=proc.returncode == 0, conflicted=conflicted)

    def replay(self, commit: str, onto: str) -> MergeResult:
        """Replay `commit`'s change onto `onto` (a cherry-pick, done in memory)."""
        parents = self.parents(commit)
        if not parents:
            raise ValueError(f"{commit} is a root commit; there is no change to replay")
        return self.merge(onto, commit, base=parents[0])

    # --- diffs ----------------------------------------------------------------------

    def changed_files(self, a: str, b: str) -> list[str]:
        text = self.out("diff", "--name-only", "--no-renames", a, b)
        return [ln for ln in text.splitlines() if ln]

    def diff_hunks(self, a: str, b: str) -> dict[str, list[Hunk]]:
        """Zero-context hunks from `a` to `b`, keyed by path, in `a`'s line numbers."""
        text = self.run("diff", "-U0", "--no-color", "--no-ext-diff", "--no-renames", a, b).stdout
        return parse_unified_diff(text)

    def ls_files(self, rev: str) -> list[str]:
        text = self.out("ls-tree", "-r", "--name-only", rev)
        return [ln for ln in text.splitlines() if ln]

    def read_blobs(self, rev: str, paths: list[str]) -> dict[str, str]:
        """Read many files at `rev` with one `git cat-file --batch` process."""
        if not paths:
            return {}
        request = "".join(f"{rev}:{p}\n" for p in paths).encode("utf-8")
        proc = subprocess.run(
            ["git", "cat-file", "--batch"], cwd=self.repo, input=request, capture_output=True
        )
        if proc.returncode != 0:
            raise GitError(["cat-file", "--batch"], proc.returncode, proc.stderr.decode())
        data = proc.stdout
        blobs: dict[str, str] = {}
        pos = 0
        for path in paths:
            end = data.index(b"\n", pos)
            header = data[pos:end].split()
            pos = end + 1
            if len(header) == 3 and header[1] == b"blob":
                size = int(header[2])
                blobs[path] = data[pos : pos + size].decode("utf-8", errors="replace")
                pos += size + 1
        return blobs

    def history_changes(self, ref: str, limit: int) -> list[tuple[str, set[str]]]:
        """(commit, files changed vs first parent) for the last `limit` first-parent commits."""
        text = self.out(
            "log",
            "--first-parent",
            "--diff-merges=first-parent",
            "--no-renames",
            f"--max-count={limit}",
            "--name-only",
            "--format=%x00%H",
            ref,
        )
        changes = []
        for chunk in text.split("\x00"):
            lines = [ln.strip() for ln in chunk.splitlines() if ln.strip()]
            if lines:
                changes.append((lines[0], set(lines[1:])))
        return changes

    # --- worktrees ------------------------------------------------------------------

    def worktree_add(self, path: Path, commit: str) -> Path:
        # Absolute, because git resolves a relative path against the repository, not the
        # caller's working directory.
        path = Path(path).resolve()
        self.run("worktree", "add", "--detach", "--force", str(path), commit)
        return path

    def worktree_remove(self, path: Path) -> None:
        self.run("worktree", "remove", "--force", str(Path(path).resolve()), check=False)

    def worktree_prune(self) -> None:
        self.run("worktree", "prune", check=False)

    def checkout_tree(self, worktree: Path, commit: str) -> None:
        """Make `worktree`'s files exactly `commit`'s tree, removing untracked files."""
        self.run("checkout", "--force", "--detach", commit, cwd=worktree)
        self.run("clean", "-ffdxq", cwd=worktree)

    def commit_worktree(self, worktree: Path, message: str) -> str | None:
        """Commit everything in `worktree`. Returns None if nothing changed."""
        self.run("add", "-A", cwd=worktree)
        head = self.out("rev-parse", "HEAD", cwd=worktree)
        tree = self.out("write-tree", cwd=worktree)
        if tree == self.tree_of(head):
            return None
        # commit-tree rather than `git commit`: no hooks, no signing, no user identity.
        return self.commit_tree(tree, [head], message)


def parse_unified_diff(text: str) -> dict[str, list[Hunk]]:
    """Parse `git diff -U0` output into old-side hunks per file.

    A file that is added or deleted is still recorded (an added file gets one insertion
    hunk at line 0), so two changes that both create the same path overlap.
    """
    hunks: dict[str, list[Hunk]] = {}
    current: str | None = None
    old_path: str | None = None
    for line in text.splitlines():
        if line.startswith("diff --git "):
            current = None
            old_path = None
        elif line.startswith("--- "):
            old_path = _strip_prefix(line[4:])
        elif line.startswith("+++ "):
            new_path = _strip_prefix(line[4:])
            current = old_path if old_path is not None else new_path
            if current is None:
                current = new_path
            if current is not None:
                hunks.setdefault(current, [])
        elif line.startswith("Binary files ") and current is None:
            match = re.match(r"Binary files (.+) and (.+) differ", line)
            if match:
                path = _strip_prefix(match.group(1)) or _strip_prefix(match.group(2))
                if path is not None:
                    hunks.setdefault(path, []).append(Hunk(1, 1))
        elif line.startswith("@@") and current is not None:
            match = _HUNK_RE.match(line)
            if match:
                start = int(match.group(1))
                length = int(match.group(2)) if match.group(2) is not None else 1
                hunks[current].append(Hunk(start, length))
    return hunks


def _strip_prefix(path: str) -> str | None:
    path = path.strip()
    if path == "/dev/null":
        return None
    if path.startswith('"') and path.endswith('"'):
        path = path[1:-1]
    if path.startswith(("a/", "b/")):
        return path[2:]
    return path
