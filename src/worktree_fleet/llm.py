"""An LLM coding agent backed by a local Ollama server.

The agent sees the task description and the files a `DescriptionPredictor` expects the task
to touch, and answers with SEARCH/REPLACE edit blocks, which are applied to its worktree.
Every generation is cached on disk under a key of (model, prompt hash, options), so an
interrupted run resumes without repeating a single call.
"""

from __future__ import annotations

import hashlib
import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .agents import AgentResult
from .gitops import Git, Hunk
from .predict import DescriptionPredictor
from .tasks import Task

DEFAULT_OPTIONS = {"temperature": 0.0, "seed": 0, "num_ctx": 16384, "num_predict": 2048}

PROMPT = """You are a software engineer working in the repository below. Make the change the
task asks for. Reply ONLY with edit blocks in exactly this format, one per edit:

path/to/file.py
<<<<<<< SEARCH
exact lines currently in the file
=======
the lines that replace them
>>>>>>> REPLACE

Rules: SEARCH must match the file exactly, including indentation. To create a new file, use
an empty SEARCH section. Keep edits minimal. Do not explain.

## Task
{task}

## Files
{files}
"""


class Client(Protocol):
    def generate(self, model: str, prompt: str) -> str: ...


# Ollama is local: never route it through HTTP(S)_PROXY from the environment, which would
# send localhost traffic to a proxy that cannot reach it.
_DIRECT = urllib.request.build_opener(urllib.request.ProxyHandler({}))


class OllamaClient:
    """Minimal `/api/generate` client with a disk cache. Standard library only."""

    def __init__(
        self,
        url: str = "http://127.0.0.1:11434",
        cache_dir: Path | None = None,
        options: dict | None = None,
        timeout: float = 900.0,
    ) -> None:
        self.url = url.rstrip("/")
        self.cache_dir = cache_dir
        self.options = dict(DEFAULT_OPTIONS if options is None else options)
        self.timeout = timeout
        self.calls = 0
        self.cache_hits = 0
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)

    def cache_key(self, model: str, prompt: str) -> str:
        blob = json.dumps(
            {
                "model": model,
                "prompt": hashlib.sha256(prompt.encode()).hexdigest(),
                "options": self.options,
            },
            sort_keys=True,
        )
        return hashlib.sha256(blob.encode()).hexdigest()

    def generate(self, model: str, prompt: str) -> str:
        key = self.cache_key(model, prompt)
        path = self.cache_dir / f"{key}.json" if self.cache_dir is not None else None
        if path is not None and path.exists():
            self.cache_hits += 1
            return json.loads(path.read_text(encoding="utf-8"))["response"]
        body = json.dumps(
            {"model": model, "prompt": prompt, "stream": False, "options": self.options}
        ).encode()
        request = urllib.request.Request(
            f"{self.url}/api/generate", data=body, headers={"Content-Type": "application/json"}
        )
        try:
            with _DIRECT.open(request, timeout=self.timeout) as resp:
                data = json.loads(resp.read())
        except urllib.error.URLError as exc:
            raise ConnectionError(f"Ollama at {self.url} is not reachable: {exc}") from exc
        self.calls += 1
        response = data.get("response", "")
        if path is not None:
            path.write_text(
                json.dumps({"model": model, "options": self.options, "response": response}),
                encoding="utf-8",
            )
        return response


@dataclass
class Edit:
    path: str
    search: str
    replace: str


_BLOCK = re.compile(
    r"^(?P<path>[^\n<>=]+?)\s*\n<<<<<<< SEARCH\n(?P<search>.*?)"
    r"^=======\n(?P<replace>.*?)^>>>>>>> REPLACE",
    re.M | re.S,
)


def parse_edits(text: str) -> list[Edit]:
    """Extract SEARCH/REPLACE blocks. Code fences around them are tolerated."""
    text = re.sub(r"^```[a-zA-Z0-9_-]*\s*$", "", text, flags=re.M)
    edits = []
    for m in _BLOCK.finditer(text):
        path = m.group("path").strip().strip("`").strip()
        edits.append(Edit(path, m.group("search"), m.group("replace")))
    return edits


def apply_edits(root: Path, edits: list[Edit]) -> list[str]:
    """Apply edits under `root`. Returns a list of problems; empty means all applied.

    An edit whose path escapes `root`, or whose SEARCH text is not found exactly once, is
    refused rather than guessed at.
    """
    problems = []
    base = root.resolve()
    for edit in edits:
        target = (root / edit.path).resolve()
        if base not in target.parents:
            problems.append(f"{edit.path}: outside the repository")
            continue
        if not edit.search.strip():
            if target.exists() and _read(target).strip():
                problems.append(f"{edit.path}: empty SEARCH on a file that already has content")
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            _write(target, edit.replace)
            continue
        if not target.exists():
            problems.append(f"{edit.path}: no such file")
            continue
        text = _read(target)
        count = text.count(edit.search)
        if count != 1:
            problems.append(f"{edit.path}: SEARCH text found {count} times, need exactly 1")
            continue
        _write(target, text.replace(edit.search, edit.replace, 1))
    return problems


# Bytes in, bytes out: text mode would rewrite every line ending on Windows and turn a
# one-line edit into a whole-file diff.
def _read(path: Path) -> str:
    return path.read_bytes().decode("utf-8", errors="replace")


def _excerpts(text: str, spans: list[Hunk], budget: int, context: int = 10) -> str:
    """Verbatim blocks of `text` around `spans` (or from the top), within `budget` chars."""
    lines = text.splitlines(keepends=True)
    ranges = [
        (max(1, h.start - context), min(len(lines), h.start + h.length + context)) for h in spans
    ] or [(1, len(lines))]
    merged: list[tuple[int, int]] = []
    for lo, hi in sorted(ranges):
        if merged and lo <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append((lo, hi))
    blocks, used = [], 0
    for lo, hi in merged:
        body = ""
        for n in range(lo, hi + 1):
            if used + len(body) + len(lines[n - 1]) > budget:
                hi = n - 1
                break
            body += lines[n - 1]
        if not body:
            break
        blocks.append(f"lines {lo}-{hi} of {len(lines)}:\n```\n{body}```")
        used += len(body)
    note = "(Only these lines are shown. SEARCH text must be copied from them exactly.)"
    return "\n".join([*blocks, note])


def _write(path: Path, text: str) -> None:
    path.write_bytes(text.encode("utf-8"))


class OllamaAgent:
    """An agent that asks a local model for SEARCH/REPLACE edits."""

    name = "ollama"

    def __init__(self, client: Client, model: str, max_files: int = 4, max_chars: int = 24000):
        self.client = client
        self.model = model
        self.max_files = max_files
        self.max_chars = max_chars

    def prompt_for(self, task: Task, worktree: Path, git: Git) -> str:
        """The task plus the files the description points at most directly.

        Files are ranked by relevance (named symbols first, a changelog last), not by name.
        A file too long for its share of the budget is shown as verbatim excerpts around the
        named symbols (or its opening lines), each labelled with its line range, and the model
        is told that SEARCH text must come from the lines shown.
        """
        head = git.out("rev-parse", "HEAD", cwd=worktree)
        ranked = DescriptionPredictor(Git(worktree)).ranked(task, head)
        chosen = [r for r in ranked if (worktree / r.path).is_file()][: self.max_files]
        share = self.max_chars // max(1, len(chosen))
        sections = []
        for item in chosen:
            text = _read(worktree / item.path)
            if len(text) <= share:
                sections.append(f"### {item.path}\n```\n{text}```")
            else:
                sections.append(f"### {item.path} (excerpts)\n{_excerpts(text, item.spans, share)}")
        return PROMPT.format(task=task.description.strip(), files="\n\n".join(sections))

    def work(self, task: Task, worktree: Path, git: Git) -> AgentResult:
        response = self.client.generate(self.model, self.prompt_for(task, worktree, git))
        edits = parse_edits(response)
        if not edits:
            return AgentResult(False, "the model returned no edit blocks")  # an agent error
        problems = apply_edits(worktree, edits)
        if problems:
            return AgentResult(False, "; ".join(problems))
        return AgentResult(True, f"{len(edits)} edit(s) applied")
