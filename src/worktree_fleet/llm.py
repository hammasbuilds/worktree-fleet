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
from .gitops import Git
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
            with urllib.request.urlopen(request, timeout=self.timeout) as resp:
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
        head = git.out("rev-parse", "HEAD", cwd=worktree)
        guess = DescriptionPredictor(Git(worktree)).footprint(task, head)
        chosen = sorted(p for p in guess.files if (worktree / p).is_file())[: self.max_files]
        budget = self.max_chars
        sections = []
        for path in chosen:
            text = (worktree / path).read_text(encoding="utf-8", errors="replace")
            if len(text) > budget:
                text = text[:budget] + "\n... (truncated)\n"
            budget -= len(text)
            sections.append(f"### {path}\n```\n{text}```")
            if budget <= 0:
                break
        return PROMPT.format(task=task.description.strip(), files="\n\n".join(sections))

    def work(self, task: Task, worktree: Path, git: Git) -> AgentResult:
        response = self.client.generate(self.model, self.prompt_for(task, worktree, git))
        edits = parse_edits(response)
        if not edits:
            return AgentResult(False, "the model returned no edit blocks")
        problems = apply_edits(worktree, edits)
        if problems:
            return AgentResult(False, "; ".join(problems))
        return AgentResult(True, f"{len(edits)} edit(s) applied")
