import json

import pytest

from worktree_fleet.fleet import Fleet
from worktree_fleet.llm import OllamaAgent, OllamaClient, apply_edits, parse_edits
from worktree_fleet.mergequeue import ACCEPTED, AGENT_FAILED, MergeQueue
from worktree_fleet.tasks import Task

REPLY = """Here you go:
```python
pkg/greet.py
<<<<<<< SEARCH
def greet(name):
    return "hi " + name
=======
def greet(name):
    return "hello " + name
>>>>>>> REPLACE
```
pkg/new.py
<<<<<<< SEARCH
=======
VALUE = 1
>>>>>>> REPLACE
"""


class FakeClient:
    """Deterministic stand-in for Ollama: returns canned replies, records prompts."""

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.prompts: list[str] = []

    def generate(self, model: str, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.reply


def test_parse_edits_tolerates_fences():
    edits = parse_edits(REPLY)
    assert [e.path for e in edits] == ["pkg/greet.py", "pkg/new.py"]
    assert edits[0].replace.endswith('"hello " + name\n')
    assert edits[1].search == ""


def test_apply_edits_refuses_to_guess(tmp_path):
    (tmp_path / "a.py").write_bytes(b"x = 1\nx = 1\n")
    problems = apply_edits(
        tmp_path, parse_edits("a.py\n<<<<<<< SEARCH\nx = 1\n=======\nx = 2\n>>>>>>> REPLACE\n")
    )
    assert problems == ["a.py: SEARCH text found 2 times, need exactly 1"]
    escape = parse_edits("../evil.py\n<<<<<<< SEARCH\n=======\nboom\n>>>>>>> REPLACE\n")
    assert apply_edits(tmp_path, escape) == ["../evil.py: outside the repository"]
    assert not (tmp_path.parent / "evil.py").exists()
    missing = parse_edits("nope.py\n<<<<<<< SEARCH\na\n=======\nb\n>>>>>>> REPLACE\n")
    assert apply_edits(tmp_path, missing) == ["nope.py: no such file"]


def test_ollama_agent_in_a_fleet_with_a_fake_model(repo, tmp_path):
    base = repo.commit("base", {"pkg/greet.py": 'def greet(name):\n    return "hi " + name\n'})
    client = FakeClient(REPLY)
    agent = OllamaAgent(client, "fake-model")

    def factory(start, ref):
        return MergeQueue(repo.git, start, ref)

    fleet = Fleet(repo.git, agent, tmp_path / "w", factory)
    report = fleet.run([Task("t1", "Make greet() say hello")], base, "parallel", run_id="llm")
    assert report.records[0].final == ACCEPTED
    assert "hello" in repo.git.show_file(report.main, "pkg/greet.py")
    assert repo.git.show_file(report.main, "pkg/new.py") == "VALUE = 1\n"
    # The prompt carried the task and the file the description predictor pointed at.
    assert "Make greet() say hello" in client.prompts[0]
    assert "### pkg/greet.py" in client.prompts[0]


def test_ollama_agent_fails_cleanly_on_garbage(repo, tmp_path):
    base = repo.commit("base", {"a.py": "x = 1\n"})

    def factory(start, ref):
        return MergeQueue(repo.git, start, ref)

    fleet = Fleet(
        repo.git, OllamaAgent(FakeClient("I cannot help."), "m"), tmp_path / "w", factory, retries=0
    )
    report = fleet.run([Task("t", "do it")], base, "parallel", run_id="g")
    assert report.records[0].first.outcome == AGENT_FAILED
    assert "no edit blocks" in report.records[0].first.note


def test_client_serves_cache_without_network(tmp_path):
    # Port 9 on localhost has nothing listening: any real call would fail loudly.
    client = OllamaClient("http://127.0.0.1:9", cache_dir=tmp_path)
    key = client.cache_key("m", "p")
    (tmp_path / f"{key}.json").write_text(json.dumps({"response": "cached!"}))
    assert client.generate("m", "p") == "cached!"
    assert client.cache_hits == 1 and client.calls == 0


def test_cache_key_depends_on_model_prompt_and_options(tmp_path):
    a = OllamaClient(cache_dir=tmp_path)
    b = OllamaClient(cache_dir=tmp_path, options={"temperature": 0.7})
    assert a.cache_key("m", "p") != a.cache_key("m2", "p")
    assert a.cache_key("m", "p") != a.cache_key("m", "q")
    assert a.cache_key("m", "p") != b.cache_key("m", "p")


def test_unreachable_server_is_a_clear_error(tmp_path):
    client = OllamaClient("http://127.0.0.1:9", cache_dir=tmp_path, timeout=5)
    with pytest.raises(ConnectionError, match="not reachable"):
        client.generate("m", "p")
