"""Target repositories for the experiment: where they live and how to test them."""

from __future__ import annotations

import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from .suite import SuiteConfig


@dataclass(frozen=True)
class Target:
    name: str
    path: Path
    url: str
    ref: str
    history: int
    python: str
    pytest_args: list[str] = field(default_factory=list)
    pythonpath: list[str] = field(default_factory=list)
    timeout: float = 600.0

    def suite(self) -> SuiteConfig:
        return SuiteConfig(
            python=self.python,
            args=list(self.pytest_args),
            pythonpath=list(self.pythonpath),
            timeout=self.timeout,
        )


def load_targets(path: Path) -> dict[str, Target]:
    """Read `targets.toml`. Relative paths resolve against the file's directory."""
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    root = path.parent
    targets = {}
    for name, spec in data.get("targets", {}).items():
        venv = root / spec["venv"]
        python = venv / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
        targets[name] = Target(
            name=name,
            path=root / spec["path"],
            url=spec["url"],
            ref=spec["ref"],
            history=int(spec["history"]),
            python=str(python),
            pytest_args=list(spec.get("pytest_args", [])),
            pythonpath=list(spec.get("pythonpath", [])),
            timeout=float(spec.get("timeout", 600)),
        )
    if not targets:
        raise ValueError(f"{path}: no [targets.<name>] tables found")
    return targets
