"""Aggregate experiment runs into rates with window-level bootstrap intervals.

Tasks inside one window share a base and interact, so they are not independent samples.
Every interval here resamples whole windows (stratified by repository when repositories are
pooled), never individual tasks.
"""

from __future__ import annotations

import json
import random
import re
from collections import defaultdict
from collections.abc import Callable, Iterable
from pathlib import Path

from .fleet import NOOP
from .gitops import Git
from .mergequeue import ACCEPTED, AGENT_ERROR, BASE_CONFLICT, SEMANTIC, TEXTUAL
from .predict import CHANGELOG

Counts = dict[str, float]
CONFLICTS = (BASE_CONFLICT, TEXTUAL)


def load_runs(runs: Path) -> list[dict]:
    records = []
    for path in sorted(runs.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(json.loads(line))
    if not records:
        raise FileNotFoundError(f"no experiment records under {runs}")
    return records


def conflict_kind(paths: list[str]) -> str:
    """What a conflict was about: any Python source or stub makes it `code`; otherwise
    `changelog` if every file is a changelog, else `other` (CI, pins, lock files, docs)."""
    if any(p.endswith((".py", ".pyi")) for p in paths):
        return "code"
    if paths and all(CHANGELOG.search(p) for p in paths):
        return "changelog"
    return "other"


def window_counts(record: dict, serial: dict | None = None) -> Counts:
    """Everything the report needs from one (window, policy) run, as summable counts.

    `serial` is the same window run serially. A task that fails its first attempt here but
    not under serial is an *excess* failure: one that running the tasks in parallel caused,
    as opposed to one the replay would have hit anyway (a change that needs a housekeeping
    commit the fleet was never given).
    """
    c: Counts = defaultdict(float)
    serial_ok = None
    if serial is not None:
        serial_ok = {t["id"]: t["first_outcome"] in (ACCEPTED, NOOP) for t in serial["tasks"]}
    size = record["size"]
    c["windows"] = 1
    c["tasks"] = size
    c["agent_runs"] = record["agent_runs"]
    c["redos"] = record["redos"]
    c["makespan"] = record["makespan"]
    c["serial_makespan"] = size
    failed_any = False
    for task in record["tasks"]:
        landed = task["final"] in (ACCEPTED, NOOP)
        c["landed"] += landed
        kind, detail = task["first_outcome"], task["first_detail"]
        if kind in (ACCEPTED, NOOP):
            continue
        failed_any = True
        c["first_fail"] += 1
        if serial_ok is not None and serial_ok.get(task["id"], False):
            c["excess_fail"] += 1
        c["rejected"] += not landed
        if kind in CONFLICTS:
            c[kind] += 1
            c[f"conflict_files_{conflict_kind(task['attempts'][0]['conflicted'])}"] += 1
        elif kind == SEMANTIC:
            c["semantic"] += 1
            c[f"semantic_{detail}"] += 1
        elif kind == AGENT_ERROR:
            c["agent_error"] += 1
    c["window_failed"] = 1 if failed_any else 0
    c["paired"] = 1 if serial_ok is not None else 0
    c["paired_tasks"] = size if serial_ok is not None else 0
    return c


def ratio(num: str, den: str) -> Callable[[Counts], float]:
    def f(total: Counts) -> float:
        return total[num] / total[den] if total[den] else float("nan")

    return f


def _wasted(total: Counts) -> float:
    """Share of agent runs whose work never landed: every failed attempt, including both
    attempts of a task that was rejected even after its redo."""
    runs = total["agent_runs"]
    return (runs - total["landed"]) / runs if runs else float("nan")


def _conflict(total: Counts) -> float:
    return (total[BASE_CONFLICT] + total[TEXTUAL]) / total["tasks"] if total["tasks"] else 0.0


METRICS: dict[str, Callable[[Counts], float]] = {
    "first_attempt_failure": ratio("first_fail", "tasks"),
    "excess_over_serial": ratio("excess_fail", "paired_tasks"),
    "conflict": _conflict,
    "base_conflict": ratio(BASE_CONFLICT, "tasks"),
    "merge_conflict": ratio(TEXTUAL, "tasks"),
    "conflict_in_code": ratio("conflict_files_code", "tasks"),
    "conflict_changelog_only": ratio("conflict_files_changelog", "tasks"),
    "conflict_other_files": ratio("conflict_files_other", "tasks"),
    "semantic_conflict": ratio("semantic", "tasks"),
    "semantic_interaction": ratio("semantic_interaction", "tasks"),
    "semantic_stale_base": ratio("semantic_stale-base", "tasks"),
    "agent_error": ratio("agent_error", "tasks"),
    "landed": ratio("landed", "tasks"),
    "rejected_after_redo": ratio("rejected", "tasks"),
    "wasted_work": _wasted,
    "window_with_a_failure": ratio("window_failed", "windows"),
    # Agent rounds on the critical path, per task offered and per task that landed. Serial
    # lands everything at 1.00; a policy that lands less must not look faster for it.
    "makespan_vs_serial": ratio("makespan", "serial_makespan"),
    "rounds_per_landed_task": ratio("makespan", "landed"),
}


def _sum(items: Iterable[Counts]) -> Counts:
    total: Counts = defaultdict(float)
    for item in items:
        for k, v in item.items():
            total[k] += v
    return total


def bootstrap_many(
    strata: dict[str, list[Counts]],
    stats: dict[str, Callable[[Counts], float]],
    rng: random.Random,
    resamples: int,
) -> dict[str, dict[str, float]]:
    """Point estimates and 95% percentile intervals for several statistics, all computed
    on the same resamples of windows (drawn within each stratum)."""
    point = _sum(c for items in strata.values() for c in items)
    draws: dict[str, list[float]] = {name: [] for name in stats}
    for _ in range(resamples):
        sample: list[Counts] = []
        for items in strata.values():
            sample.extend(rng.choice(items) for _ in items)
        total = _sum(sample)
        for name, stat in stats.items():
            value = stat(total)
            if value == value:  # drop NaN (empty denominator in this resample)
                draws[name].append(value)
    out = {}
    for name, stat in stats.items():
        values = sorted(draws[name])
        estimate = stat(point)
        if not values:
            out[name] = {"value": estimate, "lo": float("nan"), "hi": float("nan")}
            continue
        lo = values[int(0.025 * (len(values) - 1))]
        hi = values[int(0.975 * (len(values) - 1))]
        out[name] = {"value": round(estimate, 4), "lo": round(lo, 4), "hi": round(hi, 4)}
    return out


def summarise(records: list[dict], rng: random.Random, resamples: int) -> dict:
    serial = {
        (r["target"], r["mode"], r["size"], r["window"]): r
        for r in records
        if r["label"] == "serial"
    }
    groups: dict[tuple[str, str, int, str], list[Counts]] = defaultdict(list)
    for r in records:
        twin = serial.get((r["target"], r["mode"], r["size"], r["window"]))
        groups[(r["target"], r["mode"], r["size"], r["label"])].append(window_counts(r, twin))
    targets = sorted({k[0] for k in groups})
    modes = sorted({k[1] for k in groups})
    sizes = sorted({k[2] for k in groups})
    policies = sorted({k[3] for k in groups})
    out: dict = {"targets": targets, "modes": modes, "sizes": sizes, "policies": policies}
    rows = []
    for scope in [*targets, "all"]:
        for mode in modes:
            for size in sizes:
                for policy in policies:
                    strata = {
                        t: groups[(t, mode, size, policy)]
                        for t in targets
                        if scope in ("all", t) and groups.get((t, mode, size, policy))
                    }
                    if not strata:
                        continue
                    total = _sum(c for items in strata.values() for c in items)
                    row = {
                        "scope": scope,
                        "mode": mode,
                        "size": size,
                        "policy": policy,
                        "windows": int(total["windows"]),
                        "tasks": int(total["tasks"]),
                        "counts": {k: int(v) for k, v in sorted(total.items())},
                    }
                    row.update(bootstrap_many(strata, METRICS, rng, resamples))
                    rows.append(row)
    out["rows"] = rows
    return out


def predictor_quality(records: list[dict], rng: random.Random, resamples: int) -> list[dict]:
    """Task-level precision and recall of every predictor, per mode and fleet size.

    A task is predicted at risk if the predictor pairs it with any earlier task in its
    window; it actually failed if its first attempt under naive parallelism failed for a
    reason a scheduler could avoid (a conflict or a broken merge, not an agent error). The
    `+changelog-union` variants drop changelogs from the file-level footprints and are scored
    against the naive-parallel run that merged changelogs with the union driver.
    """
    by_label: dict[str, dict[tuple, dict]] = defaultdict(dict)
    for r in records:
        by_label[r["label"]][(r["target"], r["mode"], r["size"], r["window"])] = r
    plain = by_label.get("parallel", {})
    union = by_label.get("parallel+changelog-union", {})
    names = sorted(next(iter(plain.values()))["info"]["predicted"]) if plain else []
    jobs = [(name, name, plain, False) for name in names]
    jobs += [
        (f"{name}+changelog-union", name, union, True)
        for name in ("description", "oracle-files")
        if name in names
    ]
    rows = []
    for label, source, runs, drop_changelogs in jobs:
        groups: dict[tuple[str, int], dict[str, list[Counts]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for _key, record in sorted(runs.items()):
            predicted = record["info"]["predicted"][source]
            if drop_changelogs:
                pairs = _file_pairs(predicted["files"], exclude=CHANGELOG)
            else:
                pairs = {tuple(p) for p in predicted["pairs"]}
            counts = _task_counts(record, pairs)
            groups[(record["mode"], record["size"])][record["target"]].append(counts)
            groups[(record["mode"], 0)][record["target"]].append(counts)
        for (mode, size), strata in sorted(groups.items()):
            for scope in [*sorted(strata), "all"]:
                s = strata if scope == "all" else {scope: strata[scope]}
                rows.append(
                    {
                        "predictor": label,
                        "mode": mode,
                        "size": size or "all",
                        "scope": scope,
                        **bootstrap_many(
                            s,
                            {
                                "task_precision": _prec("tp", "fp"),
                                "task_recall": _prec("tp", "fn"),
                                "pairs_flagged_share": ratio("pairs_flagged", "pairs_total"),
                            },
                            rng,
                            resamples,
                        ),
                        "counts": {
                            k: int(v)
                            for k, v in sorted(_sum(x for v in s.values() for x in v).items())
                        },
                    }
                )
    return rows


def _file_pairs(files: list[list[str]], exclude: re.Pattern[str]) -> set[tuple[int, int]]:
    kept = [{f for f in fs if not exclude.search(f)} for fs in files]
    return {(i, j) for j in range(len(kept)) for i in range(j) if kept[i] & kept[j]}


def _task_counts(record: dict, pairs: set[tuple[int, int]]) -> Counts:
    c: Counts = defaultdict(float)
    risky = {j for _, j in pairs}
    for j, task in enumerate(record["tasks"]):
        failed = task["first_outcome"] in (*CONFLICTS, SEMANTIC)
        flagged = j in risky
        c["tp"] += failed and flagged
        c["fp"] += (not failed) and flagged
        c["fn"] += failed and not flagged
        c["tn"] += (not failed) and not flagged
    c["pairs_flagged"] += len(pairs)
    c["pairs_total"] += record["size"] * (record["size"] - 1) / 2
    return c


def _prec(hit: str, miss: str) -> Callable[[Counts], float]:
    def f(total: Counts) -> float:
        den = total[hit] + total[miss]
        return total[hit] / den if den else float("nan")

    return f


def history_facts(runs: Path) -> dict:
    """Per target: tasks, housekeeping left out, healthy commits, flaky tests."""
    facts = {}
    for path in sorted(runs.glob("*-history.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        results = data["results"]
        tasks = data["commits"]
        flaky = set()
        for r in results.values():
            flaky |= set(r["flaky"])
        facts[path.name.removesuffix("-history.json")] = {
            "head": data["head"],
            "task_commits": len(tasks),
            "housekeeping_commits": len(data.get("housekeeping", {})),
            "healthy_task_commits": sum(results[c]["healthy"] for c in tasks if c in results),
            "trees_tested": len(results),
            "flaky_tests": sorted(flaky),
            "median_suite_seconds": sorted(r["duration"] for r in results.values())[
                len(results) // 2
            ],
        }
    return facts


def semantic_facts(records: list[dict], repos: dict[str, Path], runs: Path) -> dict:
    """Every broken merge the test gate caught, traced to states and to changes.

    Per target and mode: how many merge candidates were states history never had (only
    those could reveal a semantic conflict), how many distinct states broke a test, and which
    task changes did it - every semantic attempt, first try or redo, with its
    classification. A change seen as both stale-base and interaction is listed under both.
    """
    facts: dict = {}
    for target in sorted({r["target"] for r in records}):
        path = repos.get(target)
        history = runs / f"{target}-history.json"
        git = Git(path) if path is not None and (path / ".git").exists() else None
        real: set[str] = set()
        if git is not None and history.exists():
            real = set(
                _trees(git, list(json.loads(history.read_text(encoding="utf-8"))["results"]))
            )
        for mode in sorted({r["mode"] for r in records if r["target"] == target}):
            mine = [r for r in records if r["target"] == target and r["mode"] == mode]
            attempts = [a for r in mine for t in r["tasks"] for a in t["attempts"]]
            candidates = sorted({a["candidate"] for a in attempts if a["candidate"]})
            tree_of = (
                dict(zip(candidates, _trees(git, candidates), strict=True))
                if git is not None
                else {c: c for c in candidates}
            )
            changes: dict[str, set[str]] = defaultdict(set)
            broke = set()
            n_semantic = 0
            for r in mine:
                for t in r["tasks"]:
                    for a in t["attempts"]:
                        if a["outcome"] == SEMANTIC:
                            n_semantic += 1
                            broke.add(tree_of[a["candidate"]])
                            changes[a["detail"] or "unclassified"].add(t["id"])
            entry = {
                "semantic_attempts": n_semantic,
                "distinct_states_that_broke_a_test": len(broke),
                "changes": {k: sorted(v) for k, v in sorted(changes.items())},
                "change_counts": {k: len(v) for k, v in sorted(changes.items())},
                "distinct_changes": len(set().union(*changes.values())) if changes else 0,
            }
            if git is not None and real:
                novel = {tree for tree in tree_of.values() if tree not in real}
                entry["distinct_candidates_tested"] = len(set(tree_of.values()))
                entry["novel_states_tested"] = len(novel)
                entry["novel_states_that_broke_a_test"] = len(broke & novel)
            facts.setdefault(target, {})[mode] = entry
    return facts


def _trees(git: Git, commits: list[str]) -> list[str]:
    trees: list[str] = []
    for i in range(0, len(commits), 400):
        chunk = commits[i : i + 400]
        trees.extend(git.out("rev-parse", *(f"{c}^{{tree}}" for c in chunk)).split())
    return trees


def missing_repos(records: list[dict], repos: dict[str, Path]) -> list[str]:
    """Targets in the runs whose git repository is not on disk."""
    return sorted(
        t
        for t in {r["target"] for r in records}
        if t not in repos or not (repos[t] / ".git").exists()
    )


def build_report(
    runs: Path, seed: int = 0, resamples: int = 2000, repos: dict[str, Path] | None = None
) -> dict:
    records = load_runs(runs)
    repos = repos or {}
    rng = random.Random(seed)
    return {
        "resamples": resamples,
        "seed": seed,
        "repos_missing": missing_repos(records, repos),
        "history": history_facts(runs),
        "semantic": semantic_facts(records, repos, runs),
        "policies": summarise(records, rng, resamples),
        "predictors": predictor_quality(records, rng, resamples),
    }


def format_table(summary: dict, scope: str = "all") -> str:
    """A plain-text table of the headline metrics for one scope."""
    head = (
        f"{'mode':<12}{'N':>3}  {'policy':<40}{'tasks':>6} {'1st fail':>9} {'excess':>7} "
        f"{'base':>7} "
        f"{'merge':>7} {'semantic':>9} {'landed':>7} {'wasted':>7} {'rounds/N':>9} "
        f"{'rounds/landed':>14}"
    )
    lines = [f"scope: {scope}", head, "-" * len(head)]
    for row in summary["policies"]["rows"]:
        if row["scope"] != scope:
            continue
        lines.append(
            f"{row['mode']:<12}{row['size']:>3}  {row['policy']:<40}{row['tasks']:>6} "
            f"{_pct(row['first_attempt_failure']):>9} {_pct(row['excess_over_serial']):>7} "
            f"{_pct(row['base_conflict']):>7} "
            f"{_pct(row['merge_conflict']):>7} {_pct(row['semantic_conflict']):>9} "
            f"{_pct(row['landed']):>7} {_pct(row['wasted_work']):>7} "
            f"{row['makespan_vs_serial']['value']:>9.2f} "
            f"{row['rounds_per_landed_task']['value']:>14.2f}"
        )
    return "\n".join(lines)


def _pct(metric: dict) -> str:
    return f"{100 * metric['value']:.1f}%"
