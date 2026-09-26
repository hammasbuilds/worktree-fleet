"""Aggregate experiment runs into rates with window-level bootstrap intervals.

Tasks inside one window share a base and interact, so they are not independent samples.
Every interval here resamples whole windows (stratified by repository when repositories are
pooled), never individual tasks.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from collections.abc import Callable, Iterable
from pathlib import Path

from .experiment import record_label
from .fleet import NOOP
from .mergequeue import ACCEPTED, AGENT_FAILED, SEMANTIC, TEXTUAL
from .predict import CHANGELOG

Counts = dict[str, float]


def load_runs(runs: Path) -> list[dict]:
    records = []
    for path in sorted(runs.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(json.loads(line))
    if not records:
        raise FileNotFoundError(f"no experiment records under {runs}")
    return records


def policy_key(record: dict) -> str:
    """The record's policy label (plain `parallel` is the shuffled headline run)."""
    return record_label(record)


def window_counts(record: dict) -> Counts:
    """Everything the report needs from one (window, policy) run, as summable counts."""
    c: Counts = defaultdict(float)
    size = record["size"]
    c["windows"] = 1
    c["tasks"] = size
    c["agent_runs"] = record["agent_runs"]
    c["redos"] = record["redos"]
    c["makespan"] = record["makespan"]
    c["serial_makespan"] = size
    failed_any = False
    for task in record["tasks"]:
        if task["final"] in (ACCEPTED, NOOP):
            c["landed"] += 1
        kind, detail = task["first_outcome"], task["first_detail"]
        if kind in (ACCEPTED, NOOP):
            continue
        failed_any = True
        c["first_fail"] += 1
        if kind in (TEXTUAL, AGENT_FAILED):
            c["textual"] += 1
            c[f"textual_{detail}"] += 1
            conflicted = task["attempts"][0]["conflicted"]
            if conflicted and all(CHANGELOG.search(p) for p in conflicted):
                c["textual_changelog_only"] += 1
        elif kind == SEMANTIC:
            c["semantic"] += 1
            c[f"semantic_{detail}"] += 1
        if task["final"] not in (ACCEPTED, NOOP):
            c["rejected"] += 1
    c["window_failed"] = 1 if failed_any else 0
    return c


def _wasted(total: Counts) -> float:
    """Share of agent runs whose work never landed: every failed attempt, including both
    attempts of a task that was rejected even after its redo."""
    runs = total["agent_runs"]
    return (runs - total["landed"]) / runs if runs else float("nan")


def ratio(num: str, den: str) -> Callable[[Counts], float]:
    def f(total: Counts) -> float:
        return total[num] / total[den] if total[den] else float("nan")

    return f


def _sum(items: Iterable[Counts]) -> Counts:
    total: Counts = defaultdict(float)
    for item in items:
        for k, v in item.items():
            total[k] += v
    return total


def bootstrap(
    strata: dict[str, list[Counts]],
    stat: Callable[[Counts], float],
    rng: random.Random,
    resamples: int,
) -> dict[str, float]:
    """Point estimate and 95% percentile interval, resampling windows within each stratum."""
    point = stat(_sum(c for items in strata.values() for c in items))
    draws = []
    for _ in range(resamples):
        sample: list[Counts] = []
        for items in strata.values():
            sample.extend(rng.choice(items) for _ in items)
        value = stat(_sum(sample))
        if value == value:  # drop NaN (empty denominator in this resample)
            draws.append(value)
    draws.sort()
    if not draws:
        return {"value": point, "lo": float("nan"), "hi": float("nan")}
    lo = draws[int(0.025 * (len(draws) - 1))]
    hi = draws[int(0.975 * (len(draws) - 1))]
    return {"value": round(point, 4), "lo": round(lo, 4), "hi": round(hi, 4)}


METRICS: dict[str, Callable[[Counts], float]] = {
    "first_attempt_failure": ratio("first_fail", "tasks"),
    "textual_conflict": ratio("textual", "tasks"),
    "semantic_conflict": ratio("semantic", "tasks"),
    "semantic_interaction": ratio("semantic_interaction", "tasks"),
    "semantic_stale_base": ratio("semantic_stale-base", "tasks"),
    "rejected_after_redo": ratio("rejected", "tasks"),
    "wasted_work": _wasted,
    "window_with_a_failure": ratio("window_failed", "windows"),
    "makespan_vs_serial": ratio("makespan", "serial_makespan"),
}


def summarise(records: list[dict], rng: random.Random, resamples: int) -> dict:
    groups: dict[tuple[str, int, str], list[Counts]] = defaultdict(list)
    for record in records:
        groups[(record["target"], record["size"], policy_key(record))].append(window_counts(record))
    targets = sorted({k[0] for k in groups})
    sizes = sorted({k[1] for k in groups})
    policies = sorted({k[2] for k in groups})
    out: dict = {"targets": targets, "sizes": sizes, "policies": policies, "rows": []}
    for scope in [*targets, "all"]:
        for size in sizes:
            for policy in policies:
                strata = {
                    t: groups[(t, size, policy)]
                    for t in targets
                    if (scope in ("all", t)) and groups.get((t, size, policy))
                }
                if not strata:
                    continue
                total = _sum(c for items in strata.values() for c in items)
                row = {
                    "scope": scope,
                    "size": size,
                    "policy": policy,
                    "windows": int(total["windows"]),
                    "tasks": int(total["tasks"]),
                    "counts": {k: int(v) for k, v in sorted(total.items())},
                }
                for name, stat in METRICS.items():
                    row[name] = bootstrap(strata, stat, rng, resamples)
                out["rows"].append(row)
    return out


def predictor_quality(records: list[dict], rng: random.Random, resamples: int) -> dict:
    """Task-level and pair-level precision/recall of every predictor.

    Task level: a task is predicted at risk if the predictor pairs it with any earlier task
    in its window; it actually failed if its first attempt under naive parallelism failed.
    Pair level: predicted pairs against pairs whose two changes, each replayed alone onto
    the base, fail to merge (pairs where a replay itself fails are left out).
    """
    parallel = [r for r in records if policy_key(r) == "parallel"]
    names = sorted(parallel[0]["info"]["predicted"]) if parallel else []
    result: dict = {}
    for name in names:
        strata: dict[str, list[Counts]] = defaultdict(list)
        for record in parallel:
            c: Counts = defaultdict(float)
            pairs = {tuple(p) for p in record["info"]["predicted"][name]["pairs"]}
            risky = {j for _, j in pairs}
            for j, task in enumerate(record["tasks"]):
                failed = task["first_outcome"] not in (ACCEPTED, NOOP)
                semantic = task["first_outcome"] == SEMANTIC
                flagged = j in risky
                c["tp"] += failed and flagged
                c["fp"] += (not failed) and flagged
                c["fn"] += failed and not flagged
                c["tn"] += (not failed) and not flagged
                c["semantic"] += semantic
                c["semantic_flagged"] += semantic and flagged
            for i, j, truth in record["info"]["pair_truth"]:
                if truth is None:
                    continue
                flagged = (i, j) in pairs
                c["ptp"] += truth and flagged
                c["pfp"] += (not truth) and flagged
                c["pfn"] += truth and not flagged
            c["pairs_flagged"] += len(pairs)
            c["pairs_total"] += record["size"] * (record["size"] - 1) / 2
            strata[record["target"]].append(c)
        result[name] = {}
        for scope in [*sorted(strata), "all"]:
            s = strata if scope == "all" else {scope: strata[scope]}
            result[name][scope] = {
                "task_precision": bootstrap(s, _prec("tp", "fp"), rng, resamples),
                "task_recall": bootstrap(s, _prec("tp", "fn"), rng, resamples),
                "semantic_recall": bootstrap(
                    s, ratio("semantic_flagged", "semantic"), rng, resamples
                ),
                "pair_precision": bootstrap(s, _prec("ptp", "pfp"), rng, resamples),
                "pair_recall": bootstrap(s, _prec("ptp", "pfn"), rng, resamples),
                "pairs_flagged_share": bootstrap(
                    s, ratio("pairs_flagged", "pairs_total"), rng, resamples
                ),
                "counts": {
                    k: int(v) for k, v in sorted(_sum(x for v in s.values() for x in v).items())
                },
            }
    return result


def _prec(hit: str, miss: str) -> Callable[[Counts], float]:
    def f(total: Counts) -> float:
        den = total[hit] + total[miss]
        return total[hit] / den if den else float("nan")

    return f


def history_facts(runs: Path) -> dict:
    """Per target: commits tested, how many were healthy, how many tests were flaky."""
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
            "healthy_task_commits": sum(results[c]["healthy"] for c in tasks),
            "trees_tested": len(results),
            "flaky_tests": sorted(flaky),
            "median_suite_seconds": sorted(r["duration"] for r in results.values())[
                len(results) // 2
            ],
        }
    return facts


def build_report(runs: Path, seed: int = 0, resamples: int = 2000) -> dict:
    records = load_runs(runs)
    rng = random.Random(seed)
    return {
        "resamples": resamples,
        "seed": seed,
        "history": history_facts(runs),
        "policies": summarise(records, rng, resamples),
        "predictors": predictor_quality(records, rng, resamples),
    }


def format_table(summary: dict, scope: str = "all") -> str:
    """A plain-text table of the headline metrics for one scope."""
    head = (
        f"{'N':>3}  {'policy':<26}{'tasks':>6}  {'1st-try fail':>13}  {'textual':>8}  "
        f"{'semantic':>8}  {'wasted':>7}  {'makespan/N':>10}"
    )
    lines = [f"scope: {scope}", head, "-" * len(head)]
    for row in summary["policies"]["rows"]:
        if row["scope"] != scope:
            continue
        lines.append(
            f"{row['size']:>3}  {row['policy']:<26}{row['tasks']:>6}  "
            f"{_pct(row['first_attempt_failure']):>13}  {_pct(row['textual_conflict']):>8}  "
            f"{_pct(row['semantic_conflict']):>8}  {_pct(row['wasted_work']):>7}  "
            f"{row['makespan_vs_serial']['value']:>10.2f}"
        )
    return "\n".join(lines)


def _pct(metric: dict) -> str:
    return f"{100 * metric['value']:.1f}%"
