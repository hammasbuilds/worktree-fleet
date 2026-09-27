# STATUS

**Status: READY-FOR-REVIEW** - headline finding produced from real history on this machine; the
LLM-agent arm is built, tested with a fake client, and queued (`scripts/run_models.sh`).

## Self-score (hostile pass, 2026-09-27)

| Points | Criterion | Score | Reason |
|---:|---|---:|---|
| 15 | Works from a clean clone | 15 | Fresh `git clone` into a temp dir: `uv sync --offline`, `uv run pytest -q` (70 passed), `uv run python demo.py` all succeed. Tests build their own git repos in `tmp_path`, use no network, no model, no data env vars. |
| 20 | Real data, real result | 19 | Five real histories (flask in two eras, click, sqlparse, more-itertools), 591 windows (3,542 task placements per policy), every merge candidate tested with the project's own suite; `results/runs/*.jsonl` + `results/summary.json`. -1: flask contributes only the commits whose suite runs in one environment per era (146/396 and 71/298). |
| 15 | Finding quality | 14 | Baselines (serial, naive parallel, history order), ablations (changelog union driver, four predictors at three information levels), window-level stratified bootstrap CIs, per-repo split, attribution against base and every real commit, novel-state denominator for semantic conflicts. -1: replay cannot generate interaction conflicts that history did not contain (stated). |
| 15 | Correctness | 14 | Every surprising number was chased (spurious `<suite-error>` "conflicts", degenerate pairwise truth, zero semantic conflicts in history order, flask 2.x failing everywhere, the more-itertools SyntaxError). -1: one-environment-per-era is a coarse way to get green history. |
| 10 | Usability | 9 | `fleet plan/run/experiment/report` with `--help`, one-line errors, `--python` validated up front, resumable experiment, dry-run for the model arm. -1: running the experiment needs two setup scripts and hours. |
| 10 | README | 10 | House format, five real Input/Output samples, NOT-do section, twelve real problems. |
| 10 | Code quality | 10 | ruff clean, typed, zero runtime dependencies, modules under ~400 lines, no dead code. |
| 5 | Honesty | 5 | Every README number is in `results/summary.json` or quoted command output; limitations stated where the numbers are. |
| | **Total** | **96** | |

## Done

- Fleet: per-task `git worktree`, parallel agents (thread pool), in-memory merge queue with a
  test gate, re-queue and redo, serial / parallel / predicted policies, three integration
  orders, JSON reports, refs under `refs/fleet/<run>/`.
- Agents: `ReplayAgent` (in-memory and worktree paths, proven equivalent by a test),
  `OllamaAgent` (SEARCH/REPLACE edits, disk-cached generations, fake-client tests).
- Predictors: description (task text + repo symbols + prior co-change), description with symbol
  spans, oracle files, oracle hunks, and a changelog-excluding wrapper.
- Experiment: windows N = 2, 4, 8, 16 over the last 300-400 first-parent commits of each target,
  up to 40 per size, eight policies per window, attribution, novel-state accounting.
- Report: window-level stratified bootstrap (2,000 resamples), per-repo and pooled, task-level
  predictor precision/recall.

## Queued for the model run

`scripts/run_models.sh` (dry run: `bash scripts/run_models.sh --dry-run`): `qwen2.5-coder:14b` as
the agent on flask and click, N = 2, 4, 8, 8 windows per size, policies serial / parallel /
predicted:description. Estimate: **672 first-attempt generations, at most 1,344 with one redo
each**; generations are cached, so the run resumes. Output: `results/runs-llm/`,
`results/summary-llm.json`.

## Known weaknesses

- The replay agent reproduces human patches; LLM patches will collide differently (model arm).
- A task is a first-parent commit; in sqlparse (direct commits) one feature's code and tests can
  be two tasks, inflating stale-base failures there.
- more-itertools' changelog (`docs/versions.rst`) is not matched by the generic changelog
  patterns, so its union-driver rows equal the plain rows.
- Wall-clock is in agent rounds, with each redo charged a full round.

## Reproduce

```bash
cd D:/github/worktree-fleet
bash scripts/fetch_targets.sh            # targets/<name>, pinned heads
bash scripts/setup_target_venvs.sh       # targets/.venvs/<name>, pinned test environments
# run from a frozen copy so edits cannot reach the worker processes mid-run
git worktree add --detach ../wf-snapshot HEAD && cd ../wf-snapshot && uv sync
for t in flask sqlparse click more-itertools flask-2x; do
  uv run fleet experiment --targets D:/github/worktree-fleet/targets.toml --target $t \
    --workers 6 --max-windows 40 --out D:/github/worktree-fleet/results/runs \
    --cache D:/github/worktree-fleet/targets/.fleet
done
cd D:/github/worktree-fleet
uv run fleet report                       # results/summary.json + the table
uv run pytest -q && uv run python demo.py
```

The README's `fleet plan` / `fleet run` samples are the exact commands shown in its Input /
Output section, run from the repository root.

The runs in `results/runs/` were produced from a frozen snapshot at commit `f755a7d`. Later
commits changed the report (shared resamples, file-kind breakdown, novel-state count), the CLI
(`--order`, `--python` validation), crash handling for agents, and dropped a per-window pairwise
field the report no longer reads; the fleet, queue, replay and attribution logic the runs used
is unchanged. `fleet report` regenerates `results/summary.json` from those runs.

## Headline numbers (pooled, naive parallel, 95% window-bootstrap CI)

| N | first attempt thrown away | with changelog union | clean merge, broken tests | wall-clock vs serial |
|---:|---|---|---|---|
| 2 | 9.0% [6.4, 11.8] | 6.4% | 0.0% | 0.59 |
| 4 | 18.7% [15.9, 21.5] | 12.3% | 0.3% | 0.44 |
| 8 | 31.4% [28.5, 34.1] | 21.9% | 0.9% | 0.44 |
| 16 | 42.5% [39.1, 46.1] | 30.5% | 1.1% | 0.49 |

Exact-line oracle at N=16: 0.5% thrown away at 0.43 wall-clock; the realistic description
predictor: 1.0% at 0.97 (it serialises on the changelog), 15.4% at 0.71 with the changelog on
`merge=union`. 6,339 never-seen states tested, 81 broke a test, from 22 changes: 20 stale-base,
one interaction pair (a clean git merge that produced a SyntaxError).
