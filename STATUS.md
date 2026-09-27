# STATUS

**Status: READY-FOR-REVIEW** - headline finding produced from real history on this machine; the
LLM-agent arm is built, tested with a fake client, and queued (`scripts/run_models.sh`).

## Self-score (hostile pass, 2026-09-27)

| Points | Criterion | Score | Reason |
|---:|---|---:|---|
| 15 | Works from a clean clone | 15 | Fresh `git clone` into a temp dir: `uv sync --offline`, `uv run pytest -q` (69 passed), `uv run python demo.py` all succeed. Tests build their own git repos in `tmp_path`, use no network, no model, no data env vars. |
| 20 | Real data, real result | 19 | Five real histories (flask in two eras, click, sqlparse, more-itertools), every merge candidate tested with the project's own suite; `results/runs/*.jsonl` + `results/summary.json`. -1: flask contributes only the commits whose suite runs in one environment per era. |
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
