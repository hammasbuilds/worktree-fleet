#!/usr/bin/env bash
# The model arm: the same windows, but an LLM agent writes each change instead of replaying
# the real commit. Queued, not run - it needs the GPU.
#
#   scripts/run_models.sh --dry-run    # print the job list and the call estimate, run nothing
#   scripts/run_models.sh              # check RAM/GPU/Ollama, then run everything
#
# Every generation is cached under results/llm-cache keyed by (model, prompt hash, options),
# so an interrupted run picks up where it stopped. Needs the replay experiment to have run
# first (it provides the cached test results the windows are planned from).
set -euo pipefail
cd "$(dirname "$0")/.."
unset VIRTUAL_ENV

MODEL="${MODEL:-qwen2.5-coder:14b}"
OLLAMA_URL="${OLLAMA_URL:-http://127.0.0.1:11434}"
TARGETS="${TARGETS:-flask click}"
SIZES="${SIZES:-2,4,8}"
WINDOWS="${WINDOWS:-8}"
POLICIES="serial,parallel,predicted:description"
MIN_FREE_RAM_GB=8
MIN_FREE_VRAM_MB=11000

dry_run=0
[ "${1:-}" = "--dry-run" ] && dry_run=1

run() {
  local extra=("$@")
  for t in $TARGETS; do
    uv run fleet experiment --target "$t" --agent ollama --model "$MODEL" \
      --ollama-url "$OLLAMA_URL" --llm-cache results/llm-cache --sizes "$SIZES" \
      --max-windows "$WINDOWS" --policies "$POLICIES" --modes consecutive \
      --out results/runs-llm --workers 1 \
      "${extra[@]}"
  done
}

if [ "$dry_run" = 1 ]; then
  echo "model: $MODEL  targets: $TARGETS  sizes: $SIZES  windows/size: $WINDOWS"
  echo "policies: $POLICIES"
  run --dry-run
  exit 0
fi

free_gb=$(powershell -NoProfile -Command \
  "[int]((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB)" 2>/dev/null \
  || awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo)
if [ "$free_gb" -lt "$MIN_FREE_RAM_GB" ]; then
  echo "only ${free_gb} GB RAM free (need ${MIN_FREE_RAM_GB}); not starting" >&2
  exit 1
fi
if command -v nvidia-smi >/dev/null; then
  free_vram=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  if [ "$free_vram" -lt "$MIN_FREE_VRAM_MB" ]; then
    echo "only ${free_vram} MB VRAM free (need ${MIN_FREE_VRAM_MB}); is another job training?" >&2
    exit 1
  fi
fi
if ! curl -sf "$OLLAMA_URL/api/tags" | grep -q "\"$MODEL\""; then
  echo "Ollama at $OLLAMA_URL is down or does not have $MODEL pulled" >&2
  exit 1
fi

run
uv run fleet report --runs results/runs-llm --out results/summary-llm.json
