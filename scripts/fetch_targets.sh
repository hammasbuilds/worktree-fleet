#!/usr/bin/env bash
# Fetch the experiment's target repositories into targets/<name>, with enough history.
#
# Usage: scripts/fetch_targets.sh [name ...]      (default: every target below)
#
# A single big clone kept dying mid-transfer on a throttled link, so history is fetched in
# small `--deepen` steps, each retried. Every target is pinned to the commit its results
# were produced from. A target whose pinned commit is already present is left alone.
set -u
cd "$(dirname "$0")/.."

# name | url | branch | pinned head | first-parent commits needed
TARGETS=(
  "flask|https://github.com/pallets/flask.git|main|d73fa1cdcbd8b1465c151db8924ba58b1dd14e35|400"
  "sqlparse|https://github.com/andialbrecht/sqlparse.git|master|60cdc649726bf1bc4f1b336050560b336da715ec|400"
  "click|https://github.com/pallets/click.git|main|06b2a678741131fd577ce170e23e5ca0aeba0309|400"
  "more-itertools|https://github.com/more-itertools/more-itertools.git|master|fbb9a98d8c7b914fcc952afd4976c1bc8deebe87|400"
)

retry() {
  local n=0
  until "$@"; do
    n=$((n + 1))
    if [ "$n" -ge 8 ]; then echo "giving up: $*" >&2; return 1; fi
    echo "retry $n: $*" >&2
    sleep $((n * 10))
  done
}

fetch_one() {
  local name=$1 url=$2 branch=$3 pin=$4 need=$5
  local dir="targets/$name"
  if [ -d "$dir/.git" ] && git -C "$dir" cat-file -e "$pin^{commit}" 2>/dev/null \
    && [ "$(git -C "$dir" rev-list --first-parent --count "$pin")" -gt "$need" ]; then
    git -C "$dir" checkout -q --detach "$pin"
    echo "[$name] already present at ${pin:0:10}"
    return 0
  fi
  mkdir -p "$dir"
  [ -d "$dir/.git" ] || git -C "$dir" init -q
  git -C "$dir" config gc.auto 0
  if ! git -C "$dir" rev-parse -q --verify FETCH_HEAD >/dev/null; then
    retry git -C "$dir" fetch -q --no-tags --depth 25 "$url" "$branch" || return 1
  fi
  # Deepen until the pinned commit is present with `need` first-parent commits behind it
  # (the branch tip may have moved past the pin since the results were produced).
  while ! git -C "$dir" cat-file -e "$pin^{commit}" 2>/dev/null \
    || [ "$(git -C "$dir" rev-list --first-parent --count "$pin")" -le "$need" ]; do
    local before
    before=$(git -C "$dir" rev-list --count FETCH_HEAD)
    retry git -C "$dir" fetch -q --no-tags --deepen 60 "$url" "$branch" || return 1
    if [ "$(git -C "$dir" rev-list --count FETCH_HEAD)" -eq "$before" ]; then break; fi
    echo "[$name] $(git -C "$dir" rev-list --first-parent --count FETCH_HEAD) first-parent commits"
  done
  git -C "$dir" checkout -q --detach "$pin"
  echo "[$name] at $(git -C "$dir" log -1 --format='%h %ad' "$pin")"
}

wanted=("$@")
for spec in "${TARGETS[@]}"; do
  IFS='|' read -r name url branch pin need <<<"$spec"
  if [ ${#wanted[@]} -gt 0 ] && [[ ! " ${wanted[*]} " =~ " $name " ]]; then continue; fi
  fetch_one "$name" "$url" "$branch" "$pin" "$need" || echo "[$name] FAILED"
done
