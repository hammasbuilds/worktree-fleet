#!/usr/bin/env bash
# Build the test environment for every experiment target, pinned to the exact versions the
# results were produced with. Test results are cached under a hash of the installed
# packages, so a different environment never reuses them.
#
# Usage: scripts/setup_target_venvs.sh [name ...]
set -euo pipefail
cd "$(dirname "$0")/../targets"
unset VIRTUAL_ENV

COMMON="pytest==8.2.2 pluggy==1.6.0 iniconfig==2.3.0 packaging==26.3 colorama==0.4.6"
declare -A EXTRA=(
  [flask]="werkzeug==3.1.8 jinja2==3.1.6 itsdangerous==2.2.0 click==8.5.0 blinker==1.9.0 \
markupsafe==3.0.3 python-dotenv==1.2.3 asgiref==3.12.1 pygments==2.21.0"
  [flask-2x]="werkzeug==2.3.7 jinja2==3.1.2 itsdangerous==2.2.0 click==8.1.7 blinker==1.9.0 markupsafe==3.0.3 python-dotenv==1.2.3 asgiref==3.12.1"
  [sqlparse]=""
  [click]=""
  [more-itertools]=""
)

names=("$@")
[ ${#names[@]} -eq 0 ] && names=(flask flask-2x sqlparse click more-itertools)
for name in "${names[@]}"; do
  venv=".venvs/$name"
  py="$venv/Scripts/python.exe"
  [ -x "$py" ] || py="$venv/bin/python"
  uv venv -q --python 3.12 "$venv"
  [ -x "$py" ] || py="$venv/bin/python"
  # shellcheck disable=SC2086
  uv pip install -q --python "$py" $COMMON ${EXTRA[$name]}
  if [ "$name" = click ]; then
    # click's test_deprecations reads click's *installed* version, but the code under test is
    # imported from the worktree via PYTHONPATH. A metadata-only stub satisfies the lookup.
    site=$("$py" -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
    mkdir -p "$site/click-0.0.0.dist-info"
    printf 'Metadata-Version: 2.1\nName: click\nVersion: 0.0.0\n' \
      >"$site/click-0.0.0.dist-info/METADATA"
    printf 'worktree-fleet\n' >"$site/click-0.0.0.dist-info/INSTALLER"
  fi
  echo "[$name] $("$py" -m pytest --version 2>&1)"
done
