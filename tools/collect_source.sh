#!/usr/bin/env bash
# Collect the working Project Praxis source tree into this repository.
#
# Usage:
#   tools/collect_source.sh <PRAXIS_DIR> [--no-models] [--no-data] [--max-data-mb N]
#
# Example:
#   tools/collect_source.sh ~/Praxis
#
# Copies only the code and artifacts that produced the reported results,
# then runs tools/collect_helper.py to make paths portable and verify
# every module compiles. The original directory is never modified.
set -euo pipefail
shopt -s nullglob

die()  { echo "ERROR: $*" >&2; exit 1; }
info() { echo "  $*"; }

SRC="${1:-}"
[[ -n "$SRC" ]] || die "usage: tools/collect_source.sh <PRAXIS_DIR> [--no-models] [--no-data] [--max-data-mb N]"
shift || true
SRC="$(cd "$SRC" 2>/dev/null && pwd)" || die "source directory not found"
[[ -d "$SRC/src/simulation" ]] || die "$SRC does not look like the Praxis tree (no src/simulation)"

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[[ "$SRC" != "$REPO" ]] || die "source and repository are the same directory"

INCLUDE_MODELS=1
INCLUDE_DATA=1
MAX_DATA_MB=25
while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-models)   INCLUDE_MODELS=0 ;;
    --no-data)     INCLUDE_DATA=0 ;;
    --max-data-mb) MAX_DATA_MB="${2:?}"; shift ;;
    *) die "unknown option: $1" ;;
  esac
  shift
done

echo "Collecting from: $SRC"
echo "Into repository: $REPO"

# ---------------------------------------------------------------- simulation
echo
echo "[copy] simulation modules"
mkdir -p "$REPO/src/simulation/_history"
for f in "$SRC"/src/simulation/*.py; do
  base="$(basename "$f")"
  case "$base" in
    data_registry.py|warehouse_env.py|gazebo_env.py|gazebo_env_lidar.py|__init__.py)
      cp -p "$f" "$REPO/src/simulation/"; info "canonical  $base" ;;
    *)
      cp -p "$f" "$REPO/src/simulation/_history/"; info "history    $base" ;;
  esac
done
[[ -f "$REPO/src/simulation/__init__.py" ]] || : > "$REPO/src/simulation/__init__.py"
[[ -f "$REPO/src/__init__.py" ]] || : > "$REPO/src/__init__.py"

# ------------------------------------------------------------------- scripts
echo
echo "[copy] scripts (layout preserved; nothing is moved or renamed)"
for f in "$SRC"/scripts/*.py; do cp -p "$f" "$REPO/scripts/"; info "scripts/$(basename "$f")"; done
for f in "$SRC"/*.py; do cp -p "$f" "$REPO/"; info "$(basename "$f")  (root)"; done

# -------------------------------------------------------------------- worlds
echo
echo "[copy] worlds"
mkdir -p "$REPO/worlds"
for f in "$SRC"/worlds/*.sdf; do cp -p "$f" "$REPO/worlds/"; info "worlds/$(basename "$f")"; done

# -------------------------------------------------------------------- assets
echo
echo "[copy] assets"
if [[ -d "$SRC/assets" ]]; then
  cp -a "$SRC/assets/." "$REPO/assets/"
  size_mb=$(du -sm "$REPO/assets" | cut -f1)
  info "assets/ copied (${size_mb} MB)"
  (( size_mb < 90 )) || info "WARN assets exceed 90 MB; consider Git LFS for meshes"
else
  info "WARN no assets/ directory found"
fi

# -------------------------------------------------------------------- models
if (( INCLUDE_MODELS )); then
  echo
  echo "[copy] trained models (reference copies kept read-only)"
  mkdir -p "$REPO/models/reference"
  chmod -R u+w "$REPO/models/reference" 2>/dev/null || true
  for f in "$SRC"/models/*.zip "$SRC"/models/*.pkl; do
    cp -p "$f" "$REPO/models/"
    cp -p "$f" "$REPO/models/reference/"
    info "models/$(basename "$f")"
  done
  chmod a-w "$REPO"/models/reference/* 2>/dev/null || true
fi

# ---------------------------------------------------------------------- data
if (( INCLUDE_DATA )); then
  echo
  echo "[copy] result data (files up to ${MAX_DATA_MB} MB; archives excluded)"
  for f in "$SRC"/data/*.csv "$SRC"/data/*.json; do
    mb=$(( ( $(stat -c%s "$f") + 1048575 ) / 1048576 ))
    if (( mb <= MAX_DATA_MB )); then
      cp -p "$f" "$REPO/data/"; info "data/$(basename "$f")  (${mb} MB)"
    else
      info "SKIP data/$(basename "$f")  (${mb} MB > ${MAX_DATA_MB} MB; regenerate via pipeline)"
    fi
  done
  for f in "$SRC"/results/*.png; do cp -p "$f" "$REPO/results/"; done
  pngs=("$REPO"/results/*.png)
  info "results/ figures: ${#pngs[@]}"
fi

# ------------------------------------------------------------ dependency pin
echo
echo "[freeze] Python dependencies"
if [[ -x "$SRC/venv/bin/pip" ]]; then
  "$SRC/venv/bin/pip" freeze > "$REPO/requirements-freeze.txt"
  info "requirements-freeze.txt ($(wc -l < "$REPO/requirements-freeze.txt") packages, exact versions)"
else
  info "WARN no venv at $SRC/venv; requirements-freeze.txt not generated"
fi
{
  echo "# Recorded by tools/collect_source.sh on $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "os=$(. /etc/os-release 2>/dev/null && echo "$PRETTY_NAME")"
  echo "kernel=$(uname -r)"
  echo "python=$("$SRC/venv/bin/python" --version 2>&1 || python3 --version 2>&1)"
  echo "gazebo=$(gz sim --version 2>/dev/null | head -1 || echo unknown)"
  echo "ros_distro=${ROS_DISTRO:-unknown (source /opt/ros/jazzy/setup.bash before collecting to record)}"
  echo "gpu=$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null || echo unknown)"
} > "$REPO/environment/RECORDED_ENVIRONMENT.txt"
info "environment/RECORDED_ENVIRONMENT.txt"

# ------------------------------------------------------------ post-process
echo
echo "[process] portability, verification, audit, provenance"
status=0
python3 "$REPO/tools/collect_helper.py" --repo "$REPO" || status=$?

echo
if (( status == 0 )); then
  echo "Collection complete. Review any WARN lines above, then:"
  echo "  cd $REPO && ./tests/run_self_tests.sh && ./run_all.sh stats"
else
  echo "Collection finished with compile failures. Fix the files listed above before committing."
fi
exit $status
