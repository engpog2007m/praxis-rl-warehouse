#!/usr/bin/env bash
# Stop Gazebo and the bridge, escalating to SIGKILL if needed.
# A wedged gz sim process ignored SIGTERM during the research and silently
# competed with a second instance, so this script verifies nothing remains.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG="$REPO/logs"
# Anchored patterns: an unanchored "gz sim" also matches any shell, editor or
# grep whose command line mentions the text, and would kill it.
GZ_PAT='(^|/)gz sim '
# Matches only the bridge binary or its ros2 launcher, anchored at argv[0].
BR_PAT='^[^ ]*/ros_gz_bridge/parameter_bridge|^[^ ]*python3?[^ ]* [^ ]*/ros2 run ros_gz_bridge parameter_bridge'

for name in bridge gz_sim; do
  pidf="$LOG/$name.pid"
  if [[ -f "$pidf" ]]; then
    kill "$(cat "$pidf")" 2>/dev/null || true
    rm -f "$pidf"
  fi
done
pkill -f "$BR_PAT" 2>/dev/null || true
pkill -f "$GZ_PAT" 2>/dev/null || true

for _ in $(seq 1 10); do
  pgrep -f "$GZ_PAT" >/dev/null 2>&1 || pgrep -f "$BR_PAT" >/dev/null 2>&1 || break
  sleep 1
done

if pgrep -f "$GZ_PAT" >/dev/null 2>&1 || pgrep -f "$BR_PAT" >/dev/null 2>&1; then
  echo "Processes ignored SIGTERM; sending SIGKILL"
  pkill -9 -f "$BR_PAT" 2>/dev/null || true
  pkill -9 -f "$GZ_PAT" 2>/dev/null || true
  sleep 2
fi

if pgrep -f "$GZ_PAT" >/dev/null 2>&1; then
  echo "ERROR: gz sim still running:" >&2
  pgrep -af "$GZ_PAT" >&2
  exit 1
fi
echo "Simulator stopped."
