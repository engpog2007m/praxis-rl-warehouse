#!/usr/bin/env bash
# Start headless Gazebo Harmonic and the ROS 2 <-> Gazebo bridge.
#
# Usage: tools/launch_sim.sh [lidar|final|legacy]
#   lidar   praxis_multiagent_lidar.sdf  - Era 2 (LiDAR, peer obstacle)  [default]
#   final   praxis_multiagent_final.sdf  - Era 1 environment, corrected physics
#   legacy  generated defect world       - Era 1 exact-number replication only
#
# Guards against the failure modes seen during the research:
#   - refuses to start if any gz sim process is already running
#   - refuses a world whose physics state does not match the mode
#   - verifies every bridge argument was created (a single mistyped
#     character silently dropped the pose bridge once)
set -euo pipefail

MODE="${1:-lidar}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$REPO/setup_env.sh" >/dev/null
LOG="$REPO/logs"
mkdir -p "$LOG"

die() { echo "ERROR: $*" >&2; exit 1; }
# Anchored to the executable so a shell, editor or grep whose command line
# merely mentions "gz sim" is never matched.
GZ_PAT='(^|/)gz sim '

case "$MODE" in
  lidar)  WORLD="$REPO/worlds/praxis_multiagent_lidar.sdf"; WANT=CORRECTED ;;
  final)  WORLD="$REPO/worlds/praxis_multiagent_final.sdf"; WANT=CORRECTED ;;
  legacy) WORLD="$REPO/worlds/praxis_multiagent_final_legacy_defect.sdf"; WANT=LEGACY
          if [[ ! -f "$WORLD" ]]; then
            python3 "$REPO/tools/world_physics.py" legacy \
              "$REPO/worlds/praxis_multiagent_final.sdf" || die "could not generate legacy world"
          fi ;;
  *) echo "usage: $0 [lidar|final|legacy]" >&2; exit 2 ;;
esac
[[ -f "$WORLD" ]] || die "world not found: $WORLD"

state="$(python3 "$REPO/tools/world_physics.py" check "$WORLD" | awk '{print $2}')" || true
[[ "$state" == "$WANT" ]] || die "$(basename "$WORLD") physics state is ${state:-UNKNOWN}, mode '$MODE' requires $WANT"

if pgrep -f "$GZ_PAT" >/dev/null 2>&1; then
  die "a gz sim process is already running; run tools/stop_sim.sh first"
fi

echo "Starting Gazebo: $(basename "$WORLD")  [physics: $state]"
nohup gz sim -r -s -v 1 "$WORLD" > "$LOG/gz_sim.log" 2>&1 &
echo $! > "$LOG/gz_sim.pid"

for _ in $(seq 1 60); do
  gz topic -l 2>/dev/null | grep -q "/world/praxis_warehouse/clock" && break
  sleep 1
done
gz topic -l 2>/dev/null | grep -q "/world/praxis_warehouse/clock" || {
  tail -20 "$LOG/gz_sim.log" >&2; die "Gazebo did not start within 60 s"; }
echo "  Gazebo up (pid $(cat "$LOG/gz_sim.pid"))"

ARGS=(
  "/world/praxis_warehouse/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock"
  "/amr_1/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist"
  "/amr_2/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist"
  "/amr_1/odom@nav_msgs/msg/Odometry[gz.msgs.Odometry"
  "/amr_2/odom@nav_msgs/msg/Odometry[gz.msgs.Odometry"
  "/world/praxis_warehouse/dynamic_pose/info@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V"
)
if [[ "$MODE" == lidar ]]; then
  ARGS+=(
    "/amr_1/scan@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan"
    "/amr_2/scan@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan"
  )
fi

nohup ros2 run ros_gz_bridge parameter_bridge "${ARGS[@]}" > "$LOG/bridge.log" 2>&1 &
echo $! > "$LOG/bridge.pid"

created=0
for _ in $(seq 1 20); do
  created=$(grep -c "Creating" "$LOG/bridge.log" 2>/dev/null || true)
  (( created >= ${#ARGS[@]} )) && break
  sleep 1
done
if grep -q "Failed to create" "$LOG/bridge.log"; then
  grep "Failed to create" "$LOG/bridge.log" >&2
  die "bridge rejected an argument"
fi
(( created == ${#ARGS[@]} )) || die "bridge created $created of ${#ARGS[@]} topics; see $LOG/bridge.log"
echo "  Bridge up: $created/${#ARGS[@]} topics (pid $(cat "$LOG/bridge.pid"))"
echo "Next: python3 tools/preflight_check.py --mode $MODE"
