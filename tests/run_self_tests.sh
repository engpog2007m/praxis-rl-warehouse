#!/usr/bin/env bash
# Self-tests for the repository tooling. No ROS or Gazebo installation needed.
#
# Section A runs anywhere. Section B starts and kills fake simulator
# processes; it is skipped automatically if a real Gazebo or bridge is
# running, so it can never interfere with a live experiment.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
T="$(mktemp -d)"
trap 'rm -rf "$T"' EXIT
STUB_BIN="$REPO/tests/stubs/bin"
STUB_PY="$REPO/tests/stubs/py"
FAKE_BIN="$REPO/tests/fakebin"
GZ_PAT='(^|/)gz sim '
BR_PAT='^[^ ]*/ros_gz_bridge/parameter_bridge'

PASS=0; FAIL=0
ok()  { PASS=$((PASS+1)); printf '  PASS  %s\n' "$1"; }
bad() { FAIL=$((FAIL+1)); printf '  FAIL  %s\n' "$1"; [[ -n "${2:-}" ]] && printf '        %s\n' "$2"; }
expect() { # label expected_exit actual_exit
  if [[ "$2" == "$3" ]]; then ok "$1"; else bad "$1" "expected exit $2, got $3"; fi; }
contains() { # label file pattern
  if grep -Eq -- "$3" "$2"; then ok "$1"; else bad "$1" "pattern not found: $3"; fi; }

mini_repo() { # copy only the tooling into a scratch repository
  local dst="$1"
  mkdir -p "$dst"
  cp -a "$REPO/tools" "$REPO/analysis" "$REPO/tests" "$REPO/run_all.sh" "$REPO/setup_env.sh" "$dst/"
  mkdir -p "$dst"/{src/simulation,scripts,worlds,assets,models,data,results,logs,docs,environment}
}

echo "Building fixtures"
python3 "$REPO/tests/make_fixtures.py" "$T/fx" >/dev/null || { echo "fixture build failed"; exit 1; }

# ===================================================================== A
echo
echo "A1. World physics classification and reversal"
W="$T/fx/corrected_world.sdf"
python3 "$REPO/tools/world_physics.py" check "$W" >"$T/o" 2>&1; expect "corrected world classified CORRECTED" 0 $?
python3 "$REPO/tools/world_physics.py" legacy "$W" >"$T/o" 2>&1; expect "legacy world generated" 0 $?
contains "round-trip verified" "$T/o" "round-trip verified"
L="$T/fx/corrected_world_legacy_defect.sdf"
python3 "$REPO/tools/world_physics.py" check "$L" >"$T/o" 2>&1; expect "legacy world classified LEGACY (exit 1)" 1 $?
contains "legacy classification text" "$T/o" "LEGACY"
python3 "$REPO/tools/world_physics.py" legacy "$L" >"$T/o" 2>&1; expect "refuses to reverse a legacy world" 1 $?
head -1 "$L" | grep -q '^<?xml' && ok "XML declaration kept on line 1" || bad "XML declaration kept on line 1"

echo
echo "A2. Source collector"
R="$T/collect_repo"; mini_repo "$R"
bash "$R/tools/collect_source.sh" "$T/fx/mock_praxis" >"$T/c1" 2>&1; expect "collector exits 0" 0 $?
contains "7 paths made portable"                "$T/c1" "total rewrites: 7"
contains "stale v6 model name normalised to v4" "$T/c1" "normalised model/curve filenames to v4"
contains "all modules compile"                  "$T/c1" "modules compiled: ([0-9]+)/\\1$"
contains "tab indentation flagged"              "$T/c1" "WARN tabs +scripts/tabbed_script.py"
contains "non-portable path flagged, not mangled" "$T/c1" "still has a non-portable"
contains "absolute path in SDF flagged"         "$T/c1" "contains an absolute /home/"
contains "oversized data file skipped"          "$T/c1" "SKIP data/dwb_lidar_demonstrations.csv"
contains "history variant separated"            "$T/c1" "history +gazebo_env_v4_backup.py"
[[ $(grep -c "matches reported configuration" "$T/c1") == 2 ]] \
  && ok "both environments match reported parameters" || bad "both environments match reported parameters"
[[ -f "$R/docs/PROVENANCE.md" ]] && grep -q four_way_lidar_results.csv "$R/docs/PROVENANCE.md" \
  && ok "provenance written" || bad "provenance written"
python3 - "$R/scripts/evaluate_four_way_lidar.py" >"$T/o" 2>&1 << 'EOF'
import os, re, sys
src = open(sys.argv[1]).read()
line = [l for l in src.splitlines() if l.startswith('g = ')][0]
os.environ.pop('PRAXIS_ROOT', None)
path = 'ppo_lidar_guided'
ns = {'os': os, 'path': path}
exec(line, ns)
assert ns['g'] == os.path.expanduser(f'~/Praxis/models/{path}'), ns['g']
os.environ['PRAXIS_ROOT'] = '/opt/x'
exec(line, ns)
assert ns['g'] == '/opt/x/models/ppo_lidar_guided', ns['g']
EOF
expect "rewritten paths behave identically when PRAXIS_ROOT is unset" 0 $?
bash "$R/tools/collect_source.sh" "$T/fx/mock_praxis" >"$T/c2" 2>&1; expect "collector re-run exits 0" 0 $?
diff <(sed -n '/\[1\/5\]/,/\[3\/5\]/p' "$T/c1") <(sed -n '/\[1\/5\]/,/\[3\/5\]/p' "$T/c2") >/dev/null \
  && ok "collector is idempotent" || bad "collector is idempotent"

echo
echo "A3. Statistics reproducer against count-exact synthetic data"
PRAXIS_ROOT="$T/fx/synth" python3 "$REPO/analysis/reproduce_statistics.py" >"$T/s" 2>&1
for label in "PPO_unguided collision p vs DWB" "BC_ml completion p vs DWB" "BC_ml collision p vs DWB" \
             "H3 collision effect size r" "H1 p \(one-sided, PPO > DWB\)" "H1 effect size r" \
             "H1 relative advantage" "H3 \(no obstacle\) completion p" "h3_test_b_ood.csv p" \
             "h3_roadmap_results.csv p" "stress tier low_load p" "stress tier congested p"; do
  contains "reproduces: $label" "$T/s" "^  OK  +$label"
done

echo
echo "A4. Preflight check (stub ROS, stub process table)"
pf() { # label expected_exit mode env...
  local label="$1" want="$2" mode="$3"; shift 3
  env PATH="$STUB_BIN:$PATH" PYTHONPATH="$STUB_PY" "$@" \
    python3 "$REPO/tools/preflight_check.py" --mode "$mode" >"$T/pf" 2>&1
  expect "$label" "$want" $?
}
pf "healthy simulator passes"                0 lidar  FAKE_MOVES=1
pf "robot not moving is blocked"             1 lidar  FAKE_MOVES=0
pf "legacy mode reports but does not block"  0 legacy FAKE_MOVES=0
pf "two Gazebo instances are blocked"        1 lidar  FAKE_GZ_COUNT=2
pf "no Gazebo instance is blocked"           1 lidar  FAKE_GZ_COUNT=0
pf "missing pose bridge is blocked"          1 lidar  FAKE_DROP_POSE=1
pf "silent LiDAR is blocked"                 1 lidar  FAKE_SCAN=0
pf "final mode does not require LiDAR"       0 final  FAKE_SCAN=0

# ===================================================================== B
echo
if pgrep -f "$GZ_PAT" >/dev/null 2>&1 || pgrep -f "$BR_PAT" >/dev/null 2>&1; then
  echo "B. SKIPPED: a real Gazebo or bridge is running; process tests would stop it."
else
  echo "B1. Simulator launch and stop (fake gz/ros2, real process management)"
  B="$T/proc_repo"; mini_repo "$B"
  cp "$W" "$B/worlds/praxis_multiagent_lidar.sdf"
  cp "$W" "$B/worlds/praxis_multiagent_final.sdf"
  export PRAXIS_NO_ROS=1
  P="$FAKE_BIN:$PATH"
  PATH="$P" "$B/tools/launch_sim.sh" lidar  >"$T/l" 2>&1; expect "launch lidar world" 0 $?
  contains "all 8 bridge topics created" "$T/l" "Bridge up: 8/8"
  PATH="$P" "$B/tools/launch_sim.sh" lidar  >"$T/l" 2>&1; expect "second launch refused" 1 $?
  PATH="$P" "$B/tools/stop_sim.sh"          >"$T/l" 2>&1; expect "stop" 0 $?
  [[ $(pgrep -fc "$GZ_PAT") == 0 ]] && ok "no Gazebo processes remain" || bad "no Gazebo processes remain"
  PATH="$P" "$B/tools/launch_sim.sh" legacy >"$T/l" 2>&1; expect "legacy world auto-generated and launched" 0 $?
  contains "legacy bridge 6/6" "$T/l" "Bridge up: 6/6"
  PATH="$P" "$B/tools/stop_sim.sh" >/dev/null 2>&1
  PATH="$P" FAKE_BRIDGE_FAIL=1 "$B/tools/launch_sim.sh" lidar >"$T/l" 2>&1; expect "rejected bridge argument detected" 1 $?
  PATH="$P" "$B/tools/stop_sim.sh" >/dev/null 2>&1
  cp "$B/worlds/praxis_multiagent_final_legacy_defect.sdf" "$B/worlds/praxis_multiagent_lidar.sdf"
  PATH="$P" "$B/tools/launch_sim.sh" lidar >"$T/l" 2>&1; expect "defect world refused in lidar mode" 1 $?
  cp "$W" "$B/worlds/praxis_multiagent_lidar.sdf"

  echo
  echo "B2. Orchestrator (fake pipeline scripts)"
  for s in train_lidar_guided train_lidar_unguided harvest_dwb_lidar train_bc_lidar evaluate_four_way_lidar; do
    printf 'print("RAN %s")\n' "$s" > "$B/scripts/$s.py"
  done
  printf 'import os\nprint("  PRE-SPECIFIED CRITERION (collision rate drop >= 0.15): " + os.environ.get("PROBE_RESULT","MET"))\n' \
    > "$B/scripts/probe_lidar_avoidance.py"
  run_b() { PATH="$P" PYTHONPATH="$STUB_PY" "$@" >"$T/r" 2>&1; }
  run_b env PROBE_RESULT=MET "$B/run_all.sh" train-lidar; expect "train-lidar completes when gate is met" 0 $?
  [[ $(grep -c "^RAN" "$T/r") == 5 ]] && ok "all five training/evaluation stages ran in order" \
    || bad "all five training/evaluation stages ran in order"
  run_b env PROBE_RESULT="NOT MET" "$B/run_all.sh" train-lidar; expect "train-lidar stops when gate is not met" 1 $?
  contains "probe ran and its criterion was read" "$T/r" "did not meet its criterion"
  grep -q "RAN train_lidar_guided" "$T/r" && bad "no training after failed gate" || ok "no training after failed gate"
  printf 'raise SystemExit(3)\n' > "$B/scripts/evaluate_four_way_lidar.py"
  run_b "$B/run_all.sh" eval-lidar; expect "crashing stage reported" 1 $?
  contains "failure attributed to the crashing script" "$T/r" "evaluate_four_way_lidar.py exited with status 3"
  [[ $(pgrep -fc "$GZ_PAT") == 0 ]] && ok "Gazebo cleaned up after crash" || bad "Gazebo cleaned up after crash"
  printf 'print("RAN evaluate")\n' > "$B/scripts/evaluate_four_way_lidar.py"
  run_b env FAKE_MOVES=0 "$B/run_all.sh" eval-lidar; expect "run blocked when robot does not move" 1 $?
  contains "blocked by the motion check specifically" "$T/r" "FAIL\\] physical_motion"
  grep -q "RAN evaluate" "$T/r" && bad "no evaluation on defective physics" || ok "no evaluation on defective physics"
  unset PRAXIS_NO_ROS
fi

echo
echo "======================================================================"
echo "SELF-TESTS: $PASS passed, $FAIL failed"
exit $(( FAIL > 0 ))
