#!/usr/bin/env bash
# Project Praxis pipeline orchestrator.
#
# Usage: ./run_all.sh <stage> [--legacy-physics] [--skip-gate]
#
# Stages (no simulator):
#   check           verify scripts, models, data and worlds are in place    ~5 s
#   stats           recompute every reported statistic from data/ CSVs     ~1 min
#   selftest        verify the repository tooling                           ~2 min
#   figures         regenerate chapter figures from data/ CSVs              ~1 min
#   restore-models  restore trained models from models/reference/
#
# Stages (simulator; each starts a fresh Gazebo and must pass preflight):
#   preflight       launch the LiDAR world and run the health check         ~2 min
#   eval-lidar      Era 2 four-way evaluation using the shipped models      ~2-3 h
#   train-lidar     Era 2 end to end: probe gate, guided, unguided,
#                   DWB demonstrations, behavioural cloning, evaluation     ~17 h
#   era1            Era 1 pipeline (7-dim observation, no obstacle)          ~20 h
#
# Options:
#   --legacy-physics  era1 only: run on the regenerated defect world so the
#                     originally reported Era 1 numbers can be reproduced.
#                     Without it, era1 runs on corrected physics and will
#                     produce different numbers by design.
#   --skip-gate       train-lidar only: continue even if the 50K-step
#                     avoidance probe does not meet its pre-specified
#                     criterion (PPO training is stochastic).
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAGE="${1:-}"
shift || true
LEGACY=0
SKIP_GATE=0
for a in "$@"; do
  case "$a" in
    --legacy-physics) LEGACY=1 ;;
    --skip-gate)      SKIP_GATE=1 ;;
    *) echo "unknown option: $a" >&2; exit 2 ;;
  esac
done

if [[ -z "$STAGE" || "$STAGE" == "-h" || "$STAGE" == "--help" ]]; then
  awk 'NR==1{next} /^#/{sub(/^# ?/,""); print; next} {exit}' "$0"
  exit 0
fi

# shellcheck disable=SC1091
source "$REPO/setup_env.sh"
mkdir -p "$REPO/logs"

die()  { echo; echo "ERROR: $*" >&2; exit 1; }
hr()   { echo; echo "======================================================================"; echo "$*"; echo "======================================================================"; }

SIM_UP=0
cleanup() { if (( SIM_UP )); then "$REPO/tools/stop_sim.sh" >/dev/null 2>&1 || true; fi; }
trap cleanup EXIT INT TERM

run_py() {
  local script="$1"; shift
  local path="$REPO/scripts/$script"
  [[ -f "$path" ]] || die "scripts/$script is missing. Run tools/collect_source.sh first."
  local log="$REPO/logs/${script%.py}.log"
  echo "  -> scripts/$script   (log: logs/$(basename "$log"))"
  set +e
  python3 -u "$path" "$@" 2>&1 | tee "$log"
  local rc=${PIPESTATUS[0]}
  set -e
  (( rc == 0 )) || die "scripts/$script exited with status $rc"
}

# Each simulator stage gets a fresh Gazebo. Long sessions accumulated
# wedged services and duplicate instances during the research.
sim_session() {
  local mode="$1"
  "$REPO/tools/stop_sim.sh" >/dev/null 2>&1 || true
  "$REPO/tools/launch_sim.sh" "$mode"
  SIM_UP=1
  python3 "$REPO/tools/preflight_check.py" --mode "$mode" \
    || die "preflight failed; not starting the run (see output above)"
}

sim_end() { "$REPO/tools/stop_sim.sh" >/dev/null 2>&1 || true; SIM_UP=0; }

notice_models() {
  echo "  NOTE: this stage overwrites trained models in models/."
  echo "        Reference copies are preserved in models/reference/;"
  echo "        restore them any time with ./run_all.sh restore-models"
}

case "$STAGE" in
  check)
    hr "Repository completeness check"
    missing=0
    # stage | script | an artifact that script is known to write
    while IFS='|' read -r stg script artifact; do
      if [[ -f "$REPO/scripts/$script" ]]; then
        printf '  OK       %-10s scripts/%s\n' "$stg" "$script"
      else
        missing=1
        cands=$(grep -l -- "$artifact" "$REPO"/scripts/*.py 2>/dev/null | xargs -r -n1 basename | tr '\n' ' ')
        printf '  MISSING  %-10s scripts/%s\n' "$stg" "$script"
        printf '           writes %s; scripts mentioning it: %s\n' "$artifact" "${cands:-none}"
      fi
    done << 'MAP'
train-lidar|probe_lidar_avoidance.py|lidar_avoid_probe.csv
train-lidar|train_lidar_guided.py|lidar_guided_training_curves.csv
train-lidar|train_lidar_unguided.py|lidar_unguided_training_curves.csv
train-lidar|harvest_dwb_lidar.py|dwb_lidar_demonstrations.csv
train-lidar|train_bc_lidar.py|bc_lidar_policy.pkl
eval-lidar|evaluate_four_way_lidar.py|four_way_lidar_results.csv
era1|train_gazebo_ppo.py|gazebo_training_curves_v4.csv
era1|evaluate_gazebo_trained.py|gazebo_trained_results_v4.csv
era1|harvest_dwb_demos.py|dwb_demonstrations.csv
era1|train_behavioural_cloning.py|bc_policy.pkl
era1|train_unguided_ppo.py|unguided_training_curves.csv
era1|evaluate_four_way.py|four_way_results.csv
MAP
    echo
    for m in ppo_lidar_guided.zip ppo_lidar_unguided.zip bc_lidar_policy.pkl; do
      if [[ -f "$REPO/models/$m" ]]; then echo "  OK       model   models/$m"
      else echo "  MISSING  model   models/$m (needed by eval-lidar)"; missing=1; fi
    done
    for d in four_way_lidar_results.csv gazebo_trained_results_v4.csv four_way_results.csv; do
      if [[ -f "$REPO/data/$d" ]]; then echo "  OK       data    data/$d"
      else echo "  MISSING  data    data/$d (needed by stats)"; missing=1; fi
    done
    for w in praxis_multiagent_lidar.sdf praxis_multiagent_final.sdf; do
      if [[ -f "$REPO/worlds/$w" ]]; then
        echo "  $(python3 "$REPO/tools/world_physics.py" check "$REPO/worlds/$w" | cut -d'(' -f1)"
      else echo "  MISSING  world   worlds/$w"; missing=1; fi
    done
    echo
    if (( missing )); then
      echo "Incomplete. For a MISSING script, either rename the script listed as"
      echo "mentioning its artifact, or edit the stage in run_all.sh to use it."
      exit 1
    fi
    echo "Repository complete."
    ;;

  stats)
    hr "Recomputing reported statistics"
    python3 "$REPO/analysis/reproduce_statistics.py"
    ;;

  selftest)
    "$REPO/tests/run_self_tests.sh"
    ;;

  figures)
    hr "Regenerating figures"
    found=0
    for s in generate_figures_v4.py generate_figures_fourway.py generate_figures_lidar.py; do
      if [[ -f "$REPO/scripts/$s" ]]; then run_py "$s"; found=1; fi
    done
    (( found )) || die "no figure scripts found in scripts/"
    ;;

  restore-models)
    [[ -d "$REPO/models/reference" ]] || die "no models/reference directory"
    cp -f "$REPO"/models/reference/* "$REPO/models/"
    chmod u+w "$REPO"/models/*.zip "$REPO"/models/*.pkl 2>/dev/null || true
    echo "Restored models from models/reference/"
    ;;

  preflight)
    hr "Simulator preflight (LiDAR world)"
    sim_session lidar
    sim_end
    ;;

  eval-lidar)
    hr "Era 2 four-way evaluation with shipped models (~2-3 h)"
    sim_session lidar
    run_py evaluate_four_way_lidar.py
    sim_end
    python3 "$REPO/analysis/reproduce_statistics.py" || true
    ;;

  train-lidar)
    hr "Era 2 full pipeline (~17 h)"
    notice_models

    hr "[1/6] Avoidance probe (50K steps, pre-specified gate)"
    sim_session lidar
    run_py probe_lidar_avoidance.py
    sim_end
    if grep -Eq 'CRITERION.*\): MET' "$REPO/logs/probe_lidar_avoidance.log"; then
      echo "  Gate: criterion MET"
    elif (( SKIP_GATE )); then
      echo "  Gate: criterion NOT met; continuing because --skip-gate was given"
    else
      die "avoidance probe did not meet its criterion. Re-run, or pass --skip-gate."
    fi

    hr "[2/6] Guided PPO, 500K steps (~6.5 h)"
    sim_session lidar; run_py train_lidar_guided.py; sim_end

    hr "[3/6] Unguided PPO, 500K steps (~6.5 h)"
    sim_session lidar; run_py train_lidar_unguided.py; sim_end

    hr "[4/6] Nav2 DWB demonstrations (~30 min)"
    sim_session lidar; run_py harvest_dwb_lidar.py; sim_end

    hr "[5/6] Behavioural cloning (no simulator, ~5 min)"
    run_py train_bc_lidar.py

    hr "[6/6] Four-way evaluation (~2-3 h)"
    sim_session lidar; run_py evaluate_four_way_lidar.py; sim_end

    python3 "$REPO/analysis/reproduce_statistics.py" || true
    ;;

  era1)
    mode=final
    if (( LEGACY )); then
      mode=legacy
      hr "Era 1 on the LEGACY DEFECT world (exact-number replication only)"
    else
      hr "Era 1 on corrected physics (numbers will differ from the praxis by design)"
    fi
    notice_models

    sim_session "$mode"; run_py train_gazebo_ppo.py;        sim_end
    sim_session "$mode"; run_py evaluate_gazebo_trained.py; sim_end
    sim_session "$mode"; run_py harvest_dwb_demos.py;       sim_end
    run_py train_behavioural_cloning.py
    sim_session "$mode"; run_py train_unguided_ppo.py;      sim_end
    sim_session "$mode"; run_py evaluate_four_way.py;       sim_end
    python3 "$REPO/analysis/reproduce_statistics.py" || true
    ;;

  *)
    die "unknown stage '$STAGE' (run ./run_all.sh --help)"
    ;;
esac
