# Reinforcement Learning and Comparative Machine Learning for Adaptive Mobile Robot Control in E-commerce Warehouses

Code, trained models, and result data for the doctoral praxis of Patriot Gumbo,
The George Washington University, School of Engineering and Applied Science (2026).

The work compares a Proximal Policy Optimization (PPO) controller with payload-adaptive
reward shaping against an unguided PPO ablation, a behavioural-cloning policy trained on
classical-planner demonstrations, and the Nav2 DWB classical planner, for autonomous
mobile robots (AMRs) in a simulated e-commerce warehouse built on Gazebo Harmonic and
ROS 2 Jazzy.

---

## Read this first: two evaluation eras

Results come from two environment generations. Both are included, and they are not
interchangeable.

| | Era 1 | Era 2 |
|---|---|---|
| Observation | 7-dim pose/goal/velocity | 31-dim: 7 pose terms + 24 LiDAR sectors |
| Obstacles | none | dynamic peer robot on the ego route |
| Robot physics | **defective** (see below) | corrected and verified |
| Position source | odometry | Gazebo ground-truth pose |
| Supports | H1, H2, Era 1 RQ1 | H3, final RQ1 |

**The Era 1 physics defect.** The original AMR models had no wheel friction and a chassis
collision cylinder resting on the ground plane. Under full commanded velocity the robot
moved about 0.5 mm in 3 s, while odometry, which integrates commanded velocity rather than
measuring displacement, reported normal travel. All Era 1 systems ran under the same defect,
so Era 1 comparisons between systems remain internally consistent, but they are not
physically validated. The correction and its consequences are documented in
[docs/KNOWN_ISSUES.md](docs/KNOWN_ISSUES.md), issue 1. `tools/preflight_check.py` now
refuses to start any run on a robot that does not physically move.

---

## Results

The Era 2 and H1 numbers below are recomputed from the episode records in `data/` by
`./run_all.sh stats`, which reports OK or DIFF against each value. The H2 figures are
quoted from the praxis and are not re-derived by that script.

### Era 2: four systems, active peer obstacle (n = 800 episodes per system)

| System | Completed | Collided | Timed out |
|---|---|---|---|
| PPO, guided | 29.5% | **18.2%** | 52.3% |
| PPO, unguided | **56.3%** | 37.4% | 6.4% |
| Behavioural cloning | 49.4% | 39.9% | 10.8% |
| Nav2 DWB | 45.4% | 41.1% | 13.5% |

- **H3, partially supported.** Guidance roughly halves the collision rate
  (18.2% vs 37.4%, Mann-Whitney one-sided p < 0.0001, r = 0.191) but also reduces task
  completion (29.5% vs 56.3%, two-sided p < 0.0001). The safety claim holds; the
  effectiveness claim does not.
- **RQ1.** Behavioural cloning is statistically indistinguishable from the planner it
  imitates (completion p = 0.109, collision p = 0.611). Unguided PPO exceeds Nav2 DWB on
  completion (p < 0.0001) with no significant difference in collision rate (p = 0.125).

### Era 1: H1, robustness under sensor noise (pre-correction physics)

Pooled at sensor noise sigma >= 0.15, PPO completed 58.2% of episodes against 51.0% for
Nav2 DWB: a 14.1% relative advantage, Mann-Whitney one-sided p = 0.011, bootstrap 95% CI
[2.3%, 27.8%]. The point estimate exceeds the hypothesised 10%; the lower confidence bound
does not.

### Era 1: H2, throughput (pre-correction physics)

Controller-level throughput improved 7.7% relative to Nav2 DWB (49.1% vs 45.6%). A
fleet-level discrete-event simulation did not converge (seed-to-seed coefficient of
variation 0.45 to 0.54) and is not reported as a result.

---

## Replication levels

| Level | What it verifies | Simulator | Time |
|---|---|---|---|
| 1. `./run_all.sh stats` | every reported statistic from the shipped episode records | no | ~1 min |
| 2. `./run_all.sh eval-lidar` | Era 2 evaluation re-run with the shipped trained models | yes | ~2-3 h |
| 3. `./run_all.sh train-lidar` | Era 2 from scratch: train, harvest, clone, evaluate | yes | ~17 h |
| 4. `./run_all.sh era1 --legacy-physics` | Era 1 from scratch on the regenerated defect world | yes | ~20 h |

Level 1 reproduces the reported numbers exactly. Levels 2 to 4 reproduce them
statistically, not exactly: PPO training is not seeded, and Gazebo steps in real time,
so episode timing varies between runs. The direction and significance of the findings
are what should replicate. See [docs/KNOWN_ISSUES.md](docs/KNOWN_ISSUES.md), issue 11.

---

## Setup

Tested on Ubuntu 24.04 LTS, ROS 2 Jazzy, Gazebo Harmonic 8.11.0, Python 3.12,
NVIDIA RTX A1000 laptop GPU. The exact software environment of the reported runs is in
`environment/RECORDED_ENVIRONMENT.txt` and `requirements-freeze.txt`.

```bash
git clone <this repository> praxis-rl-warehouse
cd praxis-rl-warehouse
environment/install_ubuntu2404.sh     # ROS 2 Jazzy, Gazebo, Python venv (skips what exists)
source setup_env.sh
./tests/run_self_tests.sh             # verifies the tooling, ~3 min, no simulator needed
./run_all.sh check                    # verifies scripts, models, data and worlds are present
./run_all.sh stats                    # replication level 1
```

Simulator stages manage Gazebo themselves. To drive it by hand:

```bash
tools/launch_sim.sh lidar             # or: final | legacy
python3 tools/preflight_check.py      # must print PREFLIGHT PASSED before any run
tools/stop_sim.sh
```

---

## Repository layout

```
run_all.sh                 pipeline orchestrator; ./run_all.sh --help
setup_env.sh               source before any run
src/simulation/            environments and dataset registry
  gazebo_env.py            Era 1, 7-dim observation
  gazebo_env_lidar.py      Era 2, 31-dim observation, peer obstacle, ground-truth pose
  warehouse_env.py         Python-layer prototype environment
  data_registry.py         Mutschler, SKU110K and YCB integration
  _history/                superseded environment variants, kept for audit
scripts/                   training, evaluation and figure scripts, as run
worlds/                    Gazebo SDF worlds (physics state: tools/world_physics.py check)
assets/                    dataset-derived maps, noise parameters, YCB models
models/                    trained policies; models/reference/ holds read-only copies
data/                      episode-level result records
results/                   chapter figures
analysis/                  reproduce_statistics.py
tools/                     simulator control, preflight check, world physics, collector
tests/                     self-tests, stubs and fixtures
docs/                      PIPELINE.md, KNOWN_ISSUES.md, PROVENANCE.md
environment/               installer and recorded environment
```

[docs/PIPELINE.md](docs/PIPELINE.md) maps each chapter result to the script and data that
produce it. [docs/PROVENANCE.md](docs/PROVENANCE.md) is generated from the code and lists
which script reads and writes each artifact.

---

## For the author: assembling this repository

The tooling in this repository was written to package the working research tree without
re-typing any code that produced results. From the repository root:

```bash
source ~/Praxis/venv/bin/activate && source /opt/ros/jazzy/setup.bash
tools/collect_source.sh ~/Praxis      # copy, make paths portable, verify, audit
./run_all.sh check                    # resolve any MISSING script it reports
./run_all.sh stats                    # every line should read OK
git init && git add -A && git commit -m "Praxis code and results"
```

The collector never modifies `~/Praxis`. Its only change to copied code is replacing
`os.path.expanduser('~/Praxis/...')` with a `PRAXIS_ROOT` lookup that resolves to the
identical path when `PRAXIS_ROOT` is unset, plus normalising an unused `v6` model filename
in `train_gazebo_ppo.py` back to `v4`. Review every `WARN` line it prints.

---

## Citation, attribution and licence

See [CITATION.cff](CITATION.cff). AI assistance in developing this code is disclosed in
[AI_ATTRIBUTION.md](AI_ATTRIBUTION.md), as required by GWU policy. Licensed under the
terms in [LICENSE](LICENSE).
