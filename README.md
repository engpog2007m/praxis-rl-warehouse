# Reinforcement Learning and Comparative Machine Learning for Adaptive Mobile Robot Control in E-commerce Warehouses

Code, trained models, and result data for the doctoral praxis of Patriot Gumbo,
The George Washington University, School of Engineering and Applied Science (2026).

The core experiment compares an unguided Proximal Policy Optimization (PPO) policy with
two baselines: a behavioural-cloning policy (traditional machine learning) trained on
demonstrations from a classical controller, and that classical controller itself, a DWB
reference controller. All three control an autonomous mobile robot (AMR) in a simulated
e-commerce warehouse built on Gazebo Harmonic and ROS 2 Jazzy. As additional
experimentation, a guided PPO variant with payload-adaptive reward shaping was evaluated
against unguided PPO (Deployment Roadmap hypothesis, H3; praxis Appendix B).

All results are from simulation. Nothing in this repository was run on a physical robot.

**About the baseline name.** The "DWB reference controller" is a Python controller modelled
on the Nav2 Dynamic Window Approach (DWB) local planner. It is not the full Nav2 navigation
stack. Some file and column names still use the label `Nav2_DWB`; they refer to the same
controller. See [docs/KNOWN_ISSUES.md](docs/KNOWN_ISSUES.md).

---

## Read this first: development stages

The work went through three stages. All are included; only the final stage supports the
hypothesis results.

| | Python layer | Stage 2 (Era 1, Gazebo) | Final stage (Era 2, Gazebo) |
|---|---|---|---|
| Simulator | kinematic, no physics | Gazebo Harmonic | Gazebo Harmonic, physics corrected |
| Observation | pose/goal/velocity | 7-dim pose/goal/velocity | 31-dim: 7 pose terms + 24 LiDAR sectors |
| Obstacles | none | none | dynamic peer robot on the ego route |
| Position source | simulated | odometry | Gazebo ground-truth pose |
| Robot physics | n/a | **defective** (see below) | corrected and verified |
| Used for | prototype, reward design | record of the defect | H1, H2, H3, RQ1-RQ3 |

**The Era 1 physics defect.** The original AMR models had no wheel friction and a chassis
collision cylinder resting on the ground plane. Under full commanded velocity the robot
moved about 0.5 mm in 3 s, while odometry, which integrates commanded velocity rather than
measuring displacement, reported normal travel. Era 1 comparisons between systems are
internally consistent but not physically validated, so no hypothesis verdict rests on them.
The correction is documented in [docs/KNOWN_ISSUES.md](docs/KNOWN_ISSUES.md), issue 1.
`tools/preflight_check.py` now refuses to start any run on a robot that does not physically move.

---

## Results (final stage, n = 800 episodes per system)

### Core experiment

| System | Completed | Collided | Timed out |
|---|---|---|---|
| PPO, unguided | **56.2%** | 37.4% | 6.4% |
| Behavioural cloning | 49.4% | 39.9% | 10.8% |
| DWB reference controller | 45.4% | 41.1% | 13.5% |

Throughput is completed tasks per 1,000 control steps, for a single controlled AMR
(controller level, not fleet level): unguided PPO 4.315, behavioural cloning 3.471,
DWB reference controller 3.226.

Hypotheses H1 and H2 were refined after advisor review; the praxis states the original and
revised wording. A hypothesis is supported when the difference is statistically significant
and the estimated improvement meets the target.

- **H1 (completion +10% vs traditional ML), supported.** Unguided PPO vs behavioural
  cloning: +13.9% relative completion, bootstrap 95% CI [4.5%, 24.7%], p = 0.006. The
  point estimate meets the target; the lower bound of the interval does not. Under sensor
  noise sigma >= 0.15, unguided PPO completed 53.25% of episodes against 40.25% for the
  DWB reference controller (+32.3%, 95% CI [15.0%, 55.7%], one-sided p = 0.0001).
- **H2 (throughput +10% vs controller-level baselines), supported at controller level.**
  Unguided PPO vs DWB reference controller: +33.7% [23.4%, 45.8%]. Vs behavioural cloning:
  +24.3% [15.5%, 34.3%]. Both intervals lie above 10%.
- **RQ1.** Unguided PPO completed significantly more tasks than both baselines, with no
  significant difference in collision rate (vs DWB p = 0.125; vs behavioural cloning
  p = 0.30). Behavioural cloning did not differ significantly from the controller it
  imitates (completion p = 0.109, collision p = 0.611).

### Additional experimentation: guided PPO (H3, praxis Appendix B)

| System | Completed | Collided | Timed out |
|---|---|---|---|
| PPO, guided | 29.5% | 18.2% | 52.2% |
| PPO, unguided | 56.2% | 37.4% | 6.4% |

- **H3, partially supported.** Across all episodes, guided PPO collided less (18.2% vs
  37.4%, p < 0.0001) but completed fewer tasks (29.5% vs 56.2%). The guided policy timed
  out on every episode in one of the two goal directions. In the direction it attempted,
  it did not differ from unguided PPO in collision (36.5% vs 35.5%, p = 0.77) or completion
  (59.0% vs 58.8%, p = 0.94). The aggregate collision reduction therefore reflects the
  direction failure, not safer driving; stratified evaluation, one of the roadmap checks,
  is what exposed this.

### Where each number comes from

| Output | Produced by |
|---|---|
| Outcome rates, noise, payload, mass, direction and pairwise tables (`results/tables/era2_*.csv`) | `python3 analysis/era2_analysis.py` |
| Unguided PPO vs behavioural cloning intervals (H1, H2) | `python3 analysis/bc_ci.py` |
| Statistics checked OK/DIFF against reported values | `./run_all.sh stats` |

### Figures

| Praxis figure | File | Script |
|---|---|---|
| 4.5 Training curves, unguided PPO | `results/figures/fig4_5_training_unguided.png` | `analysis/make_core_figures.py` |
| 4.6 Outcome rates, core systems | `results/figures/fig4_6_outcomes_core.png` | `analysis/make_core_figures.py` |
| 4.7 Completion by sensor noise | `results/figures/fig4_7_completion_noise_core.png` | `analysis/make_core_figures.py` |
| 4.8 Throughput | `results/figures/fig4_8_throughput_core.png` | `analysis/make_core_figures.py` |
| Appendix B, four-system versions and goal direction | `results/figures/fig4_5_training.png` to `fig4_9_direction.png` | `analysis/make_era2_figures.py` |

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
results/                   chapter figures (results/figures) and tables (results/tables)
analysis/                  reproduce_statistics.py, era2_analysis.py, bc_ci.py,
                           make_core_figures.py, make_era2_figures.py
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

## Use of AI tools

This repository was developed by Patriot Gumbo. An AI assistant (Anthropic's Claude) was
used during development for:

- drafting and debugging Python, Bash and ROS 2 / Gazebo code
- writing the statistical analysis and figure-generation scripts in `analysis/`
- setting up the reproducibility tooling (`run_all.sh`, self-tests, environment scripts)
- checking reference details (DOIs, volumes, pages)

All simulations, training runs and evaluations were run by the author on his own hardware,
and every reported number is produced by running the code in this repository on the
shipped data. The author reviewed, tested and is responsible for all code and results.
The praxis document itself is not part of this repository.

---

## Citation and licence

See [CITATION.cff](CITATION.cff). Licensed under the terms in [LICENSE](LICENSE).
