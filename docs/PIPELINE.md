# Pipeline

Maps each reported result to the artifact that holds it and the stage that produces it.
Stages marked with a `run_all.sh` command are automated and verified by
`./run_all.sh check`. For every other artifact, `docs/PROVENANCE.md` (generated from the
code by the collector) names the script that writes it.

## Era 2: LiDAR, dynamic peer obstacle, corrected physics

| Order | Step | Script | Output | Time |
|---|---|---|---|---|
| 1 | Avoidance probe, 50K steps, gate | `probe_lidar_avoidance.py` | `data/lidar_avoid_probe.csv` | 30 min |
| 2 | Guided PPO, 500K steps | `train_lidar_guided.py` | `models/ppo_lidar_guided.zip` | 6.5 h |
| 3 | Unguided PPO, 500K steps | `train_lidar_unguided.py` | `models/ppo_lidar_unguided.zip` | 6.5 h |
| 4 | Nav2 DWB demonstrations | `harvest_dwb_lidar.py` | `data/dwb_lidar_demonstrations.csv` | 30 min |
| 5 | Behavioural cloning | `train_bc_lidar.py` | `models/bc_lidar_policy.pkl` | 5 min |
| 6 | Four-way evaluation | `evaluate_four_way_lidar.py` | `data/four_way_lidar_results.csv` | 2-3 h |

Automated: `./run_all.sh train-lidar` (all six) or `./run_all.sh eval-lidar` (step 6 with
the shipped models). Supports H3 and the final RQ1 answer.

Configuration used by every step: 31-dim observation, `max_steps = 200`,
`collision_penalty = -20`, `collision_dist = 0.30 m`, uniform 2.0 m goals, arrival radius
0.5 m, fragile threshold 0.5, heavy threshold 0.7, peer lateral offset +/-0.75 m,
anti-freeze penalty -1.0 below 0.05 m/s measured speed, velocity reward 0.2 x measured speed.
The guided arm adds payload-adaptive shaping, proximity shaping (weight 2.0) and per-block
noise randomisation; the unguided arm removes all three.

`probe_lidar_100k.py` is retained for audit only: it ran before the peer was placed on the
ego route, so it tested goal-reaching, not avoidance.

## Era 1: 7-dim observation, no obstacle, pre-correction physics

| Step | Script | Output |
|---|---|---|
| Guided PPO training | `train_gazebo_ppo.py` | `models/ppo_gazebo_v4.zip` |
| PPO vs Nav2 DWB, 2,000 episodes | `evaluate_gazebo_trained.py` | `data/gazebo_trained_results_v4.csv` |
| DWB demonstrations | `harvest_dwb_demos.py` | `data/dwb_demonstrations.csv` |
| Behavioural cloning | `train_behavioural_cloning.py` | `models/bc_policy.pkl` |
| Unguided PPO training | `train_unguided_ppo.py` | `models/ppo_unguided.zip` |
| Four-way, 4,000 episodes | `evaluate_four_way.py` | `data/four_way_results.csv` |

Automated: `./run_all.sh era1` on corrected physics (different numbers by design), or
`./run_all.sh era1 --legacy-physics` for exact-number replication on the defect world.
`gazebo_trained_results_v4.csv` supports H1 and the Era 1 physics-layer comparison.

### Era 1 H3 tests (appendix)

Seven tests found no practically meaningful guided vs unguided difference in an
environment without a collision failure mode. Their records, recomputed by
`./run_all.sh stats`:

| Test | Output |
|---|---|
| Safety constraint under extended budget | `data/h3_test_a_safety.csv` |
| Out-of-distribution noise | `data/h3_test_b_ood.csv` |
| Roadmap followed vs skipped | `data/h3_roadmap_results.csv` |
| Operating-load stratification | `data/h3_stress_tiers.csv` (`test_h3_stress_tiers.py`) |

The remaining tests are reward, completion and noise-stratified comparisons drawn from
`data/four_way_results.csv`.

## Data preparation and Python-layer prototype

Dataset integration (Mutschler warehouse map, SKU110K density, YCB payload registry) and
the Python-layer prototype predate the Gazebo work. Their scripts are collected into
`scripts/` unchanged; consult `docs/PROVENANCE.md` for the script that writes each output
and the script's own header for its arguments. These stages are not automated by
`run_all.sh`.

## Statistics and figures

- `./run_all.sh stats` recomputes the Era 2, H1, Era 1 four-way and appendix statistics.
- `./run_all.sh figures` regenerates figures from the CSVs using the figure scripts in
  `scripts/`.
