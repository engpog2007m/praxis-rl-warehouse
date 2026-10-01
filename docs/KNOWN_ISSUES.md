# Known Issues

Each issue below was encountered during the research. Most produced output that looked
plausible while being invalid, which is why several are now enforced automatically.

## 1. Era 1 robot physics defect

**Symptom.** Commanded forward velocity of 0.4 m/s moved the robot about 0.5 mm in 3 s,
measured from Gazebo ground truth. Odometry reported normal travel throughout.

**Cause.** Two compounding faults in the AMR model: the wheel collision geometry had no
friction coefficient, so wheel torque produced no traction, and the chassis collision
cylinder rested on the ground plane, adding drag.

**Correction.** Wheel friction `mu = mu2 = 1.2`, wheel mass 0.05 kg with consistent
inertia, and the chassis collision raised (`pose z = 0.05`, length 0.09 m) with low
friction. After correction the robot covered 1.75 m in 4 s at full command.

**Consequence.** Era 1 results (H1, H2, the no-obstacle four-way comparison and the seven
Era 1 H3 tests) were produced on the defective model. Every system faced the same defect,
so between-system comparisons are internally consistent, but the results are not
physically validated. Era 2 results were produced after the correction.

**Tools.**
- `tools/world_physics.py check <world.sdf>` classifies a world as CORRECTED, LEGACY or
  UNKNOWN.
- `tools/world_physics.py legacy <world.sdf>` regenerates the defect world for exact
  Era 1 replication; `./run_all.sh era1 --legacy-physics` uses it automatically.
- `tools/preflight_check.py` drives the robot and measures displacement from ground
  truth. Any run on a non-moving robot is refused.

## 2. Odometry is not a position measurement

The Gazebo differential-drive plugin integrates commanded velocity. It does not observe
physical displacement and it does not reset when a model is teleported. During diagnosis
it reported 366 m of travel for a stationary robot.

Era 2 therefore reads all geometry (goal distance, peer separation, collision) from
`/world/praxis_warehouse/dynamic_pose/info`. The ROS bridge drops entity names from this
message, so models are identified by index in SDF declaration order: `amr_1` is index 0,
`amr_2` index 1. Reordering models in the world file breaks this silently.

Era 1 used odometry with a per-episode origin reset, which is self-consistent within an
episode only when the robot actually moves (see issue 1).

## 3. Duplicate Gazebo instances

A wedged `gz sim` process ignored SIGTERM, and a second instance was started beside it.
Both served the same topics. `tools/launch_sim.sh` refuses to start while any instance
runs; `tools/stop_sim.sh` escalates to SIGKILL and verifies nothing remains; the
preflight check fails unless exactly one instance exists.

## 4. Service layer wedges while topics keep flowing

After long sessions, `gz model --list` and `gz model -p` returned nothing while scan,
odometry and pose topics continued at full rate. Training was unaffected because it uses
topics, not services. Terminal output of the form
`NodeShared::RecvSrvRequest() error sending response: Host unreachable` accompanies this
and is cosmetic. Judge simulator health by topic rates (the preflight check does this),
not by `gz model`. `run_all.sh` restarts Gazebo before every simulator stage.

## 5. Bridge argument typos fail quietly

A trailing `~` on the pose bridge argument produced one warning line among many and left
the pose topic without data. `tools/launch_sim.sh` counts created bridges against the
argument list and fails on any `Failed to create` line.

## 6. Exit-time messages

`terminate called without an active exception` followed by `Aborted (core dumped)` at the
end of a script, and warnings about duplicate node names when several environments are
created in one process, come from ROS 2 shutdown ordering. They occur after results are
written and do not affect them.

## 7. Headless GPU LiDAR

`gpu_lidar` requires the `gz-sim-sensors-system` plugin with `render_engine` ogre2. It was
verified to publish headless (`gz sim -s`) on an NVIDIA GPU. All results used GPU LiDAR;
CPU LiDAR alternatives were not tested. The preflight check fails if scans fall below
15 Hz.

## 8. PPO standstill local optimum

With a -100 collision penalty and a 500-step budget, early training converged to standing
still: the accumulated step cost was cheaper than the expected collision cost. Only 6
terminal outcomes occurred in 100,000 steps. The fix combined a shorter horizon
(`max_steps = 200`), a -20 collision penalty, a standstill penalty on measured (not
commanded) speed, and a small reward on measured speed. `train-lidar` gates full training
on a 50,000-step probe whose collision rate must fall by at least 0.15 between the first
and last third of training.

## 9. Peer placement sensitivity

The peer is teleported to the ego route midpoint with a uniform lateral offset. A
straight-line policy collided on 10 of 10 trials at +/-0.25 m (avoidance impossible) and 8 of
10 at +/-0.75 m, which was used. The failure mode is sensitive to this parameter.

## 10. Rewards are not comparable across the guided and unguided arms

The two arms optimise different reward functions. Compare them on completion, collision
and timeout rates, never on raw reward.

## 11. Exact versus statistical replication

`./run_all.sh stats` reproduces reported values exactly from the shipped records.
Re-running evaluation or training reproduces them statistically only: PPO training was
not seeded, Gazebo steps in real time so control timing jitters, and episode resets draw
unseeded random offsets. Expect the same direction and significance, not identical rates.

## 12. Files excluded from the repository

Data files over 25 MB, notably the DWB demonstration sets, are excluded and regenerated
by the pipeline. The fleet-level discrete-event simulator did not converge and is not
included as a result-producing component.
