"""
Harvest Nav2 DWB demonstrations in the LiDAR environment.

Supervised training set for the behavioural cloning comparator (RQ1).
Observations are 31-dimensional (7 pose + 24 LiDAR sectors) and the peer
obstacle is active, so the demonstrations include DWB's behaviour when a
moving obstacle is present.

Note: Nav2 DWB has no obstacle awareness in this configuration - it steers
geometrically toward the goal and does not react to the peer. The cloned
policy therefore inherits that limitation, which is itself informative for
RQ1: a supervised controller cannot exceed its demonstrator.
"""
import sys, os, math
sys.path.insert(0, os.path.join(os.environ.get('PRAXIS_ROOT', os.path.expanduser('~/Praxis')), 'src'))
import numpy as np, pandas as pd
from simulation.gazebo_env_lidar import GazeboWarehouseEnv

TASK_IDS = [1, 2, 3, 4, 1001]
NOISE    = [0.00, 0.05, 0.15, 0.30]
MASSES   = [1.0, 2.0]
EPISODES = 12
N_OBS    = 31


class DWB:
    APPROACH_RADIUS = 0.8
    MAX_SPEED       = 1.0
    MIN_SPEED       = 0.35

    def act(self, obs):
        rx, ry, gx, gy = obs[0], obs[1], obs[2], obs[3]
        dx, dy = gx - rx, gy - ry
        n = max(math.sqrt(dx*dx + dy*dy), 1e-6)
        s = (self.MAX_SPEED if n > self.APPROACH_RADIUS
             else max(self.MIN_SPEED, self.MAX_SPEED * n / self.APPROACH_RADIUS))
        return np.array([s*dx/n, s*dy/n, 0.0], np.float32)


print("Building LiDAR environment (peer active)...", flush=True)
env = GazeboWarehouseEnv(robot='amr_1', task_id=1, noise_sigma=0.0,
                         peer_enabled=True, collision_penalty=-20.0,
                         proximity_weight=2.0, max_steps=200)
print(f"obs {env.observation_space.shape}", flush=True)

dwb = DWB()
rows = []
total = len(TASK_IDS)*len(NOISE)*len(MASSES)*EPISODES
n = 0
n_collisions = 0

for tid in TASK_IDS:
    for sigma in NOISE:
        for mass in MASSES:
            for ep in range(EPISODES):
                env.task_id, env.noise, env.mass_mult = tid, sigma, mass
                obs, _ = env.reset()
                for t in range(env.max_steps):
                    a = dwb.act(obs)
                    row = {f'o{i}': float(obs[i]) for i in range(N_OBS)}
                    row.update({'a0': float(a[0]), 'a1': float(a[1]),
                                'task_id': tid, 'noise_sigma': sigma,
                                'mass_multiplier': mass})
                    rows.append(row)
                    obs, r, term, trunc, info = env.step(a)
                    if info.get('collided'):
                        n_collisions += 1
                    if term or trunc:
                        break
                n += 1
            print(f"  task={tid} s={sigma:.2f} m={mass:.1f}  "
                  f"[{n}/{total} eps, {len(rows):,} pairs, "
                  f"{n_collisions} collisions so far]", flush=True)

df = pd.DataFrame(rows)
out = os.path.join(os.environ.get('PRAXIS_ROOT', os.path.expanduser('~/Praxis')), 'data/dwb_lidar_demonstrations.csv')
df.to_csv(out, index=False)
print(f"\nHarvested {len(df):,} state-action pairs across {n} episodes")
print(f"DWB collided with peer on {n_collisions} steps total "
      f"(expected - DWB has no obstacle awareness)")
print(f"Saved -> {out}")
env.close()
