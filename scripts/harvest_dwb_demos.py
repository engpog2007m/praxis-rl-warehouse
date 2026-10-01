"""
Harvest state-action demonstrations from the Nav2 DWB planner.

These pairs form the supervised training set for the behavioural cloning
baseline, which provides the 'traditional machine learning' comparator
required by RQ1. Unlike a Random Forest applied to episode-level features,
a behavioural cloning policy is a closed-loop controller: it maps the
current observation to a velocity command at every control step.
"""
import sys, os, math
sys.path.insert(0, os.path.join(os.environ.get('PRAXIS_ROOT', os.path.expanduser('~/Praxis')), 'src'))
import numpy as np, pandas as pd
from simulation.gazebo_env import GazeboWarehouseEnv

TASK_IDS = [1, 2, 3, 4, 1001]
NOISE    = [0.00, 0.05, 0.15, 0.30]
MASSES   = [1.0, 2.0]
EPISODES = 12          # per condition — yields ~40-60k state-action pairs
MAX_T    = 200


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


def main():
    env = GazeboWarehouseEnv(robot='amr_1', task_id=1, noise_sigma=0.0)
    dwb = DWB()
    rows = []
    total = len(TASK_IDS)*len(NOISE)*len(MASSES)*EPISODES
    n = 0

    for tid in TASK_IDS:
        for sigma in NOISE:
            for mass in MASSES:
                for ep in range(EPISODES):
                    env.task_id, env.noise, env.mass_mult = tid, sigma, mass
                    obs, _ = env.reset()
                    for t in range(MAX_T):
                        a = dwb.act(obs)
                        rows.append({
                            'o0': float(obs[0]), 'o1': float(obs[1]),
                            'o2': float(obs[2]), 'o3': float(obs[3]),
                            'o4': float(obs[4]), 'o5': float(obs[5]),
                            'o6': float(obs[6]),
                            'a0': float(a[0]), 'a1': float(a[1]),
                            'task_id': tid, 'noise_sigma': sigma,
                            'mass_multiplier': mass,
                        })
                        obs, r, term, trunc, info = env.step(a)
                        if term or trunc:
                            break
                    n += 1
                print(f"  task={tid} s={sigma:.2f} m={mass:.1f}  "
                      f"[{n}/{total} eps, {len(rows):,} pairs]", flush=True)

    df = pd.DataFrame(rows)
    out = os.path.join(os.environ.get('PRAXIS_ROOT', os.path.expanduser('~/Praxis')), 'data/dwb_demonstrations.csv')
    df.to_csv(out, index=False)
    print(f"\nHarvested {len(df):,} state-action pairs")
    print(f"Saved -> {out}")
    env.close()

main()
