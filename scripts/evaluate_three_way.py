"""
Three-way comparative evaluation in the Gazebo physics environment:

  PPO          — deep reinforcement learning
  BC           — behavioural cloning (traditional supervised ML)
  Nav2 DWB     — classical geometric planner

Directly addresses RQ1, which compares reinforcement learning against
traditional machine learning methods. Environment, protocol, and metrics
are identical to the v4 evaluation.
"""
import sys, os, math, pickle
sys.path.insert(0, os.path.join(os.environ.get('PRAXIS_ROOT', os.path.expanduser('~/Praxis')), 'src'))
import numpy as np, pandas as pd
from scipy import stats
from stable_baselines3 import PPO
from simulation.gazebo_env import GazeboWarehouseEnv

TASK_IDS = [1, 2, 3, 4, 1001]
NOISE    = [0.00, 0.05, 0.15, 0.30]
MASSES   = [1.0, 2.0]
EPISODES = 25
MAX_T    = 200
FRAGILE  = {4, 1001}

print("Loading models...", flush=True)
ppo = PPO.load(os.path.join(os.environ.get('PRAXIS_ROOT', os.path.expanduser('~/Praxis')), 'models/ppo_gazebo_v4'), device='cpu')
with open(os.path.join(os.environ.get('PRAXIS_ROOT', os.path.expanduser('~/Praxis')), 'models/bc_policy.pkl'), 'rb') as f:
    bc = pickle.load(f)
print(f"  PPO v4 loaded")
print(f"  BC loaded ({bc['model_type']}, test R2 "
      f"{bc['metrics'][bc['model_type']]['r2']:.4f})", flush=True)

env = GazeboWarehouseEnv(robot='amr_1', task_id=1, noise_sigma=0.0)


class DWB:
    APPROACH_RADIUS = 0.8; MAX_SPEED = 1.0; MIN_SPEED = 0.35
    def act(self, obs):
        rx, ry, gx, gy = obs[0], obs[1], obs[2], obs[3]
        dx, dy = gx - rx, gy - ry
        n = max(math.sqrt(dx*dx + dy*dy), 1e-6)
        s = (self.MAX_SPEED if n > self.APPROACH_RADIUS
             else max(self.MIN_SPEED, self.MAX_SPEED * n / self.APPROACH_RADIUS))
        return np.array([s*dx/n, s*dy/n, 0.0], np.float32)


def bc_act(obs):
    x = np.asarray(obs, np.float32).reshape(1, -1)
    if bc['scaler'] is not None:
        x = bc['scaler'].transform(x)
    a = bc['model'].predict(x)[0]
    return np.array([float(a[0]), float(a[1]), 0.0], np.float32)


dwb = DWB()
SYSTEMS = ['PPO_rl', 'BC_ml', 'Nav2_DWB']
records = []
total = len(TASK_IDS)*len(NOISE)*len(MASSES)*EPISODES*len(SYSTEMS)
n = 0

for tid in TASK_IDS:
    for sigma in NOISE:
        for mass in MASSES:
            for system in SYSTEMS:
                for ep in range(EPISODES):
                    env.task_id, env.noise, env.mass_mult = tid, sigma, mass
                    obs, _ = env.reset()
                    R = 0.0; steps = 0; fv = 0; kv = 0
                    speeds = []; done = False
                    for t in range(MAX_T):
                        if   system == 'PPO_rl':  a, _ = ppo.predict(obs, deterministic=True)
                        elif system == 'BC_ml':   a = bc_act(obs)
                        else:                     a = dwb.act(obs)
                        obs, r, term, trunc, info = env.step(a)
                        R += r; steps += 1
                        speeds.append(info['speed'])
                        if info['fragile_violation']: fv += 1
                        if info['kinetic_violation']: kv += 1
                        if term: done = True; break
                        if trunc: break

                    records.append({
                        'system': system, 'task_id': tid,
                        'noise_sigma': sigma, 'mass_multiplier': mass,
                        'episode': ep, 'reward': round(R, 4),
                        'cycle_time': steps,
                        'throughput': 1.0 if done and steps < 100 else 0.0,
                        'efficiency': round(R/steps, 4) if steps else 0.0,
                        'completed': done,
                        'is_fragile_task': tid in FRAGILE,
                        'fragile_violation_rate': round(fv/steps, 4) if steps else 0.0,
                        'kinetic_violation_rate': round(kv/steps, 4) if steps else 0.0,
                        'mean_speed': round(float(np.mean(speeds)), 4) if speeds else 0.0,
                    })
                    n += 1
            print(f"  task={tid} s={sigma:.2f} m={mass:.1f}  [{n}/{total}]", flush=True)

df = pd.DataFrame(records)
out = os.path.join(os.environ.get('PRAXIS_ROOT', os.path.expanduser('~/Praxis')), 'data')
df.to_csv(f'{out}/three_way_results.csv', index=False)

print(f"\n{'='*70}")
print("THREE-WAY COMPARISON — RL vs Traditional ML vs Classical Planner")
print(f"{'='*70}")
print(f"Total records: {len(df):,}\n")

summ = df.groupby('system')[['reward','efficiency','throughput',
                             'cycle_time','completed']].mean().round(4)
print(summ.to_string())

print(f"\nSafety-constraint violation rate (fragile payloads only):")
frag = df[df['is_fragile_task']]
for s in SYSTEMS:
    v = frag[frag['system']==s]['fragile_violation_rate'].mean()
    print(f"  {s:<12} {v:.4f}")

print(f"\nPairwise Wilcoxon (reward):")
for i in range(len(SYSTEMS)):
    for j in range(i+1, len(SYSTEMS)):
        a = df[df['system']==SYSTEMS[i]]['reward']
        b = df[df['system']==SYSTEMS[j]]['reward']
        W, p = stats.mannwhitneyu(a, b, alternative='two-sided')
        r = 1 - (2*W)/(len(a)*len(b))
        print(f"  {SYSTEMS[i]:<10} vs {SYSTEMS[j]:<10} "
              f"p={p:.6f}  r={r:+.3f}  medians {np.median(a):.2f} / {np.median(b):.2f}")

print(f"\nSaved -> {out}/three_way_results.csv")
env.close()
