"""
Four-way comparative evaluation — LiDAR branch.

PPO_guided    - payload-adaptive shaping + domain randomisation + peer avoidance
PPO_unguided  - progress/arrival/collision/anti-freeze only, no shaping
BC_ml         - supervised clone of DWB (R2=0.9983), inherits DWB's blindness to peer
Nav2_DWB      - classical geometric planner, no obstacle awareness

Protocol: 5 tasks x 4 noise levels x 2 mass multipliers x 20 episodes x 4
systems = 3,200 episodes. Peer obstacle active throughout.
"""
import sys, os, math, pickle
sys.path.insert(0, os.path.join(os.environ.get('PRAXIS_ROOT', os.path.expanduser('~/Praxis')), 'src'))
import numpy as np, pandas as pd
from scipy import stats
from stable_baselines3 import PPO
from simulation.gazebo_env_lidar import GazeboWarehouseEnv

TASK_IDS = [1, 2, 3, 4, 1001]
NOISE    = [0.00, 0.05, 0.15, 0.30]
MASSES   = [1.0, 2.0]
EPISODES = 20

print("Loading models...", flush=True)
g = PPO.load(os.path.join(os.environ.get('PRAXIS_ROOT', os.path.expanduser('~/Praxis')), 'models/ppo_lidar_guided'), device='cpu')
u = PPO.load(os.path.join(os.environ.get('PRAXIS_ROOT', os.path.expanduser('~/Praxis')), 'models/ppo_lidar_unguided'), device='cpu')
with open(os.path.join(os.environ.get('PRAXIS_ROOT', os.path.expanduser('~/Praxis')), 'models/bc_lidar_policy.pkl'), 'rb') as f:
    bc = pickle.load(f)
print(f"  guided, unguided, BC ({bc['model_type']}) loaded", flush=True)

env = GazeboWarehouseEnv(robot='amr_1', task_id=1, noise_sigma=0.0,
                         peer_enabled=True, collision_penalty=-20.0,
                         proximity_weight=2.0, max_steps=200)
print(f"obs {env.observation_space.shape}", flush=True)


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
SYSTEMS = ['PPO_guided', 'PPO_unguided', 'BC_ml', 'Nav2_DWB']
rows = []
total = len(TASK_IDS)*len(NOISE)*len(MASSES)*EPISODES*len(SYSTEMS)
n = 0

for tid in TASK_IDS:
    for sigma in NOISE:
        for mass in MASSES:
            for system in SYSTEMS:
                for ep in range(EPISODES):
                    env.task_id, env.noise, env.mass_mult = tid, sigma, mass
                    obs, _ = env.reset()
                    R = 0.0; steps = 0; done = False
                    for t in range(env.max_steps):
                        if   system == 'PPO_guided':   a, _ = g.predict(obs, deterministic=True)
                        elif system == 'PPO_unguided': a, _ = u.predict(obs, deterministic=True)
                        elif system == 'BC_ml':          a = bc_act(obs)
                        else:                             a = dwb.act(obs)
                        obs, r, term, trunc, info = env.step(a)
                        R += r; steps += 1
                        if term: done = True; break
                        if trunc: break
                    rows.append({
                        'system': system, 'task_id': tid,
                        'noise_sigma': sigma, 'mass_multiplier': mass,
                        'episode': ep, 'reward': round(R, 4),
                        'cycle_time': steps,
                        'completed': bool(done and info['arrived']),
                        'collided': bool(info['collided']),
                        'timed_out': bool(not done),
                    })
                    n += 1
            print(f"  task={tid} s={sigma:.2f} m={mass:.1f}  [{n}/{total}]",
                  flush=True)

df = pd.DataFrame(rows)
out = os.path.join(os.environ.get('PRAXIS_ROOT', os.path.expanduser('~/Praxis')), 'data')
df.to_csv(f'{out}/four_way_lidar_results.csv', index=False)

print(f"\n{'='*72}")
print("FOUR-WAY COMPARISON — LiDAR branch, peer obstacle active")
print(f"{'='*72}")
print(f"Total records: {len(df):,}\n")

summ = df.groupby('system')[['reward','completed','collided','timed_out']].mean()
print(summ.reindex(SYSTEMS).round(4).to_string())

print(f"\nCompletion vs collision, pairwise vs Nav2_DWB:")
dc = df[df['system']=='Nav2_DWB']['completed'].astype(float)
dcol = df[df['system']=='Nav2_DWB']['collided'].astype(float)
for s in SYSTEMS:
    if s == 'Nav2_DWB': continue
    sc = df[df['system']==s]['completed'].astype(float)
    scol = df[df['system']==s]['collided'].astype(float)
    Wc, pc = stats.mannwhitneyu(sc, dc, alternative='two-sided')
    Wcol, pcol = stats.mannwhitneyu(scol, dcol, alternative='two-sided')
    print(f"  {s:<14} completion {sc.mean():.3f} vs {dc.mean():.3f}  p={pc:.4f}   "
          f"collision {scol.mean():.3f} vs {dcol.mean():.3f}  p={pcol:.4f}")

print(f"\nGuided vs unguided (H3), completion and collision:")
gc = df[df['system']=='PPO_guided']['completed'].astype(float)
uc = df[df['system']=='PPO_unguided']['completed'].astype(float)
gcol = df[df['system']=='PPO_guided']['collided'].astype(float)
ucol = df[df['system']=='PPO_unguided']['collided'].astype(float)
Wc, pc = stats.mannwhitneyu(gc, uc, alternative='two-sided')
Wcol, pcol = stats.mannwhitneyu(gcol, ucol, alternative='less')
rc = 1 - (2*Wcol)/(len(gcol)*len(ucol))
print(f"  completion: guided {gc.mean():.3f}  unguided {uc.mean():.3f}  p={pc:.4f}")
print(f"  collision:  guided {gcol.mean():.3f}  unguided {ucol.mean():.3f}  "
      f"p={pcol:.4f}  r={rc:+.3f}  (one-sided: guided < unguided)")

print(f"\nSaved -> {out}/four_way_lidar_results.csv")
env.close()
