import sys, os, time, math
sys.path.insert(0, os.path.join(os.environ.get('PRAXIS_ROOT', os.path.expanduser('~/Praxis')), 'src'))
import numpy as np, pandas as pd
from scipy import stats
from stable_baselines3 import PPO
from simulation.gazebo_env import GazeboWarehouseEnv

TASK_IDS  = [4, 1001]
NOISE     = [0.00, 0.05, 0.15, 0.30]
MASSES    = [1.0, 2.0]
EPISODES  = 25

print("Loading Gazebo-trained PPO...", flush=True)
ppo = PPO.load(os.path.join(os.environ.get('PRAXIS_ROOT', os.path.expanduser('~/Praxis')), 'models/ppo_gazebo_v4'), device='cpu')
print("Model loaded", flush=True)

env = GazeboWarehouseEnv(robot='amr_1', task_id=1, noise_sigma=0.0)
print("Environment ready", flush=True)

class DWB:
    APPROACH_RADIUS = 0.8
    MAX_SPEED       = 1.0
    MIN_SPEED       = 0.35
    def act(self, obs):
        rx, ry = obs[0], obs[1]
        gx, gy = obs[2], obs[3]
        dx, dy = gx - rx, gy - ry
        n = max(math.sqrt(dx*dx + dy*dy), 1e-6)
        if n > self.APPROACH_RADIUS:
            s = self.MAX_SPEED
        else:
            s = max(self.MIN_SPEED, self.MAX_SPEED * n / self.APPROACH_RADIUS)
        return np.array([s*dx/n, s*dy/n, 0.0], np.float32)

dwb = DWB()
records = []
total = len(TASK_IDS)*len(NOISE)*len(MASSES)*EPISODES*2
done_n = 0

for tid in TASK_IDS:
    for sigma in NOISE:
        for mass in MASSES:
            for system in ['PPO_gazebo', 'Nav2_DWB']:
                for ep in range(EPISODES):
                    env.task_id   = tid
                    env.noise     = sigma
                    env.mass_mult = mass
                    obs, _ = env.reset()
                    R = 0.0; viols = 0; steps = 0; done = False
                    for t in range(120):
                        if system == 'PPO_gazebo':
                            a, _ = ppo.predict(obs, deterministic=True)
                        else:
                            a = dwb.act(obs)
                        obs, r, term, trunc, info = env.step(a)
                        R += r; steps += 1
                        if r < -1.5: viols += 1
                        if term:
                            done = True; break
                        if trunc: break
                    records.append({
                        'system': system, 'task_id': tid,
                        'noise_sigma': sigma, 'mass_multiplier': mass,
                        'episode': ep, 'reward': round(R, 4),
                        'cycle_time': steps,
                        'throughput': 1.0 if done and steps < 100 else 0.0,
                        'efficiency': round(R/steps, 4) if steps else 0,
                        'violation_rate': round(viols/steps, 4) if steps else 0,
                        'completed': done})
                    done_n += 1
            print(f"  task={tid} s={sigma:.2f} m={mass:.1f} "
                  f"[{done_n}/{total}]", flush=True)

df = pd.DataFrame(records)
out = os.path.join(os.environ.get('PRAXIS_ROOT', os.path.expanduser('~/Praxis')), 'data')
df.to_csv(f'{out}/gazebo_trained_results_v5.csv', index=False)

ppo_df = df[df['system']=='PPO_gazebo']
dwb_df = df[df['system']=='Nav2_DWB']

print(f"\n{'='*58}")
print("GAZEBO-TRAINED POLICY vs NAV2 DWB (physics environment)")
print(f"{'='*58}")
print(f"Total records: {len(df):,}\n")

rows = []
for m in ['reward','efficiency','throughput','cycle_time','violation_rate']:
    pv, dv = ppo_df[m].values, dwb_df[m].values
    W, p = stats.mannwhitneyu(pv, dv, alternative='two-sided')
    r = 1 - (2*W)/(len(pv)*len(dv))
    mag = ('Large' if abs(r)>=.5 else 'Medium' if abs(r)>=.3
           else 'Small' if abs(r)>=.1 else 'Negligible')
    lower = m in ['cycle_time','violation_rate']
    sup = (np.median(pv)<np.median(dv)) if lower else (np.median(pv)>np.median(dv))
    rows.append({'Metric':m,
        'PPO_median':round(float(np.median(pv)),4),
        'DWB_median':round(float(np.median(dv)),4),
        'p_value':round(float(p),6),
        'Effect_r':round(float(r),4),
        'Effect_size':mag,
        'PPO_superior':'Yes' if sup else 'No'})
    print(f"  {m:16s} PPO={np.median(pv):9.4f}  DWB={np.median(dv):9.4f}  "
          f"p={p:.5f}  r={r:+.3f} ({mag})")

pd.DataFrame(rows).to_csv(f'{out}/gazebo_trained_stats_v5.csv', index=False)

print(f"\nPPO_gazebo:  throughput={ppo_df['throughput'].mean():.1%}  "
      f"completed={ppo_df['completed'].mean():.1%}  "
      f"efficiency={ppo_df['efficiency'].mean():.4f}")
print(f"Nav2_DWB:    throughput={dwb_df['throughput'].mean():.1%}  "
      f"completed={dwb_df['completed'].mean():.1%}  "
      f"efficiency={dwb_df['efficiency'].mean():.4f}")
env.close()
