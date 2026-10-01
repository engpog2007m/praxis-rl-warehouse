"""
H3 - scenario-stratified evaluation across operating-load tiers.

Following the scenario-based testbed methodology of Garcia et al. (2026),
who found reinforcement learning achieved parity with a heuristic baseline
overall but improved order fulfilment by up to 13.5 percentage points under
congested layouts and high workloads, this evaluation stratifies performance
by operating load rather than reporting a single aggregate.

Operating load is varied through the episode step budget, which controls how
much slack a controller has to reach its goal:

    low_load    max_steps = 200   (nominal - the v4 protocol)
    medium      max_steps = 140
    high        max_steps = 100
    congested   max_steps =  70   (near the feasibility boundary)

All four control systems are evaluated under each tier with an otherwise
identical protocol. Models and environment are unchanged; only the step
budget varies.

Pre-specified support criterion: PPO_guided exceeds PPO_unguided by at least
5 percentage points task completion at the two most constrained tiers
(max_steps 100 and 70), Mann-Whitney one-sided p < 0.05.
"""
import sys, os, math, json, pickle
sys.path.insert(0, os.path.join(os.environ.get('PRAXIS_ROOT', os.path.expanduser('~/Praxis')), 'src'))
import numpy as np, pandas as pd
from scipy import stats
from stable_baselines3 import PPO
from simulation.gazebo_env import GazeboWarehouseEnv

TIERS    = [('low_load', 200), ('medium', 140), ('high', 100), ('congested', 70)]
TASK_IDS = [1, 2, 3, 4, 1001]
NOISE    = [0.00, 0.05, 0.15, 0.30]
MASSES   = [1.0, 2.0]
EPISODES = 15

print("Loading models...", flush=True)
g = PPO.load(os.path.join(os.environ.get('PRAXIS_ROOT', os.path.expanduser('~/Praxis')), 'models/ppo_gazebo_v4'), device='cpu')
u = PPO.load(os.path.join(os.environ.get('PRAXIS_ROOT', os.path.expanduser('~/Praxis')), 'models/ppo_unguided'),  device='cpu')
with open(os.path.join(os.environ.get('PRAXIS_ROOT', os.path.expanduser('~/Praxis')), 'models/bc_policy.pkl'), 'rb') as f:
    bc = pickle.load(f)
print(f"  PPO_guided, PPO_unguided, BC ({bc['model_type']}) loaded", flush=True)

env = GazeboWarehouseEnv(robot='amr_1', task_id=1, noise_sigma=0.0)
print("Environment ready", flush=True)


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
total = len(TIERS)*len(TASK_IDS)*len(NOISE)*len(MASSES)*EPISODES*len(SYSTEMS)
n = 0

for tier_name, budget in TIERS:
    env.max_steps = budget
    print(f"\n=== TIER {tier_name} (max_steps={budget}) ===", flush=True)
    for tid in TASK_IDS:
        for sigma in NOISE:
            for mass in MASSES:
                for system in SYSTEMS:
                    for ep in range(EPISODES):
                        env.task_id, env.noise, env.mass_mult = tid, sigma, mass
                        obs, _ = env.reset()
                        R = 0.0; steps = 0; done = False
                        for t in range(budget):
                            if   system == 'PPO_guided':   a, _ = g.predict(obs, deterministic=True)
                            elif system == 'PPO_unguided': a, _ = u.predict(obs, deterministic=True)
                            elif system == 'BC_ml':         a = bc_act(obs)
                            else:                            a = dwb.act(obs)
                            obs, r, term, trunc, info = env.step(a)
                            R += r; steps += 1
                            if term: done = True; break
                            if trunc: break
                        rows.append({
                            'tier': tier_name, 'max_steps': budget,
                            'system': system, 'task_id': tid,
                            'noise_sigma': sigma, 'mass_multiplier': mass,
                            'episode': ep, 'reward': round(R, 4),
                            'cycle_time': steps,
                            'efficiency': round(R/steps, 4) if steps else 0.0,
                            'completed': done,
                        })
                        n += 1
        print(f"    task={tid} done  [{n}/{total}]", flush=True)

df = pd.DataFrame(rows)
out = os.path.join(os.environ.get('PRAXIS_ROOT', os.path.expanduser('~/Praxis')), 'data')
df.to_csv(f'{out}/h3_stress_tiers.csv', index=False)

print(f"\n{'='*74}")
print("H3 - SCENARIO-STRATIFIED EVALUATION BY OPERATING LOAD")
print(f"{'='*74}")
print(f"Total records: {len(df):,}\n")

print("Task completion by tier and system:")
piv = df.pivot_table(index='tier', columns='system', values='completed',
                     aggfunc='mean').reindex([t[0] for t in TIERS])
print((piv[SYSTEMS]*100).round(1).to_string())

print("\nGuided vs unguided advantage by tier (percentage points):")
summary = {}
for tier_name, budget in TIERS:
    sub = df[df['tier']==tier_name]
    gc = sub[sub['system']=='PPO_guided']['completed'].astype(float)
    uc = sub[sub['system']=='PPO_unguided']['completed'].astype(float)
    gap_pp = (gc.mean()-uc.mean())*100
    if gc.std() > 0 or uc.std() > 0:
        W, p = stats.mannwhitneyu(gc, uc, alternative='greater')
        r = 1 - (2*W)/(len(gc)*len(uc))
    else:
        p, r = 1.0, 0.0
    summary[tier_name] = {'gap_pp': round(float(gap_pp),2),
                          'p': float(p), 'r': round(float(r),4),
                          'guided': round(float(gc.mean()),4),
                          'unguided': round(float(uc.mean()),4)}
    print(f"  {tier_name:<10} (n={budget:>3}): {gap_pp:+6.2f} pp   "
          f"p={p:.4f}   r={r:+.3f}")

crit = all(summary[t]['gap_pp'] >= 5.0 and summary[t]['p'] < 0.05
           for t in ['high', 'congested'])
print(f"\n  PRE-SPECIFIED CRITERION (>=5pp and p<0.05 at 'high' and "
      f"'congested'): {'MET' if crit else 'NOT MET'}")

print("\nRL vs classical baseline by tier (PPO_guided - Nav2_DWB, pp):")
for tier_name, budget in TIERS:
    sub = df[df['tier']==tier_name]
    gc = sub[sub['system']=='PPO_guided']['completed'].mean()
    dc = sub[sub['system']=='Nav2_DWB']['completed'].mean()
    print(f"  {tier_name:<10}: {(gc-dc)*100:+6.2f} pp")

json.dump({'test':'H3_stress_tiers','tiers':summary,
           'criterion_met':bool(crit),
           'n_per_system_per_tier':int(len(TASK_IDS)*len(NOISE)*len(MASSES)*EPISODES)},
          open(f'{out}/h3_stress_summary.json','w'), indent=2)
print(f"\nSaved -> {out}/h3_stress_tiers.csv")
env.close()
