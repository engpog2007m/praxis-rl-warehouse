"""
H3 - Deployment Roadmap vs unguided RL integration.

The Deployment Roadmap (Chapter 3) specifies staged sim-to-real validation:
Stage 1 Python prototype validation, followed by Stage 2 full physics
deployment with dataset-grounded environment parameterisation.

  roadmap_followed  - PPO trained in the Gazebo physics environment with
                      Mutschler geometry, YCB payload masses and
                      SKU110K-calibrated noise (ppo_gazebo_v4).
  roadmap_skipped   - PPO trained only in the simplified Python kinematic
                      layer and deployed directly to physics
                      (ppo_warehouse), i.e. unguided RL integration.

Both are PPO, identical architecture, 500,000 training steps. The only
variable is whether the roadmap's physics-layer validation stage was
followed. Evaluated under one identical protocol.

Pre-specified support criterion: roadmap_followed achieves at least 5%
higher task completion, Mann-Whitney one-sided p < 0.05.
"""
import sys, os, json
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

print("Loading policies...", flush=True)
rf = PPO.load(os.path.join(os.environ.get('PRAXIS_ROOT', os.path.expanduser('~/Praxis')), 'models/ppo_gazebo_v4'), device='cpu')
rs = PPO.load(os.path.join(os.environ.get('PRAXIS_ROOT', os.path.expanduser('~/Praxis')), 'models/ppo_warehouse'), device='cpu')
print("  roadmap_followed (Gazebo-trained) loaded")
print("  roadmap_skipped  (Python-trained) loaded", flush=True)

env = GazeboWarehouseEnv(robot='amr_1', task_id=1, noise_sigma=0.0)
print("Environment ready", flush=True)

rows = []
total = len(TASK_IDS)*len(NOISE)*len(MASSES)*EPISODES*2
n = 0

for tid in TASK_IDS:
    for sigma in NOISE:
        for mass in MASSES:
            for name, pol in [('roadmap_followed', rf), ('roadmap_skipped', rs)]:
                for ep in range(EPISODES):
                    env.task_id, env.noise, env.mass_mult = tid, sigma, mass
                    obs, _ = env.reset()
                    R = 0.0; steps = 0; done = False
                    for t in range(MAX_T):
                        a, _ = pol.predict(obs, deterministic=True)
                        obs, r, term, trunc, info = env.step(a)
                        R += r; steps += 1
                        if term: done = True; break
                        if trunc: break
                    rows.append({
                        'system': name, 'task_id': tid,
                        'noise_sigma': sigma, 'mass_multiplier': mass,
                        'episode': ep, 'reward': round(R, 4),
                        'cycle_time': steps,
                        'efficiency': round(R/steps, 4) if steps else 0.0,
                        'throughput': 1.0 if done and steps < 100 else 0.0,
                        'completed': done,
                    })
                    n += 1
            print(f"  task={tid} s={sigma:.2f} m={mass:.1f}  [{n}/{total}]",
                  flush=True)

df = pd.DataFrame(rows)
out = os.path.join(os.environ.get('PRAXIS_ROOT', os.path.expanduser('~/Praxis')), 'data')
df.to_csv(f'{out}/h3_roadmap_results.csv', index=False)

fc = df[df['system']=='roadmap_followed']['completed'].astype(float)
sc = df[df['system']=='roadmap_skipped']['completed'].astype(float)
fr = df[df['system']=='roadmap_followed']['reward']
sr = df[df['system']=='roadmap_skipped']['reward']

Wc, pc = stats.mannwhitneyu(fc, sc, alternative='greater')
rc = 1 - (2*Wc)/(len(fc)*len(sc))
Wr, pr = stats.mannwhitneyu(fr, sr, alternative='greater')
rr = 1 - (2*Wr)/(len(fr)*len(sr))
adv = (fc.mean()-sc.mean())/sc.mean()*100 if sc.mean() > 0 else float('inf')

print(f"\n{'='*68}")
print("H3 - DEPLOYMENT ROADMAP vs UNGUIDED RL INTEGRATION")
print(f"{'='*68}")
print(f"Episodes per system: {len(fc)}\n")
print(f"  roadmap_followed  completion {fc.mean():.4f}   reward {fr.mean():9.2f}")
print(f"  roadmap_skipped   completion {sc.mean():.4f}   reward {sr.mean():9.2f}")
print(f"\n  relative completion advantage : {adv:+.1f}%")
print(f"  completion  p = {pc:.6f}   r = {rc:+.4f}")
print(f"  reward      p = {pr:.6f}   r = {rr:+.4f}")
print(f"  PRE-SPECIFIED CRITERION (>=5%, p<0.05): "
      f"{'MET' if (adv >= 5 and pc < 0.05) else 'NOT MET'}")

print(f"\nBy noise level (completion):")
for s in NOISE:
    sub = df[df['noise_sigma']==s]
    f_ = sub[sub['system']=='roadmap_followed']['completed'].mean()
    s_ = sub[sub['system']=='roadmap_skipped']['completed'].mean()
    print(f"  sigma={s:.2f}: followed {f_:.3f}   skipped {s_:.3f}")

print(f"\nBy task (completion):")
for tid in TASK_IDS:
    sub = df[df['task_id']==tid]
    f_ = sub[sub['system']=='roadmap_followed']['completed'].mean()
    s_ = sub[sub['system']=='roadmap_skipped']['completed'].mean()
    print(f"  task {tid:>4}: followed {f_:.3f}   skipped {s_:.3f}")

json.dump({'test':'H3_roadmap',
           'followed_completion':round(float(fc.mean()),6),
           'skipped_completion':round(float(sc.mean()),6),
           'relative_advantage_pct':round(float(adv),2),
           'completion_p':float(pc),'completion_r':round(float(rc),4),
           'reward_p':float(pr),'reward_r':round(float(rr),4),
           'n_per_system':int(len(fc)),
           'criterion_met':bool(adv >= 5 and pc < 0.05)},
          open(f'{out}/h3_roadmap_summary.json','w'), indent=2)
print(f"\nSaved -> {out}/h3_roadmap_results.csv")
env.close()
