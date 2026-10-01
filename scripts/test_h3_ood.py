"""
H3 Test B - out-of-distribution generalisation.

Domain randomisation is intended to produce robustness to conditions not
encountered during training. The guided policy trained with noise sampled
across sigma in {0.00, 0.05, 0.15, 0.30}; the unguided policy trained at
sigma = 0 only.

Evaluating at those same four levels does not test generalisation, since
the guided policy saw them all. This test evaluates both policies at
sigma = 0.50 and 0.60, which neither policy encountered in training.

Pre-specified support criterion: guided completion rate significantly
higher than unguided, Mann-Whitney one-sided p < 0.05.
"""
import sys, os, json
sys.path.insert(0, os.path.join(os.environ.get('PRAXIS_ROOT', os.path.expanduser('~/Praxis')), 'src'))
import numpy as np, pandas as pd
from scipy import stats
from stable_baselines3 import PPO
from simulation.gazebo_env import GazeboWarehouseEnv

TASK_IDS = [1, 2, 3, 4, 1001]
NOISE_OOD = [0.50, 0.60]      # unseen by both policies
MASSES   = [1.0, 2.0]
EPISODES = 25
MAX_T    = 200                # v4 protocol

print("Loading policies...", flush=True)
g = PPO.load(os.path.join(os.environ.get('PRAXIS_ROOT', os.path.expanduser('~/Praxis')), 'models/ppo_gazebo_v4'), device='cpu')
u = PPO.load(os.path.join(os.environ.get('PRAXIS_ROOT', os.path.expanduser('~/Praxis')), 'models/ppo_unguided'),  device='cpu')
print("  guided + unguided loaded", flush=True)

env = GazeboWarehouseEnv(robot='amr_1', task_id=1, noise_sigma=0.0)
env.max_steps = MAX_T
print("Environment ready", flush=True)

rows = []
total = len(TASK_IDS)*len(NOISE_OOD)*len(MASSES)*EPISODES*2
n = 0

for tid in TASK_IDS:
    for sigma in NOISE_OOD:
        for mass in MASSES:
            for name, pol in [('guided', g), ('unguided', u)]:
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
                        'completed': done,
                    })
                    n += 1
            print(f"  task={tid} s={sigma:.2f} m={mass:.1f}  [{n}/{total}]",
                  flush=True)

df = pd.DataFrame(rows)
out = os.path.join(os.environ.get('PRAXIS_ROOT', os.path.expanduser('~/Praxis')), 'data')
df.to_csv(f'{out}/h3_test_b_ood.csv', index=False)

gc = df[df['system']=='guided']['completed'].astype(float)
uc = df[df['system']=='unguided']['completed'].astype(float)
W, p = stats.mannwhitneyu(gc, uc, alternative='greater')
r = 1 - (2*W)/(len(gc)*len(uc))

print(f"\n{'='*64}")
print("H3 TEST B - OUT-OF-DISTRIBUTION GENERALISATION (sigma 0.50, 0.60)")
print(f"{'='*64}")
print(f"Episodes per system : {len(gc)}")
print(f"  guided   completion : {gc.mean():.4f}")
print(f"  unguided completion : {uc.mean():.4f}")
adv = (gc.mean()-uc.mean())/uc.mean()*100 if uc.mean() > 0 else float('nan')
print(f"  relative advantage  : {adv:+.1f}%")
print(f"  Mann-Whitney one-sided p: {p:.6f}")
print(f"  effect size r           : {r:+.4f}")
print(f"  PRE-SPECIFIED CRITERION : {'MET' if p < 0.05 else 'NOT MET'}")

print(f"\nBy noise level:")
for s in NOISE_OOD:
    sub = df[df['noise_sigma']==s]
    gg = sub[sub['system']=='guided']['completed'].mean()
    uu = sub[sub['system']=='unguided']['completed'].mean()
    print(f"  sigma={s:.2f}: guided {gg:.3f}   unguided {uu:.3f}")

print(f"\nReward (context):")
for s in ['guided','unguided']:
    print(f"  {s:<9} {df[df['system']==s]['reward'].mean():8.2f}")

json.dump({'test':'B_ood','noise_levels':NOISE_OOD,
           'guided_completion':round(float(gc.mean()),6),
           'unguided_completion':round(float(uc.mean()),6),
           'relative_advantage_pct':round(float(adv),2),
           'p_value':float(p),'effect_r':round(float(r),4),
           'n_per_system':int(len(gc)),
           'criterion_met':bool(p < 0.05)},
          open(f'{out}/h3_test_b_summary.json','w'), indent=2)
print(f"\nSaved -> {out}/h3_test_b_ood.csv")
env.close()
