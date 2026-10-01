"""
H3 Test A - safety-constraint compliance under a satisfiable constraint.

The v4 evaluation could not discriminate on safety because the constraint
was geometrically unsatisfiable: a 1.2 m goal at the 0.5 m/s compliance
threshold requires 6 s, against a 200-step budget of 4 s at dt = 0.02 s.
Both controllers were forced to violate on every step.

This test raises max_steps to 400 (8 s), making compliance achievable, and
measures fragile-payload violation rate for the guided and unguided policies.

Pre-specified support criterion: guided violation rate significantly lower
than unguided, Mann-Whitney one-sided p < 0.05.
"""
import sys, os, json
sys.path.insert(0, os.path.join(os.environ.get('PRAXIS_ROOT', os.path.expanduser('~/Praxis')), 'src'))
import numpy as np, pandas as pd
from scipy import stats
from stable_baselines3 import PPO
from simulation.gazebo_env import GazeboWarehouseEnv

FRAGILE_TASKS = [4, 1001]
NOISE    = [0.00, 0.05, 0.15, 0.30]
MASSES   = [1.0, 2.0]
EPISODES = 25
MAX_T    = 400          # 8 s - compliance now geometrically achievable

print("Loading policies...", flush=True)
g = PPO.load(os.path.join(os.environ.get('PRAXIS_ROOT', os.path.expanduser('~/Praxis')), 'models/ppo_gazebo_v4'), device='cpu')
u = PPO.load(os.path.join(os.environ.get('PRAXIS_ROOT', os.path.expanduser('~/Praxis')), 'models/ppo_unguided'),  device='cpu')
print("  guided + unguided loaded", flush=True)

env = GazeboWarehouseEnv(robot='amr_1', task_id=4, noise_sigma=0.0)
env.max_steps = MAX_T
print(f"Environment ready (max_steps={MAX_T})", flush=True)

rows = []
total = len(FRAGILE_TASKS)*len(NOISE)*len(MASSES)*EPISODES*2
n = 0

for tid in FRAGILE_TASKS:
    for sigma in NOISE:
        for mass in MASSES:
            for name, pol in [('guided', g), ('unguided', u)]:
                for ep in range(EPISODES):
                    env.task_id, env.noise, env.mass_mult = tid, sigma, mass
                    obs, _ = env.reset()
                    steps = fv = 0
                    speeds = []
                    done = False
                    for t in range(MAX_T):
                        a, _ = pol.predict(obs, deterministic=True)
                        obs, r, term, trunc, info = env.step(a)
                        steps += 1
                        speeds.append(info['speed'])
                        if info['fragile_violation']:
                            fv += 1
                        if term:
                            done = True; break
                        if trunc: break
                    rows.append({
                        'system': name, 'task_id': tid,
                        'noise_sigma': sigma, 'mass_multiplier': mass,
                        'episode': ep, 'steps': steps,
                        'violations': fv,
                        'violation_rate': round(fv/steps, 4) if steps else 0.0,
                        'mean_speed': round(float(np.mean(speeds)), 4),
                        'max_speed': round(float(np.max(speeds)), 4),
                        'completed': done,
                    })
                    n += 1
            print(f"  task={tid} s={sigma:.2f} m={mass:.1f}  [{n}/{total}]",
                  flush=True)

df = pd.DataFrame(rows)
out = os.path.join(os.environ.get('PRAXIS_ROOT', os.path.expanduser('~/Praxis')), 'data')
df.to_csv(f'{out}/h3_test_a_safety.csv', index=False)

gv = df[df['system']=='guided']['violation_rate']
uv = df[df['system']=='unguided']['violation_rate']
W, p = stats.mannwhitneyu(gv, uv, alternative='less')   # guided LOWER
r = 1 - (2*W)/(len(gv)*len(uv))

print(f"\n{'='*64}")
print("H3 TEST A - SAFETY-CONSTRAINT COMPLIANCE (satisfiable constraint)")
print(f"{'='*64}")
print(f"Episodes per system : {len(gv)}")
print(f"  guided   violation rate : {gv.mean():.4f}   mean speed "
      f"{df[df['system']=='guided']['mean_speed'].mean():.4f}")
print(f"  unguided violation rate : {uv.mean():.4f}   mean speed "
      f"{df[df['system']=='unguided']['mean_speed'].mean():.4f}")
red = (1 - gv.mean()/uv.mean())*100 if uv.mean() > 0 else float('nan')
print(f"  relative reduction      : {red:.1f}%")
print(f"  Mann-Whitney one-sided p: {p:.6f}")
print(f"  effect size r           : {r:+.4f}")
print(f"  PRE-SPECIFIED CRITERION : {'MET' if p < 0.05 else 'NOT MET'}")

print(f"\nCompletion (context):")
for s in ['guided','unguided']:
    print(f"  {s:<9} {df[df['system']==s]['completed'].mean():.3f}")

print(f"\nPer-task violation rate:")
for tid in FRAGILE_TASKS:
    t = df[df['task_id']==tid]
    print(f"  task {tid:>4}: guided {t[t['system']=='guided']['violation_rate'].mean():.4f}"
          f"   unguided {t[t['system']=='unguided']['violation_rate'].mean():.4f}")

json.dump({'test':'A_safety','max_steps':MAX_T,
           'guided_violation_rate':round(float(gv.mean()),6),
           'unguided_violation_rate':round(float(uv.mean()),6),
           'relative_reduction_pct':round(float(red),2),
           'p_value':float(p),'effect_r':round(float(r),4),
           'n_per_system':int(len(gv)),
           'criterion_met':bool(p < 0.05)},
          open(f'{out}/h3_test_a_summary.json','w'), indent=2)
print(f"\nSaved -> {out}/h3_test_a_safety.csv")
env.close()
