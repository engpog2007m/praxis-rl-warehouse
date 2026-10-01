"""
H1 evaluation — direct safety-constraint violation rate.

Counts, per control step, whether the transported payload's handling
velocity constraint was exceeded. This replaces the earlier episode-level
reward proxy, which conflated kinetic cost with safety cost and lacked
the sensitivity to discriminate.

Model and environment are unchanged from v4; only the metric differs.
"""
import sys, os, math, json
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

FRAGILE = {4, 1001}          # payload profiles carrying the safety constraint

print("Loading v4 policy (unchanged)...", flush=True)
ppo = PPO.load(os.path.join(os.environ.get('PRAXIS_ROOT', os.path.expanduser('~/Praxis')), 'models/ppo_gazebo_v4'), device='cpu')
print("Loaded", flush=True)

env = GazeboWarehouseEnv(robot='amr_1', task_id=1, noise_sigma=0.0)
print("Environment ready", flush=True)


class DWB:
    """Nav2 DWB-style planner — payload-agnostic by construction."""
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


dwb = DWB()
records = []
total = len(TASK_IDS)*len(NOISE)*len(MASSES)*EPISODES*2
done_n = 0

for tid in TASK_IDS:
    for sigma in NOISE:
        for mass in MASSES:
            for system in ['PPO_gazebo', 'Nav2_DWB']:
                for ep in range(EPISODES):
                    env.task_id, env.noise, env.mass_mult = tid, sigma, mass
                    obs, _ = env.reset()

                    steps = 0
                    frag_viol = 0
                    kin_viol  = 0
                    speeds    = []
                    done      = False

                    for t in range(MAX_T):
                        if system == 'PPO_gazebo':
                            a, _ = ppo.predict(obs, deterministic=True)
                        else:
                            a = dwb.act(obs)

                        obs, r, term, trunc, info = env.step(a)
                        steps += 1
                        speeds.append(info['speed'])
                        if info['fragile_violation']: frag_viol += 1
                        if info['kinetic_violation']: kin_viol  += 1
                        if term:
                            done = True; break
                        if trunc: break

                    records.append({
                        'system': system, 'task_id': tid,
                        'noise_sigma': sigma, 'mass_multiplier': mass,
                        'episode': ep, 'steps': steps,
                        'is_fragile_task': tid in FRAGILE,
                        'fragile_violations': frag_viol,
                        'kinetic_violations': kin_viol,
                        'fragile_violation_rate': round(frag_viol/steps, 4) if steps else 0.0,
                        'kinetic_violation_rate': round(kin_viol/steps, 4) if steps else 0.0,
                        'mean_speed': round(float(np.mean(speeds)), 4) if speeds else 0.0,
                        'max_speed': round(float(np.max(speeds)), 4) if speeds else 0.0,
                        'completed': done,
                    })
                    done_n += 1
            print(f"  task={tid} s={sigma:.2f} m={mass:.1f}  [{done_n}/{total}]", flush=True)

df = pd.DataFrame(records)
out = os.path.join(os.environ.get('PRAXIS_ROOT', os.path.expanduser('~/Praxis')), 'data')
df.to_csv(f'{out}/h1_violation_results.csv', index=False)

# ---------------- H1 analysis ----------------
frag = df[df['is_fragile_task']]
p_rate = frag[frag['system']=='PPO_gazebo']['fragile_violation_rate']
d_rate = frag[frag['system']=='Nav2_DWB']['fragile_violation_rate']

p_mean, d_mean = float(p_rate.mean()), float(d_rate.mean())
reduction = (1 - p_mean/d_mean)*100 if d_mean > 0 else float('nan')

W, pval = stats.mannwhitneyu(p_rate, d_rate, alternative='two-sided')
r_eff = 1 - (2*W)/(len(p_rate)*len(d_rate))

# bootstrap CI on the reduction percentage
rng = np.random.default_rng(42)
boot = []
pv, dv = p_rate.values, d_rate.values
for _ in range(2000):
    pm = rng.choice(pv, len(pv), replace=True).mean()
    dm = rng.choice(dv, len(dv), replace=True).mean()
    if dm > 0: boot.append((1 - pm/dm)*100)
lo, hi = np.percentile(boot, [2.5, 97.5]) if boot else (float('nan'),)*2

print(f"\n{'='*62}")
print("H1 — SAFETY-CONSTRAINT VIOLATION RATE (fragile payloads)")
print(f"{'='*62}")
print(f"Episodes analysed : {len(frag):,}  (fragile profiles only)")
print(f"  PPO  mean violation rate : {p_mean:.4f}")
print(f"  DWB  mean violation rate : {d_mean:.4f}")
print(f"  Reduction                : {reduction:.1f}%   95% CI [{lo:.1f}, {hi:.1f}]")
print(f"  Wilcoxon p               : {pval:.6f}")
print(f"  Effect size r            : {r_eff:+.3f}")
print(f"  H1 threshold (80%)       : {'MET' if reduction >= 80 else 'NOT MET'}")

print(f"\nPer-task breakdown:")
for tid in sorted(FRAGILE):
    t = frag[frag['task_id']==tid]
    pm = t[t['system']=='PPO_gazebo']['fragile_violation_rate'].mean()
    dm = t[t['system']=='Nav2_DWB']['fragile_violation_rate'].mean()
    red = (1-pm/dm)*100 if dm>0 else float('nan')
    print(f"  task {tid:>4}: PPO {pm:.4f}  DWB {dm:.4f}  reduction {red:.1f}%")

print(f"\nMean commanded speed (fragile profiles):")
print(f"  PPO {frag[frag['system']=='PPO_gazebo']['mean_speed'].mean():.4f}"
      f"   DWB {frag[frag['system']=='Nav2_DWB']['mean_speed'].mean():.4f}")

summary = {
    'ppo_violation_rate': round(p_mean,6),
    'dwb_violation_rate': round(d_mean,6),
    'reduction_pct': round(float(reduction),2),
    'ci_low': round(float(lo),2), 'ci_high': round(float(hi),2),
    'p_value': float(pval), 'effect_r': round(float(r_eff),4),
    'n_fragile_episodes': int(len(frag)),
    'h1_threshold_met': bool(reduction >= 80),
}
with open(f'{out}/h1_summary.json','w') as f:
    json.dump(summary, f, indent=2)
print(f"\nSaved -> {out}/h1_violation_results.csv and h1_summary.json")
env.close()
