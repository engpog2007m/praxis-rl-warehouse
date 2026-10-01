import sys, os
sys.path.insert(0, os.path.join(os.environ.get('PRAXIS_ROOT', os.path.expanduser('~/Praxis')), 'src'))
import numpy as np, pandas as pd
from scipy import stats
from stable_baselines3 import PPO
from simulation.warehouse_env import WarehouseEnv

class Nav2DWBController:
    def __init__(self, seed=42):
        self.rng = np.random.default_rng(seed)
    def predict(self, obs):
        rx,ry,gx,gy,dist = obs[0],obs[1],obs[2],obs[3],obs[4]
        dx,dy = gx-rx, gy-ry
        norm  = max(np.sqrt(dx**2+dy**2), 1e-6)
        speed = min(0.6*dist/15.0, 1.0)
        noise = self.rng.normal(0, 0.015)
        vx = float(np.clip(speed*(dx/norm)+noise, -1.0, 1.0))
        vy = float(np.clip(speed*(dy/norm)+noise, -1.0, 1.0))
        return np.array([vx, vy, 0.0], dtype=np.float32)

NOISE_LEVELS = [0.00, 0.05, 0.15, 0.30]
MASS_MULTS   = [1.0, 1.25, 1.50, 2.0]
TASK_IDS     = [1, 2, 3, 4, 1001]
EPISODES     = 50
MAX_TICKS    = 100

MODEL_PATH = os.path.join(os.environ.get('PRAXIS_ROOT', os.path.expanduser('~/Praxis')), 'models/ppo_warehouse')
OUT_DIR    = os.path.join(os.environ.get('PRAXIS_ROOT', os.path.expanduser('~/Praxis')), 'data')
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(os.path.join(os.environ.get('PRAXIS_ROOT', os.path.expanduser('~/Praxis')), 'results'), exist_ok=True)

print("Loading trained PPO model...")
ppo_model = PPO.load(MODEL_PATH)
dwb_ctrl  = Nav2DWBController()
print(f"Model loaded from {MODEL_PATH}.zip")

def run_episode(ctrl, env, is_ppo):
    obs, _ = env.reset()
    R=0.0; viols=0; ticks=0; done=False
    for _ in range(MAX_TICKS):
        if is_ppo:
            a, _ = ctrl.predict(obs, deterministic=True)
        else:
            a = ctrl.predict(obs)
        obs,r,term,trunc,_ = env.step(a)
        R+=r; ticks+=1
        if r < -5.0: viols+=1
        if term: done=True; break
    return {
        'reward':         round(R,4),
        'cycle_time':     ticks if done else MAX_TICKS,
        'throughput':     1.0 if (done and ticks<90) else 0.0,
        'efficiency':     round(R/ticks,4) if ticks else 0.0,
        'violation_rate': round(viols/ticks,4) if ticks else 0.0,
        'completed':      done,
    }

records = []
total   = len(TASK_IDS)*len(NOISE_LEVELS)*len(MASS_MULTS)*EPISODES*2
done_n  = 0

print(f"\nRunning {total:,} evaluation episodes...")
for tid in TASK_IDS:
    for sig in NOISE_LEVELS:
        for mass in MASS_MULTS:
            ep_ppo = WarehouseEnv(noise_sigma=sig, mass_multiplier=mass, task_id=tid)
            ep_dwb = WarehouseEnv(noise_sigma=sig, mass_multiplier=mass, task_id=tid)
            for ep in range(EPISODES):
                for ctrl,env,sname,ip in [
                    (ppo_model, ep_ppo, 'PPO',      True),
                    (dwb_ctrl,  ep_dwb, 'Nav2_DWB', False)]:
                    res = run_episode(ctrl, env, ip)
                    records.append({'system':sname,'task_id':tid,
                        'noise_sigma':sig,'mass_multiplier':mass,
                        'episode':ep, **res})
                    done_n+=1
            if done_n % 400 == 0:
                print(f"  {done_n:,}/{total:,}  ({done_n/total*100:.0f}%)")

df = pd.DataFrame(records)
df.to_csv(f'{OUT_DIR}/evaluation_results.csv', index=False)
print(f"\nSaved {len(df):,} records to {OUT_DIR}/evaluation_results.csv")

print("\nRunning Wilcoxon rank-sum tests (a = 0.05)...")
ppo_df = df[df['system']=='PPO']
dwb_df = df[df['system']=='Nav2_DWB']

stat_rows = []
for m in ['reward','cycle_time','throughput','efficiency','violation_rate']:
    pv,dv = ppo_df[m].values, dwb_df[m].values
    W,p   = stats.mannwhitneyu(pv, dv, alternative='two-sided')
    r     = 1 - (2*W)/(len(pv)*len(dv))
    mag   = ('Large'  if abs(r)>=0.5 else
             'Medium' if abs(r)>=0.3 else
             'Small'  if abs(r)>=0.1 else 'Negligible')
    lower   = m in ['cycle_time','violation_rate']
    ppo_sup = (np.median(pv)<np.median(dv)) if lower else (np.median(pv)>np.median(dv))
    stat_rows.append({
        'Metric':m,
        'PPO_median':round(float(np.median(pv)),4),
        'PPO_IQR':round(float(np.percentile(pv,75)-np.percentile(pv,25)),4),
        'DWB_median':round(float(np.median(dv)),4),
        'DWB_IQR':round(float(np.percentile(dv,75)-np.percentile(dv,25)),4),
        'W_statistic':round(float(W),1),
        'p_value':round(float(p),6),
        'Significant':'Yes' if p<0.05 else 'No',
        'Effect_r':round(float(r),4),
        'Effect_size':mag,
        'PPO_superior':'Yes' if ppo_sup else 'No',
    })
    flag = 'YES' if p<0.05 else 'NO'
    sup  = 'PPO better' if ppo_sup else 'DWB better'
    print(f"  {m:20s}  p={p:.6f}  sig={flag}  r={r:.3f} ({mag})  {sup}")

pd.DataFrame(stat_rows).to_csv(f'{OUT_DIR}/statistical_analysis.csv', index=False)
print(f"\nStatistical analysis saved to {OUT_DIR}/statistical_analysis.csv")

print("\n" + "="*55)
print("RESULTS SUMMARY")
print("="*55)
for sname, sdf in [('PPO', ppo_df), ('Nav2 DWB', dwb_df)]:
    print(f"\n{sname}:")
    print(f"  Throughput rate : {sdf['throughput'].mean():.1%}")
    print(f"  Efficiency index: {sdf['efficiency'].mean():.4f}")
    print(f"  Violation rate  : {sdf['violation_rate'].mean():.4f}")
    print(f"  Cycle time      : {sdf['cycle_time'].mean():.1f} ticks")
