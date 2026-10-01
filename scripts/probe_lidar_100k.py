"""
100K-step learning probe for the LiDAR branch.

Checkpoint before committing to full training: does collision rate fall
at all over 100,000 steps? A flat curve means avoidance is not learnable
in this compute budget and 500K would not help either.
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.environ.get('PRAXIS_ROOT', os.path.expanduser('~/Praxis')), 'src'))
import numpy as np, pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from simulation.gazebo_env_lidar import GazeboWarehouseEnv

TOTAL_STEPS = 100_000

class Probe(BaseCallback):
    def __init__(self):
        super().__init__()
        self.rows=[]; self._rew=[]; self._len=[]; self._coll=[]
    def _on_step(self):
        for info in self.locals.get('infos', []):
            if 'collided' in info and (info.get('collided') or
                                       info.get('arrived')):
                self._coll.append(1.0 if info['collided'] else 0.0)
            if 'episode' in info:
                self._rew.append(info['episode']['r'])
                self._len.append(info['episode']['l'])
                if len(self._rew) % 10 == 0:
                    w = self._coll[-30:] if self._coll else [0.0]
                    r = {'timestep': self.num_timesteps,
                         'mean_reward': round(float(np.mean(self._rew[-20:])),2),
                         'mean_length': round(float(np.mean(self._len[-20:])),1),
                         'collision_rate': round(float(np.mean(w)),3)}
                    self.rows.append(r)
                    print(f"  step {r['timestep']:6d}  reward={r['mean_reward']:8.2f}"
                          f"  len={r['mean_length']:6.1f}"
                          f"  collision_rate={r['collision_rate']:.3f}", flush=True)
        return True

print("Building LiDAR environment (31-dim, peer obstacle)...", flush=True)
env = GazeboWarehouseEnv(robot='amr_1', task_id=1, noise_sigma=0.05,
                         peer_enabled=True, collision_penalty=-100.0,
                         proximity_weight=2.0)
print(f"obs dim {env.observation_space.shape}  max_steps {env.max_steps}",
      flush=True)

model = PPO('MlpPolicy', env,
    learning_rate=3e-4, n_steps=512, batch_size=64, n_epochs=10,
    gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01,
    vf_coef=0.5, max_grad_norm=0.5, verbose=0, device='cpu')

cb = Probe(); t0 = time.time()
model.learn(TOTAL_STEPS, callback=cb)
mins = (time.time()-t0)/60

model.save(os.path.join(os.environ.get('PRAXIS_ROOT', os.path.expanduser('~/Praxis')), 'models/ppo_lidar_probe'))
df = pd.DataFrame(cb.rows)
df.to_csv(os.path.join(os.environ.get('PRAXIS_ROOT', os.path.expanduser('~/Praxis')), 'data/lidar_probe_curves.csv'), index=False)

print(f"\n{'='*60}")
print(f"PROBE COMPLETE — {mins:.1f} min")
print(f"{'='*60}")
if len(df) >= 6:
    first = df.head(len(df)//3)
    last  = df.tail(len(df)//3)
    print(f"  collision rate  first third {first['collision_rate'].mean():.3f}"
          f"   last third {last['collision_rate'].mean():.3f}")
    print(f"  mean reward     first third {first['mean_reward'].mean():8.2f}"
          f"   last third {last['mean_reward'].mean():8.2f}")
    fell = last['collision_rate'].mean() < first['collision_rate'].mean() - 0.05
    rose = last['mean_reward'].mean() > first['mean_reward'].mean() + 10
    print(f"\n  collision rate falling : {'YES' if fell else 'NO'}")
    print(f"  reward improving       : {'YES' if rose else 'NO'}")
    print(f"  VERDICT: {'PROCEED to full training' if (fell or rose) else 'STOP - not learning'}")
env.close()
