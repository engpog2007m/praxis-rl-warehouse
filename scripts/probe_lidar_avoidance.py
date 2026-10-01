"""
100K-step avoidance probe — LiDAR branch with an active peer obstacle.

The earlier probe ran while the peer sat 27 m away, so it demonstrated only
that the 31-dimensional observation does not impede goal-reaching. It did
not test avoidance, because there was nothing to avoid.

The peer is now teleported onto the ego's route at every episode reset. A
straight-line baseline collides on 8 of 10 episodes and arrives on 2, so
both outcomes are reachable and a naive policy is penalised.

Pre-specified criterion: collision rate falls by at least 0.15 between the
first and last third of training. Flat or rising means avoidance is not
learnable in this budget and 500K would not change that.
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.environ.get('PRAXIS_ROOT', os.path.expanduser('~/Praxis')), 'src'))
import numpy as np, pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from simulation.gazebo_env_lidar import GazeboWarehouseEnv

TOTAL_STEPS = 50_000


class AvoidanceProbe(BaseCallback):
    def __init__(self):
        super().__init__()
        self.rows = []
        self._rew, self._len = [], []
        self._out = []          # 1 = collided, 0 = arrived or timed out

    def _on_step(self):
        for info in self.locals.get('infos', []):
            if 'collided' in info and (info.get('collided') or
                                       info.get('arrived')):
                self._out.append(1.0 if info['collided'] else 0.0)
            if 'episode' in info:
                self._rew.append(info['episode']['r'])
                self._len.append(info['episode']['l'])
                if len(self._rew) % 10 == 0:
                    w = self._out[-40:] if self._out else [0.0]
                    r = {'timestep': self.num_timesteps,
                         'mean_reward': round(float(np.mean(self._rew[-20:])), 2),
                         'mean_length': round(float(np.mean(self._len[-20:])), 1),
                         'collision_rate': round(float(np.mean(w)), 3),
                         'n_outcomes': len(self._out)}
                    self.rows.append(r)
                    print(f"  step {r['timestep']:6d}  "
                          f"reward={r['mean_reward']:8.2f}  "
                          f"len={r['mean_length']:6.1f}  "
                          f"collision_rate={r['collision_rate']:.3f}",
                          flush=True)
        return True


print("Building LiDAR environment with active peer obstacle...", flush=True)
env = GazeboWarehouseEnv(robot='amr_1', task_id=1, noise_sigma=0.05,
                         peer_enabled=True, collision_penalty=-20.0,
                         proximity_weight=2.0, max_steps=200)
print(f"obs {env.observation_space.shape}  max_steps {env.max_steps}  "
      f"collision_dist {env.collision_dist} m", flush=True)

model = PPO('MlpPolicy', env,
            learning_rate=3e-4, n_steps=512, batch_size=64, n_epochs=10,
            gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01,
            vf_coef=0.5, max_grad_norm=0.5, verbose=0, device='cpu')

cb = AvoidanceProbe()
t0 = time.time()
model.learn(TOTAL_STEPS, callback=cb)
mins = (time.time() - t0) / 60

model.save(os.path.join(os.environ.get('PRAXIS_ROOT', os.path.expanduser('~/Praxis')), 'models/ppo_lidar_avoid_probe'))
df = pd.DataFrame(cb.rows)
df.to_csv(os.path.join(os.environ.get('PRAXIS_ROOT', os.path.expanduser('~/Praxis')), 'data/lidar_avoid_probe.csv'), index=False)

print(f"\n{'='*64}")
print(f"AVOIDANCE PROBE COMPLETE — {mins:.1f} min, "
      f"{len(cb._out)} terminal outcomes")
print(f"{'='*64}")

if len(df) >= 6:
    k = len(df) // 3
    first, last = df.head(k), df.tail(k)
    c0, c1 = first['collision_rate'].mean(), last['collision_rate'].mean()
    r0, r1 = first['mean_reward'].mean(), last['mean_reward'].mean()
    drop = c0 - c1
    print(f"  collision rate  first {c0:.3f}  ->  last {c1:.3f}   "
          f"drop {drop:+.3f}")
    print(f"  mean reward     first {r0:8.2f}  ->  last {r1:8.2f}")
    print(f"  episode length  first {first['mean_length'].mean():6.1f}  ->  "
          f"last {last['mean_length'].mean():6.1f}")
    met = drop >= 0.15
    print(f"\n  PRE-SPECIFIED CRITERION (collision rate drop >= 0.15): "
          f"{'MET' if met else 'NOT MET'}")
    print(f"  VERDICT: {'PROCEED to full training' if met else 'STOP — write up'}")
else:
    print("  too few logged episodes to assess")

env.close()
