"""
Guided PPO — LiDAR branch, full 500K-step training.

Payload-adaptive reward shaping (C_kinetic, C_safety) plus domain
randomisation across sensor noise levels, with a dynamic peer obstacle
and 31-dimensional observation (7 pose + 24 LiDAR sectors).

Configuration validated by the 100K-step avoidance probe:
  collision_penalty=-20.0, max_steps=200, freeze_penalty=-1.0,
  velocity_weight=0.2 — collision rate fell 0.552 -> 0.394 (105 outcomes).

This is the "roadmap followed" arm for H3 and the guided arm for RQ1/H1.
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.environ.get('PRAXIS_ROOT', os.path.expanduser('~/Praxis')), 'src'))
import numpy as np, pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from simulation.gazebo_env_lidar import GazeboWarehouseEnv

TOTAL_STEPS = 500_000
TASK_IDS    = [1, 2, 3, 4, 1001]
STEPS_EACH  = TOTAL_STEPS // len(TASK_IDS)


class Logger(BaseCallback):
    def __init__(self):
        super().__init__()
        self.rows = []
        self._rew, self._len, self._out = [], [], []

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
                         'collision_rate': round(float(np.mean(w)), 3)}
                    self.rows.append(r)
                    print(f"  step {r['timestep']:6d}  "
                          f"reward={r['mean_reward']:8.2f}  "
                          f"len={r['mean_length']:6.1f}  "
                          f"collision_rate={r['collision_rate']:.3f}",
                          flush=True)
        return True


print("Building guided LiDAR environment...", flush=True)
env = GazeboWarehouseEnv(
    robot='amr_1', task_id=1, noise_sigma=0.05,
    peer_enabled=True,
    collision_penalty=-20.0,
    proximity_weight=2.0,
    freeze_penalty=-1.0,
    velocity_weight=0.2,
    freeze_speed=0.05,
    max_steps=200,
)
print(f"obs {env.observation_space.shape}  max_steps {env.max_steps}  "
      f"collision_dist {env.collision_dist} m", flush=True)

model = PPO('MlpPolicy', env,
    learning_rate=3e-4, n_steps=512, batch_size=64, n_epochs=10,
    gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01,
    vf_coef=0.5, max_grad_norm=0.5, verbose=0, device='cpu')

logger = Logger()
t_start = time.time()

for tid in TASK_IDS:
    print(f"\n=== Guided training task_id={tid} "
          f"({STEPS_EACH:,} steps) ===", flush=True)
    env.task_id = tid
    # domain randomisation: noise varies across the training set
    env.noise = float(np.random.choice([0.0, 0.05, 0.15, 0.30]))
    model.learn(STEPS_EACH, callback=logger, reset_num_timesteps=False)
    elapsed = (time.time() - t_start) / 60
    print(f"    elapsed: {elapsed:.1f} min", flush=True)

os.makedirs(os.path.join(os.environ.get('PRAXIS_ROOT', os.path.expanduser('~/Praxis')), 'models'), exist_ok=True)
model.save(os.path.join(os.environ.get('PRAXIS_ROOT', os.path.expanduser('~/Praxis')), 'models/ppo_lidar_guided'))
pd.DataFrame(logger.rows).to_csv(
    os.path.join(os.environ.get('PRAXIS_ROOT', os.path.expanduser('~/Praxis')), 'data/lidar_guided_training_curves.csv'),
    index=False)

print(f"\nTraining complete in {(time.time()-t_start)/60:.1f} min", flush=True)
print("Model: ~/Praxis/models/ppo_lidar_guided.zip", flush=True)
env.close()
