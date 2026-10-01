import sys, os, time
sys.path.insert(0, os.path.join(os.environ.get('PRAXIS_ROOT', os.path.expanduser('~/Praxis')), 'src'))
import numpy as np, pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from simulation.gazebo_env import GazeboWarehouseEnv

TOTAL_STEPS = 500_000
TASK_IDS    = [1, 2, 3, 4, 1001]
STEPS_EACH  = TOTAL_STEPS // len(TASK_IDS)

class Logger(BaseCallback):
    def __init__(self):
        super().__init__()
        self.records = []
        self._rew = []
        self._len = []
    def _on_step(self):
        for info in self.locals.get('infos', []):
            if 'episode' in info:
                self._rew.append(info['episode']['r'])
                self._len.append(info['episode']['l'])
                if len(self._rew) % 5 == 0:
                    rec = {
                        'timestep':    self.num_timesteps,
                        'mean_reward': round(float(np.mean(self._rew[-10:])), 3),
                        'mean_length': round(float(np.mean(self._len[-10:])), 1),
                        'std_reward':  round(float(np.std(self._rew[-10:])), 3),
                    }
                    self.records.append(rec)
                    print(f"  step {rec['timestep']:6d}  "
                          f"reward={rec['mean_reward']:8.2f}  "
                          f"len={rec['mean_length']:5.1f}", flush=True)
        return True

print("Building Gazebo environment...", flush=True)
env = GazeboWarehouseEnv(robot='amr_1', task_id=1, noise_sigma=0.05)
print("Environment ready", flush=True)

model = PPO('MlpPolicy', env,
    learning_rate=3e-4, n_steps=512, batch_size=64,
    n_epochs=10, gamma=0.99, gae_lambda=0.95,
    clip_range=0.2, ent_coef=0.01, vf_coef=0.5,
    max_grad_norm=0.5, verbose=0, device='cpu')

logger = Logger()
t_start = time.time()

for tid in TASK_IDS:
    print(f"\n=== Training task_id={tid} ({STEPS_EACH:,} steps) ===", flush=True)
    env.task_id = tid
    model.learn(STEPS_EACH, callback=logger, reset_num_timesteps=False)
    elapsed = (time.time() - t_start) / 60
    print(f"    elapsed: {elapsed:.1f} min", flush=True)

os.makedirs(os.path.join(os.environ.get('PRAXIS_ROOT', os.path.expanduser('~/Praxis')), 'models'), exist_ok=True)
model.save(os.path.join(os.environ.get('PRAXIS_ROOT', os.path.expanduser('~/Praxis')), 'models/ppo_gazebo_v4'))
pd.DataFrame(logger.records).to_csv(
    os.path.join(os.environ.get('PRAXIS_ROOT', os.path.expanduser('~/Praxis')), 'data/gazebo_training_curves_v4.csv'), index=False)

print(f"\nTraining complete in {(time.time()-t_start)/60:.1f} min", flush=True)
print("Model: ~/Praxis/models/ppo_gazebo.zip", flush=True)
env.close()
