import sys, os
sys.path.insert(0, os.path.join(os.environ.get('PRAXIS_ROOT', os.path.expanduser('~/Praxis')), 'src'))
import numpy as np, pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from simulation.warehouse_env import WarehouseEnv

TASK_IDS    = [1, 2, 3, 4, 1001]
TOTAL_STEPS = 500_000
STEPS_EACH  = TOTAL_STEPS // len(TASK_IDS)

class Logger(BaseCallback):
    def __init__(self):
        super().__init__()
        self.records=[]; self._rew=[]; self._len=[]
    def _on_step(self):
        for info in self.locals.get('infos', []):
            if 'episode' in info:
                self._rew.append(info['episode']['r'])
                self._len.append(info['episode']['l'])
                if len(self._rew) % 10 == 0:
                    self.records.append({
                        'timestep':    self.num_timesteps,
                        'mean_reward': round(float(np.mean(self._rew[-20:])), 3),
                        'mean_length': round(float(np.mean(self._len[-20:])), 1),
                        'std_reward':  round(float(np.std(self._rew[-20:])),  3),
                    })
        return True

os.makedirs(os.path.join(os.environ.get('PRAXIS_ROOT', os.path.expanduser('~/Praxis')), 'models'), exist_ok=True)
os.makedirs(os.path.join(os.environ.get('PRAXIS_ROOT', os.path.expanduser('~/Praxis')), 'data'),   exist_ok=True)
os.makedirs(os.path.join(os.environ.get('PRAXIS_ROOT', os.path.expanduser('~/Praxis')), 'logs'),   exist_ok=True)

model = PPO('MlpPolicy',
    make_vec_env(lambda: WarehouseEnv(noise_sigma=0.05, task_id=1), n_envs=4),
    learning_rate=3e-4, n_steps=2048, batch_size=64, n_epochs=10,
    gamma=0.99, gae_lambda=0.95, clip_range=0.2,
    ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5,
    verbose=1, tensorboard_log=os.path.join(os.environ.get('PRAXIS_ROOT', os.path.expanduser('~/Praxis')), 'logs/'))

logger = Logger()
print(f"Training on: {model.device}")

for tid in TASK_IDS:
    print(f"\n→ task_id={tid}  ({STEPS_EACH:,} steps)")
    model.set_env(make_vec_env(
        lambda t=tid: WarehouseEnv(noise_sigma=0.05, task_id=t), n_envs=4))
    model.learn(STEPS_EACH, callback=logger, reset_num_timesteps=False,
                tb_log_name=f'PPO_task{tid}')
    if logger.records:
        print(f"   Latest reward: {logger.records[-1]['mean_reward']:.3f}")

model.save(os.path.join(os.environ.get('PRAXIS_ROOT', os.path.expanduser('~/Praxis')), 'models/ppo_warehouse'))
pd.DataFrame(logger.records).to_csv(
    os.path.join(os.environ.get('PRAXIS_ROOT', os.path.expanduser('~/Praxis')), 'data/ppo_training_curves.csv'), index=False)
print("\n✅ Training complete — model saved to ~/Praxis/models/ppo_warehouse.zip")
