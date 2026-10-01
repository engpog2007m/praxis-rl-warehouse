"""
H3 baseline — unguided PPO.

Trains the same PPO architecture without the Deployment Roadmap's two
guidance mechanisms:
  1. payload-adaptive reward shaping (C_kinetic, C_safety disabled)
  2. domain randomisation (sensor noise fixed at zero during training)

Progress reward and arrival bonus are retained, so the policy still learns
to navigate. What it does not learn is payload-appropriate behaviour or
robustness to sensor uncertainty. Evaluated against the guided policy under
identical noisy, payload-varied conditions, this isolates the contribution
of the roadmap's guidance components.
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.environ.get('PRAXIS_ROOT', os.path.expanduser('~/Praxis')), 'src'))
import numpy as np, pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from simulation.gazebo_env import GazeboWarehouseEnv

TOTAL_STEPS = 500_000
TASK_IDS    = [1, 2, 3, 4, 1001]
STEPS_EACH  = TOTAL_STEPS // len(TASK_IDS)


class UnguidedEnv(GazeboWarehouseEnv):
    """Strips reward shaping; retains progress and arrival terms only."""

    def step(self, action):
        task  = self.registry.get_task(self.task_id)
        speed = float(np.linalg.norm(action[:2]))

        vx = float(action[0]) * 0.4
        vy = float(action[1]) * 0.4
        # NO payload-adaptive velocity scaling

        self.bridge.drive(vx, vy)
        time.sleep(self.dt)

        rel  = self.bridge.pos - self.origin
        dist = float(np.linalg.norm(self.goal_rel - rel))

        r_prog   = 10.0 * (self._prev_dist - dist)
        r_arrive = 150.0 if dist < 0.8 else 0.0
        # NO C_kinetic, NO C_safety
        reward   = r_prog + r_arrive - 0.1

        self._prev_dist = dist
        self._step_n   += 1
        terminated = dist < 0.8
        truncated  = self._step_n >= self.max_steps
        if terminated or truncated:
            self.bridge.halt()

        fragile_violation = bool(task['is_fragile'] and speed > 0.5)
        kinetic_violation = bool(task['is_heavy'] and speed > 0.7)
        return self._obs(), reward, terminated, truncated, {
            'distance': dist, 'speed': speed,
            'fragile_violation': fragile_violation,
            'kinetic_violation': kinetic_violation,
        }


class Logger(BaseCallback):
    def __init__(self):
        super().__init__(); self.records=[]; self._rew=[]; self._len=[]
    def _on_step(self):
        for info in self.locals.get('infos', []):
            if 'episode' in info:
                self._rew.append(info['episode']['r'])
                self._len.append(info['episode']['l'])
                if len(self._rew) % 5 == 0:
                    rec = {'timestep': self.num_timesteps,
                           'mean_reward': round(float(np.mean(self._rew[-10:])),3),
                           'mean_length': round(float(np.mean(self._len[-10:])),1)}
                    self.records.append(rec)
                    print(f"  step {rec['timestep']:6d}  "
                          f"reward={rec['mean_reward']:8.2f}  "
                          f"len={rec['mean_length']:5.1f}", flush=True)
        return True


print("Building unguided environment (no reward shaping, no noise)...", flush=True)
env = UnguidedEnv(robot='amr_1', task_id=1, noise_sigma=0.0)   # noise OFF
print("Environment ready", flush=True)

model = PPO('MlpPolicy', env,
    learning_rate=3e-4, n_steps=512, batch_size=64, n_epochs=10,
    gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01,
    vf_coef=0.5, max_grad_norm=0.5, verbose=0, device='cpu')

logger = Logger(); t0 = time.time()
for tid in TASK_IDS:
    print(f"\n=== Unguided training task_id={tid} ({STEPS_EACH:,} steps) ===", flush=True)
    env.task_id = tid
    model.learn(STEPS_EACH, callback=logger, reset_num_timesteps=False)
    print(f"    elapsed: {(time.time()-t0)/60:.1f} min", flush=True)

model.save(os.path.join(os.environ.get('PRAXIS_ROOT', os.path.expanduser('~/Praxis')), 'models/ppo_unguided'))
pd.DataFrame(logger.records).to_csv(
    os.path.join(os.environ.get('PRAXIS_ROOT', os.path.expanduser('~/Praxis')), 'data/unguided_training_curves.csv'), index=False)
print(f"\nComplete in {(time.time()-t0)/60:.1f} min")
print("Model: ~/Praxis/models/ppo_unguided.zip", flush=True)
env.close()
