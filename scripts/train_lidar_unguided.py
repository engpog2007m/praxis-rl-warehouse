"""
Unguided PPO — LiDAR branch, full 500K-step training.

H3 baseline. Identical architecture, step budget, and environment to the
guided arm, with the Deployment Roadmap's guidance mechanisms removed:

  - payload-adaptive reward shaping disabled (C_kinetic, C_safety)
  - payload-adaptive velocity scaling disabled
  - peer-proximity shaping disabled
  - domain randomisation disabled (trained at sigma = 0 only)

Retained: progress reward, arrival bonus, collision penalty, anti-freeze
terms. The policy still learns to navigate and to avoid the peer; what it
does not receive is payload-appropriate guidance or noise exposure.

Note on comparison: raw reward is NOT comparable across arms, since the
reward functions differ. H3 is assessed on completion rate, collision
rate, and constraint violation rate.
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.environ.get('PRAXIS_ROOT', os.path.expanduser('~/Praxis')), 'src'))
import numpy as np, pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from simulation.gazebo_env_lidar import GazeboWarehouseEnv, _RobotBridge

TOTAL_STEPS = 500_000
TASK_IDS    = [1, 2, 3, 4, 1001]
STEPS_EACH  = TOTAL_STEPS // len(TASK_IDS)


class UnguidedEnv(GazeboWarehouseEnv):
    """Strips payload-adaptive shaping and velocity scaling."""

    def step(self, action):
        task  = self.registry.get_task(self.task_id)
        speed = float(np.linalg.norm(action[:2]))

        vx = float(action[0]) * 0.4
        vy = float(action[1]) * 0.4
        # NO payload-adaptive velocity scaling

        if self.peer_driver is not None:
            self.peer_driver.step()

        self.bridge.drive(vx, vy)
        time.sleep(self.dt)

        rel  = self.ego_pos() - self.origin
        dist = float(np.linalg.norm(self.goal_rel - rel))

        sep = self.peer_separation()
        collided = sep < self.collision_dist
        if collided:
            self.n_collisions += 1

        speed_actual = float(np.linalg.norm(self.bridge.vel))

        r_prog   = 10.0 * (self._prev_dist - dist)
        r_arrive = 150.0 if dist < self.ARRIVAL_RADIUS else 0.0
        r_coll   = self.collision_penalty if collided else 0.0
        r_freeze = self.freeze_penalty if speed_actual < self.FREEZE_SPEED else 0.0
        r_vel    = self.velocity_weight * speed_actual
        # NO c_kin, NO c_safe, NO r_prox

        reward = r_prog + r_arrive + r_coll + r_freeze + r_vel - 0.1

        self._prev_dist = dist
        self._step_n   += 1

        terminated = (dist < self.ARRIVAL_RADIUS) or collided
        truncated  = self._step_n >= self.max_steps
        if terminated or truncated:
            self.bridge.halt()
            if self.peer is not None:
                self.peer.halt()

        min_obs = float(self.bridge.scan.min() * _RobotBridge.RANGE_MAX)
        return self._obs(), reward, terminated, truncated, {
            'distance': dist,
            'speed': speed,
            'speed_actual': speed_actual,
            'min_obstacle_dist': min_obs,
            'peer_separation': sep,
            'collided': bool(collided),
            'arrived': bool(dist < self.ARRIVAL_RADIUS),
            'n_collisions': self.n_collisions,
            'fragile_violation': bool(task['is_fragile'] and
                                      speed > self.FRAGILE_THRESHOLD),
            'kinetic_violation': bool(task['is_heavy'] and
                                      speed > self.HEAVY_THRESHOLD),
        }


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

                    # health check: a dead simulator pins episode length at
                    # the cap and freezes collision rate. Halt rather than
                    # train for hours against a stalled Gazebo.
                    if len(self.rows) >= 8:
                        recent = self.rows[-8:]
                        if (all(x['mean_length'] >= 199.9 for x in recent) and
                            len({x['collision_rate'] for x in recent}) == 1):
                            print("\n*** HEALTH CHECK FAILED: episode length "
                                  "pinned and collision rate frozen. "
                                  "Simulator likely stalled. Halting. ***",
                                  flush=True)
                            return False
        return True


print("Building unguided LiDAR environment...", flush=True)
env = UnguidedEnv(
    robot='amr_1', task_id=1, noise_sigma=0.0,   # no domain randomisation
    peer_enabled=True,
    collision_penalty=-20.0,
    proximity_weight=0.0,                        # no proximity shaping
    freeze_penalty=-1.0,
    velocity_weight=0.2,
    freeze_speed=0.05,
    max_steps=200,
)
print(f"obs {env.observation_space.shape}  max_steps {env.max_steps}", flush=True)

model = PPO('MlpPolicy', env,
    learning_rate=3e-4, n_steps=512, batch_size=64, n_epochs=10,
    gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01,
    vf_coef=0.5, max_grad_norm=0.5, verbose=0, device='cpu')

logger = Logger()
t_start = time.time()

for tid in TASK_IDS:
    print(f"\n=== Unguided training task_id={tid} "
          f"({STEPS_EACH:,} steps) ===", flush=True)
    env.task_id = tid
    env.noise = 0.0        # fixed; no randomisation
    model.learn(STEPS_EACH, callback=logger, reset_num_timesteps=False)
    print(f"    elapsed: {(time.time()-t_start)/60:.1f} min", flush=True)

model.save(os.path.join(os.environ.get('PRAXIS_ROOT', os.path.expanduser('~/Praxis')), 'models/ppo_lidar_unguided'))
pd.DataFrame(logger.rows).to_csv(
    os.path.join(os.environ.get('PRAXIS_ROOT', os.path.expanduser('~/Praxis')), 'data/lidar_unguided_training_curves.csv'),
    index=False)

print(f"\nTraining complete in {(time.time()-t_start)/60:.1f} min", flush=True)
print("Model: ~/Praxis/models/ppo_lidar_unguided.zip", flush=True)
env.close()
