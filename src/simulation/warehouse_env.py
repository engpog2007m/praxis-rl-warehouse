import numpy as np
import gymnasium as gym
from gymnasium import spaces
from src.simulation.data_registry import DataRegistry

class RobotState:
    def __init__(self):
        self.position = np.zeros(2, dtype=np.float32)

TASK_REGISTRY = {
    1:    {"name": "standard_box",     "is_heavy": False, "is_fragile": False, "weight": 5.0},
    2:    {"name": "lightweight_pkg",  "is_heavy": False, "is_fragile": False, "weight": 2.0},
    3:    {"name": "heavy_pallet",     "is_heavy": True,  "is_fragile": False, "weight": 45.0},
    4:    {"name": "fragile_glassware","is_heavy": False, "is_fragile": True,  "weight": 3.5},
    1001: {"name": "heavy_fragile",    "is_heavy": True,  "is_fragile": True,  "weight": 30.0},
}

class WarehouseEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, noise_sigma=0.0, mass_multiplier=1.0, task_id=1):
        super(WarehouseEnv, self).__init__()
        self.registry          = DataRegistry()
        self.noise_sigma       = noise_sigma
        self.mass_multiplier   = mass_multiplier
        self.current_task_id   = task_id
        self.max_episode_steps = 100
        self.current_step      = 0
        self.action_space      = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=np.array([0,0,0,0,0,-1,-1], dtype=np.float32),
            high=np.array([15,15,15,15,21.2,1,1], dtype=np.float32))
        self.robot_pos    = np.zeros(2, dtype=np.float32)
        self.goal_pos     = np.array([10.0, 10.0], dtype=np.float32)
        self.prev_dist    = 0.0
        self.velocity     = np.zeros(2, dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.robot_pos    = np.random.uniform(0.5, 2.5, size=2).astype(np.float32)
        self.goal_pos     = np.random.uniform(10.0, 14.5, size=2).astype(np.float32)
        self.velocity     = np.zeros(2, dtype=np.float32)
        self.prev_dist    = float(np.linalg.norm(self.goal_pos - self.robot_pos))
        return self._obs(), {}

    def step(self, action):
        self.current_step += 1
        task  = TASK_REGISTRY.get(self.current_task_id, TASK_REGISTRY[1])
        speed = float(np.linalg.norm(action[:2]))

        eff = action.copy()
        if task["is_heavy"] and self.mass_multiplier > 1.25:
            eff[:2] *= 0.75

        self.velocity  = eff[:2].astype(np.float32)
        self.robot_pos = np.clip(
            self.robot_pos + self.velocity * 0.1, 0.0, 15.0
        ).astype(np.float32)

        dist   = float(np.linalg.norm(self.goal_pos - self.robot_pos))
        reward = self._reward(dist, speed, task)
        self.prev_dist = dist

        terminated = dist < 0.5
        truncated  = self.current_step >= self.max_episode_steps
        return self._obs(), reward, terminated, truncated, {}

    def _reward(self, dist, speed, task):
        return (10.0 * (self.prev_dist - dist)
              + (-0.5 * speed if task["is_heavy"] else 0.0)
              + (-2.0 if task["is_fragile"] and speed > 0.5 else 0.0)
              + (100.0 if dist < 0.5 else 0.0)
              - 0.1)

    def _obs(self):
        o = np.array([
            self.robot_pos[0], self.robot_pos[1],
            self.goal_pos[0],  self.goal_pos[1],
            float(np.linalg.norm(self.goal_pos - self.robot_pos)),
            self.velocity[0],  self.velocity[1],
        ], dtype=np.float32)
        if self.noise_sigma > 0:
            o += np.random.normal(0, self.noise_sigma, size=o.shape).astype(np.float32)
            o  = np.clip(o, self.observation_space.low, self.observation_space.high)
        return o
