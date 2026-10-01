"""
GazeboWarehouseEnv — Gymnasium environment backed by Gazebo Harmonic physics.
"""
import os, sys, math, time, threading
import numpy as np
import gymnasium as gym
from gymnasium import spaces

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry

sys.path.insert(0, os.path.join(os.environ.get('PRAXIS_ROOT', os.path.expanduser('~/Praxis')), 'src'))
from simulation.data_registry import DataRegistry

_RCLPY_STARTED = False
_EXECUTOR = None
_SPIN_THREAD = None


def _ensure_rclpy():
    global _RCLPY_STARTED, _EXECUTOR, _SPIN_THREAD
    if not _RCLPY_STARTED:
        rclpy.init()
        _EXECUTOR = rclpy.executors.MultiThreadedExecutor(num_threads=4)
        _SPIN_THREAD = threading.Thread(target=_EXECUTOR.spin, daemon=True)
        _SPIN_THREAD.start()
        _RCLPY_STARTED = True
        time.sleep(2.0)
    return _EXECUTOR


class _RobotBridge(Node):
    def __init__(self, robot='amr_1'):
        super().__init__(f'{robot}_train_bridge')
        self.pos = np.zeros(2, dtype=np.float64)
        self.vel = np.zeros(2, dtype=np.float64)
        self.got_odom = False
        self.pub = self.create_publisher(Twist, f'/{robot}/cmd_vel', 10)
        self.create_subscription(Odometry, f'/{robot}/odom', self._cb, 10)

    def _cb(self, msg):
        self.pos[0] = msg.pose.pose.position.x
        self.pos[1] = msg.pose.pose.position.y
        self.vel[0] = msg.twist.twist.linear.x
        self.vel[1] = msg.twist.twist.linear.y
        self.got_odom = True

    def drive(self, vx, vy):
        m = Twist()
        m.linear.x = float(np.clip(vx, -0.4, 0.4))
        m.linear.y = float(np.clip(vy, -0.4, 0.4))
        self.pub.publish(m)

    def halt(self):
        self.drive(0.0, 0.0)


class GazeboWarehouseEnv(gym.Env):
    metadata = {"render_modes": []}

    GOAL_OFFSETS = {
        1:    np.array([3.0, 0.0]),
        2:    np.array([3.5, 0.0]),
        3:    np.array([2.5, 0.0]),
        4:    np.array([1.2, 0.0]),
        1001: np.array([1.5, 0.0]),
    }

    def __init__(self, robot='amr_1', task_id=1,
                 noise_sigma=0.05, mass_multiplier=1.0,
                 control_hz=50.0, max_steps=200):
        super().__init__()
        _ensure_rclpy()
        self.registry  = DataRegistry()
        self.robot     = robot
        self.task_id   = task_id
        self.noise     = noise_sigma
        self.mass_mult = mass_multiplier
        self.dt        = 1.0 / control_hz
        self.max_steps = max_steps

        self.bridge = _RobotBridge(robot)
        _EXECUTOR.add_node(self.bridge)

        t0 = time.time()
        while not self.bridge.got_odom and time.time() - t0 < 10.0:
            time.sleep(0.05)

        self.origin     = self.bridge.pos.copy()
        self.goal_rel   = self.GOAL_OFFSETS[task_id].copy()
        self._step_n    = 0
        self._prev_dist = None
        self._direction = 1

        self.observation_space = spaces.Box(
            low=np.array([-10, -10, -10, -10, 0, -1, -1], np.float32),
            high=np.array([10, 10, 10, 10, 20, 1, 1], np.float32))
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.bridge.halt()
        time.sleep(0.15)
        self.origin = self.bridge.pos.copy()
        self._direction *= -1
        self.goal_rel = self.GOAL_OFFSETS[self.task_id] * self._direction
        self._step_n = 0
        self._prev_dist = float(np.linalg.norm(self.goal_rel))
        return self._obs(), {}

    def step(self, action):
        task  = self.registry.get_task(self.task_id)
        speed = float(np.linalg.norm(action[:2]))
        vx = float(action[0]) * 0.4
        vy = float(action[1]) * 0.4
        if task['is_heavy'] and self.mass_mult > 1.25:
            vx *= 0.75
            vy *= 0.75
        self.bridge.drive(vx, vy)
        time.sleep(self.dt)

        rel  = self.bridge.pos - self.origin
        dist = float(np.linalg.norm(self.goal_rel - rel))
        r_prog   = 10.0 * (self._prev_dist - dist)
        c_kin    = -0.5 * speed if task['is_heavy'] else 0.0
        c_safe   = -0.5 if (task['is_fragile'] and speed > 0.5) else 0.0
        r_arrive = 150.0 if dist < 0.8 else 0.0
        reward   = r_prog + c_kin + c_safe + r_arrive - 0.1

        self._prev_dist = dist
        self._step_n += 1
        terminated = dist < 0.8
        truncated  = self._step_n >= self.max_steps
        if terminated or truncated:
            self.bridge.halt()
        return self._obs(), reward, terminated, truncated, {'distance': dist}

    def _obs(self):
        rel  = self.bridge.pos - self.origin
        dist = float(np.linalg.norm(self.goal_rel - rel))
        o = np.array([rel[0], rel[1],
                      self.goal_rel[0], self.goal_rel[1],
                      dist, self.bridge.vel[0], self.bridge.vel[1]],
                     dtype=np.float32)
        if self.noise > 0:
            o += np.random.normal(0, self.noise, 7).astype(np.float32)
        return np.clip(o, self.observation_space.low,
                       self.observation_space.high)

    def close(self):
        try:
            self.bridge.halt()
        except Exception:
            pass
