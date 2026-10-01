"""
GazeboWarehouseEnv — Gymnasium environment backed by Gazebo Harmonic physics.
"""
import os, sys, math, time, threading, subprocess, re
import numpy as np
import gymnasium as gym
from gymnasium import spaces

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from tf2_msgs.msg import TFMessage

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
    N_SECTORS = 24
    RANGE_MAX = 12.0

    def __init__(self, robot='amr_1'):
        super().__init__(f'{robot}_train_bridge')
        self.pos = np.zeros(2, dtype=np.float64)
        self.vel = np.zeros(2, dtype=np.float64)
        self.got_odom = False
        # 24-sector LiDAR, normalised to [0,1]; 1.0 == nothing within range
        self.scan = np.ones(self.N_SECTORS, dtype=np.float32)
        self.got_scan = False
        self.pub = self.create_publisher(Twist, f'/{robot}/cmd_vel', 10)
        self.create_subscription(Odometry, f'/{robot}/odom', self._cb, 10)
        self.create_subscription(LaserScan, f'/{robot}/scan', self._scan_cb, 10)

    def _scan_cb(self, msg):
        v = np.asarray(msg.ranges, dtype=np.float32)
        v = np.nan_to_num(v, posinf=self.RANGE_MAX, nan=self.RANGE_MAX)
        v = np.clip(v, 0.0, self.RANGE_MAX)
        n = len(v)
        if n % self.N_SECTORS == 0:
            # per-sector minimum preserves the nearest-obstacle signal
            self.scan = (v.reshape(self.N_SECTORS, n // self.N_SECTORS)
                          .min(axis=1) / self.RANGE_MAX).astype(np.float32)
        else:
            idx = np.linspace(0, n, self.N_SECTORS + 1).astype(int)
            self.scan = np.array([v[idx[i]:idx[i+1]].min()
                                  for i in range(self.N_SECTORS)],
                                 dtype=np.float32) / self.RANGE_MAX
        self.got_scan = True

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


# World spawn positions from praxis_multiagent_lidar.sdf. Odometry is
# reported relative to each robot's own spawn, so physical separation
# requires adding this offset.
class _WorldPose(Node):
    """Ground-truth world poses, subscribed from Gazebo's dynamic pose feed.

    The differential-drive plugin integrates commanded wheel velocity and
    accumulates without bound; it does not measure physical displacement.
    Observed drift of 5 m and 14 m while both robots were stationary at
    spawn confirmed odometry is unusable for inter-robot geometry.

    Gazebo publishes /world/<w>/dynamic_pose/info as gz.msgs.Pose_V. The
    ros_gz bridge maps this to tf2_msgs/TFMessage but drops the entity
    name field, so entries are identified by index. Order follows entity
    creation in the SDF: model poses first, then link poses. Indices 0 and
    1 are amr_1 and amr_2 respectively; this holds while the model
    declaration order in praxis_multiagent_lidar.sdf is unchanged.
    """
    TOPIC     = '/world/praxis_warehouse/dynamic_pose/info'
    MODEL_IDX = {'amr_1': 0, 'amr_2': 1}

    def __init__(self):
        super().__init__('world_pose_reader')
        self.pose = {n: np.zeros(2) for n in self.MODEL_IDX}
        self.got  = False
        self.create_subscription(TFMessage, self.TOPIC, self._cb, 10)

    def _cb(self, msg):
        tf = msg.transforms
        for name, idx in self.MODEL_IDX.items():
            if idx < len(tf):
                v = tf[idx].transform.translation
                self.pose[name][0] = v.x
                self.pose[name][1] = v.y
        self.got = True

    def refresh(self, force=False):
        return self.pose


class _PeerDriver:
    """Drives amr_2 as a moving obstacle during amr_1's episodes.

    The peer follows a DWB-style proportional controller toward randomly
    reassigned goals, so it behaves as an independently tasked warehouse
    robot rather than a scripted obstacle. Following Li et al. (2024),
    a dynamic peer provides the failure mode that a safety-shaped reward
    is intended to prevent; static geometry alone cannot.
    """
    APPROACH_RADIUS = 0.8
    MAX_SPEED       = 0.9
    MIN_SPEED       = 0.35
    GOAL_RANGE      = 1.2      # metres; keeps the peer in the ego's corridor
    REPLAN_DIST     = 0.6      # reassign goal within this distance

    def __init__(self, bridge, seed=0):
        self.b   = bridge
        self.rng = np.random.default_rng(seed)
        self.origin = bridge.pos.copy()
        self.goal   = self._new_goal()

    def _new_goal(self):
        # Bias goals lateral to the ego's travel axis so the peer crosses
        # rather than parallels it. Ego goals are +/- x; peer goals favour y.
        ang = self.rng.uniform(-np.pi, np.pi)
        rad = self.rng.uniform(0.5, self.GOAL_RANGE)
        v = np.array([rad*np.cos(ang), rad*np.sin(ang)])
        v[1] *= 1.5            # lateral motion across the ego's path
        v[0] *= 0.3            # limited along-track drift
        return self.origin + v

    def reset(self):
        self.origin = self.b.pos.copy()
        self.goal   = self._new_goal()

    def step(self):
        d = self.goal - self.b.pos
        n = float(np.linalg.norm(d))
        if n < self.REPLAN_DIST:
            self.goal = self._new_goal()
            d = self.goal - self.b.pos
            n = max(float(np.linalg.norm(d)), 1e-6)
        s = (self.MAX_SPEED if n > self.APPROACH_RADIUS
             else max(self.MIN_SPEED, self.MAX_SPEED * n / self.APPROACH_RADIUS))
        self.b.drive(s * d[0] / n * 0.4, s * d[1] / n * 0.4)


class GazeboWarehouseEnv(gym.Env):
    metadata = {"render_modes": []}

    # ---- parameters derived from measured kinematics -----------------
    # Action-to-velocity is linear: v = action * 0.4, giving 0.403 m/s at
    # full command. Measured on the corrected physics model (wheel friction
    # mu=1.2, chassis raised clear of the floor) using Gazebo ground-truth
    # poses rather than integrated odometry.
    #
    #   action 0.5 -> 0.202 m/s -> 497 steps for 2.0 m  (compliant)
    #   action 1.0 -> 0.403 m/s -> 249 steps for 2.0 m  (violating)
    #
    # Both fit the 500-step budget, so constraint compliance and task
    # completion are simultaneously achievable and the policy faces a
    # genuine speed-versus-compliance trade-off.
    EGO_START         = (8.0, 18.0)   # open corridor between shelf bands
    ARRIVAL_RADIUS    = 0.5     # m
    FRAGILE_THRESHOLD = 0.5     # action norm; 0.202 m/s
    HEAVY_THRESHOLD   = 0.7     # action norm; 0.283 m/s
    PROX_ENVELOPE     = 1.2     # m; graded peer-proximity cost within

    # Uniform 2.0 m goal distance across all payload profiles, so that
    # profiles differ only in their constraints. Derived from measured
    # kinematics on the corrected physics model: 0.403 m/s at full command,
    # giving 249 steps at full speed and 497 steps at the compliant
    # threshold of 0.5, both inside the 500-step budget.
    GOAL_OFFSETS = {
        1:    np.array([2.0, 0.0]),
        2:    np.array([2.0, 0.0]),
        3:    np.array([2.0, 0.0]),
        4:    np.array([2.0, 0.0]),
        1001: np.array([2.0, 0.0]),
    }

    def __init__(self, robot='amr_1', task_id=1,
                 noise_sigma=0.05, mass_multiplier=1.0,
                 control_hz=50.0, max_steps=500,
                 peer_enabled=True, collision_dist=0.30,
                 collision_penalty=-100.0, proximity_weight=0.0,
                 freeze_penalty=-1.0, velocity_weight=0.2,
                 freeze_speed=0.05):
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

        # dynamic peer (amr_2) acting as a moving obstacle
        self.robot = robot
        self.peer_enabled = peer_enabled
        self.peer = None
        self.peer_name = None
        self.peer_driver = None
        if self.peer_enabled:
            peer_name = 'amr_2' if robot == 'amr_1' else 'amr_1'
            self.peer_name = peer_name
            self.peer = _RobotBridge(peer_name)
            _EXECUTOR.add_node(self.peer)

        self.collision_dist    = collision_dist
        self.collision_penalty = collision_penalty
        self.proximity_weight  = proximity_weight
        self.n_collisions      = 0
        self.freeze_penalty    = freeze_penalty
        self.velocity_weight   = velocity_weight
        self.FREEZE_SPEED      = freeze_speed
        self._debug_steps      = 0

        t0 = time.time()
        while not (self.bridge.got_odom and self.bridge.got_scan) \
                and time.time() - t0 < 15.0:
            time.sleep(0.05)
        if not self.bridge.got_scan:
            print('WARNING: no LiDAR data received; scan will read as clear',
                  flush=True)
        if self.peer is not None:
            t1 = time.time()
            while not self.peer.got_odom and time.time() - t1 < 10.0:
                time.sleep(0.05)
            self.peer_driver = _PeerDriver(self.peer, seed=17)
        self._world = _WorldPose()
        _EXECUTOR.add_node(self._world)
        t2 = time.time()
        while not self._world.got and time.time() - t2 < 10.0:
            time.sleep(0.05)
        if not self._world.got:
            print('WARNING: no world-pose data; separation will read as zero',
                  flush=True)

        self.origin     = self.ego_pos()
        self.goal_rel   = self.GOAL_OFFSETS[task_id].copy()
        self._step_n    = 0
        self._prev_dist = None
        self._direction = 1

        # 7 pose/goal terms + 24 normalised LiDAR sectors
        lo = np.concatenate([np.array([-10,-10,-10,-10, 0,-1,-1], np.float32),
                             np.zeros(_RobotBridge.N_SECTORS, np.float32)])
        hi = np.concatenate([np.array([ 10, 10, 10, 10,20, 1, 1], np.float32),
                             np.ones(_RobotBridge.N_SECTORS, np.float32)])
        self.observation_space = spaces.Box(low=lo, high=hi, dtype=np.float32)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.bridge.halt()
        time.sleep(0.15)
        self.origin = self.ego_pos()
        self._direction *= -1
        self.goal_rel = self.GOAL_OFFSETS[self.task_id] * self._direction
        self._step_n = 0
        self._prev_dist = float(np.linalg.norm(self.goal_rel))
        self.n_collisions = 0

        if self.peer_driver is not None:
            # Reposition both robots each episode so the peer lies across
            # the ego's route. Without this the two drift apart over a
            # training run and the collision failure mode never activates.
            self.peer.halt()
            self.bridge.halt()
            time.sleep(0.1)

            ex, ey = self.EGO_START
            gx = ex + float(self.goal_rel[0])
            gy = ey + float(self.goal_rel[1])
            # peer placed near the midpoint, offset laterally so it must be
            # passed rather than sitting exactly on the goal
            mx, my = (ex + gx) / 2.0, (ey + gy) / 2.0
            off = self.np_random.uniform(-0.75, 0.75)

            self._teleport(self.robot, ex, ey)
            self._teleport(self.peer_name, mx, my + off)
            time.sleep(0.35)

            self.origin = self.ego_pos()
            self._prev_dist = float(np.linalg.norm(self.goal_rel))
            self.peer_driver.reset()

        return self._obs(), {}

    def step(self, action):
        task  = self.registry.get_task(self.task_id)
        speed = float(np.linalg.norm(action[:2]))

        vx = float(action[0]) * 0.4
        vy = float(action[1]) * 0.4
        if task['is_heavy'] and self.mass_mult > 1.25:
            vx *= 0.75
            vy *= 0.75

        # advance the dynamic peer alongside the ego robot
        if self.peer_driver is not None:
            self.peer_driver.step()

        self.bridge.drive(vx, vy)
        time.sleep(self.dt)

        rel  = self.ego_pos() - self.origin
        dist = float(np.linalg.norm(self.goal_rel - rel))

        # --- peer proximity and collision (Li et al. 2024) ---
        sep = self.peer_separation()
        collided = sep < self.collision_dist
        if collided:
            self.n_collisions += 1

        r_prox = 0.0
        if self.proximity_weight > 0.0 and sep < self.PROX_ENVELOPE:
            r_prox = -self.proximity_weight * (self.PROX_ENVELOPE - sep) \
                     / self.PROX_ENVELOPE

        # Measured velocity, not commanded action. A policy pressed against
        # the peer or a shelf still commands motion, so the anti-freeze term
        # must read what the robot actually does.
        speed_actual = float(np.linalg.norm(self.bridge.vel))

        r_prog   = 10.0 * (self._prev_dist - dist)
        c_kin    = -0.5 * speed if task['is_heavy'] else 0.0
        c_safe   = -0.5 if (task['is_fragile'] and
                            speed > self.FRAGILE_THRESHOLD) else 0.0
        r_arrive = 150.0 if dist < self.ARRIVAL_RADIUS else 0.0
        r_coll   = self.collision_penalty if collided else 0.0

        # Anti-freeze: standing still previously dominated, because the
        # accumulated step cost was cheaper than the expected collision
        # penalty. A direct standstill penalty plus a small velocity reward
        # removes that local optimum.
        r_freeze = self.freeze_penalty if speed_actual < self.FREEZE_SPEED else 0.0
        r_vel    = self.velocity_weight * speed_actual

        reward = (r_prog + c_kin + c_safe + r_arrive + r_prox
                  + r_coll + r_freeze + r_vel - 0.1)

        if self._debug_steps < 20:
            print(f"[DEBUG] t={self._step_n:3d} cmd={speed:.3f} "
                  f"act={speed_actual:.3f} prog={r_prog:+.2f} "
                  f"frz={r_freeze:+.1f} vel={r_vel:+.2f} "
                  f"coll={r_coll:+.1f} R={reward:+.2f}", flush=True)
            self._debug_steps += 1

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

    def _teleport(self, name, x, y):
        """Place a model at a world position via Gazebo's set_pose service."""
        req = (f"name: '{name}', position: {{x: {x:.3f}, y: {y:.3f}, z: 0.08}}, "
               f"orientation: {{w: 1.0}}")
        try:
            subprocess.run(['gz', 'service', '-s',
                            '/world/praxis_warehouse/set_pose',
                            '--reqtype', 'gz.msgs.Pose',
                            '--reptype', 'gz.msgs.Boolean',
                            '--timeout', '2000', '--req', req],
                           capture_output=True, timeout=3)
        except Exception:
            pass

    def ego_pos(self):
        """Ego position in world coordinates (Gazebo ground truth).
    
        Odometry integrates commanded wheel velocity, does not reset
        on teleport, and diverged to 366 m while the robot stood at
        (8, 18). All episode geometry therefore uses world poses.
        """
        return self._world.pose[self.robot].copy()
    
    def peer_separation(self):
        """Physical distance between ego and peer in world coordinates."""
        if self.peer is None:
            return float('inf')
        p = self._world.refresh()
        return float(np.linalg.norm(p[self.robot] - p[self.peer_name]))

    def _obs(self):
        rel  = self.ego_pos() - self.origin
        dist = float(np.linalg.norm(self.goal_rel - rel))
        pose = np.array([rel[0], rel[1],
                         self.goal_rel[0], self.goal_rel[1],
                         dist, self.bridge.vel[0], self.bridge.vel[1]],
                        dtype=np.float32)
        if self.noise > 0:
            # sensor noise perturbs pose/velocity estimation; the LiDAR
            # carries its own Gaussian noise model in the SDF sensor block
            pose += np.random.normal(0, self.noise, 7).astype(np.float32)
        o = np.concatenate([pose, self.bridge.scan]).astype(np.float32)
        return np.clip(o, self.observation_space.low,
                       self.observation_space.high)

    def close(self):
        try:
            self.bridge.halt()
        except Exception:
            pass
