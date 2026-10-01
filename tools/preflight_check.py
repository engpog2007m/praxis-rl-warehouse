#!/usr/bin/env python3
"""Verify the simulator is genuinely healthy before any training or evaluation.

Every check here corresponds to a failure observed during the research that
produced silent, plausible-looking, invalid results:

  1. single_instance  Two gz sim processes compete for the same topics and
                      services; runs appeared healthy while reading from a
                      wedged instance.
  2. topics           A mistyped bridge argument (a trailing '~') dropped the
                      pose bridge; geometry silently read zeros.
  3. ground_truth     Pose messages must arrive and be non-degenerate.
  4. lidar_rate       Scan must publish near its configured 20 Hz.
  5. physical_motion  THE critical check. Commands forward velocity and
                      measures displacement from Gazebo ground truth, never
                      from odometry. Odometry integrates commanded velocity
                      and reported normal travel while the robot moved 0.5 mm.

Usage:
  python3 tools/preflight_check.py              # lidar world (default)
  python3 tools/preflight_check.py --mode final # 7-dim corrected world
  python3 tools/preflight_check.py --mode legacy
                                                # defect world: motion check
                                                # is expected to FAIL and is
                                                # reported, not enforced

Exit status 0 only if every enforced check passes.
"""
import argparse
import subprocess
import sys
import threading
import time

WORLD = 'praxis_warehouse'
POSE_TOPIC = f'/world/{WORLD}/dynamic_pose/info'
# Index of each model in the Pose_V -> TFMessage bridge. The bridge drops
# entity names, so order follows model declaration order in the SDF.
MODEL_IDX = {'amr_1': 0, 'amr_2': 1}
TEST_POS = (8.0, 18.0)     # open corridor between shelf bands
SPAWN = {'amr_1': (2.0, 2.0), 'amr_2': (4.0, 2.0)}
DRIVE_V = 0.4              # m/s, full command after the 0.4 action scaling
DRIVE_S = 2.5              # seconds
MIN_DISP = 0.6             # metres; healthy physics gives about 1.0 m

results = []


def record(name, ok, detail, enforced=True):
    results.append((name, ok, detail, enforced))
    tag = 'PASS' if ok else ('FAIL' if enforced else 'NOTE')
    print(f'  [{tag}] {name:<16} {detail}', flush=True)


GZ_PAT = r'(^|/)gz sim '   # anchored: never match shells/editors mentioning it


def gz_sim_instances():
    try:
        out = subprocess.run(['pgrep', '-af', GZ_PAT], capture_output=True,
                             text=True, timeout=5).stdout
    except Exception:
        return []
    return [ln for ln in out.splitlines() if ln.strip()]


def teleport(name, x, y):
    req = (f"name: '{name}', position: {{x: {x}, y: {y}, z: 0.08}}, "
           f"orientation: {{w: 1.0}}")
    try:
        out = subprocess.run(
            ['gz', 'service', '-s', f'/world/{WORLD}/set_pose',
             '--reqtype', 'gz.msgs.Pose', '--reptype', 'gz.msgs.Boolean',
             '--timeout', '3000', '--req', req],
            capture_output=True, text=True, timeout=6).stdout
        return 'true' in out
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['lidar', 'final', 'legacy'], default='lidar')
    args = ap.parse_args()
    print(f'Preflight check (mode={args.mode})', flush=True)

    inst = gz_sim_instances()
    record('single_instance', len(inst) == 1,
           f'{len(inst)} gz sim process(es)' +
           ('' if len(inst) == 1 else ' -- run tools/stop_sim.sh then tools/launch_sim.sh'))
    if len(inst) != 1:
        return summarise()

    import rclpy
    from rclpy.node import Node
    from geometry_msgs.msg import Twist
    from sensor_msgs.msg import LaserScan
    from tf2_msgs.msg import TFMessage
    import numpy as np

    rclpy.init()

    class Probe(Node):
        def __init__(self):
            super().__init__('praxis_preflight')
            self.pose = {n: None for n in MODEL_IDX}
            self.n_pose = 0
            self.n_scan = 0
            self.pub = self.create_publisher(Twist, '/amr_1/cmd_vel', 10)
            self.create_subscription(TFMessage, POSE_TOPIC, self._pose_cb, 10)
            self.create_subscription(LaserScan, '/amr_1/scan', self._scan_cb, 10)

        def _pose_cb(self, msg):
            self.n_pose += 1
            for name, idx in MODEL_IDX.items():
                if idx < len(msg.transforms):
                    t = msg.transforms[idx].transform.translation
                    self.pose[name] = np.array([t.x, t.y])

        def _scan_cb(self, _msg):
            self.n_scan += 1

        def drive(self, vx):
            m = Twist()
            m.linear.x = float(vx)
            self.pub.publish(m)

    node = Probe()
    ex = rclpy.executors.SingleThreadedExecutor()
    ex.add_node(node)
    th = threading.Thread(target=ex.spin, daemon=True)
    th.start()

    try:
        # ---- topics
        time.sleep(1.0)
        names = {n for n, _ in node.get_topic_names_and_types()}
        need = ['/amr_1/cmd_vel', '/amr_1/odom', POSE_TOPIC]
        if args.mode == 'lidar':
            need += ['/amr_1/scan', '/amr_2/scan']
        missing = [t for t in need if t not in names]
        record('topics', not missing,
               'all required topics present' if not missing
               else f'missing {missing} -- check bridge arguments for typos')

        # ---- ground truth
        n0 = node.n_pose
        time.sleep(2.0)
        rate = (node.n_pose - n0) / 2.0
        p = node.pose['amr_1']
        ok = rate > 10 and p is not None and float(np.linalg.norm(p)) > 0.1
        record('ground_truth', ok,
               f'{rate:.0f} msg/s, amr_1 at {None if p is None else np.round(p, 2)}')

        # ---- lidar
        if args.mode == 'lidar':
            s0 = node.n_scan
            time.sleep(2.0)
            srate = (node.n_scan - s0) / 2.0
            record('lidar_rate', srate >= 15, f'{srate:.1f} Hz (expect ~20)')

        # ---- physical motion (ground truth only; never odometry)
        t0 = time.time()
        while node.pub.get_subscription_count() == 0 and time.time() - t0 < 5:
            time.sleep(0.1)
        if node.pub.get_subscription_count() == 0 or node.pose['amr_1'] is None:
            record('physical_motion', False, 'cannot test: no cmd_vel subscriber or no pose',
                   enforced=args.mode != 'legacy')
        else:
            if not teleport('amr_1', *TEST_POS):
                record('physical_motion', False, 'teleport via set_pose failed',
                       enforced=args.mode != 'legacy')
            else:
                time.sleep(1.0)
                start = node.pose['amr_1'].copy()
                t_end = time.time() + DRIVE_S
                while time.time() < t_end:
                    node.drive(DRIVE_V)
                    time.sleep(0.05)
                node.drive(0.0)
                time.sleep(0.6)
                disp = float(np.linalg.norm(node.pose['amr_1'] - start))
                expected = DRIVE_V * DRIVE_S
                ok = disp >= MIN_DISP
                detail = (f'moved {disp:.3f} m of ~{expected:.1f} m commanded '
                          f'(ground truth)')
                if not ok:
                    detail += (' -- robot is not physically moving; see '
                               'KNOWN_ISSUES.md issue 1')
                if args.mode == 'legacy':
                    detail += ' [expected on legacy defect world]'
                record('physical_motion', ok, detail,
                       enforced=args.mode != 'legacy')
                teleport('amr_1', *SPAWN['amr_1'])
    finally:
        node.drive(0.0)
        ex.shutdown()
        node.destroy_node()
        rclpy.shutdown()

    return summarise()


def summarise():
    enforced_fail = [r for r in results if r[3] and not r[1]]
    print()
    if enforced_fail:
        print(f'PREFLIGHT FAILED ({len(enforced_fail)} check(s)). '
              f'Do not start a run.', flush=True)
        return 1
    print('PREFLIGHT PASSED. Safe to start a run.', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
